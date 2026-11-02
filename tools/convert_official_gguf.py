"""Stream verified official FP8 shards into native Q8_0 and Q4_K expert packages.

Every model tensor, including the full output head, comes from the new checkpoint.
Only the unchanged codec and embedded runtime resources use a verified template.
No complete floating-point intermediate or second source bank is allocated.
"""
from contextlib import ExitStack
import ctypes
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / '.cache/drama3-update-20261031'
WORK.mkdir(parents=True, exist_ok=True)
for key in ('TEMP', 'TMP', 'TMPDIR', 'HF_HOME', 'XDG_CACHE_HOME', 'TORCH_HOME'):
    os.environ[key] = str(WORK)
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / '.cache/q8-tools/python')]
import numpy as np
import torch
from gguf import GGMLQuantizationType as Q
from gguf.quants import dequantize, quantize
from drama_low_vram.checkpoint_reader import iter_readonly_checkpoint_tensors
from drama_low_vram.gguf_archive import Archive, BLOCKS, scalar_metadata, write_header

SOURCE = ROOT / 'checkpoints/drama-3-20261029-fp8'
TEMPLATE = ROOT / 'checkpoints/drama-3-gguf-q8/drama3-experts-Q8_0-bf16.gguf'
EXPERT = re.compile(r'model_weights/(model\.language_model\.layers\.\d+\.mlp\.experts)\.(gate_proj|up_proj|down_proj)\.weight$')


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, path)


def digest_file(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


class Quantizer:
    def __init__(self):
        self.library = ctypes.CDLL(str(WORK / 'quantizer.dll'))
        self.run = self.library.drama_quantize_rows
        self.run.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                             ctypes.c_int64, ctypes.c_int64, ctypes.c_int]
        self.run.restype = ctypes.c_int

    def __call__(self, data, kind):
        data = np.ascontiguousarray(data, dtype=np.float32)
        block, size = BLOCKS[kind]
        result = np.empty((*data.shape[:-1], data.shape[-1] // block * size), dtype=np.uint8)
        if self.run(kind, data.ctypes.data, result.ctypes.data,
                    data.size // data.shape[-1], data.shape[-1], 8):
            raise RuntimeError('Native reference quantizer failed')
        return result


def expert_groups(values, prefix, projection):
    for start in range(0, 256, 4):
        group = []
        for expert in range(start, min(start + 4, 256)):
            name = f'{prefix}.{expert}.{projection}.weight'
            weight, scale = values[name], values[name + '_scale_inv']
            rows, cols = weight.shape
            if scale.shape != (rows // 128, cols // 128):
                raise ValueError('FP8 block scale geometry mismatch')
            group.append((weight.float().view(rows // 128, 128, cols // 128, 128)
                          * scale[:, None, :, None]).reshape(rows, cols))
        data = torch.stack(group).numpy()
        if not np.isfinite(data).all():
            raise ValueError('Nonfinite official FP8 tensor')
        yield data


def main():
    torch.set_num_threads(8)
    receipt = json.loads((SOURCE / 'download-verification.json').read_text(encoding='utf-8'))
    if receipt['equivalent_huggingface_weight_revision'] != '78e5ab79e49aa9a25abe8799eff2c4d73a1c3216':
        raise ValueError('Unexpected source revision')
    if not all(row['verified'] for row in receipt['files']):
        raise ValueError('Unverified download')
    expected_template = json.loads(TEMPLATE.with_suffix('.conversion.json').read_text(encoding='utf-8'))
    print('Verifying template package SHA256', flush=True)
    if digest_file(TEMPLATE) != expected_template['output_sha256']:
        raise ValueError('Template package changed')
    if next(row['sha256'] for row in receipt['files'] if row['name'] == 'codec.pth') != '74fc41c5a7151c6f350af8bd7e5d6e3accfcc7f3dfbfac23afd35af07052bb2f':
        raise ValueError('Codec changed; template reuse is forbidden')
    native = Quantizer()
    probe = np.random.default_rng(123).normal(size=(16, 2048)).astype(np.float32)
    if not np.array_equal(native(probe, 8), quantize(probe, Q.Q8_0)):
        raise ValueError('Native Q8 reference parity failed')
    template = Archive(TEMPLATE)
    values = dict(iter_readonly_checkpoint_tensors(SOURCE))
    expected = set()
    targets, plans, reports = {}, {}, {}
    total_required = 0
    for kind, label in [(8, 'q8'), (12, 'q4')]:
        target = ROOT / f'checkpoints/drama-3-20261029-gguf-{label}' / f'drama3-20261029-experts-{("Q8_0" if kind == 8 else "Q4_K_S")}-bf16.gguf'
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.with_suffix('.gguf.partial').exists():
            raise FileExistsError(f'Output already exists: {target}')
        plan = []
        for tensor in template.tensors:
            match = EXPERT.fullmatch(tensor.logical)
            output_kind = kind if match else tensor.kind
            if tensor.logical.startswith('model_weights/'):
                if match:
                    prefix, projection = match.groups()
                    for expert in range(tensor.shape[0]):
                        name = f'{prefix}.{expert}.{projection}.weight'
                        if tuple(values[name].shape) != tensor.shape[1:]:
                            raise ValueError(f'Source shape mismatch: {name}')
                        expected.update((name, name + '_scale_inv'))
                else:
                    name = tensor.logical.removeprefix('model_weights/')
                    shape = tuple(values[name].shape)
                    singleton_gate = (name.endswith('.mlp.shared_expert_gate.weight')
                                      and shape == (1, *tensor.shape))
                    if (shape != tensor.shape and not singleton_gate) or values[name].dtype not in (torch.float32, torch.bfloat16, torch.float16):
                        raise ValueError(f'Source shape/dtype mismatch: {name}')
                    expected.add(name)
            block, size = BLOCKS[output_kind]
            plan.append(replace(tensor, kind=output_kind, offset=0,
                                nbytes=int(np.prod(tensor.shape)) // block * size))
        total_required += sum(row.nbytes + 32 for row in plan) + 128 * 1024**2
        targets[kind], plans[kind] = target, plan
        reports[kind] = dict(status='planned', source_revision=receipt['equivalent_huggingface_weight_revision'],
                             source_directory=str(SOURCE), source_files=receipt['files'], output=str(target),
                             template=str(TEMPLATE), template_sha256=expected_template['output_sha256'],
                             template_reuse='Unchanged codec and embedded tokenizer/config runtime assets only',
                             recipe=f'{Q(kind).name} experts from original FP8/scales; source BF16 dense/tower/full head; FP32 routers and vectors; template FP16 codec',
                             large_intermediate_created=False, tensors=[])
    if set(values) != expected:
        raise ValueError(f'Source tensor coverage mismatch: {set(values) ^ expected}')
    if shutil.disk_usage(ROOT).free < total_required + 2 * 1024**3:
        raise RuntimeError('Insufficient disk for both outputs plus safety margin')
    print(json.dumps(dict(event='plan', source_tensors=len(values), output_tensors=len(template.tensors), bytes=total_required)), flush=True)
    started = time.monotonic()
    with ExitStack() as stack:
        copied = stack.enter_context(TEMPLATE.open('rb'))
        outputs, sizes = {}, {}
        for kind, target in targets.items():
            metadata = [(name, raw) for name, raw in template.metadata if name not in {
                'general.name', 'audiocpp.weight_type', 'audiocpp.source_format', 'drama3.source_sha256'}]
            for name, value in {
                'general.name': f'Drama 3 2026-10-29 {Q(kind).name} experts BF16 backbone/tower/head',
                'audiocpp.weight_type': Q(kind).name.lower(),
                'audiocpp.source_format': 'official-fp8/bf16',
                'drama3.source_repository': 'fishaudio/drama-3',
                'drama3.source_revision': receipt['equivalent_huggingface_weight_revision'],
            }.items():
                metadata.append((name, scalar_metadata(name, value)))
            outputs[kind] = stack.enter_context(target.with_suffix('.gguf.partial').open('xb'))
            sizes[kind] = write_header(outputs[kind], metadata, plans[kind], alignment=template.alignment)
        for index, original in enumerate(template.tensors):
            match = EXPERT.fullmatch(original.logical)
            hashes = {kind: hashlib.sha256() for kind in targets}
            began = time.monotonic()
            errors = {kind: [0., 0.] for kind in targets}
            for kind in outputs:
                outputs[kind].seek(plans[kind][index].offset)
            if match:
                for data in expert_groups(values, *match.groups()):
                    for kind, output in outputs.items():
                        packed = native(data, kind)
                        packed.tofile(output); hashes[kind].update(packed)
                        sample = data.reshape(-1, data.shape[-1])[:1]
                        restored = dequantize(packed.reshape(-1, packed.shape[-1])[:1], Q(kind))
                        errors[kind][0] += float(np.sum((sample.astype(np.float64) - restored) ** 2))
                        errors[kind][1] += float(np.sum(sample.astype(np.float64) ** 2))
            elif original.logical.startswith('model_weights/'):
                value = values[original.logical.removeprefix('model_weights/')].reshape(original.shape)
                data = (value.float().numpy() if original.kind == 0 else
                        value.to(torch.bfloat16).contiguous().view(torch.uint16).numpy() if original.kind == 30 else
                        value.half().numpy())
                for kind, output in outputs.items():
                    if data.nbytes != plans[kind][index].nbytes:
                        raise ValueError('Dense output byte count mismatch')
                    data.tofile(output); hashes[kind].update(data)
            else:
                copied.seek(original.offset)
                remaining = original.nbytes
                while remaining:
                    data = copied.read(min(8 * 1024**2, remaining))
                    if not data: raise EOFError('Template truncated')
                    for kind, output in outputs.items():
                        output.write(data); hashes[kind].update(data)
                    remaining -= len(data)
            for kind, output in outputs.items():
                if output.tell() != plans[kind][index].offset + plans[kind][index].nbytes:
                    raise ValueError('Written tensor length mismatch')
                reports[kind]['tensors'].append(dict(name=original.logical, kind=plans[kind][index].kind,
                    bytes=plans[kind][index].nbytes, sha256=hashes[kind].hexdigest(),
                    action='new_source_experts' if match else 'new_source_float' if original.logical.startswith('model_weights/') else 'unchanged_codec',
                    sampled_relative_rmse=(errors[kind][0] / max(errors[kind][1], 1e-30)) ** .5 if match else None))
            if match or index % 100 == 0:
                for kind in outputs:
                    outputs[kind].flush()
                    reports[kind].update(status='converting', seconds=time.monotonic() - started)
                    save(targets[kind].with_suffix('.conversion.json'), reports[kind])
                print(json.dumps(dict(event='tensor', index=index+1, total=len(template.tensors), name=original.logical,
                                      seconds=time.monotonic()-began, elapsed=time.monotonic()-started)), flush=True)
        for kind, output in outputs.items():
            output.truncate(sizes[kind]); output.flush(); os.fsync(output.fileno())
    for kind, target in targets.items():
        partial = target.with_suffix('.gguf.partial')
        archive = Archive(partial)
        for tensor, record in zip(archive.tensors, reports[kind]['tensors'], strict=True):
            if tensor.logical != record['name'] or tensor.kind != record['kind']:
                raise ValueError('Read-back tensor directory mismatch')
        print('Verifying every output tensor:', target.name, flush=True)
        with partial.open('rb') as file:
            for tensor, record in zip(archive.tensors, reports[kind]['tensors'], strict=True):
                file.seek(tensor.offset); remaining=tensor.nbytes; digest=hashlib.sha256()
                while remaining:
                    data=file.read(min(8*1024**2,remaining))
                    if not data: raise EOFError('Truncated output')
                    digest.update(data); remaining-=len(data)
                if digest.hexdigest()!=record['sha256']:
                    raise ValueError('Read-back payload mismatch: '+tensor.logical)
        partial.rename(target)
        reports[kind].update(status='complete', output_bytes=target.stat().st_size,
                             output_sha256=digest_file(target), seconds=time.monotonic()-started,
                             independent_payload_verification=True)
        save(target.with_suffix('.conversion.json'), reports[kind])
        print(json.dumps(dict(event='complete', output=str(target), bytes=target.stat().st_size,
                              sha256=reports[kind]['output_sha256'])), flush=True)


if __name__ == '__main__':
    main()
