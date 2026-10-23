"""Create audio.cpp Q8 experts from the verified shared official FP8 source.

Only experts are quantized. BF16/F32 backbone and tower tensors are read from
the source unchanged; the verified Q4 package supplies its unchanged BF16 full
head, FP16 codec and tokenizer/package metadata. No Q4 expert bytes are reused.
No full FP16 intermediate model is written.
"""

from __future__ import annotations

import argparse
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
sys.path[:0] = [
    str(ROOT / ".cache/q8-tools/python"), str(ROOT / "src"),
    str(ROOT / "vendor/fish-speech"),
]
import numpy as np
import torch
from gguf import GGMLQuantizationType as Q
from gguf.quants import quantize, dequantize
from drama_low_vram.shared_checkpoint import descriptor, iter_packed, virtual_tensors
from gguf_archive import Archive, BLOCKS, scalar_metadata, write_header

EXPERT = re.compile(
    r"model_weights/model\.language_model\.layers\.(\d+)\.mlp\.experts\."
    r"(gate_proj|up_proj|down_proj)\.weight$")


def emit(event, **values):
    print(json.dumps({"event": event, **values}), flush=True)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def native_float(source):
    if source.dtype == torch.bfloat16:
        return source.contiguous().view(torch.uint16).numpy(), 30
    if source.dtype == torch.float32:
        return source.contiguous().numpy(), 0
    if source.dtype == torch.float16:
        return source.contiguous().numpy(), 1
    raise ValueError(f"Unsupported source float tensor: {source.dtype}")


def expert_chunks(values, layer, projection, group=4):
    prefix = f"model.layers.{layer}.mlp.experts."
    name = "down" if projection == "down_proj" else "gate_up"
    weight, scale = values[prefix + name + "_weight"], values[prefix + name + "_scale"]
    width = weight.shape[1] // 2 if name == "gate_up" else weight.shape[1]
    offset = width if projection == "up_proj" else 0
    for start in range(0, weight.shape[0], group):
        w = weight[start:start + group, offset:offset + width].float()
        scales = scale[start:start + group, offset // 128:(offset + width) // 128]
        # Preserve the official FP32 multiplication before quantization.
        w *= scales.repeat_interleave(128, 1).repeat_interleave(128, 2)
        if not bool(torch.isfinite(w).all()):
            raise ValueError("Nonfinite dequantized official expert")
        yield start, w.numpy()


def copy_range(source, target, size, digest):
    remaining = size
    while remaining:
        data = source.read(min(8 * 1024**2, remaining))
        if not data:
            raise ValueError("Unexpected end of source model")
        target.write(data)
        digest.update(data)
        remaining -= len(data)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    checkpoint = ROOT / "checkpoints/drama-3"
    source = descriptor(checkpoint)
    parity = json.loads((ROOT / "research/phase28-q8-quantizer-parity.json").read_text())
    if not parity["all_bytes_equal"]:
        raise ValueError("Q8 reference quantizer parity was not established")
    pin = json.loads((ROOT / "research/phase19-gguf-q4-quality-download-pinned.json").read_text())
    template = Archive(Path(pin["destination"]) / pin["files"][0]["name"])
    values = dict(virtual_tensors(checkpoint))
    experts = dict(iter_packed(checkpoint, include_prefixes=("model.layers.",)))
    experts = {key: value for key, value in experts.items() if ".mlp.experts." in key}
    config = json.loads((checkpoint / "config.json").read_text())
    eos, first, last = (config["eos_token_id"], config["semantic_start_token_id"],
                        config["semantic_end_token_id"])
    ids = np.sort(np.r_[np.array([eos], dtype=np.int64), np.arange(first, last + 1)])
    compact = dict(iter_packed(checkpoint, include_prefixes=("audio_head.",)))["audio_head.weight"]
    full_head = template.by_name["model_weights/lm_head.weight"]
    mapped_head = np.memmap(template.path, mode="r", dtype=np.uint16,
                            offset=full_head.offset, shape=full_head.shape)
    if not np.array_equal(mapped_head[ids], compact.view(torch.uint16).numpy()):
        raise ValueError("Full BF16 head candidates differ from original shared source")

    plans, actions = [], []
    for tensor in template.tensors:
        match = EXPERT.fullmatch(tensor.logical)
        if match:
            if tensor.kind != 12:
                raise ValueError("Template experts must be Q4_K (never used as quantization input)")
            kind, action = 8, "q8_from_fp8"
        elif tensor.logical.startswith("codec_weights/") or tensor.logical == "model_weights/lm_head.weight":
            kind, action = tensor.kind, "copy_template"
        else:
            original = values[tensor.logical.removeprefix("model_weights/")]
            if original.numel() != np.prod(tensor.shape):
                raise ValueError(f"Source tensor shape mismatch: {tensor.logical}")
            _, kind = native_float(original)
            if tensor.kind == 0:
                kind = 0  # router/vector FP32 contains exact BF16 source values
            action = "original_float"
        block, byte_count = BLOCKS[kind]
        size = int(np.prod(tensor.shape)) // block * byte_count
        plans.append(replace(tensor, kind=kind, offset=0, nbytes=size))
        actions.append(action)
    if sum(action == "q8_from_fp8" for action in actions) != 3 * config["text_config"]["num_hidden_layers"]:
        raise ValueError("Incomplete expert conversion plan")
    target = args.output.resolve()
    temporary = target.with_suffix(".gguf.partial")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or temporary.exists():
        raise FileExistsError("Choose a fresh model output; interrupted partials are preserved")
    required = sum(t.nbytes + 32 for t in plans) + 512 * 1024**2
    if shutil.disk_usage(target.parent).free < required:
        raise RuntimeError(f"Need at least {required / 1024**3:.2f}GiB free")
    report_path = target.with_suffix(".conversion.json")
    started = time.perf_counter()
    report = {
        "status": "planned", "output": str(target), "source": source,
        "template": str(template.path), "template_sha256": pin["files"][0]["sha256"],
        "recipe": "Q8_0 experts from dequantized official FP8; original BF16/F32 backbone/tower; exact BF16 full head from verified template; template FP16 codec and package sidecars.",
        "full_head_candidates_equal": True, "expert_tensor_count": actions.count("q8_from_fp8"),
        "estimated_output_bytes": required - 512 * 1024**2,
        "large_intermediate_created": False, "tensor_records": [],
    }
    save(report_path, report)
    emit("plan", bytes=report["estimated_output_bytes"], expert_tensors=report["expert_tensor_count"])
    if args.validate_only:
        return
    replaced = {"general.name", "audiocpp.weight_type", "audiocpp.source_format"}
    metadata = [(name, raw) for name, raw in template.metadata if name not in replaced]
    metadata.extend([
        ("general.name", scalar_metadata("general.name", "Drama3 Q8_0 experts original BF16 backbone and audio tower")),
        ("audiocpp.weight_type", scalar_metadata("audiocpp.weight_type", "q8_0")),
        ("audiocpp.source_format", scalar_metadata("audiocpp.source_format", "original-shared-fp8/bf16")),
        ("drama3.source_sha256", scalar_metadata("drama3.source_sha256", source["sha256"])),
        ("drama3.codec_precision", scalar_metadata("drama3.codec_precision", "f16 from verified audio.cpp template")),
    ])
    try:
        report["status"] = "converting"
        save(report_path, report)
        with temporary.open("xb") as output, template.path.open("rb") as copied:
            final_size = write_header(output, metadata, plans, alignment=template.alignment)
            for index, (original, tensor, action) in enumerate(zip(template.tensors, plans, actions, strict=True)):
                output.seek(tensor.offset)
                digest = hashlib.sha256()
                stats = {}
                tensor_started = time.perf_counter()
                if action == "q8_from_fp8":
                    layer, projection = EXPERT.fullmatch(tensor.logical).groups()
                    squared_error, squared_signal, maximum_error = 0.0, 0.0, 0.0
                    written = 0
                    for _, data in expert_chunks(experts, int(layer), projection):
                        packed = quantize(data, Q.Q8_0)
                        packed.tofile(output)
                        digest.update(packed)
                        written += packed.nbytes
                        # Bounded diagnostics: one complete row per expert chunk.
                        sample = data.reshape(-1, data.shape[-1])[:1]
                        restored = dequantize(quantize(sample, Q.Q8_0), Q.Q8_0)
                        error = sample - restored
                        squared_error += float(np.sum(error.astype(np.float64) ** 2))
                        squared_signal += float(np.sum(sample.astype(np.float64) ** 2))
                        maximum_error = max(maximum_error, float(np.max(np.abs(error))))
                    if written != tensor.nbytes:
                        raise ValueError("Incorrect Q8 tensor byte count")
                    stats = {
                        "sampled_relative_rmse": (squared_error / max(squared_signal, 1e-30)) ** 0.5,
                        "sampled_max_absolute_error": maximum_error,
                        "sample_scope": "first row per four-expert group; not full-model quality",
                    }
                elif action == "original_float":
                    value = values[tensor.logical.removeprefix("model_weights/")]
                    data = (
                        value.float().reshape(tensor.shape).contiguous().numpy()
                        if tensor.kind == 0 else native_float(value.reshape(tensor.shape))[0]
                    )
                    if data.nbytes != tensor.nbytes:
                        raise ValueError("Incorrect original tensor size")
                    data.tofile(output)
                    digest.update(data)
                else:
                    copied.seek(original.offset)
                    copy_range(copied, output, tensor.nbytes, digest)
                record = {
                    "name": tensor.logical, "action": action, "kind": tensor.kind,
                    "bytes": tensor.nbytes, "sha256": digest.hexdigest(),
                    "seconds": time.perf_counter() - tensor_started, **stats,
                }
                report["tensor_records"].append(record)
                if action == "q8_from_fp8" or index % 100 == 0:
                    output.flush()
                    report["seconds"] = time.perf_counter() - started
                    save(report_path, report)
                    emit("tensor", index=index + 1, count=len(plans),
                         name=tensor.logical, seconds=record["seconds"],
                         total_seconds=report["seconds"])
            if output.tell() < final_size:
                output.write(bytes(final_size - output.tell()))
            output.flush()
            os.fsync(output.fileno())
        verified = Archive(temporary)
        if len(verified.tensors) != len(plans) or any(
            actual.logical != expected.logical or actual.kind != expected.kind
            or actual.shape != expected.shape or actual.nbytes != expected.nbytes
            for actual, expected in zip(verified.tensors, plans, strict=True)
        ):
            raise ValueError("Final GGUF metadata validation failed")
        temporary.rename(target)
        report.update(status="complete", seconds=time.perf_counter() - started,
                      output_bytes=target.stat().st_size,
                      output_sha256=hashlib.file_digest(target.open("rb"), "sha256").hexdigest())
        save(report_path, report)
        emit("complete", output=str(target), seconds=report["seconds"],
             bytes=report["output_bytes"], sha256=report["output_sha256"])
    except BaseException as error:
        report.update(status="error", seconds=time.perf_counter() - started,
                      error=f"{type(error).__name__}: {error}")
        save(report_path, report)
        raise


if __name__ == "__main__":
    main()
