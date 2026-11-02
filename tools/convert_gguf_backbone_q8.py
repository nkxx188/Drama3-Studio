"""Quantize only Drama 3 backbone projections; preserve all other tensor bytes."""

from __future__ import annotations

import argparse
from collections import Counter
import ctypes
from dataclasses import replace
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".cache/q8-tools/python")]

import numpy as np

from drama_low_vram.gguf_archive import (
    Archive,
    BLOCKS,
    read_number,
    read_string,
    scalar_metadata,
    write_header,
)

REVISION = "78e5ab79e49aa9a25abe8799eff2c4d73a1c3216"
PROJECTIONS = re.compile(
    r"model_weights/model\.language_model\.layers\.\d+\."
    r"(?P<module>self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
    r"|linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)"
    r"|mlp\.shared_expert\.(?:gate_proj|up_proj|down_proj))\.weight$"
)
EXPECTED_GROUPS = {"self_attn": 40, "linear_attn": 150, "shared_expert": 120}
BUFFER_BYTES = 8 * 1024**2


def projection_group(name):
    match = PROJECTIONS.fullmatch(name)
    if match is None:
        return None
    module = match["module"]
    return "shared_expert" if module.startswith("mlp.") else module.split(".")[0]


def build_plan(archive):
    result = []
    for tensor in archive.tensors:
        if projection_group(tensor.logical) is not None:
            if tensor.kind != 30 or len(tensor.shape) != 2 or tensor.shape[-1] % 32:
                raise ValueError("Expected a block-aligned BF16 projection: " + tensor.logical)
            size = math.prod(tensor.shape) // BLOCKS[8][0] * BLOCKS[8][1]
            result.append(replace(tensor, kind=8, nbytes=size, offset=0))
        else:
            result.append(replace(tensor, offset=0))
    return result


def string_metadata(archive, name):
    raw = next((raw for key, raw in archive.metadata if key == name), None)
    if raw is None:
        return None
    file = BytesIO(raw)
    if read_string(file) != name or read_number(file, "I") != 8:
        raise ValueError("Expected string metadata: " + name)
    return read_string(file)


def bf16_to_float32(raw, shape):
    bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    bits <<= 16
    return bits.view(np.float32).reshape(shape)


class NativeQuantizer:
    def __init__(self, library, threads):
        self.library = ctypes.CDLL(str(library))
        self.function = self.library.drama_quantize_rows
        self.function.argtypes = [
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int,
        ]
        self.function.restype = ctypes.c_int
        self.threads = threads

    def __call__(self, values):
        values = np.ascontiguousarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[-1] % 32 or not np.isfinite(values).all():
            raise ValueError("Q8 input must be a finite, block-aligned matrix")
        rows, columns = values.shape
        output = np.empty((rows, columns // 32 * 34), dtype=np.uint8)
        status = self.function(
            8, values.ctypes.data, output.ctypes.data, rows, columns, self.threads
        )
        if status:
            raise RuntimeError(f"Native Q8 quantizer failed: {status}")
        return output


def read_exact(file, length, *digests):
    raw = file.read(length)
    if len(raw) != length:
        raise EOFError("Truncated tensor data")
    for digest in digests:
        digest.update(raw)
    return raw


def hash_until(file, target, digest):
    if file.tell() > target:
        raise ValueError("Tensor payloads must be in non-overlapping file order")
    while file.tell() < target:
        read_exact(file, min(BUFFER_BYTES, target - file.tell()), digest)


def save_report(path, report):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def verify_output(path, plan, records):
    archive = Archive(path)
    if len(archive.tensors) != len(plan):
        raise ValueError("Output tensor count changed")
    full_digest = hashlib.sha256()
    with path.open("rb") as file:
        for actual, expected, record in zip(archive.tensors, plan, records, strict=True):
            if (
                actual.logical != expected.logical or actual.physical != expected.physical
                or actual.shape != expected.shape or actual.kind != expected.kind
                or actual.offset != expected.offset or actual.nbytes != expected.nbytes
            ):
                raise ValueError("Output tensor directory mismatch: " + expected.logical)
            hash_until(file, actual.offset, full_digest)
            digest = hashlib.sha256()
            remaining = actual.nbytes
            while remaining:
                count = min(BUFFER_BYTES, remaining)
                read_exact(file, count, full_digest, digest)
                remaining -= count
            if digest.hexdigest() != record["sha256"]:
                raise ValueError("Output tensor verification failed: " + actual.logical)
        hash_until(file, path.stat().st_size, full_digest)
    return full_digest.hexdigest()


def convert(source, destination, quantizer):
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize, quantize

    source, destination = source.resolve(), destination.resolve()
    if source == destination:
        raise ValueError("Source and destination must differ")
    partial = destination.with_suffix(".gguf.partial")
    report_path = destination.with_suffix(".conversion.json")
    if any(path.exists() for path in (destination, partial, report_path)):
        raise FileExistsError("Refusing to overwrite an existing model or conversion")
    archive = Archive(source)
    receipt = json.loads(source.with_suffix(".conversion.json").read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "complete"
        or receipt.get("source_revision") != REVISION
        or not receipt.get("independent_payload_verification")
        or receipt.get("output_bytes") != source.stat().st_size
        or string_metadata(archive, "drama3.source_revision") != REVISION
    ):
        raise ValueError("Expected a verified updated-checkpoint GGUF source")
    expected_records = receipt["tensors"]
    if len(expected_records) != len(archive.tensors):
        raise ValueError("Incomplete source tensor manifest")
    groups = Counter(filter(None, (projection_group(t.logical) for t in archive.tensors)))
    if groups != EXPECTED_GROUPS:
        raise ValueError(f"Unexpected backbone projection layout: {groups}")
    plan = build_plan(archive)
    required = sum(t.nbytes + archive.alignment for t in plan) + archive.data_offset
    if shutil.disk_usage(destination.parent if destination.parent.exists() else ROOT).free < required + 2 * 1024**3:
        raise RuntimeError("Insufficient destination disk space, including safety margin")
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = [
        (name, raw) for name, raw in archive.metadata
        if name not in {"general.name", "drama3.backbone_weight_type", "drama3.parent_sha256"}
    ]
    name = string_metadata(archive, "general.name")
    if not name or "BF16 backbone/tower/head" not in name:
        raise ValueError("Unexpected parent model description")
    name = name.replace("BF16 backbone/tower/head", "Q8_0 backbone BF16 tower/head")
    metadata.extend(
        (key, scalar_metadata(key, value)) for key, value in {
            "general.name": name,
            "drama3.backbone_weight_type": "q8_0",
            "drama3.parent_sha256": receipt["output_sha256"],
        }.items()
    )
    report = {
        "status": "converting", "source_revision": REVISION,
        "source": str(source), "source_sha256": receipt["output_sha256"],
        "output": str(destination), "recipe": name,
        "quantized_tensor_groups": dict(groups),
        "unchanged_policy": "Byte-identical experts, tower, codec, embedding, full head, routers and norms",
        "tensors": [],
    }
    save_report(report_path, report)
    source_digest = hashlib.sha256()
    started = time.monotonic()
    print(json.dumps({"event": "start", "source": str(source), "output": str(destination),
                      "estimated_bytes": required, "quantized_tensors": sum(groups.values())}), flush=True)
    with source.open("rb") as input_file, partial.open("xb") as output_file:
        final_size = write_header(output_file, metadata, plan, alignment=archive.alignment)
        for index, (original, target, expected) in enumerate(
            zip(archive.tensors, plan, expected_records, strict=True)
        ):
            if (
                original.logical != expected["name"] or original.kind != expected["kind"]
                or original.nbytes != expected["bytes"]
            ):
                raise ValueError("Source tensor manifest mismatch: " + original.logical)
            hash_until(input_file, original.offset, source_digest)
            output_file.seek(target.offset)
            input_digest, output_digest = hashlib.sha256(), hashlib.sha256()
            changed = original.kind != target.kind
            error = None
            if changed:
                raw = read_exact(input_file, original.nbytes, source_digest, input_digest)
                values = bf16_to_float32(raw, original.shape)
                packed = quantizer(values)
                # Check every converted matrix's leading rows against the
                # independent NumPy implementation, not just a synthetic probe.
                sample = values[:min(4, len(values))]
                if not np.array_equal(
                    packed[:len(sample)], quantize(sample, GGMLQuantizationType.Q8_0)
                ):
                    raise ValueError("Independent Q8 parity failed: " + original.logical)
                restored = dequantize(packed[:len(sample)], GGMLQuantizationType.Q8_0)
                denominator = max(float(np.sum(sample.astype(np.float64) ** 2)), 1e-30)
                error = float(np.sqrt(np.sum(
                    (sample.astype(np.float64) - restored) ** 2
                ) / denominator))
                data = packed.tobytes()
                if len(data) != target.nbytes:
                    raise ValueError("Quantized byte count mismatch")
                output_file.write(data)
                output_digest.update(data)
                del raw, values, packed, data
            else:
                remaining = original.nbytes
                while remaining:
                    count = min(BUFFER_BYTES, remaining)
                    data = read_exact(input_file, count, source_digest, input_digest)
                    output_file.write(data)
                    output_digest.update(data)
                    remaining -= count
            if input_digest.hexdigest() != expected["sha256"]:
                raise ValueError("Source payload hash mismatch: " + original.logical)
            report["tensors"].append({
                "name": target.logical, "shape": target.shape,
                "source_kind": original.kind, "kind": target.kind, "bytes": target.nbytes,
                "sha256": output_digest.hexdigest(),
                "source_sha256": input_digest.hexdigest(),
                "action": "backbone_bf16_to_q8_0" if changed else "unchanged",
                "sampled_relative_rmse": error,
            })
            if index % 50 == 0 or index + 1 == len(plan):
                report["seconds"] = time.monotonic() - started
                save_report(report_path, report)
                print(json.dumps({"event": "tensor", "index": index + 1, "total": len(plan),
                                  "seconds": report["seconds"]}), flush=True)
        hash_until(input_file, source.stat().st_size, source_digest)
        if source_digest.hexdigest() != receipt["output_sha256"]:
            raise ValueError("Complete source GGUF SHA256 mismatch")
        output_file.truncate(final_size)
        output_file.flush()
        os.fsync(output_file.fileno())
    report["status"] = "verifying"
    save_report(report_path, report)
    print(json.dumps({"event": "verify", "output": str(destination)}), flush=True)
    output_hash = verify_output(partial, plan, report["tensors"])
    if destination.exists():
        raise FileExistsError("Destination appeared during conversion")
    partial.rename(destination)
    report.update(
        status="complete", output_sha256=output_hash, output_bytes=final_size,
        independent_payload_verification=True, source_sha256_verified=True,
        changed_tensors=sum(row["action"] != "unchanged" for row in report["tensors"]),
        unchanged_tensors=sum(row["action"] == "unchanged" for row in report["tensors"]),
        seconds=time.monotonic() - started,
    )
    save_report(report_path, report)
    print(json.dumps({"event": "complete", "output": str(destination),
                      "bytes": final_size, "sha256": output_hash,
                      "seconds": report["seconds"]}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", choices=("q4", "q8"), nargs="+", default=["q4", "q8"])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--quantizer", type=Path,
                        default=ROOT / ".cache/drama3-update-20261031/quantizer.dll")
    args = parser.parse_args()
    if args.threads < 1 or len(args.models) != len(set(args.models)):
        parser.error("Positive thread count and distinct models are required")
    temporary = ROOT / ".cache/drama3-backbone-q8-20261031"
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ.update(TEMP=str(temporary), TMP=str(temporary), TMPDIR=str(temporary))
    pairs = []
    for label in args.models:
        expert = "Q4_K_S" if label == "q4" else "Q8_0"
        directory = ROOT / f"checkpoints/drama-3-20261029-gguf-{label}"
        source = directory / f"drama3-20261029-experts-{expert}-bf16.gguf"
        destination = ROOT / (
            f"checkpoints/drama-3-20261029-gguf-{label}-backbone-q8/"
            f"drama3-20261029-experts-{expert}-backbone-Q8_0-bf16.gguf"
        )
        pairs.append((source, destination))
    required = 0
    for source, destination in pairs:
        archive = Archive(source)
        required += sum(t.nbytes + archive.alignment for t in build_plan(archive)) + archive.data_offset
        if destination.exists() or destination.with_suffix(".gguf.partial").exists():
            raise FileExistsError(destination)
    if shutil.disk_usage(ROOT).free < required + 2 * 1024**3:
        raise RuntimeError("Insufficient D: space for all selected outputs plus safety margin")
    quantizer = NativeQuantizer(args.quantizer, args.threads)
    for source, destination in pairs:
        convert(source, destination, quantizer)


if __name__ == "__main__":
    main()
