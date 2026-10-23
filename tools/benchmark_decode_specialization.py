"""Matched official/direct-route-slot FP8 decode on captured real MoE inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor/fish-speech")]
for key, name in (("TRITON_CACHE_DIR", "triton"), ("TEMP", "tmp"), ("TMP", "tmp")):
    directory = ROOT / ".cache" / name
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(directory)

import torch

from drama_low_vram.expert_cache_file import ExpertCacheFile
from drama_low_vram.offload import (
    EXPERT_BUFFER_NAMES, ExpertOffloadManager, ExpertStore, OffloadStatistics,
    TritonExpertBackend, _StagingPool,
)


@torch.inference_mode()
def benchmark(args):
    data = torch.load(args.cases, weights_only=True)
    cases = data["cases"]
    inputs = [(c["x"].cuda(), c["ids"].cuda(), c["weights"].cuda()) for c in cases]
    source = ExpertCacheFile.open(ROOT / ".cache/drama3-fp8-experts")
    store = ExpertStore(**source.layer_buffers(data["layer"]))
    backends = [TritonExpertBackend(decode_specialization=value) for value in (False, True)]
    pool = _StagingPool(8, True, False, OffloadStatistics())
    validations, resident_samples = [], [[], []]
    manager_results = []
    try:
        for index, (case, (x, _, weights)) in enumerate(zip(cases, inputs, strict=True)):
            ids = torch.arange(8, device=x.device).reshape(1, 8)
            with pool.session(store, x.device):
                bank = pool.stage(store, case["ids"].reshape(-1).contiguous())
                outputs = [backend.forward(x, ids, weights, bank) for backend in backends]
                torch.cuda.synchronize()
                validations.append({
                    "case": index,
                    "official_equals_capture": bool(torch.equal(outputs[0].cpu(), case["output"])),
                    "direct_equals_official": bool(torch.equal(outputs[1], outputs[0])),
                })
                for repeat in range(args.repeats):
                    for variant in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                        started = time.perf_counter_ns()
                        backends[variant].forward(x, ids, weights, bank)
                        torch.cuda.synchronize()
                        resident_samples[variant].append((time.perf_counter_ns() - started) / 1e6)
        for cache in args.cache_experts:
            managers = [
                ExpertOffloadManager(backend=backend, cache_experts=cache,
                                    combined_decode_bank=True)
                for backend in backends
            ]
            module = SimpleNamespace(
                fp8=True, **{name: getattr(store, name) for name in EXPERT_BUFFER_NAMES})
            samples, outputs = [[], []], {}
            try:
                for manager in managers:
                    for _ in range(2):
                        for values in inputs:
                            manager.forward(module, *values)
                    manager.synchronize()
                    torch.cuda.synchronize()
                for repeat in range(args.repeats):
                    for variant in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                        manager = managers[variant]
                        for index, values in enumerate(inputs):
                            started = time.perf_counter_ns()
                            result = manager.forward(module, *values)
                            torch.cuda.synchronize()
                            samples[variant].append((time.perf_counter_ns() - started) / 1e6)
                            if repeat == 0:
                                outputs[variant, index] = result.cpu()
                row = {
                    "cache_experts": cache,
                    "official_median_ms": statistics.median(samples[0]),
                    "direct_median_ms": statistics.median(samples[1]),
                    "official_mean_ms": statistics.mean(samples[0]),
                    "direct_mean_ms": statistics.mean(samples[1]),
                    "same_outputs": all(torch.equal(outputs[0, i], outputs[1, i]) for i in range(len(inputs))),
                    "samples_ms": samples,
                    "stats": [manager.stats.as_dict() for manager in managers],
                }
                manager_results.append(row)
                print(json.dumps({key: value for key, value in row.items() if key not in ("samples_ms", "stats")}), flush=True)
            finally:
                for manager in managers:
                    manager.close()
        return {
            "scope": "Matched single-layer expert comparison; same weights, dtype, cache policy and all routes.",
            "torch": torch.__version__, "device": torch.cuda.get_device_name(),
            "torch_intraop_threads": torch.get_num_threads(), "repeats": args.repeats,
            "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
            "validation": validations, "manager_results": manager_results,
            "resident_kernel_path": {
                "official_median_ms": statistics.median(resident_samples[0]),
                "direct_median_ms": statistics.median(resident_samples[1]),
                "samples_ms": resident_samples,
            },
            "notes": [
                "The candidate removes GPU sorting/alignment for a single row and addresses route slots directly.",
                "Quantization, FP8 dot block geometry, BF16 gate/up and FP32 reduction remain the official arithmetic.",
                "Timing includes synchronous completion; component gains do not imply complete-speech speedup.",
            ],
        }
    finally:
        pool.close()
        source.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "research/native-expert-cases.pt")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--cache-experts", type=int, nargs="+", default=[0, 2, 4])
    parser.add_argument("--output", type=Path, default=ROOT / "research/decode-specialization-benchmark.json")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("Positive repeat count required")
    torch.set_num_threads(1)
    report = benchmark(args)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["resident_kernel_path"] | {"samples_ms": "saved"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
