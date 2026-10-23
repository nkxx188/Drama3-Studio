"""Validate and time native W8A8 experts on captured real Drama MoE inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor/fish-speech")]
for key, directory in (
    ("TRITON_CACHE_DIR", "triton"), ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
    ("TORCH_EXTENSIONS_DIR", "torch_extensions"), ("TEMP", "tmp"), ("TMP", "tmp"),
):
    path = ROOT / ".cache" / directory
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)

import torch

from drama_low_vram.expert_cache_file import ExpertCacheFile
from drama_low_vram.fp8_cpu_reference import expert_pairs, quantize_fp8_blocks
from drama_low_vram.native_backend import NativeExpertExecutor, NativeLibrary
from drama_low_vram.offload import (
    EXPERT_BUFFER_NAMES, ExpertStore, OffloadStatistics, StagedExperts,
    TorchExpertBackend, TritonExpertBackend, _StagingPool,
)


def difference(actual, expected):
    actual, expected = actual.float(), expected.float()
    delta = actual - expected
    return {
        "bitwise_equal": bool(torch.equal(actual, expected)),
        "relative_l2_error": float(torch.linalg.vector_norm(delta) /
                                   torch.linalg.vector_norm(expected).clamp_min(1e-20)),
        "max_abs_error": float(delta.abs().max()),
        "cosine_similarity": float(torch.nn.functional.cosine_similarity(
            actual.reshape(1, -1), expected.reshape(1, -1))),
    }


@torch.inference_mode()
def benchmark(args):
    data = torch.load(args.cases, map_location="cpu", weights_only=True)
    cases = data["cases"][:args.cases_count]
    if not cases:
        raise ValueError("Capture real MoE cases with tools/profile_engine_runtime.py first")
    source = ExpertCacheFile.open(ROOT / ".cache/drama3-fp8-experts")
    store = ExpertStore(**source.layer_buffers(data["layer"]))
    library = NativeLibrary()
    rows, numerical = [], []
    prepared = [
        (case["x"].contiguous(), case["ids"].reshape(-1).contiguous(),
         case["weights"].reshape(-1).float().contiguous())
        for case in cases
    ]
    gpu = TritonExpertBackend()
    device = torch.device("cuda:0")
    gpu._check(prepared[0][0].to(device))
    gpu_quantized = [
        tuple(value.cpu() for value in gpu.kernel.quantize_groups(
            gpu.kernel.quant_kernel, x.to(device), store.hidden
        )) for x, _, _ in prepared
    ]
    try:
        for implementation, mask in (("scalar", 1), ("avx2", 2), ("avx512", 4)):
            if not library.features & mask:
                continue
            with NativeExpertExecutor(
                store, capacity=8, threads=1, implementation=implementation,
                library=library,
            ) as executor:
                for index, (x, ids, weights) in enumerate(prepared):
                    actual = executor.forward_pairs(x, ids, weights)
                    shared_q = executor.forward_quantized_pairs(
                        *gpu_quantized[index], ids, weights
                    )
                    q, s = library.quantize(x)
                    expected_q, expected_s = quantize_fp8_blocks(x)
                    pool = _StagingPool(8, True, False, OffloadStatistics())
                    try:
                        with pool.session(store, device):
                            staged = pool.stage(store, ids)
                            gpu_partials = gpu.forward_pairs(
                                x.repeat(8, 1).to(device),
                                torch.arange(8, device=device).reshape(8, 1),
                                weights.to(device).reshape(8, 1), staged,
                            ).cpu()
                    finally:
                        pool.close()
                    row = {
                        "implementation": implementation, "case": index,
                        "input_quantization_bytes_equal": bool(torch.equal(q, expected_q.view(torch.uint8))),
                        "input_scales_equal": bool(torch.equal(s, expected_s)),
                        "input_quantization_matches_gpu": bool(torch.equal(
                            q, gpu_quantized[index][0].view(torch.uint8))),
                        "input_scales_match_gpu": bool(torch.equal(
                            s, gpu_quantized[index][1].reshape(-1))),
                        "native_vs_official_gpu_partials": difference(actual, gpu_partials),
                        "shared_gpu_quantization_vs_official_partials": difference(shared_q, gpu_partials),
                        "shared_gpu_quantization_sum_vs_captured": difference(
                            shared_q.sum(0).to(torch.bfloat16), cases[index]["output"].reshape(-1)),
                        "native_sum_vs_captured_output": difference(
                            actual.sum(0).to(torch.bfloat16), cases[index]["output"].reshape(-1)),
                    }
                    if index < 2:
                        row["native_vs_explicit_cpu_reference"] = difference(
                            actual, expert_pairs(store, x, ids, weights))
                    numerical.append(row)
        for implementation, mask in (("avx2", 2), ("avx512", 4)):
            if not library.features & mask:
                continue
            for threads in args.threads:
                with NativeExpertExecutor(
                    store, capacity=8, threads=threads,
                    implementation=implementation, library=library,
                ) as executor:
                    for count in args.experts:
                        output = torch.empty(count, store.hidden)
                        for x, ids, weights in prepared:
                            executor.forward_pairs(x, ids[:count], weights[:count], out=output)
                        samples = []
                        for _ in range(args.repeats):
                            started = time.perf_counter()
                            for x, ids, weights in prepared:
                                executor.forward_pairs(x, ids[:count], weights[:count], out=output)
                            samples.append((time.perf_counter() - started) * 1000 / len(prepared))
                        row = {"implementation": implementation, "threads": threads,
                               "experts": count, "median_ms": statistics.median(samples),
                               "samples_ms": samples}
                        rows.append(row)
                        print(json.dumps(row), flush=True)
        gpu_rows = []
        for count in args.experts:
            pool = _StagingPool(count, True, True, OffloadStatistics())
            route_ids = torch.arange(count, device=device).reshape(count, 1)
            inputs = [(x.repeat(count, 1).to(device), ids[:count],
                       weights[:count].to(device).reshape(count, 1))
                      for x, ids, weights in prepared]
            try:
                def run():
                    for x, ids, weights in inputs:
                        with pool.session(store, device):
                            staged = pool.stage(store, ids)
                            gpu.forward_pairs(x, route_ids, weights, staged)
                    pool.synchronize()
                    torch.cuda.synchronize(device)
                run()
                samples = []
                for _ in range(args.repeats):
                    started = time.perf_counter()
                    run()
                    samples.append((time.perf_counter() - started) * 1000 / len(inputs))
                gpu_rows.append({
                    "experts": count, "median_ms": statistics.median(samples),
                    "samples_ms": samples, "stats": pool.stats.as_dict(),
                })
            finally:
                pool.close()
        legacy_rows = []
        cpu_fallback = TorchExpertBackend()
        bank = StagedExperts(**{name: getattr(store, name) for name in EXPERT_BUFFER_NAMES})
        for count in args.experts:
            samples = []
            for x, ids, weights in prepared:
                cpu_fallback.forward(x, ids[:count].reshape(1, count),
                                     weights[:count].reshape(1, count), bank)
            for _ in range(args.repeats):
                started = time.perf_counter()
                for x, ids, weights in prepared:
                    cpu_fallback.forward(x, ids[:count].reshape(1, count),
                                         weights[:count].reshape(1, count), bank)
                samples.append((time.perf_counter() - started) * 1000 / len(prepared))
            legacy_rows.append({"experts": count, "threads": torch.get_num_threads(),
                                "median_ms": statistics.median(samples),
                                "samples_ms": samples})
        return {
            "scope": "Native batch-one W8A8 partials on real layer-zero MoE inputs; not an integrated CPU/GPU scheduler or speech-quality result.",
            "library": str(library.path),
            "library_sha256": hashlib.sha256(library.path.read_bytes()).hexdigest(),
            "features": library.features, "source_layer": data["layer"],
            "captured_cases": len(cases), "repeats": args.repeats,
            "geometry": {"hidden": store.hidden, "intermediate": store.intermediate},
            "numerical": numerical, "cpu_results": rows, "gpu_results": gpu_rows,
            "legacy_w16a16_cpu_results": legacy_rows,
            "quality_gate": {
                "default_engine_integration": False,
                "reason": "Real GPU partial/hidden errors are measured but no full-model speech-quality gate is established.",
                "all_shared_quantized_inputs_from_official_gpu": True,
            },
            "notes": [
                "CPU native input is already in RAM; the eventual hybrid path adds D2H and partial-result H2D.",
                "GPU includes host gather, H2D and completed expert execution; route synchronization is excluded.",
                "Native SIMD accumulation and C++ exp may round differently from GPU tensor cores/Triton exp.",
                "CPU owns only bounded scratch and borrows the existing original FP8 bank.",
            ],
        }
    finally:
        source.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "research/native-expert-cases.pt")
    parser.add_argument("--cases-count", type=int, default=6)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--experts", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "research/native-expert-validation.json")
    args = parser.parse_args()
    if min(args.cases_count, args.repeats, *args.threads, *args.experts) < 1 or max(args.experts) > 8:
        parser.error("Positive counts and up to eight experts are required")
    result = benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
