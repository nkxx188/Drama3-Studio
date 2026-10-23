"""Probe existing CPU expert costs; the W16A16 fallback is not official W8A8."""

from __future__ import annotations

import argparse
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
from drama_low_vram.offload import (
    EXPERT_BUFFER_NAMES, ExpertStore, OffloadStatistics, StagedExperts,
    TorchExpertBackend, TritonExpertBackend, _StagingPool,
)


@torch.inference_mode()
def benchmark(args):
    original_threads = torch.get_num_threads()
    source = ExpertCacheFile.open(ROOT / ".cache/drama3-fp8-experts")
    store = ExpertStore(**source.layer_buffers(args.layer))
    bank = StagedExperts(**{name: getattr(store, name) for name in EXPERT_BUFFER_NAMES})
    hidden = torch.load(ROOT / "research/profile-hidden.pt", map_location="cpu", weights_only=True)
    x = hidden[:1].contiguous()
    generator = torch.Generator().manual_seed(20261015)
    groups = {
        count: [
            torch.randperm(store.num_experts, generator=generator)[:count].reshape(1, count)
            for _ in range(args.calls)
        ]
        for count in args.experts
    }
    weights = {
        count: torch.softmax(torch.randn(1, count, generator=generator), -1)
        for count in args.experts
    }
    backend = TorchExpertBackend()
    rows = []

    def measure(function, routes):
        for ids in routes:
            function(ids)
        samples = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            for ids in routes:
                function(ids)
            samples.append((time.perf_counter() - started) * 1000 / len(routes))
        return {"median_ms": statistics.median(samples), "samples_ms": samples}

    try:
        for threads in args.threads:
            torch.set_num_threads(threads)
            for count, routes in groups.items():
                values = measure(
                    lambda ids: backend.forward(x, ids, weights[count], bank), routes
                )
                row = {"lane": "cpu_torch_w16a16", "threads": threads,
                       "experts": count, **values}
                rows.append(row)
                print(json.dumps(row), flush=True)
        torch.set_num_threads(original_threads)
        device = torch.device("cuda:0")
        gpu = TritonExpertBackend()
        gpu._check(x.to(device))
        gpu_x = x.to(device)
        gpu_rows = []
        numerical = []
        for count, routes in groups.items():
            pool = _StagingPool(count, True, True, OffloadStatistics())
            local_ids = torch.arange(count, device=device).reshape(1, count)
            gpu_weights = weights[count].to(device)
            try:
                def invoke(ids):
                    with pool.session(store, device):
                        staged = pool.stage(store, ids.reshape(-1))
                        output = gpu.forward(gpu_x, local_ids, gpu_weights, staged)
                    return output

                for ids in routes:
                    invoke(ids)
                pool.synchronize()
                torch.cuda.synchronize(device)
                samples = []
                for _ in range(args.repeats):
                    started = time.perf_counter()
                    for ids in routes:
                        invoke(ids)
                    pool.synchronize()
                    torch.cuda.synchronize(device)
                    samples.append((time.perf_counter() - started) * 1000 / len(routes))
                gpu_rows.append({
                    "lane": "gpu_official_w8a8_with_host_gather_and_h2d",
                    "experts": count, "median_ms": statistics.median(samples),
                    "samples_ms": samples, "stats": pool.stats.as_dict(),
                })
                target = invoke(routes[0]).float().cpu()
                actual = backend.forward(x, routes[0], weights[count], bank).float()
                delta = actual - target
                numerical.append({
                    "experts": count, "bitwise_equal": bool(torch.equal(actual, target)),
                    "relative_l2_error": float(torch.linalg.vector_norm(delta) /
                                               torch.linalg.vector_norm(target).clamp_min(1e-20)),
                    "max_abs_error": float(delta.abs().max()),
                    "cosine_similarity": float(torch.nn.functional.cosine_similarity(actual, target)),
                })
            finally:
                pool.close()
        return {
            "scope": "Warm real FP8 layer source. CPU is existing dequantize-and-BF16-matvec fallback, not the official W8A8 activation arithmetic. Not an integrated CPU/GPU scheduler, native-kernel bound, or speech-quality result.",
            "input": "Captured backbone hidden row, not a captured routed-expert input",
            "cpu_isa": torch.backends.cpu.get_cpu_capability(),
            "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
            "source_layer": args.layer, "expert_bytes": store.nbytes // store.num_experts,
            "hidden": store.hidden, "intermediate": store.intermediate,
            "calls_per_repeat": args.calls, "repeats": args.repeats,
            "cpu_results": rows, "gpu_results": gpu_rows,
            "cpu_vs_gpu_numerics": numerical,
            "notes": [
                "CPU and GPU are timed separately; no overlap speedup is measured.",
                "CPU input is already in host RAM; a hybrid engine would also pay D2H, worker dispatch and output H2D.",
                "The GPU path includes gathering and copying weights; CPU includes on-demand BF16 dequantization.",
                "Neither path expands the whole expert bank to BF16.",
                "Sampled numerical error does not establish perceptual speech quality.",
            ],
        }
    finally:
        torch.set_num_threads(original_threads)
        source.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--experts", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--calls", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "research/cpu-expert-feasibility.json")
    args = parser.parse_args()
    if min(*args.threads, *args.experts, args.calls, args.repeats) < 1:
        parser.error("Counts must be positive")
    report = benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
