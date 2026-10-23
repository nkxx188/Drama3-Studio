"""Compare four-buffer and packed expert staging using real FP8 source bytes."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor/fish-speech")]
for key, directory in (("TRITON_CACHE_DIR", "triton"), ("TEMP", "tmp"), ("TMP", "tmp")):
    path = ROOT / ".cache" / directory
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)

import torch

from drama_low_vram.expert_cache_file import ExpertCacheFile
from drama_low_vram.offload import (
    EXPERT_BUFFER_NAMES,
    FP8_DTYPE,
    ExpertStore,
    OffloadStatistics,
    TritonExpertBackend,
    _StagingPool,
)


def benchmark(args):
    device = torch.device(args.device)
    cache = ExpertCacheFile.open(args.cache)
    store = ExpertStore(**cache.layer_buffers(args.layer))
    generator = torch.Generator().manual_seed(20261015)
    routes = {
        count: [
            torch.randperm(store.num_experts, generator=generator)[:count]
            for _ in range(args.calls)
        ]
        for count in args.experts
    }
    # Touch source pages before timing. This does not measure cold disk access.
    checksum = sum(
        int(getattr(store, name).reshape(-1).view(torch.uint8)[::4096].sum())
        for name in EXPERT_BUFFER_NAMES
    )
    results = []
    pools = {}

    def finish(pool):
        pool._finish_copy()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    try:
        for enabled in (False, True):
            pools[enabled] = _StagingPool(
                max(args.experts), args.pin_memory, False, OffloadStatistics(),
                packed_staging=enabled,
            )
        for count, groups in routes.items():
            durations = {False: [], True: []}
            parity = {}
            for enabled, pool in pools.items():
                with pool.session(store, device):
                    for ids in groups[:4]:
                        pool.stage(store, ids)
                    finish(pool)
                    staged = pool.stage(store, groups[-1])
                    finish(pool)
                    parity[enabled] = all(
                        torch.equal(
                            getattr(staged, name).cpu().view(torch.uint8),
                            getattr(store, name).view(torch.uint8).index_select(0, groups[-1]),
                        )
                        for name in EXPERT_BUFFER_NAMES
                    )
            for repeat in range(args.repeats):
                for enabled in ((False, True) if repeat % 2 == 0 else (True, False)):
                    pool = pools[enabled]
                    with pool.session(store, device):
                        started = time.perf_counter()
                        for ids in groups:
                            pool.stage(store, ids)
                        finish(pool)
                        durations[enabled].append(
                            (time.perf_counter() - started) * 1000 / len(groups)
                        )
            median = {
                enabled: statistics.median(samples)
                for enabled, samples in durations.items()
            }
            results.append({
                "experts": count,
                "bytes_per_call": count * store.nbytes // store.num_experts,
                "baseline_ms": median[False],
                "packed_ms": median[True],
                "baseline_to_packed_ratio": median[False] / median[True],
                "baseline_samples_ms": durations[False],
                "packed_samples_ms": durations[True],
                "byte_parity": all(parity.values()),
            })
        kernel_parity = []
        if device.type == "cuda":
            backend = TritonExpertBackend()
            x = torch.randn((1, store.hidden), generator=generator).to(device, torch.bfloat16)
            for count, groups in routes.items():
                weights = torch.softmax(
                    torch.randn((1, count), generator=generator), -1
                ).to(device)
                local_ids = torch.arange(count, device=device).reshape(1, count)
                outputs = []
                for pool in pools.values():
                    with pool.session(store, device):
                        staged = pool.stage(store, groups[0])
                        outputs.append(backend.forward(x, local_ids, weights, staged).cpu())
                kernel_parity.append({
                    "experts": count,
                    "bitwise_equal": bool(torch.equal(*outputs)),
                    "max_abs_error": float((outputs[0].float() - outputs[1].float()).abs().max()),
                })
        return {
            "scope": "Warm real layer CPU gather plus completed device copy; excludes cold SSD and full-model compute.",
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "pin_memory_requested": args.pin_memory,
            "pin_memory_actual": all(pool.pin_memory for pool in pools.values()) if device.type == "cuda" else False,
            "cpu_threads": torch.get_num_threads(),
            "torch_version": torch.__version__,
            "calls_per_repeat": args.calls,
            "repeats": args.repeats,
            "source_checksum": checksum,
            "capacity": max(args.experts),
            "results": results,
            "kernel_parity": kernel_parity,
            "statistics": {
                "packed" if enabled else "baseline": pool.stats.as_dict()
                for enabled, pool in pools.items()
            },
        }
    finally:
        for pool in pools.values():
            pool.close()
        del store
        cache.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache/drama3-fp8-experts")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--experts", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--calls", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, default=ROOT / "research/benchmark-staging-arena.json")
    args = parser.parse_args()
    if min(args.calls, args.repeats, *args.experts) < 1:
        parser.error("Counts must be positive")
    result = benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if (
        all(row["byte_parity"] for row in result["results"])
        and all(row["bitwise_equal"] for row in result["kernel_parity"])
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
