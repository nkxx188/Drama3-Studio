"""Calibrate CPU-only copying of real official FP8 expert banks."""

from __future__ import annotations

import argparse
import datetime
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "vendor" / "fish-speech"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import torch  # noqa: E402

from drama_low_vram.expert_cache_file import ExpertCacheFile  # noqa: E402
from drama_low_vram.offload import EXPERT_BUFFER_NAMES, FP8_DTYPE  # noqa: E402


def run_benchmark(
    cache_dir: Path,
    *,
    thread_counts: list[int],
    counts: list[int],
    route_groups: int,
    repeats: int,
) -> dict:
    cache = ExpertCacheFile.open(cache_dir)
    if max(counts) > cache.num_experts:
        raise ValueError("Requested staging group exceeds the model's expert count")
    buffers = cache.layer_buffers(0)
    sources = {
        name: (
            tensor.view(torch.uint8) if tensor.dtype == FP8_DTYPE else tensor
        )
        for name, tensor in buffers.items()
    }
    generator = torch.Generator().manual_seed(20261015)
    routes = {
        count: [
            torch.randperm(cache.num_experts, generator=generator)[:count]
            for _ in range(route_groups)
        ]
        for count in counts
    }
    per_expert_bytes = sum(
        source[0].numel() * source.element_size() for source in sources.values()
    )
    warm_started = time.perf_counter()
    # Touch the first layer once. Subsequent runs isolate RAM copy overhead
    # rather than SSD page faults, and use no other layer's payload.
    warm_checksum = sum(
        int(source.reshape(-1).view(torch.uint8)[::4096].sum().item())
        for source in sources.values()
    )
    warm_seconds = time.perf_counter() - warm_started
    original_threads = torch.get_num_threads()
    results = []
    try:
        for threads in thread_counts:
            torch.set_num_threads(threads)
            for count in counts:
                targets = {
                    name: torch.empty(
                        (count, *source.shape[1:]),
                        dtype=source.dtype,
                        device="cpu",
                    )
                    for name, source in sources.items()
                }

                def copy(ids):
                    for name in EXPERT_BUFFER_NAMES:
                        torch.index_select(
                            sources[name], 0, ids, out=targets[name]
                        )

                for ids in routes[count]:
                    copy(ids)
                durations = []
                for _ in range(repeats):
                    started = time.perf_counter()
                    for ids in routes[count]:
                        copy(ids)
                    durations.append(time.perf_counter() - started)
                total_groups = repeats * route_groups
                total_seconds = sum(durations)
                row = {
                    "num_threads": threads,
                    "experts_per_group": count,
                    "route_groups_per_repeat": route_groups,
                    "repeats": repeats,
                    "total_groups": total_groups,
                    "bytes_per_group": per_expert_bytes * count,
                    "mean_ms_per_group": total_seconds * 1000 / total_groups,
                    "median_ms_per_group": (
                        statistics.median(durations) * 1000 / route_groups
                    ),
                    "min_ms_per_group": min(durations) * 1000 / route_groups,
                    "max_ms_per_group": max(durations) * 1000 / route_groups,
                    "logical_read_gib_per_second": (
                        per_expert_bytes
                        * count
                        * total_groups
                        / total_seconds
                        / 1024**3
                    ),
                }
                results.append(row)
                print(
                    f"threads={threads:2d} experts={count:2d}: "
                    f"{row['mean_ms_per_group']:.3f} ms/group, "
                    f"{row['logical_read_gib_per_second']:.2f} GiB/s",
                    flush=True,
                )
    finally:
        torch.set_num_threads(original_threads)
        cache.close()
    recommendations = {}
    for count in counts:
        rows = [row for row in results if row["experts_per_group"] == count]
        best = min(rows, key=lambda row: row["median_ms_per_group"])
        baseline = next(
            (row for row in rows if row["num_threads"] == original_threads),
            max(rows, key=lambda row: row["num_threads"]),
        )
        recommendations[str(count)] = {
            "best_num_threads": best["num_threads"],
            "best_median_ms_per_group": best["median_ms_per_group"],
            "baseline_num_threads": baseline["num_threads"],
            "baseline_median_ms_per_group": baseline["median_ms_per_group"],
            "baseline_to_best_ratio": (
                baseline["median_ms_per_group"] / best["median_ms_per_group"]
            ),
        }
    return {
        "date": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        "model_fingerprint": cache.model_fingerprint,
        "cache_dir": str(cache_dir.resolve()),
        "torch_version": torch.__version__,
        "original_num_threads": original_threads,
        "source_layer": 0,
        "source_layer_bytes": sum(
            source.numel() * source.element_size() for source in sources.values()
        ),
        "warm_seconds": warm_seconds,
        "warm_checksum": warm_checksum,
        "host_buffer": "ordinary pageable CPU memory",
        "gpu_operations": False,
        "notes": [
            "Measures only four CPU index_select(out=...) copies, matching host gathering in the staging pool.",
            "The source layer is warmed before measuring; SSD cold faults and PCIe transfers are excluded.",
            "Logical bandwidth counts copied payload once, not total read plus write traffic.",
            "Route groups model one 40-layer decode step but all use first-layer data.",
            "These results alone do not establish full-model generation speed or an optimal CPU lm_head thread count.",
        ],
        "results": results,
        "recommendations": recommendations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / ".cache" / "drama3-fp8-experts",
    )
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--experts", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--route-groups", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "research" / "host-staging-threads.json",
    )
    args = parser.parse_args()
    if min(args.route_groups, args.repeats, *args.threads, *args.experts) < 1:
        parser.error("Counts, threads and repeats must be positive")
    result = run_benchmark(
        args.cache,
        thread_counts=args.threads,
        counts=args.experts,
        route_groups=args.route_groups,
        repeats=args.repeats,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
