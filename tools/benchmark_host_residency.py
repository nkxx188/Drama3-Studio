"""CPU-only gather comparison; warm mapped pages do not measure SSD latency."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from drama_low_vram.expert_cache_file import ExpertCacheFile
from drama_low_vram.host_residency import HostResidency
from drama_low_vram.offload import EXPERT_BUFFER_NAMES, FP8_DTYPE, ExpertStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--calls", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.top_k, args.calls, args.repeats) < 1:
        parser.error("top-k, calls and repeats must be positive")
    cache = ExpertCacheFile.open(args.cache_dir)
    store = ExpertStore(**cache.layer_buffers(args.layer))
    if args.top_k * 2 > store.num_experts:
        parser.error("layer must contain at least twice top-k experts")
    host = HostResidency({args.layer: store})
    setup_started = time.perf_counter()
    host.configure(host.expert_bytes * args.top_k, [(args.layer, i) for i in range(args.top_k)])
    setup_seconds = time.perf_counter() - setup_started
    targets = {
        name: torch.empty((args.top_k, *getattr(store, name).shape[1:]), dtype=getattr(store, name).dtype)
        for name in EXPERT_BUFFER_NAMES
    }

    def original(ids):
        for name in EXPERT_BUFFER_NAMES:
            source, target = getattr(store, name), targets[name]
            if source.dtype == FP8_DTYPE:
                source, target = source.view(torch.uint8), target.view(torch.uint8)
            torch.index_select(source, 0, ids, out=target)

    def resident(ids):
        host.copy_to(args.layer, ids, targets, observe=False)

    routes = {
        "all_resident": torch.arange(args.top_k, dtype=torch.int64),
        "mixed_alternating": torch.tensor(
            [i if i % 2 == 0 else i + args.top_k for i in range(args.top_k)], dtype=torch.int64
        ),
        "all_mapped": torch.arange(args.top_k, args.top_k * 2, dtype=torch.int64),
    }
    results = {}
    for name, ids in routes.items():
        original(ids)
        expected = {key: tensor.view(torch.uint8).clone() for key, tensor in targets.items()}
        resident(ids)
        exact = all(torch.equal(targets[key].view(torch.uint8), value) for key, value in expected.items())
        times = {"mapped_direct_ms": [], "host_residency_ms": []}
        for repeat in range(args.repeats):
            # Alternate order so one method is not always measured after the other.
            methods = [("mapped_direct_ms", original), ("host_residency_ms", resident)]
            if repeat % 2:
                methods.reverse()
            for method_name, function in methods:
                function(ids)
                started = time.perf_counter()
                for _ in range(args.calls):
                    function(ids)
                times[method_name].append((time.perf_counter() - started) * 1000 / args.calls)
        medians = {key: statistics.median(value) for key, value in times.items()}
        results[name] = {
            **medians,
            "speedup": medians["mapped_direct_ms"] / medians["host_residency_ms"],
            "bitwise_equal": exact,
            "samples": times,
        }
    report = {
        "scope": "CPU gather from warm OS file pages versus explicit pageable RAM arena; not physical SSD I/O",
        "layer": args.layer,
        "shape": {name: list(getattr(store, name).shape) for name in EXPERT_BUFFER_NAMES},
        "top_k": args.top_k,
        "calls_per_repeat": args.calls,
        "repeats": args.repeats,
        "setup_seconds": setup_seconds,
        "results": results,
        "residency": host.as_dict(),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    host.close()
    cache.close()


if __name__ == "__main__":
    main()
