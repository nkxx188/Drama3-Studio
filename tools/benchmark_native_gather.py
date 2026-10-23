"""Interleaved warm official expert byte gathers into bounded pinned targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from benchmark_engine_decode import torch
from drama_low_vram.expert_cache_file import ExpertCacheFile
from drama_low_vram.native_backend import NativeLibrary
from drama_low_vram.offload import EXPERT_BUFFER_NAMES, ExpertStore, FP8_DTYPE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--groups", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, default=ROOT / "research/phase6-host-gather.json")
    args = parser.parse_args()
    if min(*args.experts, args.groups, args.repeats) < 1:
        parser.error("Positive expert, group and repetition counts required")
    torch.set_num_threads(1)
    cache = ExpertCacheFile.open(ROOT / ".cache/drama3-fp8-experts")
    library = NativeLibrary()
    report = {
        "scope": "Warm first-layer CPU gather only, pinned targets; complete native Python validation included. No GPU transfer or speech speed claim.",
        "configuration": {"torch_threads": 1, "groups": args.groups, "repeats": args.repeats},
        "results": [],
    }
    try:
        store = ExpertStore(**cache.layer_buffers(0))
        if max(args.experts) > store.num_experts:
            raise ValueError("Expert group exceeds bank size")
        rng = torch.Generator().manual_seed(20261016)
        for count in args.experts:
            routes = [torch.randperm(store.num_experts, generator=rng)[:count]
                      for _ in range(args.groups)]
            targets = {
                name: torch.empty(
                    (count, *getattr(store, name).shape[1:]),
                    dtype=getattr(store, name).dtype, pin_memory=True,
                ) for name in EXPERT_BUFFER_NAMES
            }

            def original(ids):
                for name in EXPERT_BUFFER_NAMES:
                    source, target = getattr(store, name), targets[name]
                    if source.dtype == FP8_DTYPE:
                        source, target = source.view(torch.uint8), target.view(torch.uint8)
                    torch.index_select(source, 0, ids, out=target)

            def native(ids):
                library.gather_experts(store, ids, targets)

            # Warm every used source page and compare bytes outside timing.
            for ids in routes:
                original(ids)
                expected = {key: value.view(torch.uint8).clone() for key, value in targets.items()}
                native(ids)
                if not all(torch.equal(expected[key], targets[key].view(torch.uint8)) for key in targets):
                    raise RuntimeError("Native gather changed expert bytes")
            samples = {"torch": [], "native": []}
            for repeat in range(args.repeats):
                order = [("torch", original), ("native", native)]
                if repeat % 2:
                    order.reverse()
                for label, function in order:
                    started = time.perf_counter()
                    for ids in routes:
                        function(ids)
                    samples[label].append((time.perf_counter() - started) / args.groups)
            row = {
                "experts": count, "bytes_equal": True, "samples_seconds_per_group": samples,
                "medians_ms": {key: statistics.median(value) * 1000 for key, value in samples.items()},
            }
            report["results"].append(row)
            print(json.dumps(row), flush=True)
    finally:
        cache.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
