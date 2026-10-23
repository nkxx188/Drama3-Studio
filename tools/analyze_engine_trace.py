"""Summarize actual device intervals without double-counting annotated scopes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path


def union_duration(intervals):
    end = None
    total = 0.0
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    opener = gzip.open if args.trace.suffix == ".gz" else open
    with opener(args.trace, "rt", encoding="utf-8") as handle:
        events = json.load(handle)["traceEvents"]
    groups = defaultdict(lambda: {"calls": 0, "duration_us": 0.0, "bytes": 0})
    device_intervals = []
    for event in events:
        category = event.get("cat")
        if event.get("ph") != "X" or category not in (
            "kernel", "gpu_memcpy", "gpu_memset", "cuda_runtime", "cuda_driver"
        ):
            continue
        entry = groups[(category, event["name"])]
        entry["calls"] += 1
        entry["duration_us"] += event.get("dur", 0)
        entry["bytes"] += event.get("args", {}).get("bytes", 0)
        if category in ("kernel", "gpu_memcpy", "gpu_memset"):
            device_intervals.append((event["ts"], event["ts"] + event.get("dur", 0)))
    start = min(x[0] for x in device_intervals)
    stop = max(x[1] for x in device_intervals)
    busy = union_duration(device_intervals)
    rows = [{"category": category, "name": name, **values}
            for (category, name), values in groups.items()]
    totals = defaultdict(lambda: {"calls": 0, "duration_us": 0.0, "bytes": 0})
    for row in rows:
        for key in totals[row["category"]]:
            totals[row["category"]][key] += row[key]
    report = {
        "scope": "One-GPU diagnostic trace. Raw kernels/copies only; annotations are excluded.",
        "trace": str(args.trace.resolve()),
        "event_count": len(events), "category_totals": dict(totals),
        "device_span_seconds": (stop - start) / 1e6,
        "device_interval_union_seconds": busy / 1e6,
        "device_busy_fraction_in_span": busy / (stop - start),
        "device_gap_seconds": (stop - start - busy) / 1e6,
        "top_host_runtime": sorted(
            [x for x in rows if x["category"] in ("cuda_runtime", "cuda_driver")],
            key=lambda x: x["duration_us"], reverse=True,
        )[:15],
        "top_device": sorted(
            [x for x in rows if x["category"] in ("kernel", "gpu_memcpy")],
            key=lambda x: x["duration_us"], reverse=True,
        )[:20],
        "notes": [
            "Kernel/copy sums may overlap; only interval union measures their combined active span.",
            "Runtime calls overlap GPU work; do not add host waits and device durations as exclusive costs.",
            "Device gaps include host gather/dispatch, dependencies and profiler effects, not just Python interpretation.",
            "Warm short partial generation only, not a full TTS/clone latency or an uninstrumented utilization estimate.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "category_totals", "device_span_seconds", "device_interval_union_seconds",
        "device_busy_fraction_in_span", "device_gap_seconds",
    )}, indent=2))


if __name__ == "__main__":
    main()
