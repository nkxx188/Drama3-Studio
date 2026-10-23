"""Completed single-layer cooperation: routing, D2H, CPU/GPU work, H2D and merge."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import gzip
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
from torch.profiler import ProfilerActivity, profile, record_function

from drama_low_vram.expert_cache_file import ExpertCacheFile
from drama_low_vram.hybrid_layer import HybridLayerExperiment
from drama_low_vram.native_backend import NativeLibrary
from drama_low_vram.fp8_decode import build_gpu_silu_table
from drama_low_vram.offload import EXPERT_BUFFER_NAMES, ExpertOffloadManager, ExpertStore


def difference(actual, expected):
    delta = actual.float() - expected.float()
    return {
        "equal": bool(torch.equal(actual, expected)),
        "relative_l2": float(delta.norm() / expected.float().norm().clamp_min(1e-30)),
        "max_abs": float(delta.abs().max()),
    }


def interval_union(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def clock_anchors(records, count=16):
    for _ in range(count):
        before = time.perf_counter_ns()
        with record_function("drama.clock_anchor"):
            inside = time.perf_counter_ns()
            exiting = time.perf_counter_ns()
        after = time.perf_counter_ns()
        records.append((before, inside, exiting, after))


def overlap_trace(path, records, anchors):
    trace = json.loads(path.read_text(encoding="utf-8"))
    anchor_events = [
        event for event in trace["traceEvents"]
        if event.get("ph") == "X" and event.get("cat") == "user_annotation"
        and event.get("name") == "drama.clock_anchor"
    ]
    # Windows wall-clock reads can be quantized near 1ms. Bound the mapping
    # from high-resolution perf_counter to Kineto using traced enter/exit calls.
    lower, upper = [], []
    for event, (before, inside, exiting, after) in zip(anchor_events, anchors, strict=True):
        first, last = event["ts"], event["ts"] + event["dur"]
        lower.extend((first - inside / 1000, last - after / 1000))
        upper.extend((first - before / 1000, last - exiting / 1000))
    offset_low, offset_high = max(lower), min(upper)
    offset = (offset_low + offset_high) / 2
    uncertainty = (offset_high - offset_low) / 2
    gpu = interval_union([
        (event["ts"], event["ts"] + event["dur"])
        for event in trace["traceEvents"]
        if event.get("ph") == "X" and event.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
    ])
    kernels = interval_union([
        (event["ts"], event["ts"] + event["dur"])
        for event in trace["traceEvents"]
        if event.get("ph") == "X" and event.get("cat") == "kernel"
    ])
    copies = interval_union([
        (event["ts"], event["ts"] + event["dur"])
        for event in trace["traceEvents"]
        if event.get("ph") == "X" and event.get("cat") == "gpu_memcpy"
    ])
    scopes = [
        event for event in trace["traceEvents"]
        if event.get("ph") == "X" and event.get("cat") == "user_annotation"
        and event.get("name") == "drama.hybrid.single_layer"
    ]
    aligned = len(scopes) == len(records) and 0 <= uncertainty <= 50
    overlap_us = conservative_us = cpu_us = 0.0
    kernel_overlap_us = copy_overlap_us = 0.0
    for index, report in enumerate(records):
        cpu = report["cpu_interval"]
        if cpu is None:
            continue
        first = cpu["start_ns"] / 1000 + offset
        last = cpu["end_ns"] / 1000 + offset
        if index < len(scopes):
            scope = scopes[index]
            aligned = aligned and (
                scope["ts"] - 1000 <= first <= last <= scope["ts"] + scope["dur"] + 1000
            )
        cpu_us += last - first
        overlap_us += sum(max(0.0, min(last, b) - max(first, a)) for a, b in gpu)
        kernel_overlap_us += sum(max(0.0, min(last, b) - max(first, a)) for a, b in kernels)
        copy_overlap_us += sum(max(0.0, min(last, b) - max(first, a)) for a, b in copies)
        conservative_us += sum(
            max(0.0, min(last - uncertainty, b) - max(first + uncertainty, a))
            for a, b in gpu)
        trace["traceEvents"].append({
            "ph": "X", "cat": "native_cpu_window", "name": "drama.cpu_expert_call",
            "pid": os.getpid(), "tid": cpu["thread_id"], "ts": first,
            "dur": last - first, "args": {"includes": "Python binding validation and synchronous native call"},
        })
    compressed = path.with_suffix(path.suffix + ".gz")
    with gzip.open(compressed, "wt", encoding="utf-8") as handle:
        json.dump(trace, handle)
    path.unlink()
    return {
        "trace": str(compressed), "cpu_clock_alignment_checked": aligned,
        "clock_mapping": "perf_counter bounded by traced annotation entry/exit; no coarse wall clock",
        "clock_mapping_uncertainty_us": uncertainty,
        "cpu_expert_call_window_ms": cpu_us / 1000,
        "actual_gpu_work_overlap_ms": overlap_us / 1000 if aligned else None,
        "conservative_gpu_work_overlap_ms": conservative_us / 1000 if aligned else None,
        "gpu_kernel_overlap_ms": kernel_overlap_us / 1000 if aligned else None,
        "gpu_memcpy_overlap_ms": copy_overlap_us / 1000 if aligned else None,
        "cpu_call_window_overlap_fraction": overlap_us / cpu_us if aligned and cpu_us else None,
        "scope": "Instrumented run only; raw GPU interval union intersected with CPU expert-call windows, including binding validation.",
    }


def launch_comparison(args, store, inputs, library, silu_table=None):
    """Compare early CPU work during DMA with GPU-compute-gated CPU work."""
    launch_modes = ("gather", "dma", "compute")
    reports = []
    with ExitStack() as stack:
        layers = [
            stack.enter_context(HybridLayerExperiment(
                store, cpu_jobs=2, threads=args.threads, cache_experts=2,
                parallel=True, cpu_launch=mode, library=library,
                native_arithmetic=args.arithmetic, silu_table=silu_table,
            )) for mode in launch_modes
        ]
        for layer in layers:
            for _ in range(2):
                for values in inputs:
                    layer.forward(*values)
        samples = [[] for _ in launch_modes]
        outputs = {}
        for repeat in range(args.repeats):
            for variant in ((slot + repeat) % len(layers) for slot in range(len(layers))):
                for index, values in enumerate(inputs):
                    start = time.perf_counter_ns()
                    output, _ = layers[variant].forward(*values)
                    samples[variant].append((time.perf_counter_ns() - start) / 1e6)
                    if not repeat:
                        outputs[variant, index] = output.cpu()
        for variant, (mode, layer) in enumerate(zip(launch_modes, layers, strict=True)):
            records, anchors = [], []
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
                clock_anchors(anchors)
                for values in inputs:
                    with record_function("drama.hybrid.single_layer"):
                        _, record = layer.forward(*values)
                        records.append(record)
                clock_anchors(anchors)
            path = args.trace if mode == "gather" else args.trace.with_name(
                args.trace.stem + "." + mode + args.trace.suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            profiler.export_chrome_trace(str(path))
            reports.append({
                "cpu_launch": mode,
                "completed_median_ms": statistics.median(samples[variant]),
                "samples_ms": samples[variant],
                "profiled_calls": records, "profile_clock_anchors": anchors,
                "overlap": overlap_trace(path, records, anchors),
            })
    return {
        "cpu_jobs": 2, "cache_experts": 2,
        "output_bitwise_equal": all(
            torch.equal(outputs[0, index], outputs[variant, index])
            for variant in range(1, len(launch_modes)) for index in range(len(inputs))),
        "variants": reports,
    }


@torch.inference_mode()
def benchmark(args):
    data = torch.load(args.cases, map_location="cpu", weights_only=True)
    cases = data["cases"][:args.cases_count]
    if not cases:
        raise ValueError("No captured real cases")
    source = ExpertCacheFile.open(ROOT / ".cache/drama3-fp8-experts")
    store = ExpertStore(**source.layer_buffers(data["layer"]))
    library = NativeLibrary()
    silu_table = build_gpu_silu_table() if args.gpu_silu_table else None
    device = torch.device("cuda:0")
    inputs = [
        (case["x"].to(device), case["ids"].to(device), case["weights"].to(device))
        for case in cases
    ]
    expected = [case["output"].to(device) for case in cases]
    configs = [(0, True), *[(count, parallel) for count in args.cpu_jobs if count
                            for parallel in (False, True)]]
    results = []
    raw = []
    production_rows = []
    profiler_result = None
    try:
        for cache in args.cache_experts:
            with ExitStack() as stack:
                production = ExpertOffloadManager(
                    cache_experts=cache, max_staged_experts=8,
                    combined_decode_bank=True)
                stack.callback(production.close)
                module = SimpleNamespace(
                    fp8=True, **{name: getattr(store, name) for name in EXPERT_BUFFER_NAMES})
                layers = [
                    stack.enter_context(HybridLayerExperiment(
                        store, cpu_jobs=count, threads=args.threads,
                        cache_experts=cache, parallel=parallel, library=library,
                        native_arithmetic=args.arithmetic, silu_table=silu_table,
                    )) for count, parallel in configs
                ]
                for layer in layers:
                    for _ in range(2):
                        for values in inputs:
                            layer.forward(*values)
                for _ in range(2):
                    for values in inputs:
                        production.forward(module, *values)
                torch.cuda.synchronize()
                samples = [[] for _ in configs]
                outputs = {}
                production_samples, production_errors = [], []
                for repeat in range(args.repeats):
                    for index, values in enumerate(inputs):
                        start = time.perf_counter_ns()
                        output = production.forward(module, *values)
                        torch.cuda.current_stream(device).synchronize()
                        production_samples.append((time.perf_counter_ns() - start) / 1e6)
                        if repeat == 0:
                            production_errors.append(difference(output, expected[index]))
                    order = [(slot + repeat) % len(configs) for slot in range(len(configs))]
                    for variant in order:
                        layer = layers[variant]
                        for case_index, values in enumerate(inputs):
                            start = time.perf_counter_ns()
                            out, report = layer.forward(*values)
                            report["completed_call_ms"] = (time.perf_counter_ns() - start) / 1e6
                            report.update(cache_experts=cache, cpu_jobs=configs[variant][0],
                                          parallel=configs[variant][1], repeat=repeat, case=case_index)
                            samples[variant].append(report)
                            if repeat == 0:
                                outputs[variant, case_index] = out.cpu()
                production_rows.append({
                    "cache_experts": cache,
                    "completed_median_ms": statistics.median(production_samples),
                    "completed_mean_ms": statistics.mean(production_samples),
                    "official_gpu_output_errors": production_errors,
                    "samples_ms": production_samples,
                })
                for variant, (count, parallel) in enumerate(configs):
                    measured = samples[variant]
                    times = [entry["completed_call_ms"] for entry in measured]
                    mean_repeats = [
                        statistics.mean(entry["completed_call_ms"] for entry in measured if entry["repeat"] == repeat)
                        for repeat in range(args.repeats)
                    ]
                    numerical = [
                        difference(outputs[variant, index], expected[index].cpu())
                        for index in range(len(inputs))
                    ]
                    serial_variant = configs.index((count, False)) if count else variant
                    row = {
                        "cache_experts": cache, "cpu_jobs": count,
                        "gpu_jobs": 8 - count, "parallel": parallel,
                        "cpu_threads": args.threads if count else 0,
                        "completed_median_ms": statistics.median(times),
                        "completed_mean_ms": statistics.mean(times),
                        "repeat_means_ms": mean_repeats,
                        "median_components_ms": {
                            key: statistics.median(entry[key] for entry in measured)
                            for key in ("handoff_host_ms", "gpu_submission_host_ms",
                                        "cpu_remaining_wait_host_ms", "cpu_job_ms",
                                        "gpu_compute_and_cache_ms", "cpu_result_h2d_ms",
                                        "merge_ms", "host_gather_ms")
                        },
                        "mean_gpu_expert_h2d_bytes": statistics.mean(entry["gpu_expert_h2d_bytes"] for entry in measured),
                        "control_d2h_bytes": measured[0]["control_d2h_bytes"],
                        "cpu_result_h2d_bytes": measured[0]["cpu_result_h2d_bytes"],
                        "official_gpu_output_errors": numerical,
                        "serial_parallel_bitwise_equal": all(
                            torch.equal(outputs[variant, index], outputs[serial_variant, index])
                            for index in range(len(inputs))
                        ),
                    }
                    results.append(row)
                    raw.extend(measured)
                    print(json.dumps({
                        key: row[key] for key in (
                            "cache_experts", "cpu_jobs", "parallel",
                            "completed_median_ms", "serial_parallel_bitwise_equal",
                        )
                    }), flush=True)
        launches = launch_comparison(args, store, inputs, library, silu_table)
        profiler_result = launches["variants"][0]["overlap"]
        for row in results:
            baseline = next(r for r in results if r["cache_experts"] == row["cache_experts"] and not r["cpu_jobs"])
            row["speedup_vs_gpu_only"] = baseline["completed_median_ms"] / row["completed_median_ms"]
            reference = next(r for r in production_rows if r["cache_experts"] == row["cache_experts"])
            row["speedup_vs_existing_manager"] = reference["completed_median_ms"] / row["completed_median_ms"]
            if row["cpu_jobs"]:
                serial = next(r for r in results if (
                    r["cache_experts"] == row["cache_experts"]
                    and r["cpu_jobs"] == row["cpu_jobs"] and not r["parallel"]))
                row["speedup_vs_same_split_serial"] = serial["completed_median_ms"] / row["completed_median_ms"]
        return {
            "scope": "Single real MoE routed-expert branch; all eight routes retained, including GPU route handoff, CPU/GPU work, result H2D and ordered FP32 merge. Not full speech or a full transformer block.",
            "cases": len(cases), "repeats": args.repeats,
            "source_layer": data["layer"],
            "native_arithmetic": args.arithmetic, "gpu_silu_table": args.gpu_silu_table,
            "library_sha256": hashlib.sha256(library.path.read_bytes()).hexdigest(),
            "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
            "torch": torch.__version__, "device": torch.cuda.get_device_name(),
            "torch_intraop_threads": torch.get_num_threads(),
            "gpu_tensor_peak_allocated": torch.cuda.max_memory_allocated(device),
            "results": results, "existing_manager_baselines": production_rows,
            "calls": raw, "overlap": profiler_result,
            "launch_comparison": launches,
            "adoption_gate": {
                "required": "Official bitwise equality until a separate speech-quality tolerance is established.",
                "single_layer_exact_output_gate_passed": all(
                    error["equal"] for row in results for error in row["official_gpu_output_errors"]),
                "passed": False,
                "reason": "Single-layer data alone does not validate full-engine speech or all hardware.",
                "integrated_in_default_engine": False,
            },
            "notes": [
                "The CPU owns low-ranked cold routes first; cache hits stay on the GPU when possible.",
                "Shared original GPU FP8 input bytes/scales remove input quantization differences.",
                "Serial and parallel runs use the same split and arithmetic; official GPU numerical errors remain explicit.",
                "Captured GPU inputs/routes are prepared before timing; router GEMM, attention and shared expert are excluded.",
                "Expert mmap pages are warm; no physical SSD miss benchmark.",
                "Variant order rotates each repeat; component times can overlap and must not be summed.",
                "CUDA-event intervals include stream scheduling; actual overlap is separately checked with raw profiler intervals.",
                "This isolated experiment sets Torch intra-op threads to one for all managers; full-engine threading and GPU memory pressure are outside the measurement.",
            ],
        }
    finally:
        source.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "research/native-expert-cases.pt")
    parser.add_argument("--cases-count", type=int, default=11)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--arithmetic", choices=("reference", "sm89"), default="reference")
    parser.add_argument("--gpu-silu-table", action="store_true")
    parser.add_argument("--cpu-jobs", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--cache-experts", type=int, nargs="+", default=[0, 2, 4])
    parser.add_argument("--output", type=Path, default=ROOT / "research/hybrid-layer-benchmark.json")
    parser.add_argument("--trace", type=Path, default=ROOT / "research/hybrid-layer.trace.json")
    args = parser.parse_args()
    if (min(args.cases_count, args.repeats, args.threads) < 1
            or any(not 0 <= count <= 8 for count in args.cpu_jobs)
            or any(not 0 <= count <= 8 for count in args.cache_experts)):
        parser.error("Positive counts and at most eight CPU jobs/cache experts required")
    torch.set_num_threads(1)
    report = benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {args.output}", flush=True)
    print(json.dumps(report["overlap"], indent=2), flush=True)


if __name__ == "__main__":
    main()
