"""Production decode-only profiler: exclude prefill, snapshots, codec and graphs."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import gzip
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from benchmark_engine_decode import DramaEngine, EngineOptions, build_drama3_prompt, seed_sampling, torch
from drama_low_vram.verification import DecodeCursor
from torch.profiler import ProfilerActivity, profile, record_function


def labeled(function, label):
    def run(*args, **kwargs):
        with record_function(label):
            return function(*args, **kwargs)
    return run


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--cpu-experts", type=int, default=2)
    parser.add_argument("--native-threads", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "research/decode-only-profile.json")
    parser.add_argument("--trace", type=Path, default=ROOT / "research/decode-only-profile.trace.json.gz")
    args = parser.parse_args()
    if args.frames < 2:
        parser.error("At least two frames required")
    torch.set_num_threads(1)
    with DramaEngine(EngineOptions(
        checkpoint=ROOT / "checkpoints/drama-3", expert_storage="mmap",
        cache_experts=1, staged_experts=8, pair_rows=64, prefill_chunk=32,
        backbone_graph=True, combined_decode_bank=True,
        cpu_experts=args.cpu_experts, native_threads=args.native_threads, host_gather="auto",
        profile_transfers=True,
    )) as engine:
        engine.load()
        prompt = build_drama3_prompt(
            "<|speaker:0|>我们继续测量这套引擎，让声音保持清楚自然。",
            engine.model.tokenizer, engine.model.config)
        engine._prepare(512)
        seed_sampling(123)
        cursor = DecodeCursor.prefill(
            engine.model, prompt, tower_graph=engine.graph, backbone_graph=engine.backbone_graph)
        base = cursor.capture()

        def run():
            return [cursor.step() for _ in range(args.frames)]

        run()  # Compile and warm actual routes outside the measured samples.
        cursor.restore(base)
        engine.manager.clear_hot_cache()
        torch.cuda.synchronize(engine.device)
        started = time.perf_counter()
        expected = run()
        torch.cuda.synchronize(engine.device)
        uninstrumented = time.perf_counter() - started
        cursor.restore(base)
        engine.manager.clear_hot_cache()
        before = engine.manager.stats.as_dict()
        with ExitStack() as stack:
            scopes = [
                (engine.backbone_graph, "forward", "decode.backbone"),
                (engine.graph, "step", "decode.tower"),
                (engine.manager, "forward", "decode.experts"),
                (engine.manager.pool, "stage", "decode.host_gather_and_h2d"),
                (engine.manager.pool, "combine_resident", "decode.combine_resident"),
                (engine.manager.backend, "forward", "decode.expert_kernel_submission"),
            ]
            if engine.manager.hybrid_decode is not None:
                scopes += [
                    (engine.manager.hybrid_decode, "forward", "decode.hybrid"),
                    (engine.manager.hybrid_decode, "_cpu", "decode.native_cpu"),
                ]
            for owner, name, label in scopes:
                stack.enter_context(patch.object(owner, name, labeled(getattr(owner, name), label)))
            started = time.perf_counter()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
                actual = run()
                torch.cuda.synchronize(engine.device)
            wall = time.perf_counter() - started
        engine.manager.synchronize()
        after = engine.manager.stats.as_dict()
        equal = all(
            torch.equal(a[key], b[key]) for a, b in zip(actual, expected)
            for key in ("frame", "semantic_logits"))
        rows = [{
            "name": event.key, "calls": event.count,
            "self_cpu_us": event.self_cpu_time_total, "inclusive_cpu_us": event.cpu_time_total,
        } for event in profiler.key_averages()]
        temporary = args.trace.with_suffix("")
        profiler.export_chrome_trace(str(temporary))
        with temporary.open("rb") as source, gzip.open(args.trace, "wb") as target:
            while data := source.read(1024**2):
                target.write(data)
        temporary.unlink()
        excluded = {
            "cpu_expert_bytes", "pinned_bytes", "device_buffer_bytes", "hot_cache_bytes",
            "hot_cache_capacity_experts", "peak_staged_experts",
        }
        report = {
            "scope": "Fixed warm decode-only window. Prefill/graph capture/codec/snapshot restore outside timers. Profiler alters timing; component host scopes are inclusive.",
            "frames": args.frames, "cpu_experts": args.cpu_experts,
            "native_threads": args.native_threads,
            "uninstrumented_seconds": uninstrumented, "profiled_wall_seconds": wall,
            "frames_and_semantic_logits_equal": equal,
            "offload_delta": {key: after[key] - before[key] for key in before if key not in excluded},
            "top_self_cpu": sorted(rows, key=lambda row: row["self_cpu_us"], reverse=True)[:35],
            "trace": str(args.trace),
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        if not equal:
            raise RuntimeError("Profiler diagnostic changed continuation")


if __name__ == "__main__":
    main()
