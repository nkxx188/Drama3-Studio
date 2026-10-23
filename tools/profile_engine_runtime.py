"""Profile the current segmented-graph engine and capture real MoE inputs."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import gzip
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

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
from torch.profiler import ProfilerActivity, profile, record_function

from drama_low_vram.engine import DramaEngine, EngineOptions, seed_sampling
from drama_low_vram.generation import generate
from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt


def labeled(function, name):
    def wrapped(*args, **kwargs):
        with record_function(name):
            return function(*args, **kwargs)
    return wrapped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--cache-experts", type=int, default=3)
    parser.add_argument("--cpu-experts", type=int, default=0)
    parser.add_argument("--native-threads", type=int, default=4)
    parser.add_argument("--host-threads", type=int, default=1)
    parser.add_argument("--prefix-cache-mib", type=int, default=0)
    parser.add_argument("--clone", action="store_true")
    parser.add_argument("--capacity", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "research/profile-engine-native-baseline.json")
    parser.add_argument("--cases", type=Path, default=ROOT / "research/native-expert-cases.pt")
    parser.add_argument("--trace", type=Path, default=ROOT / "research/profile-engine-native-baseline.trace.json.gz")
    args = parser.parse_args()
    if args.frames < 2:
        parser.error("At least two frames are needed for a decode-layer capture")
    if args.host_threads < 1:
        parser.error("Host threads must be positive")
    torch.set_num_threads(args.host_threads)
    with DramaEngine(EngineOptions(
        checkpoint=ROOT / "checkpoints/drama-3", expert_storage="mmap",
        cache_experts=args.cache_experts, staged_experts=8, pair_rows=64,
        prefill_chunk=32, backbone_graph=True, combined_decode_bank=True,
        profile_transfers=True,
        cpu_experts=args.cpu_experts, native_threads=args.native_threads,
        prefix_cache_mib=args.prefix_cache_mib,
    )) as engine:
        engine.load()
        text = "<|speaker:0|>你好，这是官方 FP8 质量优先测试。"
        codes = None
        if args.clone:
            codes, _ = engine.prepare_reference(
                ROOT / "outputs/quality-official-fp8.wav", text)
        prompt = build_drama3_prompt(
            text,
            engine.model.tokenizer, engine.model.config,
            prompt_texts=[text] if args.clone else [],
            prompt_tokens=[codes] if args.clone else [],
        )
        capacity = args.capacity or len(prompt) + max(256, args.frames)
        plan, _ = engine._prepare(capacity)
        manager = engine.manager
        first_module = next(iter(manager.modules.values()))
        original = manager.forward
        cases = []

        def capture(module, x, ids, weights):
            result = original(module, x, ids, weights)
            if module is first_module and x.shape[0] == 1 and len(cases) < args.frames:
                cases.append({
                    "x": x.detach().cpu().clone(),
                    "ids": ids.detach().cpu().clone(),
                    "weights": weights.detach().cpu().clone(),
                    "output": result.detach().cpu().clone(),
                })
            return result

        def run():
            seed_sampling(123)
            return generate(
                engine.model, prompt, max_new_tokens=args.frames,
                prefill_chunk_size=plan.prefill_chunk,
                tower_graph=engine.graph, backbone_graph=engine.backbone_graph,
                prefix_cache=engine.prefix_cache,
            )

        with patch.object(manager, "forward", capture):
            warm = run()
        args.cases.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "source": "Real layer-zero MoE decode input/output captured before profiling.",
            "layer": 0, "checkpoint": str(engine.checkpoint),
            "cases": cases,
        }, args.cases)
        # One uninstrumented, already-warm run separates profiler overhead.
        baseline = run()
        before = manager.stats.as_dict()
        with ExitStack() as stack:
            scopes = [
                (manager, "forward", "drama.expert_dispatch"),
                (manager.pool, "stage", "drama.host_gather_and_h2d"),
                (manager.backend, "forward", "drama.expert_kernel_submission"),
                (manager.backend, "forward_pairs", "drama.expert_pair_kernel_submission"),
                (engine.backbone_graph, "forward", "drama.backbone_graph"),
                (engine.graph, "step", "drama.audio_tower_graph"),
                (engine.model.backbone, "forward", "drama.backbone_prefill"),
                (engine.model.audio_decoder, "prefill", "drama.audio_tower_prefill"),
            ]
            if manager.hybrid_decode is not None:
                scopes.append(
                    (manager.hybrid_decode, "forward", "drama.hybrid_dispatch"))
            for owner, method, label in scopes:
                stack.enter_context(patch.object(owner, method, labeled(getattr(owner, method), label)))
            started = time.perf_counter()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
                measured = run()
                torch.cuda.synchronize(engine.device)
            wall = time.perf_counter() - started
        manager.synchronize()
        after = manager.stats.as_dict()
        rows = [{
            "name": event.key, "calls": event.count,
            "self_cpu_us": event.self_cpu_time_total, "inclusive_cpu_us": event.cpu_time_total,
            "self_device_us": event.self_device_time_total,
            "inclusive_device_us": event.device_time_total,
        } for event in profiler.key_averages()]
        trace_temp = args.trace.with_suffix("")
        trace_temp.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(trace_temp))
        with trace_temp.open("rb") as src, gzip.open(args.trace, "wb") as dst:
            while block := src.read(1024 * 1024):
                dst.write(block)
        trace_temp.unlink()
        report = {
            "scope": "Short diagnostic only; no codec or complete speech claim. Instrumentation changes timings.",
            "configuration": {"cache_experts": args.cache_experts, "staged_experts": 8,
                              "pair_rows": 64, "prefill_chunk": 32,
                              "backbone_graph": True, "combined_decode_bank": True,
                              "profile_transfers": True},
            "runtime_plan": plan.as_dict(), "warm_capture_run": warm.as_dict(),
            "uninstrumented_warm": baseline.as_dict(), "instrumented": measured.as_dict(),
            "same_frames": bool(torch.equal(baseline.frames, measured.frames)),
            "profiler_wall_seconds": wall, "captured_real_expert_cases": len(cases),
            "capture": str(args.cases), "trace": str(args.trace),
            "offload_delta": {key: after[key] - before[key] for key in before if key not in
                              ("cpu_expert_bytes", "pinned_bytes", "device_buffer_bytes",
                               "hot_cache_bytes", "hot_cache_capacity_experts", "peak_staged_experts")},
            "top_self_cpu": sorted(rows, key=lambda row: row["self_cpu_us"], reverse=True)[:30],
            "top_self_device": sorted(rows, key=lambda row: row["self_device_us"], reverse=True)[:30],
            "notes": [
                "Host scopes are inclusive and overlap; do not sum them as exclusive phases.",
                "CUDA event H2D is measured with profiling enabled and includes stream queue behavior.",
                "Real MoE captures occur in the unprofiled warm run, not inside the trace.",
                "The current graphs are captured before warmup and profiling.",
            ],
        }
        report["configuration"].update({
            "cpu_experts": args.cpu_experts, "native_threads": args.native_threads,
            "host_threads": args.host_threads, "prefix_cache_mib": args.prefix_cache_mib,
            "clone": args.clone,
            "requested_capacity": capacity,
            "actual_capacity": engine.model.cache_capacity,
        })
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: report[key] for key in
                          ("uninstrumented_warm", "instrumented", "same_frames",
                           "offload_delta", "captured_real_expert_cases")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
