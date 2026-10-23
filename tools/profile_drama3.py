"""Profile the official FP8 path and collect representative hidden states."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor" / "fish-speech")]

from drama_low_vram.autotune import autotune
from drama_low_vram.model import (
    configure_model_runtime,
    load_offloaded_drama3,
    release_drama3,
)
from fish_speech.models.text2semantic.fish_omni.generate import (
    build_drama3_prompt,
    generate_drama3,
)
from fish_speech.models.text2semantic.sampling import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)
from drama_low_vram.sampling import DEFAULT_TOP_K


def instrument(module, method, label, timers, hidden_states=None):
    original = getattr(module, method)

    def wrapped(self, *args, **kwargs):
        started = time.perf_counter()
        with record_function(label):
            result = original(*args, **kwargs)
        entry = timers.setdefault(label, {"calls": 0, "host_seconds": 0.0})
        entry["calls"] += 1
        entry["host_seconds"] += time.perf_counter() - started
        if hidden_states is not None:
            hidden_states.append(args[0].detach().to("cpu").clone())
        return result

    setattr(module, method, types.MethodType(wrapped, module))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/drama-3")
    parser.add_argument("--text", default="<|speaker:0|>你好，这是官方 FP8 质量优先测试。")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--output", type=Path, default=ROOT / "research/profile-official.json")
    parser.add_argument("--hidden-output", type=Path, default=ROOT / "research/profile-hidden.pt")
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    os.environ["FISH_DRAMA3_CUDA_GRAPH"] = "0"
    runtime = autotune(device="cuda:0")
    if not runtime.can_run:
        raise RuntimeError("; ".join(runtime.warnings))
    model, manager = load_offloaded_drama3(
        args.checkpoint,
        device=runtime.device,
        lm_head_device=runtime.lm_head_device,
        expert_backend=runtime.backend,
        max_staged_experts=runtime.max_staged_experts,
        max_pair_rows=runtime.max_pair_rows,
        pin_memory=runtime.pin_memory,
    )
    timers, hidden = {}, []
    try:
        prompt = build_drama3_prompt(args.text, model.tokenizer, model.config)
        plan = configure_model_runtime(
            model, manager, sequence_capacity=len(prompt) + args.max_new_tokens,
            runtime_config=runtime,
        )
        instrument(model.backbone, "compute_logits", "drama.lm_head", timers, hidden)
        instrument(model.backbone, "forward", "drama.backbone", timers)
        instrument(model.backbone, "embed", "drama.embedding", timers)
        instrument(model.audio_decoder, "decode_step", "drama.tower_decode", timers)
        instrument(model.audio_decoder, "prefill", "drama.tower_prefill", timers)
        instrument(model.audio_decoder, "embed_backbone", "drama.audio_embedding", timers)
        torch.manual_seed(123)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
            frames, reason = generate_drama3(
                model, prompt, max_new_tokens=args.max_new_tokens,
                temperature=DEFAULT_TEMPERATURE, top_p=DEFAULT_TOP_P,
                top_k=DEFAULT_TOP_K, prefill_chunk_size=plan.prefill_chunk,
            )
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        events = profiler.key_averages()
        rows = [
            {
                "name": event.key,
                "calls": event.count,
                "self_cpu_us": event.self_cpu_time_total,
                "cpu_us": event.cpu_time_total,
                "self_cuda_us": event.self_device_time_total,
                "cuda_us": event.device_time_total,
            }
            for event in events
        ]
        result = {
            "scope": "Instrumented prefill and decode; profiling affects timing.",
            "frames": len(frames),
            "finish_reason": reason,
            "elapsed_seconds": elapsed,
            "timers": timers,
            "top_self_cpu": sorted(rows, key=lambda row: row["self_cpu_us"], reverse=True)[:25],
            "top_self_cuda": sorted(rows, key=lambda row: row["self_cuda_us"], reverse=True)[:25],
            "runtime_plan": plan.as_dict(),
            "offload": manager.stats.as_dict(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        torch.save(torch.cat(hidden), args.hidden_output)
        if args.trace:
            profiler.export_chrome_trace(str(args.trace))
        print(json.dumps(result, indent=2), flush=True)
    finally:
        release_drama3(model, manager)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
