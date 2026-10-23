"""Completed voice-clone A/B of bounded reference-state reuse."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from benchmark_engine_decode import DramaEngine, EngineOptions, torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--cpu-experts", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "research/engine-prefix-comparison.json")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("Positive repetitions required")
    torch.set_num_threads(1)
    short = "<|speaker:0|>你好，这是官方 FP8 质量优先测试。"
    text = "<|speaker:0|>今天我们继续测试这套本地语音引擎。希望普通电脑也能保持清楚自然的声音，读完一整句话，而不需要购买昂贵的专业显卡。接下来还会仔细检查语音克隆、长句停顿和声音的稳定性。"
    report = {
        "scope": "Same loaded official model, CPU share, seed, reference and full utterance. Only reference-state reuse changes. Common forced codec parking.",
        "configuration": {
            "cache_experts": 2, "staged_experts": 8, "pair_rows": 64, "prefill_chunk": 32,
            "cpu_experts": args.cpu_experts, "native_threads": 4, "torch_threads": 1,
            "prefix_cache_mib": 256, "codec_policy": "park",
        },
        "requests": [], "warmups": [],
    }

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with DramaEngine(EngineOptions(
        checkpoint=ROOT / "checkpoints/drama-3", expert_storage="mmap",
        cache_experts=2, staged_experts=8, pair_rows=64, prefill_chunk=32,
        backbone_graph=True, combined_decode_bank=True,
        prefix_cache_mib=256, cpu_experts=args.cpu_experts, codec_policy="park",
    )) as engine:
        engine.load()
        for repeat in range(-1, args.repeats):
            order = (True,) if repeat < 0 else ((False, True) if repeat % 2 == 0 else (True, False))
            for enabled in order:
                label = "cache" if enabled else "baseline"
                suffix = "warmup" if repeat < 0 else f"{label}-{repeat + 1}"
                output = ROOT / "outputs" / f"engine-prefix-clone-{suffix}.wav"
                last = time.perf_counter()

                def progress(frames, _limit):
                    nonlocal last
                    now = time.perf_counter()
                    if now - last >= 20:
                        print(json.dumps({"phase": suffix, "frames": frames}), flush=True)
                        last = now

                with nullcontext() if enabled else engine.prefix_cache.suspend():
                    result = engine.synthesize(
                        text, output, seed=123, max_new_tokens=768,
                        reference_audio=ROOT / "outputs/quality-official-fp8.wav",
                        reference_text=short, progress=progress)
                row = {
                    "variant": label, "repeat": repeat + 1, "status": result["status"],
                    "report_file": str(output.with_suffix(".json")), "audio": result["audio"],
                    "frames": [chunk["frames"] for chunk in result["chunks"]],
                    "code_hashes": [chunk["semantic_codes_sha256"] for chunk in result["chunks"]],
                    "generation_seconds": sum(chunk["generation_seconds"] for chunk in result["chunks"]),
                    "prefill_seconds": sum(chunk["prefill_seconds"] for chunk in result["chunks"]),
                    "decode_seconds": sum(chunk["decode_seconds"] for chunk in result["chunks"]),
                    "first_frame_seconds": [chunk["first_frame_seconds"] for chunk in result["chunks"]],
                    "runtime_prepare_seconds": sum(chunk["runtime_prepare_seconds"] for chunk in result["chunks"]),
                    "total_seconds": result["total_seconds"],
                    "h2d_bytes": result["offload_delta"]["h2d_bytes"],
                    "lifecycle": result["lifecycle_delta"], "memory": result["memory"],
                    "prefix_cache": [chunk["prefix_cache"] for chunk in result["chunks"]],
                    "prefix_residency": engine.prefix_cache.report(),
                    "wav_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                    "peak_tensor_bytes": max(chunk["peak_allocated_bytes"] for chunk in result["chunks"]),
                }
                report["warmups" if repeat < 0 else "requests"].append(row)
                save()
                print(json.dumps({key: row[key] for key in (
                    "variant", "repeat", "frames", "generation_seconds", "prefill_seconds",
                    "decode_seconds", "total_seconds", "prefix_cache", "wav_sha256")}), flush=True)
                if result["status"] != "complete":
                    raise RuntimeError("Incomplete clone; no performance claim")
                peers = report["warmups"] + report["requests"]
                if any(item["code_hashes"] != row["code_hashes"] or item["wav_sha256"] != row["wav_sha256"]
                       for item in peers):
                    raise RuntimeError("Prefix reuse changed complete speech codes/WAV")
        rows = report["requests"]
        report["summary"] = {
            **{
                f"{label}_{key}": statistics.median(row[key] for row in rows if row["variant"] == label)
                for label in ("baseline", "cache")
                for key in ("prefill_seconds", "generation_seconds", "decode_seconds", "total_seconds")
            },
            "all_codes_equal": len({tuple(row["code_hashes"]) for row in rows}) == 1,
            "all_wavs_equal": len({row["wav_sha256"] for row in rows}) == 1,
            "all_lifecycle_counts_equal": len({
                tuple(row["lifecycle"][key] for key in (
                    "runtime_reuses", "runtime_rebuilds", "park_calls", "resume_calls",
                    "codec_coexist_calls", "codec_park_calls", "codec_cache_reclaims"))
                for row in rows}) == 1,
        }
        summary = report["summary"]
        summary["total_time_reduction_fraction"] = 1 - summary["cache_total_seconds"] / summary["baseline_total_seconds"]
        save()
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
