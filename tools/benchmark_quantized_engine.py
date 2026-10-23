"""Actual DramaEngine requests and resources for one explicitly chosen format."""

import argparse
import cProfile
from pathlib import Path
import json
import pstats
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor/fish-speech")]
from drama_low_vram.engine import DramaEngine, EngineOptions
import torch

SHORT = "<|speaker:0|>你好，这是官方 FP8 质量优先测试。"
LONG = (
    "<|speaker:0|>今天我们继续测试这套本地语音引擎。希望普通电脑也能保持清楚自然的声音，"
    "读完一整句话，而不需要购买昂贵的专业显卡。接下来还会仔细检查语音克隆、"
    "长句停顿和声音的稳定性。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cache-experts", type=int)
    parser.add_argument("--prefill-chunk", type=int, default=32)
    parser.add_argument("--cpu-experts", type=int, default=0)
    parser.add_argument("--pair-rows", type=int, default=64)
    parser.add_argument("--staged-experts", type=int, default=8)
    parser.add_argument("--baseline-tiles", action="store_true")
    parser.add_argument("--profile-codec", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cases", nargs="+", choices=("clone-short", "clone-long", "tts", "tts-short"),
                        default=["clone-short", "clone-long", "clone-long", "tts"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = {"label": args.label, "checkpoint": str(args.checkpoint.resolve()),
              "scope": "Complete real requests unless smoke. Formats can emit different "
              "durations/codes; compare frame rate and RTF as well as total time.",
              "results": [], "smoke": args.smoke}
    started = time.perf_counter()
    with DramaEngine(EngineOptions(
        checkpoint=args.checkpoint, expert_storage="auto",
        cache_experts=args.cache_experts, staged_experts=args.staged_experts,
        pair_rows=args.pair_rows,
        prefill_chunk=args.prefill_chunk, backbone_graph=True,
        combined_decode_bank=True, host_gather="auto", cpu_experts=args.cpu_experts,
        native_threads=4, prefix_cache_mib=256,
    )) as engine:
        engine.load()
        if args.profile_codec:
            original_decode = engine._decode_audio_segments
            codec_profiles = report["codec_profiles"] = []
            def profiled_decode(segments):
                profiler = cProfile.Profile()
                started_codec = time.perf_counter()
                value = profiler.runcall(original_decode, segments)
                rows = [
                    {"file": file, "line": line, "function": function, "calls": total,
                     "self_seconds": own, "cumulative_seconds": cumulative}
                    for (file, line, function), (primitive, total, own, cumulative, callers)
                    in pstats.Stats(profiler).stats.items()
                ]
                row = {"seconds": time.perf_counter() - started_codec,
                       "hotspots": sorted(rows, key=lambda row: -row["self_seconds"])[:30]}
                codec_profiles.append(row)
                print(json.dumps({"phase": "codec-profile", **row}), flush=True)
                return value
            engine._decode_audio_segments = profiled_decode
        if args.baseline_tiles:
            engine.manager.backend.tuned_tiles = False
        report["cold_load_seconds"] = time.perf_counter() - started
        report["baseline_tiles"] = args.baseline_tiles
        cases = ["clone-short"] if args.smoke else args.cases
        for index, case in enumerate(cases):
            text = SHORT if case in ("clone-short", "tts-short") else LONG
            path = ROOT / "outputs" / f"quant-{args.label}-{index}.wav"
            last = time.perf_counter()
            def progress(frames, limit):
                nonlocal last
                if time.perf_counter() - last >= 20:
                    print(json.dumps({"phase": "quant-progress", "label": args.label,
                                      "index": index, "frames": frames}), flush=True)
                    last = time.perf_counter()
            reference = {} if case.startswith("tts") else {
                "reference_audio": ROOT / "outputs/quality-official-fp8.wav",
                "reference_text": SHORT,
            }
            result = engine.synthesize(
                text, path, seed=args.seed,
                max_new_tokens=4 if args.smoke else args.max_new_tokens,
                allow_partial=args.smoke or args.allow_partial, progress=progress, **reference)
            result["benchmark_case"] = case
            report["results"].append(result)
            (ROOT / "research" / f"quant-{args.label}-benchmark.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"phase": "quant-result", "label": args.label,
                              "index": index, "status": result["status"],
                              "total_seconds": result["total_seconds"],
                              "audio": result["audio"],
                              "decode_seconds": sum(c["decode_seconds"] for c in result["chunks"]),
                              "frames": sum(c["frames"] for c in result["chunks"]),
                              "generation_peak_allocated_bytes": max(
                                  c["peak_allocated_bytes"] for c in result["chunks"]),
                              "generation_peak_reserved_bytes": max(
                                  c["peak_reserved_bytes"] for c in result["chunks"]),
                              "memory": result["memory"]}), flush=True)
        if not args.smoke and any(r["status"] != "complete" for r in report["results"]):
            raise RuntimeError("Some quantized requests did not reach normal EOS; see report")


if __name__ == "__main__":
    main()
