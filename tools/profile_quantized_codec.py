"""Isolate codec latency while retaining a real compressed Drama3 backbone."""

import argparse
import cProfile
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pstats
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor/fish-speech")]
import torch
from drama_low_vram.engine import DramaEngine, EngineOptions


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/drama-3-int8")
    parser.add_argument("--label", default="int8")
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = {"scope": "Codec diagnostic using repeated reference codes and a real resident "
              "backbone, not generated speech. cProfile host time includes CUDA waits.",
              "label": args.label, "rows": []}
    with DramaEngine(EngineOptions(
        checkpoint=args.checkpoint, staged_experts=8, pair_rows=64, prefill_chunk=32,
        backbone_graph=True, combined_decode_bank=True, host_gather="auto",
        prefix_cache_mib=256,
    )) as engine:
        engine.load()
        codes, _ = engine.prepare_reference(
            ROOT / "outputs/quality-official-fp8.wav",
            "<|speaker:0|>你好，这是官方 FP8 质量优先测试。")
        codes = codes.repeat(1, (375 + codes.shape[1] - 1) // codes.shape[1])[:, :375]
        for index, placement in enumerate(("auto", "auto", "full", "auto")):
            engine.options = replace(engine.options, codec_placement=placement)
            engine._prepare(947)
            before = dict(engine._lifecycle)
            profiler = cProfile.Profile()
            start = time.perf_counter()
            profiler.enable()
            audio = engine._decode_audio_segments([codes])
            profiler.disable()
            seconds = time.perf_counter() - start
            stats = pstats.Stats(profiler)
            rows = []
            for (file, line, function), (primitive, total, own, cumulative, callers) in stats.stats.items():
                rows.append({"file": file, "line": line, "function": function,
                             "calls": total, "self_seconds": own, "cumulative_seconds": cumulative})
            row = {
                "placement": placement, "seconds": seconds,
                "waveform_sha256": hashlib.sha256(audio.view(torch.uint8).numpy().tobytes()).hexdigest(),
                "lifecycle": {key: value - before[key] for key, value in engine._lifecycle.items()},
                "hotspots": sorted(rows, key=lambda value: -value["self_seconds"])[:35],
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
            report["rows"].append(row)
            (ROOT / "research" / f"phase19-codec-profile-{args.label}.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"placement": placement, "seconds": seconds,
                              "hotspots": row["hotspots"][:8], "lifecycle": row["lifecycle"]}), flush=True)
            profiler.dump_stats(str(ROOT / "research" / f"phase19-codec-{args.label}-{index}.prof"))


if __name__ == "__main__":
    main()
