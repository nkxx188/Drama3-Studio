"""Measure production decode candidates and record an optional compute profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from benchmark_engine_decode import DramaEngine, EngineOptions, build_drama3_prompt, torch
from drama_low_vram.calibration import calibrate_compute_windows, apply_compute_selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--offsets", type=int, nargs="+", default=[0, 16, 48])
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "research/engine-compute-calibration.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with DramaEngine(EngineOptions(
        checkpoint=ROOT / "checkpoints/drama-3", expert_storage="mmap",
        cache_experts=1, staged_experts=8, pair_rows=64, prefill_chunk=32,
        backbone_graph=True, combined_decode_bank=True,
    )) as engine:
        engine.load()
        text = "<|speaker:0|>你好，这是一次完整的语音推理性能校准，请保持清楚自然的声音。"
        prompt = build_drama3_prompt(text, engine.model.tokenizer, engine.model.config)
        report = calibrate_compute_windows(
            engine, prompt, frames=args.frames, repeats=args.repeats,
            offsets=args.offsets,
            progress=lambda row: print(json.dumps(row), flush=True), output=args.output)
        print(json.dumps(report["selection"], indent=2), flush=True)
        if args.apply:
            apply_compute_selection(engine, report["selection"])


if __name__ == "__main__":
    main()
