"""First-run calibration and fresh-engine cached policy acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from benchmark_engine_decode import DramaEngine, EngineOptions, torch
from drama_low_vram.workflows import DramaWorkflows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path,
                        default=ROOT / ".cache/compute-profiles-validation")
    parser.add_argument("--output", type=Path, default=ROOT / "research/auto-compute-acceptance.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = {
        "scope": "Real TTS through automatic production preparation. Exact generated VQ and matching cached policy are gates. Cross-cold-start codec WAV identity is measured separately, not assumed.",
        "requests": [],
    }
    codes, wav_hash = None, None
    for repetition in range(2):
        with DramaEngine(EngineOptions(
            checkpoint=ROOT / "checkpoints/drama-3", expert_storage="mmap",
            cache_experts=1, staged_experts=8, pair_rows=64, prefill_chunk=32,
            backbone_graph=True, combined_decode_bank=True,
            prefix_cache_mib=256, compute_policy="auto",
            compute_profile_dir=args.profile_dir,
        )) as engine:
            engine.status_callback = lambda value: print(
                json.dumps(value, ensure_ascii=False), flush=True)
            output = ROOT / "outputs" / f"auto-compute-{repetition + 1}.wav"
            result = DramaWorkflows(engine).run(
                "tts", "<|speaker:0|>你好，这是一次语音测试。", output,
                seed=123, max_new_tokens=256)
            with np.load(result["take_file"], allow_pickle=False) as take:
                actual = take["codes"].copy()
            actual_hash = hashlib.sha256(output.read_bytes()).hexdigest()
            if repetition == 0:
                codes, wav_hash = actual, actual_hash
            row = {
                "repetition": repetition + 1, "status": result["status"],
                "audio": result["audio"], "compute_policy": engine.compute_report,
                "total_seconds": result["total_seconds"],
                "prepare": result["chunks"][0]["runtime_prepare_breakdown"],
                "codes_equal_to_first": bool(np.array_equal(actual, codes)),
                "wav_equal_to_first": actual_hash == wav_hash,
                "report_file": str(output.with_suffix(".json")),
            }
            report["requests"].append(row)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if result["status"] != "complete" or not row["codes_equal_to_first"]:
                raise RuntimeError("Automatic compute policy output gate failed")
            if repetition == 1 and engine.compute_report["status"] != "cached":
                raise RuntimeError("A fresh matching engine failed to reuse its profile")
    report["compute_policy_gates_passed"] = True
    report["all_wav_bytes_equal"] = all(row["wav_equal_to_first"] for row in report["requests"])
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
