"""Recheck actual files for text-stream prefix comparisons and cache hits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path,
                        default=ROOT / "research/text-stream-prefix-admitted-validation.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "research/text-stream-prefix-output-audit.json")
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    original = None
    rows = []
    for request in [report["warmup"], *report["requests"]]:
        codes, frames, parts = [], [], []
        complete = True
        for path in request["report_files"]:
            detail = json.loads(Path(path).read_text(encoding="utf-8"))
            complete &= detail["status"] == "complete" and all(
                chunk["finish_reason"] == "eos" for chunk in detail["chunks"])
            with np.load(detail["take_file"], allow_pickle=False) as take:
                codes.append(take["codes"].copy())
                frames.append(take["frames"].copy())
            parts.append(sf.read(detail["output"], dtype="int16")[0])
        if original is None:
            original = codes, frames
        actual, rate = sf.read(request["output"], dtype="int16")
        rows.append({
            "variant": request["variant"], "repetition": request["repetition"],
            "all_eos": complete,
            "codes_equal": all(np.array_equal(a, b) for a, b in zip(codes, original[0])),
            "all_frames_equal": all(np.array_equal(a, b) for a, b in zip(frames, original[1])),
            "exported_pcm_matches": np.array_equal(actual, np.concatenate(parts)),
            "wav_hash_matches": hashlib.sha256(Path(request["output"]).read_bytes()).hexdigest()
                                == request["wav_sha256"],
            "sample_rate": rate,
            "cache_hit": request["prefix"][1]["hit"],
            "reused_rows": request["prefix"][1]["reused_rows"],
        })
    result = {
        "scope": "Saved VQ/all-frame byte equality, EOS, actual delivered PCM export and real cache hits. No perceptual comparison.",
        "input_report": str(args.report), "rows": rows,
        "passed": all(
            row["all_eos"] and row["codes_equal"] and row["all_frames_equal"]
            and row["exported_pcm_matches"] and row["wav_hash_matches"] and row["sample_rate"] == 44100
            and (row["cache_hit"] if row["variant"] == "cached" else not row["cache_hit"])
            for row in rows),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise RuntimeError("Text stream prefix artifacts failed re-audit")


if __name__ == "__main__":
    main()
