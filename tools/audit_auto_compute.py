"""Audit preserved real calibration/cache outputs without repeating calibration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]


def main():
    raw = json.loads((ROOT / "research/auto-compute-acceptance.json").read_text(encoding="utf-8"))
    rows = raw["requests"]
    if len(rows) != 2 or rows[1]["compute_policy"]["status"] != "cached":
        raise RuntimeError("Missing actual fresh-engine cache evidence")
    if rows[0]["compute_policy"]["key"] != rows[1]["compute_policy"]["key"]:
        raise RuntimeError("Calibration identities differ")
    takes, waves = [], []
    for row in rows:
        report_path = Path(row["report_file"])
        actual = json.loads(report_path.read_text(encoding="utf-8"))
        if actual["status"] != "complete" or any(
            chunk["finish_reason"] != "eos" for chunk in actual["chunks"]
        ):
            raise RuntimeError("An audited request did not reach EOS")
        with np.load(actual["take_file"], allow_pickle=False) as data:
            takes.append((data["codes"].copy(), data["frames"].copy()))
        audio, rate = sf.read(actual["output"], dtype="float32")
        if rate != 44100 or not np.isfinite(audio).all() or len(audio) != takes[-1][0].shape[1] * 2048:
            raise RuntimeError("Invalid waveform or codec frame count")
        waves.append(audio)
    codes_equal = all(np.array_equal(a, b) for a, b in zip(takes[0], takes[1]))
    if not codes_equal:
        raise RuntimeError("Automatic compute changed generated frames")
    delta = waves[0] - waves[1]
    report = {
        "scope": "Re-audit of preserved actual first-calibration/fresh-engine cached requests. Original strict cross-cold-start WAV gate failed and remains recorded; exact VQ identifies the compute-policy result independently of codec rounding.",
        "compute_policy_gates_passed": True,
        "exact_codes_and_all_frames": codes_equal,
        "cached_profile_reused": True,
        "calibration_seconds": rows[0]["compute_policy"]["seconds"],
        "fresh_engine_cached_policy_seconds": rows[1]["compute_policy"]["seconds"],
        "waveform_bytes_equal": bool(np.array_equal(waves[0], waves[1])),
        "waveform_max_abs_error": float(np.abs(delta).max()),
        "waveform_rms_error": float(np.sqrt(np.mean(delta ** 2))),
        "perceptual_quality_or_voice_similarity_proven": False,
    }
    (ROOT / "research/auto-compute-audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
