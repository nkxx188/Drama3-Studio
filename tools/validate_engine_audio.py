"""Check engine reports against decoded PCM; this is not a listening evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def validate(report_path: Path):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    path = Path(report["output"])
    samples, sample_rate = sf.read(path, dtype="float64", always_2d=True)
    expected_samples = sum(chunk["frames"] for chunk in report["chunks"]) * 2048
    finite = bool(np.isfinite(samples).all())
    complete = (
        report["status"] == "complete"
        and all(chunk["finish_reason"] == "eos" for chunk in report["chunks"])
    )
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    rms = float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0
    valid = (
        complete and finite and sample_rate == 44100 and samples.shape[1] == 1
        and samples.shape[0] == expected_samples and peak > 0 and rms > 1e-6
        and report["audio"]["samples"] == expected_samples
    )
    return {
        "report": str(report_path.resolve()),
        "audio": str(path.resolve()),
        "valid": valid,
        "all_chunks_eos": complete,
        "samples": int(samples.shape[0]),
        "expected_samples": expected_samples,
        "sample_rate": sample_rate,
        "channels": int(samples.shape[1]),
        "duration_seconds": samples.shape[0] / sample_rate,
        "finite": finite,
        "peak": peak,
        "rms": rms,
        "clipped_sample_fraction": float(np.mean(np.abs(samples) >= 32767 / 32768)),
        "zero_sample_fraction": float(np.mean(samples == 0)),
        "wav_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [validate(path) for path in args.reports]
    result = {
        "scope": "PCM integrity, EOS, sample count and non-silence only; no ASR or voice-similarity claim.",
        "all_valid": all(item["valid"] for item in results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["all_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
