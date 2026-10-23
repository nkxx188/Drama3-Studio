"""Audit saved tensor bytes, completed speech hashes and PCM integrity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from validate_native_teacher_forcing import torch
import numpy as np
import soundfile as sf


def same_bytes(a, b):
    return (
        a.dtype == b.dtype and a.shape == b.shape
        and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numerics", type=Path, default=ROOT / "research/hybrid-decode-numerics.pt")
    parser.add_argument("--benchmarks", type=Path, nargs="+", default=[
        ROOT / "research/engine-hybrid-comparison.json",
        ROOT / "research/engine-hybrid-warm-comparison.json",
        ROOT / "research/engine-hybrid-steady-short-comparison.json",
    ])
    parser.add_argument("--output", type=Path, default=ROOT / "research/hybrid-output-audit.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    artifacts = torch.load(args.numerics, weights_only=True, map_location="cpu")
    windows = []
    for condition in artifacts:
        for label, actual in condition["variants"].items():
            windows.append({
                "condition": condition["condition"], "variant": label,
                **{f"{name}_bytes_equal": same_bytes(actual[name], condition["reference"][name])
                   for name in ("hidden", "logits", "routes")},
            })
    audio, summaries, all_requests = [], [], []
    for benchmark in args.benchmarks:
        report = json.loads(benchmark.read_text(encoding="utf-8"))
        all_requests.extend(report["requests"])
        for request in report["requests"]:
            detail = json.loads(Path(request["report_file"]).read_text(encoding="utf-8"))
            path = Path(detail["output"])
            values, sample_rate = sf.read(path, dtype="float32")
            value_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            audio.append({
                "file": str(path), "status": detail["status"],
                "eos": all(row["finish_reason"] == "eos" for row in detail["chunks"]),
                "sample_rate": sample_rate, "samples": len(values),
                "sample_count_matches_report": len(values) == detail["audio"]["samples"],
                "finite": bool(np.isfinite(values).all()),
                "peak": float(np.abs(values).max()),
                "rms": float(np.sqrt(np.mean(values.astype(np.float64) ** 2))),
                "wav_hash_matches_report": value_hash == request["wav_sha256"],
            })
        for case in dict.fromkeys(row["case"] for row in report["requests"]):
            rows = [row for row in report["requests"] if row["case"] == case]
            summaries.append({
                "benchmark": str(benchmark), "case": case, "requests": len(rows),
                "all_codes_equal": len({tuple(row["code_hashes"]) for row in rows}) == 1,
                "all_wavs_equal": len({row["wav_sha256"] for row in rows}) == 1,
                "all_complete": all(row["status"] == "complete" for row in rows),
            })
    across_runs = [{
        "case": case,
        "all_codes_equal": len({
            tuple(row["code_hashes"]) for row in all_requests if row["case"] == case
        }) == 1,
        "all_wavs_equal": len({
            row["wav_sha256"] for row in all_requests if row["case"] == case
        }) == 1,
    } for case in dict.fromkeys(row["case"] for row in all_requests)]
    passed = (
        all(all(row[f"{name}_bytes_equal"] for name in ("hidden", "logits", "routes")) for row in windows)
        and all(row["all_codes_equal"] and row["all_wavs_equal"] and row["all_complete"] for row in summaries)
        and all(row["all_codes_equal"] and row["all_wavs_equal"] for row in across_runs)
        and all(row["finite"] and row["eos"] and row["wav_hash_matches_report"]
                and row["sample_count_matches_report"] and row["sample_rate"] == 44100 for row in audio)
    )
    report = {
        "passed": passed, "numerical_windows": windows, "benchmarks": summaries,
        "across_benchmark_runs": across_runs,
        "audio": audio,
        "scope": "Byte equality and PCM integrity. No new ASR, speaker-similarity or listening assessment.",
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": passed, "windows": len(windows), "audio_files": len(audio)}), flush=True)
    if not passed:
        raise RuntimeError("Saved hybrid outputs failed audit")


if __name__ == "__main__":
    main()
