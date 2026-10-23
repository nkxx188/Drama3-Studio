"""Audit saved state/oracle artifacts and completed prefix-cache speech."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from validate_state_prefix import byte_equal, torch
import numpy as np
import soundfile as sf


def main():
    torch.set_num_threads(1)
    artifacts = torch.load(ROOT / "research/state-oracle-validation.pt", weights_only=True)
    tensor_checks = []
    for label, row in artifacts.items():
        for name, values in row["actual"].items():
            tensor_checks.append({
                "variant": label, "tensor": name,
                "bytes_equal": byte_equal(values, row["reference"][name]),
            })
    benchmark = json.loads((ROOT / "research/engine-prefix-comparison.json").read_text(encoding="utf-8"))
    rows = benchmark["warmups"] + benchmark["requests"]
    audio = []
    for row in rows:
        detail = json.loads(Path(row["report_file"]).read_text(encoding="utf-8"))
        output = Path(detail["output"])
        values, sample_rate = sf.read(output, dtype="float32")
        audio.append({
            "file": str(output), "eos": all(chunk["finish_reason"] == "eos" for chunk in detail["chunks"]),
            "samples": len(values), "sample_rate": sample_rate,
            "sample_count_matches": len(values) == detail["audio"]["samples"],
            "finite": bool(np.isfinite(values).all()),
            "hash_matches": hashlib.sha256(output.read_bytes()).hexdigest() == row["wav_sha256"],
            "peak": float(np.abs(values).max()),
            "rms": float(np.sqrt(np.mean(values.astype(np.float64) ** 2))),
        })
    report = {
        "scope": "Byte/logit and PCM integrity audit. Oracle proposals are perfect/free; no draft or acceptance sampler.",
        "oracle_tensor_checks": tensor_checks, "audio": audio,
        "all_speech_codes_equal": len({tuple(row["code_hashes"]) for row in rows}) == 1,
        "all_speech_wavs_equal": len({row["wav_sha256"] for row in rows}) == 1,
    }
    prefix = json.loads((ROOT / "research/state-prefix-validation.json").read_text(encoding="utf-8"))
    oracle = json.loads((ROOT / "research/state-oracle-validation.json").read_text(encoding="utf-8"))
    windows = json.loads((ROOT / "research/oracle-window-direct-benchmark.json").read_text(encoding="utf-8"))
    report["prefix_numerical_cases"] = len(prefix["prefix"])
    report["prefix_numerical_cases_passed"] = all(
        all(row[key] for key in (
            "frames_bytes_equal", "hidden_bytes_equal", "logits_bytes_equal", "routes_bytes_equal"))
        for row in prefix["prefix"])
    report["rollback_cases"] = len(oracle["rollback"])
    report["rollback_cases_passed"] = all(row["equal"] for row in oracle["rollback"])
    window_rows = [row for condition in windows["conditions"] for row in condition["windows"]]
    report["oracle_windows"] = len(window_rows)
    report["oracle_windows_passed"] = all(
        row["performance_gate"] and all(row[key] for key in (
            "semantic_logits_bytes_equal", "acoustic_logits_bytes_equal",
            "hidden_bytes_equal", "state_bytes_equal"))
        for row in window_rows)
    report["passed"] = (
        all(row["bytes_equal"] for row in tensor_checks)
        and all(row["eos"] and row["finite"] and row["hash_matches"]
                and row["sample_count_matches"] and row["sample_rate"] == 44100 for row in audio)
        and report["all_speech_codes_equal"] and report["all_speech_wavs_equal"]
        and report["prefix_numerical_cases_passed"] and report["rollback_cases_passed"]
        and report["oracle_windows_passed"])
    (ROOT / "research/state-prefix-audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "audio_files": len(audio), "tensor_checks": len(tensor_checks)}))
    if not report["passed"]:
        raise RuntimeError("Saved artifacts failed audit")


if __name__ == "__main__":
    main()
