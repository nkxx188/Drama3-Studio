"""Recheck full speech hashes and the lifecycle behind codec coexistence gains."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "research/phase7-codec-lifecycle-comparison.json")
    parser.add_argument("--output", type=Path, default=ROOT / "research/phase7-codec-lifecycle-audit.json")
    args = parser.parse_args()
    comparison = json.loads(args.input.read_text(encoding="utf-8"))
    rows = []
    for sample in comparison["warmups"] + comparison["requests"]:
        report = json.loads(Path(sample["report_file"]).read_text(encoding="utf-8"))
        chunks = report["chunks"]
        wav = Path(report["output"])
        row = {
            "variant": sample["variant"], "repeat": sample["repeat"],
            "complete": report["status"] == "complete" and all(chunk["complete"] for chunk in chunks),
            "wav_sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
            "code_hashes": [chunk["semantic_codes_sha256"] for chunk in chunks],
            "audio": report["audio"],
            "batching": [
                {key: chunk["runtime_plan"][key] for key in (
                    "cache_experts", "max_staged_experts", "max_pair_rows", "prefill_chunk")}
                for chunk in chunks
            ],
            "prefix_hits": [chunk["prefix_cache"]["hit"] for chunk in chunks],
            "lifecycle": report["lifecycle_delta"],
            "runtime_prepare_breakdown": [chunk["runtime_prepare_breakdown"] for chunk in chunks],
            "h2d_bytes": report["offload_delta"]["h2d_bytes"],
            "prefill_seconds": sum(chunk["prefill_seconds"] for chunk in chunks),
            "decode_seconds": sum(chunk["decode_seconds"] for chunk in chunks),
            "generation_seconds": sum(chunk["generation_seconds"] for chunk in chunks),
            "runtime_prepare_seconds": sum(chunk["runtime_prepare_seconds"] for chunk in chunks),
            "codec_seconds": report["codec_seconds"], "total_seconds": report["total_seconds"],
        }
        if row["code_hashes"] != sample["code_hashes"] or row["wav_sha256"] != sample["wav_sha256"]:
            raise RuntimeError("Saved speech/report hash mismatch")
        rows.append(row)
    timed = [row for row in rows if row["repeat"] > 0]
    baseline = [row for row in timed if row["variant"] == "full"]
    optimized = [row for row in timed if row["variant"] == "auto"]
    checks = {
        "two_timed_requests_per_path": len(baseline) == len(optimized) == 2,
        "all_complete": all(row["complete"] for row in rows),
        "all_codes_equal": len({tuple(row["code_hashes"]) for row in rows}) == 1,
        "all_wavs_equal": len({row["wav_sha256"] for row in rows}) == 1,
        "all_batching_equal": len({json.dumps(row["batching"], sort_keys=True) for row in timed}) == 1,
        "all_timed_prefix_hits": all(all(row["prefix_hits"]) for row in timed),
        "baseline_parks_and_resumes": all(
            row["lifecycle"]["park_calls"] == row["lifecycle"]["resume_calls"] == 1
            for row in baseline),
        "optimized_keeps_model_resident": all(
            row["lifecycle"]["park_calls"] == row["lifecycle"]["resume_calls"] == 0
            for row in optimized),
        "optimized_no_graph_recapture": all(
            item.get("backbone_graph_seconds", 0) < 0.001
            and item.get("tower_graph_seconds", 0) < 0.001
            for row in optimized for item in row["runtime_prepare_breakdown"]),
        "split_decode_exercised": all(row["lifecycle"]["codec_split_decode_calls"] == 1 for row in optimized),
        "no_codec_oom_retries": all(row["lifecycle"]["codec_oom_retries"] == 0 for row in timed),
    }
    metrics = (
        "prefill_seconds", "decode_seconds", "generation_seconds",
        "runtime_prepare_seconds", "codec_seconds", "total_seconds",
    )
    medians = {
        label: {key: statistics.median(row[key] for row in timed if row["variant"] == label)
                for key in metrics}
        for label in ("full", "auto")
    }
    report = {
        "scope": "Same-model continuous completed clones, full placement vs serial operation placement. Three excluded block warmups establish each path's previous-request lifecycle.",
        "checks": checks, "all_checks_passed": all(checks.values()),
        "medians": medians,
        "total_time_reduction_fraction": 1 - medians["auto"]["total_seconds"] / medians["full"]["total_seconds"],
        "h2d_byte_counts": {
            label: [row["h2d_bytes"] for row in timed if row["variant"] == label]
            for label in ("full", "auto")
        },
        "notes": [
            "Retained hot caches can slightly change expert H2D bytes; this is part of the intended lifecycle change, not an identical-transfer claim.",
            "Generation and codec timings include the actual cache retention/reclamation behavior.",
            "Graph retention is audited using park/resume and graph capture timings. The first benchmark process's graph identity flag compared None with None on parked paths; that flag is not used as evidence here and the tool has been corrected.",
            "Two measured requests per path on one machine are local evidence, not a broad hardware or text guarantee.",
        ],
        "requests": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "requests"}, indent=2))
    if not report["all_checks_passed"]:
        raise RuntimeError("Codec lifecycle comparison audit failed")


if __name__ == "__main__":
    main()
