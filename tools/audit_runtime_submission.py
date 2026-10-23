"""Audit matched completed gathering comparisons against saved engine reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "research/phase6-complete-gather.json")
    parser.add_argument("--output", type=Path, default=ROOT / "research/phase6-complete-audit.json")
    args = parser.parse_args()
    comparison = json.loads(args.input.read_text(encoding="utf-8"))
    measured = comparison["requests"]
    if len(measured) < 4:
        raise RuntimeError("At least two completed samples per path required")
    rows = []
    lifecycle_keys = (
        "runtime_reuses", "runtime_rebuilds", "park_calls", "resume_calls",
        "codec_coexist_calls", "codec_park_calls", "codec_cache_reclaims",
    )
    for sample in comparison["warmups"] + measured:
        report_path = Path(sample["report_file"])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        chunks = report["chunks"]
        wav = Path(report["output"])
        row = {
            "variant": sample["variant"], "repeat": sample["repeat"],
            "complete": report["status"] == "complete" and all(chunk["complete"] for chunk in chunks),
            "audio": report["audio"],
            "wav_sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
            "code_hashes": [chunk["semantic_codes_sha256"] for chunk in chunks],
            "batching": [
                {key: chunk["runtime_plan"][key] for key in
                 ("cache_experts", "max_staged_experts", "max_pair_rows", "prefill_chunk")}
                for chunk in chunks
            ],
            "prefix": [chunk["prefix_cache"] for chunk in chunks],
            "lifecycle_counts": {key: report["lifecycle_delta"][key] for key in lifecycle_keys},
            "codec_seconds": report["codec_seconds"],
            "prefill_seconds": sum(chunk["prefill_seconds"] for chunk in chunks),
            "decode_seconds": sum(chunk["decode_seconds"] for chunk in chunks),
            "generation_seconds": sum(chunk["generation_seconds"] for chunk in chunks),
            "runtime_prepare_seconds": sum(chunk["runtime_prepare_seconds"] for chunk in chunks),
            "total_seconds": report["total_seconds"],
            "h2d_bytes": report["offload_delta"]["h2d_bytes"],
            "host_gather_seconds": report["offload_delta"]["host_gather_seconds"],
            "native_gather_calls": report["offload_delta"]["native_gather_calls"],
            "torch_gather_calls": report["offload_delta"]["torch_gather_calls"],
            "peak_tensor_bytes": max(chunk["peak_allocated_bytes"] for chunk in chunks),
            "peak_reserved_bytes": max(chunk["peak_reserved_bytes"] for chunk in chunks),
        }
        if row["code_hashes"] != sample["code_hashes"] or row["wav_sha256"] != sample["wav_sha256"]:
            raise RuntimeError("Saved request/report hash mismatch")
        rows.append(row)
    timed = [row for row in rows if row["repeat"] > 0]
    checks = {
        "all_complete": all(row["complete"] for row in rows),
        "all_codes_equal": len({tuple(row["code_hashes"]) for row in rows}) == 1,
        "all_wavs_equal": len({row["wav_sha256"] for row in rows}) == 1,
        "all_batching_equal": len({json.dumps(row["batching"], sort_keys=True) for row in timed}) == 1,
        "all_prefix_hits": all(item["hit"] for row in timed for item in row["prefix"]),
        "all_lifecycle_counts_equal": len({tuple(row["lifecycle_counts"].values()) for row in timed}) == 1,
        "all_h2d_bytes_equal": len({row["h2d_bytes"] for row in timed}) == 1,
        "native_path_exercised": all(row["native_gather_calls"] > 0 for row in timed if row["variant"] == "native_gather"),
        "torch_baseline_retained": all(row["native_gather_calls"] == 0 for row in timed if row["variant"] == "torch_gather"),
    }
    metrics = (
        "prefill_seconds", "decode_seconds", "generation_seconds", "runtime_prepare_seconds",
        "codec_seconds", "total_seconds", "host_gather_seconds", "peak_tensor_bytes",
    )
    medians = {
        label: {
            key: statistics.median(row[key] for row in timed if row["variant"] == label)
            for key in metrics
        } for label in ("torch_gather", "native_gather")
    }
    result = {
        "scope": "Complete clone, forced parking, warm reference prefix, same original weights and top-8. Two timed repetitions per path; local evidence only.",
        "checks": checks, "all_checks_passed": all(checks.values()),
        "medians": medians,
        "total_time_reduction_fraction": 1 - medians["native_gather"]["total_seconds"] / medians["torch_gather"]["total_seconds"],
        "requests": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "requests"}, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError("Completed gathering comparison audit failed")


if __name__ == "__main__":
    main()
