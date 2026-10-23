"""Measure native CPU reference encoding without autoregressive generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

import psutil


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {"default": "--encode", "fp32": "--encode-f32"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 2:
        raise ValueError("Use at least two alternating repeats")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment["PATH"] = str(args.probe.resolve().parent) + os.pathsep + environment.get("PATH", "")
    report = {
        "scope": "Native CPU default vs FP32 reference encoding; no TTS generation",
        "model": str(args.model.resolve()), "probe": str(args.probe.resolve()),
        "probe_sha256": hashlib.sha256(args.probe.read_bytes()).hexdigest(),
        "threads": 8, "repeats": args.repeats, "samples": [], "summaries": [],
        "notes": [
            "A fresh probe process is used per sample. Codec loading and encode graph construction are measured separately.",
            "encode_ms includes first-call graph construction and reference encoding, not backbone generation.",
            "Working-set peak is sampled per process and is not a system-wide committed-memory measurement.",
        ],
        "status": "running",
    }

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        for index, reference in enumerate(args.reference):
            hashes = {}
            for repeat in range(args.repeats):
                order = list(VARIANTS) if (index + repeat) % 2 == 0 else list(reversed(VARIANTS))
                for variant in order:
                    stem = f"reference-{index + 1}-{variant}-{repeat + 1}"
                    log_file = output / (stem + ".log")
                    codes = output / (stem + ".safetensors")
                    command = [str(args.probe.resolve()), VARIANTS[variant], str(args.model.resolve()),
                               str(reference.resolve()), str(codes), "cpu", "8"]
                    started = time.perf_counter()
                    print(json.dumps({"stage": "encode", "case": stem}), flush=True)
                    with log_file.open("wb") as log:
                        child = subprocess.Popen(
                            command, env=environment, cwd=ROOT / ".cache/audio-cpp-phase19",
                            stdout=log, stderr=subprocess.STDOUT,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                        process = psutil.Process(child.pid)
                        peak = 0
                        try:
                            while child.poll() is None:
                                if time.perf_counter() - started > 180:
                                    raise TimeoutError("Reference encoder benchmark timed out")
                                try:
                                    memory = process.memory_info()
                                    peak = max(peak, memory.rss, getattr(memory, "peak_wset", 0))
                                except psutil.NoSuchProcess:
                                    break
                                time.sleep(0.05)
                            child.wait(timeout=10)
                        finally:
                            if child.poll() is None:
                                child.terminate()
                                child.wait(timeout=10)
                    if child.returncode:
                        raise RuntimeError(f"Probe failed: {log_file}")
                    log = log_file.read_text(encoding="utf-8", errors="replace")
                    match = re.search(r"probe.load_ms=(\S+) encode_ms=(\S+) books=(\d+) frames=(\d+)", log)
                    buffer = re.search(r"Fish DAC codec.buffer_mb (\S+)", log)
                    if not match or not buffer:
                        raise ValueError(f"Missing timings in {log_file}")
                    digest = hashlib.sha256(codes.read_bytes()).hexdigest()
                    if variant in hashes and hashes[variant] != digest:
                        raise ValueError(f"Repeated encoding produced different codes: {stem}")
                    hashes[variant] = digest
                    row = {
                        "reference": str(reference.resolve()), "reference_index": index + 1,
                        "variant": variant, "repeat": repeat + 1,
                        "load_ms": float(match[1]), "encode_ms": float(match[2]),
                        "codebooks": int(match[3]), "frames": int(match[4]),
                        "codec_buffer_mib": float(buffer[1]), "peak_working_set_mib": peak / 1024**2,
                        "process_seconds": time.perf_counter() - started,
                        "codes_sha256": digest,
                    }
                    report["samples"].append(row)
                    save()
                    print(json.dumps(row), flush=True)
            medians = {}
            for variant in VARIANTS:
                rows = [row for row in report["samples"]
                        if row["reference_index"] == index + 1 and row["variant"] == variant]
                medians[variant] = {
                    key: statistics.median(row[key] for row in rows)
                    for key in ("load_ms", "encode_ms", "codec_buffer_mib", "peak_working_set_mib")
                }
                medians[variant]["encode_min_ms"] = min(row["encode_ms"] for row in rows)
                medians[variant]["encode_max_ms"] = max(row["encode_ms"] for row in rows)
            report["summaries"].append({
                "reference": str(reference.resolve()), "medians": medians,
                "default_encode_time_reduction_percent": (
                    1 - medians["default"]["encode_ms"] / medians["fp32"]["encode_ms"]) * 100,
                "default_load_and_encode_time_reduction_percent": (
                    1 - (medians["default"]["load_ms"] + medians["default"]["encode_ms"]) /
                    (medians["fp32"]["load_ms"] + medians["fp32"]["encode_ms"])) * 100,
            })
            save()
        report["status"] = "complete"
        save()
    except BaseException as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    main()
