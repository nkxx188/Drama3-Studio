"""Exact reference-prefill reuse for a complete two-chunk text stream."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from benchmark_engine_decode import DramaEngine, EngineOptions, torch
from drama_low_vram.autotune import inspect_hardware
from drama_low_vram.protocols import VoiceReference
from drama_low_vram.text_stream import TextStreamSession
from drama_low_vram.workflows import DramaWorkflows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--prefix", default="text-stream-prefix")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "research/text-stream-prefix-validation.json")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 4:
        parser.error("repeats must be from one to four")
    torch.set_num_threads(1)
    reference = VoiceReference(
        str(ROOT / "outputs/quality-official-fp8.wav"),
        "你好，这是官方 FP8 质量优先测试。")
    report = {
        "scope": "Real complete reference-conditioned lookahead streams. Same prompt chunks, seed, batching and model. Cache misses/hits are compared with suspended-cache baselines using exact VQ/all frames and actual PCM export. Completed-request times include codec; no broad quality claim.",
        "requests": [],
    }
    expected_codes, expected_frames = None, None
    with DramaEngine(EngineOptions(
        checkpoint=ROOT / "checkpoints/drama-3", expert_storage="mmap",
        cache_experts=1, staged_experts=8, pair_rows=64, prefill_chunk=32,
        backbone_graph=True, combined_decode_bank=True,
        cpu_experts=2, native_threads=4, host_gather="auto", prefix_cache_mib=256,
    )) as engine:
        admissions = []
        admit = engine.prefix_cache.admission

        def admission(size):
            hardware = inspect_hardware(str(engine.device))
            result = admit(size)
            admissions.append({
                "requested_bytes": size, "admitted": result,
                "ram_available_bytes": hardware.ram_available_bytes,
                "commit_available_bytes": hardware.commit_available_bytes,
                "future_parking_bytes": max(
                    0, engine.host_plan.requirements.parking_bytes - sum(
                        value.numel() * value.element_size() for value in engine._cpu_mirrors.values())),
                "buffer_bytes": engine.host_plan.buffer_bytes,
            })
            return result

        engine.prefix_cache.admission = admission
        engine.prepare_reference(Path(reference.audio), reference.transcript())
        engine.load()

        def request(label, repetition):
            nonlocal expected_codes, expected_frames
            directory = ROOT / "outputs" / f"{args.prefix}-{label}-{repetition}"
            admission_start = len(admissions)
            session = TextStreamSession(
                DramaWorkflows(engine), directory,
                options={"max_new_tokens": 192, "seed": 123, "references": [reference]})
            session.push("早上好。")
            session.push("我们开始今天的测试。")
            session.finish()
            rows = list(session.run())
            if session.status != "complete" or len(rows) != 2:
                raise RuntimeError("Every reference-conditioned stream chunk must reach EOS")
            output = session.export(directory.with_suffix(".wav"))
            details = [json.loads(Path(row["path"]).with_suffix(".json").read_text(
                encoding="utf-8")) for row in rows]
            codes, frames = [], []
            for row in rows:
                with np.load(row["take_file"], allow_pickle=False) as take:
                    codes.append(take["codes"].copy())
                    frames.append(take["frames"].copy())
            if expected_codes is None:
                expected_codes, expected_frames = codes, frames
            actual, _ = sf.read(output, dtype="int16")
            delivered = np.concatenate([sf.read(row["path"], dtype="int16")[0] for row in rows])
            value = {
                "variant": label, "repetition": repetition, "status": session.status,
                "total_seconds": sum(detail["total_seconds"] for detail in details),
                "prefill_seconds": sum(
                    chunk["prefill_seconds"] for detail in details for chunk in detail["chunks"]),
                "prepare_seconds": sum(
                    chunk["runtime_prepare_seconds"] for detail in details for chunk in detail["chunks"]),
                "prefix": [chunk["prefix_cache"] for detail in details for chunk in detail["chunks"]],
                "snapshot_admissions": admissions[admission_start:],
                "codes_equal": all(np.array_equal(a, b) for a, b in zip(codes, expected_codes)),
                "all_frames_equal": all(np.array_equal(a, b) for a, b in zip(frames, expected_frames)),
                "exported_pcm_matches": bool(np.array_equal(actual, delivered)),
                "output": str(output),
                "report_files": [str(Path(row["path"]).with_suffix(".json")) for row in rows],
                "code_hashes": [hashlib.sha256(value.tobytes()).hexdigest() for value in codes],
                "wav_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "lifecycle": [detail["lifecycle_delta"] for detail in details],
            }
            print(json.dumps(value, ensure_ascii=False), flush=True)
            return value

        # Complete both chunks before timing so weights, codec and cache shapes
        # have traversed their actual lifecycle at least once.
        with engine.prefix_cache.suspend():
            report["warmup"] = request("warmup", 0)
        for repetition in range(1, args.repeats + 1):
            labels = ("baseline", "cached") if repetition % 2 else ("cached", "baseline")
            for label in labels:
                if label == "baseline":
                    with engine.prefix_cache.suspend():
                        row = request(label, repetition)
                else:
                    engine.prefix_cache.clear()
                    row = request(label, repetition)
                report["requests"].append(row)
                args.output.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        baseline = [row["total_seconds"] for row in report["requests"] if row["variant"] == "baseline"]
        cached = [row["total_seconds"] for row in report["requests"] if row["variant"] == "cached"]
        report["summary"] = {
            "baseline_median_seconds": statistics.median(baseline),
            "cached_median_seconds": statistics.median(cached),
            "local_time_reduction_fraction": 1 - statistics.median(cached) / statistics.median(baseline),
            "timed_requests_per_variant": len(baseline),
        }
        report["passed"] = (
            all(row["codes_equal"] and row["all_frames_equal"] and row["exported_pcm_matches"]
                for row in report["requests"])
            and all(row["prefix"][1]["hit"]
                    for row in report["requests"] if row["variant"] == "cached"))
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if not report["passed"]:
            raise RuntimeError("Text-stream prefix cache failed its exact-output/hit gate")


if __name__ == "__main__":
    main()
