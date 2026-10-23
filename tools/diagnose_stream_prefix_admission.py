"""Measure the exact memory gate that accepts/refuses stream-prefix snapshots."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from benchmark_engine_decode import DramaEngine, EngineOptions, torch
from drama_low_vram.autotune import inspect_hardware
from drama_low_vram.protocols import VoiceReference
from drama_low_vram.workflows import DramaWorkflows


def main():
    torch.set_num_threads(1)
    output = ROOT / "research/stream-prefix-admission-diagnosis.json"
    report = {"scope": "Actual host readings at prefix snapshot admission, unchanged memory policy.", "admissions": []}
    with DramaEngine(EngineOptions(
        checkpoint=ROOT / "checkpoints/drama-3", expert_storage="mmap",
        cache_experts=1, staged_experts=8, pair_rows=64, prefill_chunk=32,
        backbone_graph=True, combined_decode_bank=True,
        cpu_experts=2, native_threads=4, host_gather="auto", prefix_cache_mib=256,
    )) as engine:
        original = engine.prefix_cache.admission

        def admission(size):
            hardware = inspect_hardware(str(engine.device))
            mirrors = sum(value.numel() * value.element_size() for value in engine._cpu_mirrors.values())
            row = {
                "requested_bytes": size, "ram_available_bytes": hardware.ram_available_bytes,
                "commit_available_bytes": hardware.commit_available_bytes,
                "existing_mirror_bytes": mirrors, "codec_loaded": engine.codec is not None,
                "planned_parking_bytes": engine.host_plan.requirements.parking_bytes,
                "planned_codec_build_bytes": engine.host_plan.requirements.codec_build_bytes,
                "buffer_bytes": engine.host_plan.buffer_bytes, "admitted": original(size),
            }
            report["admissions"].append(row)
            output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(row), flush=True)
            return row["admitted"]

        engine.prefix_cache.admission = admission
        reference = VoiceReference(
            str(ROOT / "outputs/quality-official-fp8.wav"),
            "你好，这是官方 FP8 质量优先测试。")
        workflows = DramaWorkflows(engine)
        for index in range(2):
            result = workflows.run(
                "stream", "早上好。", ROOT / "outputs" / f"prefix-admission-stream-{index}.wav",
                references=[reference], lookahead="我们开始吧。", max_new_tokens=128, seed=123)
            report.setdefault("requests", []).append({
                "status": result["status"], "prefix": result["chunks"][0]["prefix_cache"],
                "seconds": result["total_seconds"]})
            report["cache"] = engine.prefix_cache.report()
            output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(report["requests"][-1]), flush=True)


if __name__ == "__main__":
    main()
