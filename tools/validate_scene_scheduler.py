"""Complete multi-character scene via explicit per-line real-reference synthesis."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from benchmark_engine_decode import DramaEngine, EngineOptions, torch
from drama_low_vram.protocols import Character, SceneLine, VoiceReference
from drama_low_vram.scenes import SceneSession


def main():
    torch.set_num_threads(1)
    report = {
        "scope": "Explicit per-line scene scheduling, not native audio-separator generation. Real voice references and first/last actual take anchors; all lines must reach EOS.",
        "segments": [],
    }
    with DramaEngine(EngineOptions(
        checkpoint=ROOT / "checkpoints/drama-3", expert_storage="mmap",
        cache_experts=1, staged_experts=8, pair_rows=64, prefill_chunk=32,
        backbone_graph=True, combined_decode_bank=True,
        cpu_experts=2, native_threads=4, host_gather="auto", prefix_cache_mib=256,
    )) as engine:
        scene = SceneSession(
            engine, ROOT / "outputs/scene-scheduled.wav",
            scene="两位朋友在清晨的车站碰面，准备一起出发。",
            characters=[Character("林", "声音清楚，语气从容。", 0),
                        Character("周", "声音轻松，自然回应。", 1)],
            lines=[SceneLine("林", "早上好，你准备好了吗？"),
                   SceneLine("周", "准备好了，我们现在出发。"),
                   SceneLine("林", "好，那就一起走吧。")],
            references=[
                VoiceReference(str(ROOT / "outputs/quality-official-fp8.wav"),
                               "你好，这是官方 FP8 质量优先测试。", 0),
                VoiceReference(str(ROOT / "outputs/engine-workflow-tts.wav"),
                               "你好，这是一次语音测试。", 1),
            ],
            max_new_tokens=256, seed=123)
        last = time.perf_counter()

        def progress(event):
            nonlocal last
            now = time.perf_counter()
            if event["stage"] != "generate" or now - last >= 20:
                print(json.dumps(event, ensure_ascii=False), flush=True)
                last = now

        for row in scene.run(progress=progress):
            report["segments"].append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        report["scene"] = scene.report
        if scene.status != "complete" or len(scene.results) != 3:
            raise RuntimeError("Scheduled scene did not complete")
        expected = np.concatenate([sf.read(row["path"], dtype="int16")[0]
                                   for row in scene.results])
        actual, rate = sf.read(scene.output, dtype="int16")
        report["exported_pcm_matches_lines"] = bool(np.array_equal(actual, expected))
        report["sample_rate"] = rate
        report["same_character_reuses_true_history"] = scene.results[2]["history_chunks"] == 1
        report["gates_passed"] = (
            report["exported_pcm_matches_lines"] and report["same_character_reuses_true_history"])
        (ROOT / "research/scene-scheduler-validation.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if not report["gates_passed"]:
            raise RuntimeError("Scene scheduling boundaries or history failed")


if __name__ == "__main__":
    main()
