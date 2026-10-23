"""Real cross-language, controls, editing and long-clone functional evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from benchmark_engine_decode import DramaEngine, EngineOptions, torch
from drama_low_vram.protocols import VoiceReference
from drama_low_vram.workflows import DramaWorkflows, load_take


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+",
                        choices=("cross-language", "scoped-control", "events", "edit", "long-clone"),
                        default=["cross-language", "scoped-control", "events", "edit", "long-clone"])
    parser.add_argument("--output", type=Path,
                        default=ROOT / "research/engine-capabilities-validation.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = {
        "scope": "Original model on native Windows 8GB. Completion, true conditioning and finite audio are functional gates. ASR/listening must separately assess content, controls and speaker similarity.",
        "cases": [],
    }
    reference = VoiceReference(
        str(ROOT / "outputs/quality-official-fp8.wav"),
        "你好，这是官方 FP8 质量优先测试。", 0)
    long = (
        "今天的天气很好，我们准备沿着河边慢慢走一段路。"
        "你可以听见远处的鸟鸣，也可以看看水面上轻轻摇晃的树影。"
        "等太阳再升高一些，我们就在前面的小店坐下来，喝杯热茶，聊聊最近发生的事情。"
        "这段话会完整生成，不需要在句子中间突然停下。"
        "希望每一位使用普通电脑的朋友，都能方便地创作属于自己的声音。")
    cases = [
        ("cross-language", "tts",
         "Good morning. We can now create natural speech on this computer.",
         dict(references=[reference], language="en", max_new_tokens=384)),
        ("scoped-control", "tts",
         "<whisper>请轻一点说话，别吵醒他们。</whisper>",
         dict(references=[reference], max_new_tokens=256)),
        ("events", "tts", "[gasps]怎么会这样？[sighs]我们再试一次。",
         dict(references=[reference], max_new_tokens=256)),
        ("edit", "edit", "你好，这是一次完整测试。",
         dict(max_new_tokens=256, direction="Keep the original voice. Read the corrected line.")),
        ("long-clone", "tts", long,
         dict(references=[reference], max_new_tokens=768, max_chunk_chars=240)),
    ]
    with DramaEngine(EngineOptions(
        checkpoint=ROOT / "checkpoints/drama-3", expert_storage="mmap",
        cache_experts=1, staged_experts=8, pair_rows=64, prefill_chunk=32,
        backbone_graph=True, combined_decode_bank=True,
        cpu_experts=2, native_threads=4, host_gather="auto", prefix_cache_mib=256,
    )) as engine:
        engine.load()
        workflows = DramaWorkflows(engine)
        for label, mode, text, options in cases:
            if label not in args.cases:
                continue
            if mode == "edit":
                previous, _, _ = load_take(
                    ROOT / "outputs/engine-workflow-tts.take.npz", engine.model.config)
                options["previous_takes"] = [previous]
            last = time.perf_counter()

            def progress(event):
                nonlocal last
                now = time.perf_counter()
                if event["stage"] != "generate" or now - last >= 20:
                    print(json.dumps({"case": label, **event}, ensure_ascii=False), flush=True)
                    last = now

            output = ROOT / "outputs" / f"capability-{label}.wav"
            result = workflows.run(
                mode, text, output, seed=123, progress=progress, **options)
            row = {
                "case": label, "mode": mode, "text": text,
                "status": result["status"], "audio": result["audio"],
                "total_seconds": result["total_seconds"], "output": str(output),
                "frames": [chunk["frames"] for chunk in result["chunks"]],
                "chunks": len(result["chunks"]), "lifecycle": result["lifecycle_delta"],
                "report_file": str(output.with_suffix(".json")),
            }
            report["cases"].append(row)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if result["status"] != "complete":
                raise RuntimeError(f"{label} did not complete; retain diagnostic evidence")
        report["all_cases_complete"] = all(row["status"] == "complete" for row in report["cases"])
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")


if __name__ == "__main__":
    main()
