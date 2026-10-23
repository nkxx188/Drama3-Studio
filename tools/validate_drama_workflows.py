"""Exercise native task protocols with the real official 8GB engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from benchmark_engine_decode import DramaEngine, EngineOptions, torch
from drama_low_vram.protocols import Character, InfillSpan, PreviousTake, SceneLine, VoiceReference
from drama_low_vram.workflows import DramaWorkflows, load_take


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+",
                        choices=("tts", "revision", "infill", "scene", "stream", "references"),
                        default=["tts", "revision", "infill", "scene", "stream", "references"])
    parser.add_argument("--output", type=Path, default=ROOT / "research/engine-workflows-validation.json")
    parser.add_argument("--continue-on-incomplete", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = {
        "scope": "Real official model task protocols on native Windows 8GB. EOS, token/code boundaries and finite audio are checked. Expressiveness/content still require separate ASR/listening evaluation.",
        "cases": [],
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with DramaEngine(EngineOptions(
        checkpoint=ROOT / "checkpoints/drama-3", expert_storage="mmap",
        cache_experts=1, staged_experts=8, pair_rows=64, prefill_chunk=32,
        backbone_graph=True, combined_decode_bank=True, cpu_experts=2,
        host_gather="auto", prefix_cache_mib=256,
    )) as engine:
        workflows = DramaWorkflows(engine)
        basic_text = "<|speaker:0|>你好，这是一次语音测试。"
        previous = None
        for mode in ["tts", *[mode for mode in args.cases if mode != "tts"]]:
            if mode == "tts" and mode not in args.cases and (
                ROOT / "outputs/engine-workflow-tts.take.npz").is_file():
                engine.load()
                previous, _, _ = load_take(ROOT / "outputs/engine-workflow-tts.take.npz", engine.model.config)
                continue
            text = basic_text
            kwargs = {"seed": 123, "max_new_tokens": 256}
            actual_mode = mode
            if mode == "revision":
                kwargs.update(previous_takes=[previous], direction="Speak more softly and warmly than the previous take.")
            elif mode == "infill":
                count = previous.codes.shape[1]
                text = "<|speaker:0|>你好，这是一次声音测试。"
                kwargs.update(
                    infill=InfillSpan(previous.codes, count // 3, count * 2 // 3,
                                      max(1, count // 3)),
                    max_new_tokens=128,
                )
            elif mode == "scene":
                text = ""
                kwargs.update(
                    scene="A quiet room in the morning. Two friends greet each other.",
                    characters=[Character("Ada", "calm female voice", 0),
                                Character("Bo", "warm male voice", 1)],
                    lines=[SceneLine("Ada", "Hi, Bo."), SceneLine("Bo", "Good morning, Ada.")],
                    max_new_tokens=384,
                )
            elif mode == "stream":
                text = "早上好。"
                kwargs.update(lookahead="我们开始吧。", max_new_tokens=128)
            elif mode == "references":
                actual_mode = "tts"
                text = "<|speaker:0|>你好。<|speaker:1|>我们开始测试。"
                kwargs.update(references=[
                    VoiceReference(str(ROOT / "outputs/quality-official-fp8.wav"),
                                   "<|speaker:0|>你好，这是官方 FP8 质量优先测试。", 0),
                    VoiceReference(str(ROOT / "outputs/engine-workflow-tts.wav"), basic_text, 1),
                ])
                # The second reference is the same transcript assigned to a
                # second speaker; write its canonical speaker before validation.
                kwargs["references"][1] = VoiceReference(
                    str(ROOT / "outputs/engine-workflow-tts.wav"),
                    "你好，这是一次语音测试。", 1)
            last = time.perf_counter()

            def progress(event):
                nonlocal last
                now = time.perf_counter()
                if event["stage"] != "generate" or now - last >= 15:
                    print(json.dumps({"case": mode, **event}, ensure_ascii=False), flush=True)
                    last = now

            result = workflows.run(
                actual_mode, text, ROOT / "outputs" / f"engine-workflow-{mode}.wav",
                progress=progress, **kwargs)
            row = {
                "case": mode, "status": result["status"], "audio": result["audio"],
                "total_seconds": result["total_seconds"],
                "report_file": str(ROOT / "outputs" / f"engine-workflow-{mode}.json"),
                "chunks": result["chunks"], "lifecycle": result["lifecycle_delta"],
            }
            report["cases"].append(row)
            save()
            print(json.dumps({key: row[key] for key in
                              ("case", "status", "audio", "total_seconds")}, ensure_ascii=False), flush=True)
            if mode == "tts" and result["status"] == "complete":
                previous, _, _ = load_take(result["take_file"], engine.model.config)
            if result["status"] != "complete":
                if not args.continue_on_incomplete:
                    raise RuntimeError(f"Real task validation failed for {mode}")
        report["all_cases_complete"] = all(row["status"] == "complete" for row in report["cases"])
        save()


if __name__ == "__main__":
    main()
