"""Bounded real GGUF design, voice reuse and unsplit joint-dialogue acceptance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor/fish-speech"), str(ROOT)]

import ui.server as ui
from drama_low_vram.gguf_takes import load_native_take


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("q4", "q8"), default="q8")
    parser.add_argument("--transcribe", action="store_true")
    args = parser.parse_args()
    originals = ui.STATE.list_voices()
    ui.STATE.shutdown()
    folder = ROOT / ".cache/studio-generation-acceptance" / args.model
    state = ui.AppState(data_root=folder / "ui_data")
    state.checkpoint = next((ROOT / f"checkpoints/drama-3-gguf-{args.model}").glob("*.gguf")).resolve()
    receipt = ROOT / "research" / f"2026-10-19-studio-generation-{args.model}.json"
    report = {"model": str(state.checkpoint), "started": time.time(), "checks": {}, "jobs": {},
              "scope": "Real generation and codec artifacts; naturalness still requires listening."}

    def save():
        receipt.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def run(name, payload):
        print("RUN", name, flush=True)
        identifier = state.create_job(payload)
        limit = time.monotonic() + 900
        while time.monotonic() < limit:
            result = state.job(identifier)
            if result["status"] not in ("queued", "running"):
                break
            time.sleep(.3)
        else:
            state.cancel(identifier)
            raise TimeoutError(name)
        report["jobs"][name] = result
        save()
        print("RESULT", name, result["status"], result.get("error"),
              result.get("result", {}).get("total_seconds"), flush=True)
        assert result["status"] == "complete", (name, result.get("error"), result.get("result"))
        assert Path(result["output"]).is_file() and result["take_file"]
        return result

    try:
        designed = run("design", {
            "mode": "design", "text": "别着急，我们还有时间。坐下来慢慢说。",
            "voice_description": "A clear, warm adult female voice, natural Mandarin, medium pitch and relaxed delivery.",
            "character_name": "Lin", "seed": 456, "max_new_tokens": 384,
            "studio": {"work_id": "design-acceptance", "segment_id": "first", "layout": "design"},
        })
        take, metadata, chunks = load_native_take(designed["take_file"])
        report["checks"]["design_has_real_single_take"] = len(chunks) == 1 and take.codes.shape[1] > 0
        report["checks"]["description_separate_from_speech"] = (
            metadata["spoken_text"] == "别着急，我们还有时间。坐下来慢慢说。"
            and metadata["voice_description"].startswith("A clear"))
        upload = state.reference_from_version({"source_version": {"job_id": designed["id"]}})
        run("reuse_designed_voice", {
            "text": "好的，那我们就明天见。",
            "uploaded_references": [{
                "id": upload["id"], "text": metadata["spoken_text"], "speaker": 0}],
            "seed": 789, "max_new_tokens": 256,
        })
        if len(originals) < 2:
            raise RuntimeError("Two existing demo references are required for the joint-dialogue test")
        bindings = []
        for speaker, source in enumerate(originals[:2]):
            audio = ROOT / "ui_data/voices" / source["id"] / source["audio"]
            voice = state.add_voice(
                f"Acceptance {speaker}", source["transcript"], 0, audio.name, audio.read_bytes())
            bindings.append({"voice_id": voice["id"], "speaker": speaker})
        dialogue = run("joint_dialogue", {
            "mode": "scene", "scene_strategy": "joint",
            "scene": "Two friends look for a key before leaving home.",
            "characters": [{"name": "小林", "description": "Calm, clear voice.", "speaker": 0},
                           {"name": "小周", "description": "Warm, relaxed voice.", "speaker": 1}],
            "lines": [{"character": "小林", "text": "你把钥匙放在哪里了？"},
                      {"character": "小周", "text": "就在桌上，别着急。"},
                      {"character": "小林", "text": "找到了，我们走吧。"}],
            "voice_bindings": bindings, "seed": 123, "max_new_tokens": 512,
            "studio": {"work_id": "joint-acceptance", "segment_id": "first", "layout": "scene"},
        })
        _, _, chunks = load_native_take(dialogue["take_file"])
        result = dialogue["result"]
        report["checks"]["joint_is_one_decode_with_all_references"] = (
            len(chunks) == 1 and len(dialogue["recipe"]["reference_snapshot"]) == 2
            and chunks[0].text.count("<|speaker:0|>") == 2 and "<|speaker:1|>" in chunks[0].text
            and result["strategy"] == "joint" and not result["segments"])
        assert all(report["checks"].values())
        state._engine.close()
        if args.transcribe:
            report["transcripts"] = {}
            for name, job in report["jobs"].items():
                task = state.asr.start(Path(job["output"]), language="auto")
                while task["status"] in ("queued", "running"):
                    time.sleep(.4)
                    task = state.asr.get(task["id"])
                report["transcripts"][name] = task
                save()
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        state.shutdown()
        report["total_seconds"] = time.time() - report["started"]
        save()
    print("ACCEPTANCE", report["status"], flush=True)


if __name__ == "__main__":
    main()
