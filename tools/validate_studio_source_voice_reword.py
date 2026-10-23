"""Real no-original-reference rewrite through a checked generated voice reference."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor/fish-speech"), str(ROOT)]

import ui.server as ui
from drama_low_vram.gguf_takes import load_native_take


def main():
    original = json.loads((ROOT / "research/2026-10-19-studio-generation-q4.json").read_text(encoding="utf-8"))
    source = original["jobs"]["design"]
    expected = source["recipe"]["text"]
    recognized = original["transcripts"]["design"]["result"]["text"]
    normalize = lambda text: re.sub(r"[\W_]+", "", text)
    if normalize(expected) != normalize(recognized):
        raise ValueError("The original transcript needs human confirmation before reuse")
    ui.STATE.shutdown()
    state = ui.AppState(data_root=ROOT / ".cache/studio-generation-acceptance/q4/ui_data")
    state.checkpoint = Path(original["model"])
    receipt = ROOT / "research/2026-10-19-studio-source-voice-reword-q4.json"
    report = {
        "source": source["id"], "started": time.time(), "checks": {}, "jobs": {},
        "scope": "Real Q4 rewrite from a generated voice without original references. ASR checks words, not identity or naturalness.",
    }
    def save():
        receipt.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        upload = state.reference_from_version({"source_version": {"job_id": source["id"]}})
        target = "明天早上见，记得带上钥匙。"
        identifier = state.create_job({
            "source_version": {"job_id": source["id"]}, "operation": "reword", "text": target,
            "reword_references": [{"id": upload["id"], "text": expected, "speaker": 0}],
            "reference_transcripts_checked": True, "seed": 123, "max_new_tokens": 256,
        })
        print("RUN source-voice-reword", identifier, flush=True)
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            job = state.job(identifier)
            if job["status"] not in ("queued", "running"):
                break
            time.sleep(.4)
        else:
            state.cancel(identifier)
            raise TimeoutError("source voice rewrite")
        report["jobs"]["reword"] = job
        save()
        if job["status"] != "complete":
            raise RuntimeError(job.get("error") or job["status"])
        take, metadata, chunks = load_native_take(job["take_file"])
        report["checks"]["original_had_no_references"] = not source["recipe"]["reference_snapshot"]
        report["checks"]["uses_confirmed_audio_reference_without_old_history"] = (
            job["recipe"]["conditioning"] == "frozen_reference_regeneration"
            and not job["recipe"].get("previous_take")
            and job["recipe"]["reference_snapshot"][0]["text"] == expected)
        report["checks"]["new_text_has_real_codes_and_one_take"] = (
            metadata["spoken_text"] == target and take.codes.shape[1] > 0 and len(chunks) == 1)
        state._engine.close()
        state._engine = None
        task = state.asr.start(Path(job["output"]), language="auto")
        deadline = time.monotonic() + 300
        while task["status"] in ("queued", "running"):
            if time.monotonic() > deadline:
                raise TimeoutError("ASR")
            time.sleep(.4)
            task = state.asr.get(task["id"])
        report["transcription"] = task
        report["checks"]["spoken_words_match_new_target"] = (
            task["status"] == "complete" and normalize(task["result"]["text"]) == normalize(target))
        if not all(report["checks"].values()):
            raise AssertionError(report["checks"])
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        state.shutdown()
        report["total_seconds"] = time.time() - report["started"]
        save()
    print("ACCEPTANCE", report["status"], flush=True)


if __name__ == "__main__":
    main()
