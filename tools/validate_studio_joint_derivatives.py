"""Real joint-dialogue revision/reword checks using an existing acceptance take."""

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
    parser.add_argument("--model", choices=("q4", "q8"), default="q4")
    parser.add_argument("--transcribe", action="store_true")
    parser.add_argument("--seed", type=int, default=789)
    parser.add_argument("--direction", default="Relaxed, natural conversation between friends.")
    parser.add_argument("--operation", choices=("both", "revision", "reword"), default="both")
    args = parser.parse_args()
    source = json.loads((ROOT / f"research/2026-10-19-studio-generation-{args.model}.json").read_text(
        encoding="utf-8"))["jobs"]["joint_dialogue"]
    ui.STATE.shutdown()
    state = ui.AppState(data_root=ROOT / ".cache/studio-generation-acceptance" / args.model / "ui_data")
    state.checkpoint = Path(source["recipe"]["model"])
    receipt = ROOT / f"research/2026-10-19-studio-joint-derivatives-{args.model}-seed{args.seed}.json"
    report = {
        "model": str(state.checkpoint), "source_job": source["id"], "started": time.time(),
        "scope": "Real GGUF generation and saved codec takes. ASR is a content check, not a naturalness judgment.",
        "checks": {}, "jobs": {},
    }

    def save():
        receipt.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    try:
        for operation in (("revision", "reword") if args.operation == "both" else (args.operation,)):
            payload = {
                "source_version": {"job_id": source["id"]}, "operation": operation,
                "direction": args.direction, "seed": args.seed,
                "max_new_tokens": 512,
            }
            if operation == "reword":
                payload["lines"] = [
                    {"character": "小林", "text": "你把钥匙放在哪里了？"},
                    {"character": "小周", "text": "就在桌上，我们不着急。"},
                    {"character": "小林", "text": "找到了，我们走吧。"},
                ]
                payload["line_ids"] = ["question", "answer", "leaving"]
            print("RUN", operation, flush=True)
            identifier = state.create_job(payload)
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                job = state.job(identifier)
                if job["status"] not in ("queued", "running"):
                    break
                time.sleep(.4)
            else:
                state.cancel(identifier)
                raise TimeoutError(operation)
            report["jobs"][operation] = job
            save()
            report["checks"][operation + "_completed"] = job["status"] == "complete"
            if not job.get("output"):
                raise RuntimeError(f"{operation}: {job.get('error') or job['status']}")
            take, metadata, chunks = load_native_take(
                job.get("take_file") or job["result"]["native_take_file"])
            report["checks"][operation + "_keeps_both_frozen_voices"] = (
                job["recipe"]["reference_snapshot"] == source["recipe"]["reference_snapshot"])
            report["checks"][operation + "_uses_correct_conditioning"] = (
                job["recipe"].get("previous_take") == source["take_file"] if operation == "revision" else
                not job["recipe"].get("previous_take")
                and job["recipe"].get("conditioning") == "frozen_reference_regeneration")
            report["checks"][operation + "_is_one_full_tagged_take"] = (
                len(chunks) == 1 and take.codes.shape[1] > 0
                and metadata["spoken_text"].count("<|speaker:0|>") == 2
                and metadata["spoken_text"].count("<|speaker:1|>") == 1)
            if operation == "reword":
                report["checks"]["reword_saves_updated_script"] = (
                    "我们不着急" in metadata["spoken_text"]
                    and job["recipe"]["lines"][1]["text"] == payload["lines"][1]["text"])
            print("RESULT", operation, job["status"], job["result"].get("total_seconds"), flush=True)
            save()
        if state._engine:
            state._engine.close()
            state._engine = None
        if args.transcribe:
            report["transcripts"] = {}
            for operation, job in report["jobs"].items():
                task = state.asr.start(Path(job["output"]), language="auto")
                deadline = time.monotonic() + 300
                while task["status"] in ("queued", "running"):
                    if time.monotonic() > deadline:
                        raise TimeoutError("ASR " + operation)
                    time.sleep(.4)
                    task = state.asr.get(task["id"])
                report["transcripts"][operation] = task
                save()
                if task["status"] != "complete":
                    raise RuntimeError("ASR " + operation)
                actual = task["result"]["text"]
                report["checks"][operation + "_asr_content"] = (
                    "你把钥匙放在哪里" in actual and "找到了" in actual
                    and ("我们不着急" in actual if operation == "reword" else "别着急" in actual))
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
