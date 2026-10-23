"""Bounded real Q4 acceptance for adopted per-line history updates."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "vendor/fish-speech")]

import torch
import ui.server as ui
from drama_low_vram.gguf_takes import load_conditioning_take


def main():
    ui.STATE.shutdown()
    source = json.loads((ROOT / "research/2026-10-19-studio-generation-q4.json").read_text(encoding="utf-8"))
    folder = ROOT / ".cache/studio-context-acceptance" / uuid.uuid4().hex[:10]
    state = ui.AppState(data_root=folder / "ui_data")
    state.checkpoint = Path(source["model"])
    receipt = ROOT / "research/2026-10-19-studio-context-generation-q4.json"
    report = {
        "started_at": time.time(), "model": str(state.checkpoint), "folder": str(folder),
        "scope": "Real Q4 inference, adopted codes and ASR content; naturalness and similarity need listening.",
        "checks": {}, "jobs": {},
    }

    def save():
        receipt.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def run(label, payload):
        print("RUN", label, flush=True)
        identifier = state.create_job(payload)
        deadline = time.monotonic() + 600
        while True:
            job = state.job(identifier)
            if job["status"] not in ("queued", "running"):
                break
            if time.monotonic() >= deadline:
                state.cancel(identifier)
                raise TimeoutError(label)
            time.sleep(.3)
        report["jobs"][label] = job
        save()
        print("RESULT", label, job["status"], job.get("error"), flush=True)
        assert job["status"] == "complete", job
        return job

    try:
        bindings = []
        for reference in source["jobs"]["joint_dialogue"]["recipe"]["reference_snapshot"]:
            path = Path(reference["audio"])
            upload = state.add_reference_upload(path.name, path.read_bytes())
            bindings.append({"id": upload["id"], "text": reference["text"], "speaker": reference["speaker"]})
        baseline = run("baseline_scene", {
            "mode": "scene", "scene_strategy": "per_line", "scene": "Two friends make plans.",
            "characters": [{"name": "林", "description": "Calm and natural.", "speaker": 0},
                           {"name": "周", "description": "Relaxed and friendly.", "speaker": 1}],
            "lines": [{"character": "林", "text": "明天几点出发？"},
                      {"character": "周", "text": "早上九点，我们在门口见。"},
                      {"character": "林", "text": "好，那我们明天见。"}],
            "line_ids": ["first", "second", "third"], "voice_bindings": bindings,
            "seed": 123, "max_new_tokens": 384,
            "studio": {"work_id": "context-acceptance", "segment_id": "dialogue", "layout": "scene"},
        })
        first = run("new_first_line", {
            "source_version": {"job_id": baseline["id"], "part_index": 0}, "operation": "retry", "seed": 456,
        })
        state.adopt_version({
            "work_id": "context-acceptance", "segment_id": "dialogue", "line_id": "first",
            "version": {"job_id": first["id"]},
        })
        report["checks"]["new_preceding_adoption_marks_old_third_line"] = (
            state.scene_context_status({"job_id": baseline["id"]})["lines"][2]["changed"])
        original = state.version({"job_id": baseline["id"], "part_index": 2})["recipe"]
        updated = run("updated_third_line", {
            "source_version": {"job_id": baseline["id"], "part_index": 2},
            "operation": "update_context", "seed": 125,
        })
        recipe = updated["recipe"]
        report["checks"]["new_history_points_to_actual_adopted_codes"] = (
            recipe["previous_sources"][0]["take_file"] == first["take_file"]
            and recipe["conditioning"] == "adopted_scene_history"
            and recipe["text"] == original["text"]
            and recipe["reference_snapshot"] == original["reference_snapshot"])
        expected, _, chunks = load_conditioning_take(first["take_file"])
        expected = chunks[0] if chunks else expected
        history = state._previous(recipe, state._engine)
        report["checks"]["restored_anchor_codes_and_transcript_match"] = (
            torch.equal(history[0].codes, expected.codes) and history[0].text == expected.text)
        part = state.studio.snapshot()["works"]["context-acceptance"]["segments"][0]
        report["checks"]["updated_candidate_not_automatically_adopted"] = "third" not in part.get("line_versions", {})
        state.adopt_version({
            "work_id": "context-acceptance", "segment_id": "dialogue", "line_id": "third",
            "version": {"job_id": updated["id"]},
        })
        report["checks"]["adopting_update_clears_dependency_warning"] = (
            not state.scene_context_status({"job_id": baseline["id"]})["lines"][2]["changed"])
        report["composition"] = state.export_work({"work_id": "context-acceptance"})
        report["checks"]["composition_exists"] = Path(report["composition"]["output"]).is_file()
        state._engine.close()
        task = state.asr.start(Path(updated["output"]), language="auto")
        deadline = time.monotonic() + 300
        while task["status"] in ("queued", "running"):
            if time.monotonic() >= deadline:
                raise TimeoutError("ASR content check")
            time.sleep(.4)
            task = state.asr.get(task["id"])
        report["transcript"] = task
        report["checks"]["updated_content_matches_target"] = (
            task["status"] == "complete"
            and "".join(character for character in task.get("result", {}).get("text", "") if character.isalnum())
            == "好那我们明天见")
        report["status"] = "passed" if all(report["checks"].values()) else "failed"
        save()
        assert report["status"] == "passed", report["checks"]
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        state.shutdown()
        report["total_seconds"] = time.time() - report["started_at"]
        save()
    print("ACCEPTANCE", report["status"], flush=True)


if __name__ == "__main__":
    main()
