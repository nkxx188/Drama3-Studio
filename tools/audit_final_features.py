"""Inspect saved runtime artifacts for the official features and local UI.

Successful summaries alone are insufficient: inspect persisted jobs, take arrays,
WAV contents and the current regression report. Perceptual quality is separate.
"""

from __future__ import annotations

import json
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
REGRESSION = next((ROOT / f"research/phase{phase}-pytest.xml" for phase in (13, 12, 11)
                   if (ROOT / f"research/phase{phase}-pytest.xml").is_file()))


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def audio_valid(path, expected_hash=None):
    path = Path(path)
    samples, rate = sf.read(path, dtype="float32")
    return (
        rate == 44100 and samples.ndim == 1 and samples.size > 0
        and bool(np.isfinite(samples).all()) and bool(np.abs(samples).max() > 0)
        and (expected_hash is None or hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash)
    )


def take_valid(path):
    with np.load(path, allow_pickle=False) as take:
        codes, frames = take["codes"], take["frames"]
        return (
            codes.ndim == 2 and codes.shape[0] == 10 and codes.shape[1] > 0
            and frames.ndim == 2 and frames.shape[1] == 11
            and bool((codes >= 0).all()) and bool((codes < 4096).all())
            and isinstance(json.loads(str(take["metadata"].item())), dict)
        )


def job_valid(summary_path):
    summary = read_json(summary_path)
    job = read_json(f"ui_data/jobs/{summary['job_id']}/job.json")
    return (
        summary.get("status") == job.get("status") == "complete"
        and job.get("output") == summary["output"]
        and audio_valid(summary["output"], summary.get("wav_sha256"))
        and (not job.get("take_file") or take_valid(job["take_file"]))
    )


def workflow_valid(path):
    result = read_json(path)
    return (
        result.get("status") == "complete" and result.get("chunks")
        and all(chunk["finish_reason"] == "eos" for chunk in result["chunks"])
        and audio_valid(result["output"]) and take_valid(result["take_file"])
    )


def regression_test_passed(name):
    root = ET.parse(REGRESSION).getroot()
    matched = [case for case in root.iter("testcase") if case.attrib.get("name") == name]
    return bool(matched) and all(
        not any(case.find(kind) is not None for kind in ("failure", "error", "skipped"))
        for case in matched)


def reference_clone_valid():
    summary = read_json("research/ui-resident-audio-stream-validation.json")
    job = read_json(f"ui_data/jobs/{summary['stream_job_id']}/job.json")
    baseline = read_json(f"ui_data/jobs/{summary['baseline_job_id']}/job.json")
    with np.load(job["take_file"], allow_pickle=False) as take, np.load(
        baseline["take_file"], allow_pickle=False
    ) as reference:
        metadata = json.loads(str(take["metadata"].item()))
        equal = np.array_equal(take["codes"], reference["codes"])
        equal &= np.array_equal(take["frames"], reference["frames"])
    return (
        job["status"] == baseline["status"] == "complete" and equal
        and len(metadata["references"]) > 0
        and audio_valid(job["output"]) and take_valid(job["take_file"])
    )


def multiple_references_valid():
    report = read_json("research/same-speaker-multireference-validation.json")
    if not report["passed"] or not report["all_reference_codes_enter_actual_prompt"]:
        return False
    if len({row["audio_sha256"] for row in report["references"]}) != 2:
        return False
    for row in report["requests"]:
        if (not workflow_valid(row["report_file"])
                or not audio_valid(row["output"], row["wav_sha256"])):
            return False
        with np.load(row["take_file"], allow_pickle=False) as take:
            references = json.loads(str(take["metadata"].item()))["references"]
        if ([reference["speaker"] for reference in references] != [0] * row["reference_count"]):
            return False
    return {row["reference_count"] for row in report["requests"]} == {1, 2}


def live_text_valid():
    report = read_json("research/ui-live-text-stream-validation.json")
    job = read_json(f"ui_data/jobs/{report['job_id']}/job.json")
    rows = job["result"]["segments"]
    delivered = np.concatenate([sf.read(row["path"], dtype="int16")[0] for row in rows])
    final, rate = sf.read(job["output"], dtype="int16")
    return (
        report["passed"] is True and job["status"] == "complete" and len(rows) == 3
        and rate == 44100 and np.array_equal(delivered, final)
        and [row["lookahead"] for row in rows] == [
            "我们开始今天的测试。", "所有内容都能按顺序听到。", ""]
        and all(take_valid(row["take_file"]) for row in rows)
        and all(workflow_valid(Path(row["path"]).with_suffix(".json")) for row in rows)
    )


def frame_stream_valid():
    report = read_json("research/ui-resident-audio-stream-validation.json")
    rows = report["segments"]
    delivered = np.concatenate([sf.read(row["path"], dtype="int16")[0] for row in rows])
    final, _ = sf.read(report["output"], dtype="int16")
    return (
        reference_clone_valid() and np.array_equal(delivered, final)
        and any(row["before_eos"] and row["sequence_preservation"] == "resident" for row in rows)
        and report["lifecycle"]["park_calls"] == 0
        and not report["deferred_audio_for_host_budget"]
    )


def cancelled_valid():
    report = read_json("research/ui-live-text-stream-validation.json")
    job = read_json(f"ui_data/jobs/{report['waiting_job_id']}/job.json")
    directory = ROOT / "ui_data/jobs" / job["id"]
    return (
        job["status"] == "cancelled" and job.get("cancel_requested") is True
        and job["output"] is None and not (directory / "result.wav").exists()
        and regression_test_passed("test_cancel_wakes_a_live_consumer_and_releases_the_serial_queue")
    )


def main():
    rows = []

    def add(feature, evidence, check):
        row = {"feature": feature, "evidence": evidence}
        try:
            row["status"] = "verified" if check() else "needs_evidence"
        except (KeyError, ValueError, OSError, TypeError, ET.ParseError) as exc:
            row.update(status="needs_evidence", error=f"{type(exc).__name__}: {exc}")
        rows.append(row)

    add("ordinary_tts", "research/ui-real-job-validation.json",
        lambda: job_valid("research/ui-real-job-validation.json"))
    add("voice_cloning", "research/ui-resident-audio-stream-validation.json", reference_clone_valid)
    add("multi_reference", "research/same-speaker-multireference-validation.json", multiple_references_valid)
    add("cross_language", "outputs/capability-cross-language.json",
        lambda: workflow_valid("outputs/capability-cross-language.json"))
    add("director_revision", "research/ui-real-revision-validation.json",
        lambda: job_valid("research/ui-real-revision-validation.json"))
    add("continuation_edit", "outputs/capability-edit.json",
        lambda: workflow_valid("outputs/capability-edit.json"))
    add("infill", "research/ui-real-infill-validation.json",
        lambda: job_valid("research/ui-real-infill-validation.json") and all(
            read_json("research/ui-real-infill-validation.json")["infill"][key]
            for key in ("prefix_codes_equal", "suffix_codes_equal")))
    add("scene_scheduler", "research/ui-real-scene-validation.json",
        lambda: job_valid("research/ui-real-scene-validation.json") and read_json(
            "research/ui-real-scene-validation.json")["native_audio_separators_used"] is False)
    add("text_stream", "research/ui-live-text-stream-validation.json", live_text_valid)
    add("audio_stream", "research/ui-resident-audio-stream-validation.json", frame_stream_valid)
    add("cancellation", "research/ui-live-text-stream-validation.json", cancelled_valid)
    add("voice_upload_validation", str(REGRESSION.relative_to(ROOT)),
        lambda: regression_test_passed("test_voice_storage_keeps_audio_and_transcript_in_a_project_local_record")
        and regression_test_passed("test_voice_upload_rejects_structural_transcript_content"))
    add("persistent_jobs", str(REGRESSION.relative_to(ROOT)),
        lambda: regression_test_passed("test_persisted_running_job_is_recovered_as_interrupted_after_restart"))
    add("browser_interactions", "research/ui-browser-audit.json",
        lambda: read_json("research/ui-browser-audit.json")["passed"] is True)
    add("point_and_scoped_control_inputs", "research/engine-capabilities-validation.json",
        lambda: all(workflow_valid(f"outputs/capability-{label}.json")
                    for label in ("events", "scoped-control")))
    add("long_clone", "outputs/capability-long-clone.json",
        lambda: workflow_valid("outputs/capability-long-clone.json") and read_json(
            "outputs/capability-long-clone.json")["audio"]["duration_seconds"] >= 20)
    add("model_setup_and_verification", "research/ui-model-setup-validation.json",
        lambda: read_json("research/ui-model-setup-validation.json")["passed"] is True
        and read_json("research/ui-model-setup-validation.json")["verification_task"]["status"] == "complete"
        and regression_test_passed("test_model_selection_is_persistent_and_closes_only_an_idle_previous_engine"))
    add("resumable_download", "research/model-download-live-metadata-validation.json",
        lambda: read_json("research/model-download-live-metadata-validation.json")["all_hashes_match"] is True
        and regression_test_passed("test_cancelled_transfer_resumes_only_completed_chunks_then_verifies")
        and regression_test_passed("test_full_hash_mismatch_never_publishes_a_model_file"))
    native = read_json("research/scene-paper-joined-diagnosis.json")
    rows.append({
        "feature": "native_scene_separator", "status": "known_limit",
        "evidence": "research/scene-paper-joined-diagnosis.json",
        "diagnostic_contains_failed_runs": bool(native["runs"]) and all(
            not run["boundaries"] and run["finish_reason"] != "eos" for run in native["runs"]),
    })
    report = {
        "scope": "Functional runtime artifact audit. Reads actual jobs, code/take arrays, EOS reports, PCM and regression cases. This does not prove perceptual quality, speaker identity or direction adherence.",
        "rows": rows,
        "verified_count": sum(row["status"] == "verified" for row in rows),
        "known_limits": [
            "Native audio_separator output is not reliable on the published checkpoint.",
            "Perceptual listening and speaker-similarity evidence is not established.",
            "Multi-reference conditioning is verified; improved perceived similarity from more references is not established.",
            "Only the local Windows/NVIDIA host has end-to-end evidence.",
        ],
        "objective_complete": False,
    }
    target = ROOT / "research/final-feature-audit.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "verified_count": report["verified_count"],
        "total": len(rows),
        "objective_complete": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
