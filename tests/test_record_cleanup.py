"""Deleting a row must not break the remaining recipes or their audio."""

from pathlib import Path

import pytest

import ui.server as ui
from test_studio import context, finish, state  # noqa: F401


def test_delete_source_keeps_the_retained_edit_and_reclaims_its_chain(state):
    voice = state.add_voice("Test voice", "Reference.", 0, "reference.wav", b"RIFF" * 32)
    voice_path = Path(state._voice(voice["id"])["audio_path"])
    original = state.create_job({"text": "Original.", "studio": context(), "voice_ids": [voice["id"]]})
    take = finish(state, original)
    revision = state.create_job({"source_version": {"job_id": original}, "operation": "revision"})
    finish(state, revision)
    latest = state.create_job({"source_version": {"job_id": revision}, "operation": "retry"})
    finish(state, latest)
    state.save_draft("single", {"form": {"text": "Keep this draft."}, "active_job_id": original})

    assert state.delete_records("jobs", [original, revision])["deleted"] == sorted([original, revision])
    assert state.delete_records("voices", [voice["id"]])["deleted"] == [voice["id"]]
    assert [job["id"] for job in state.jobs()] == [latest]
    assert [job["id"] for job in state.work_jobs("work")["jobs"]] == [latest]
    assert not state.list_voices() and voice_path.is_file() and take.is_file()
    assert state.prepare_version_restore({"job_id": latest})["references"]
    saved = state.studio.snapshot()["drafts"]["single"]["value"]
    assert saved == {"form": {"text": "Keep this draft."}, "active_job_id": None}

    reopened = ui.AppState(data_root=state.data_root)
    try:
        assert not reopened.list_voices() and [job["id"] for job in reopened.jobs()] == [latest]
        assert reopened.delete_records("jobs", [latest])["deleted"] == [latest]
        assert not take.exists() and not voice_path.exists() and not reopened.jobs()
        assert not reopened.studio.snapshot()["works"]
    finally:
        reopened.shutdown()


def test_bulk_cleanup_protects_running_jobs_and_drops_unreferenced_records(state):
    original = state.create_job({"text": "Original."})
    finish(state, original)
    dependent = state.create_job({"source_version": {"job_id": original}, "operation": "retry"})
    finish(state, dependent)
    active = state.create_job({"text": "Active."})
    result = state.delete_records("jobs", [original, active, dependent])
    assert result == {"deleted": sorted([original, dependent]), "busy": [active]}
    assert state.job(original) is None and state.job(dependent) is None and state.job(active)
    assert not (state.jobs_root / original).exists()
    assert (state.jobs_root / active).is_dir()


def test_voice_kept_for_draft_until_unbound_and_unsafe_ids_rejected(state):
    voice = state.add_voice("测试 音色", "Reference.", 0, "reference.wav", b"RIFF" * 32)
    identifier = voice["id"]
    state.save_draft("single", {"voices": [identifier]})
    with pytest.raises(ValueError, match="Invalid voice id"):
        state.delete_records("voices", [identifier, "../uploads"])
    assert len(state.list_voices()) == 1
    deleted = state.delete_records("voices", [identifier])
    assert not state.list_voices() and [row["id"] for row in deleted["retained"]] == [identifier]
    assert Path(state._voice(identifier)["audio_path"]).is_file()
    state.save_draft("single", {"voices": []})
    assert not (state.voices_root / identifier).exists()
