"""Immutable recipes, creative operations and full-script scheduling."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
from threading import Event, RLock
import time
from types import SimpleNamespace

import pytest
import torch
import numpy as np
import soundfile as sf

import ui.server as ui
from drama_low_vram.gguf_takes import load_native_take, save_native_take
from drama_low_vram.gguf_workflows import GGUFWorkflows
from drama_low_vram.protocols import Character, PreviousTake, SceneLine, VoiceReference, build_design_prompt
from drama_low_vram.scenes import SceneSession
from drama_low_vram.studio_store import StudioStore
from test_gguf_workflows import Native
from test_protocols import Tokenizer
from test_ui_server import defer_workers


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "ROOT", tmp_path)
    monkeypatch.setattr(ui, "load_native_runtime", lambda _root: {"protocol": 4, "sha256": "synthetic-runtime"})
    value = ui.AppState()
    value.checkpoint = tmp_path / "model.gguf"
    defer_workers(value)
    yield value
    value.shutdown()


def context():
    return {"work_id": "work", "segment_id": "part-1", "layout": "single"}


def finish(state, job, text="Hello."):
    from drama_low_vram.codec import write_wav
    output = state.jobs_root / job / "result.wav"
    write_wav(torch.zeros(4096), output)
    take = save_native_take(output, PreviousTake(text, torch.zeros(10, 2, dtype=torch.int64)))
    state._record_result(job, {"status": "complete", "output": str(output), "take_file": str(take)})
    return take


def test_job_recipe_freezes_voice_text_before_queue_and_survives_restart(state):
    voice = state.add_voice("First", "Old reference.", 0, "reference.wav", b"RIFF" * 32)
    payload = {"text": "Original.", "direction": "Warm.", "voice_ids": [voice["id"]], "studio": context()}
    job = state.create_job(payload)
    payload["text"] = "Mutated outside."
    metadata = state.voices_root / voice["id"] / "voice.json"
    metadata.write_text(metadata.read_text(encoding="utf-8").replace("Old reference.", "New reference."))
    saved = state.job(job)
    assert saved["recipe"]["text"] == "Original."
    assert saved["recipe"]["reference_snapshot"][0]["text"] == "Old reference."
    recovered = ui.AppState(data_root=state.data_root)
    try:
        assert recovered.job(job)["recipe"] == saved["recipe"]
        assert recovered.studio.snapshot()["works"]["work"]["segments"][0]["versions"] == [job]
    finally:
        recovered.shutdown()


def test_reference_bindings_override_library_speaker_without_modifying_voice(state):
    voice = state.add_voice("Voice", "Reference.", 0, "reference.wav", b"RIFF" * 32)
    job = state.create_job({"text": "<|speaker:3|> Hello.", "voice_bindings": [
        {"voice_id": voice["id"], "speaker": 3}]})
    assert state.job(job)["recipe"]["reference_snapshot"][0]["speaker"] == 3
    assert state._voice(voice["id"])["speaker"] == 0


def test_retry_uses_source_conditions_and_does_not_adopt_new_candidate(state):
    first = state.create_job({"text": "Original.", "direction": "Warm.", "studio": context()})
    finish(state, first, "Original.")
    second = state.create_job({
        "source_version": {"job_id": first}, "operation": "retry",
        "text": "Ignored.", "direction": "Ignored.", "seed": 456})
    finish(state, second)
    recipe = state.job(second)["recipe"]
    assert recipe["text"] == "Original." and recipe["direction"] == "Warm." and recipe["seed"] == 456
    work = state.studio.snapshot()["works"]["work"]
    assert work["segments"][0]["adopted"]["job_id"] == first
    state.studio.adopt("work", "part-1", {"job_id": second})
    assert state.studio.snapshot()["works"]["work"]["segments"][0]["adopted"]["job_id"] == second


@pytest.mark.parametrize("operation", ["revision", "reword", "continue", "infill"])
def test_derived_actions_bind_source_codes_and_keep_original_recipe(state, operation):
    first = state.create_job({"text": "Original.", "direction": "Warm.", "studio": context()})
    take = finish(state, first, "Original.")
    derived = state.create_job({
        "source_version": {"job_id": first}, "operation": operation,
        "text": "New.", "direction": "Softer.", "start_frame": 0, "end_frame": 1})
    recipe = state.job(derived)["recipe"]
    assert state.job(first)["recipe"]["text"] == "Original."
    if operation == "infill":
        assert recipe["source_take"] == str(take) and not recipe["reference_snapshot"]
        assert "direction" not in recipe and "previous_take" not in recipe
    else:
        assert recipe["previous_take"] == str(take)
        assert recipe["text"] == ("Original." if operation == "revision" else "New.")


def test_source_model_mismatch_is_not_silently_switched(state):
    first = state.create_job({"text": "Original.", "studio": context()})
    finish(state, first)
    state.checkpoint = state.checkpoint.with_name("different.gguf")
    with pytest.raises(ValueError, match="different model"):
        state.create_job({"source_version": {"job_id": first}, "operation": "retry"})
    assert state.checkpoint.name == "different.gguf"


def test_unknown_or_unfinished_source_is_not_used(state):
    job = state.create_job({"text": "Original."})
    with pytest.raises(ValueError, match="complete"):
        state.create_job({"source_version": {"job_id": job}, "operation": "retry"})
    with pytest.raises(ValueError, match="does not exist"):
        state.create_job({"source_version": {"job_id": "missing"}, "operation": "retry"})


def test_fresh_request_does_not_inherit_another_version(state):
    first = state.create_job({"text": "Original.", "studio": context()})
    finish(state, first)
    second = state.create_job({"text": "New.", "studio": {"work_id": "new", "segment_id": "p"}})
    assert not state.job(second)["recipe"].get("previous_take")
    assert not state.job(second)["recipe"].get("source_version")


def test_draft_persists_text_reference_and_scene_state_separately(tmp_path):
    store = StudioStore(tmp_path)
    store.save_draft("single", {"text": "One.", "upload": {"id": "saved"}})
    store.save_draft("scene", {"roles": [{"id": "a"}], "strategy": "joint"})
    restored = StudioStore(tmp_path).snapshot()
    assert restored["drafts"]["single"]["value"]["upload"]["id"] == "saved"
    assert restored["drafts"]["scene"]["value"]["strategy"] == "joint"
    with pytest.raises(ValueError):
        store.save_draft("../outside", {})
    with pytest.raises(ValueError):
        store.save_draft("single", {"invalid": float("nan")})


def test_joint_dialogue_batches_complete_lines_with_only_active_references(tmp_path):
    engine = Native()
    session = SceneSession(
        engine, tmp_path / "joint.wav", scene="A room.",
        characters=[Character("A", "Calm."), Character("B", "Bright.", 1)],
        lines=[SceneLine("A", "Hello. " * 20), SceneLine("B", "Hi."), SceneLine("A", "Again.")],
        references=[VoiceReference("a.wav", "A.", 0), VoiceReference("b.wav", "B.", 1)],
        strategy="joint", max_chunk_chars=6)
    assert list(session.run()) == []
    assert session.report["strategy"] == "joint" and session.report["take_file"]
    assert len(engine.calls) == 3
    assert all(len(options["references"]) == 1 and options["task_mode"] == "joint"
               for _text, options in engine.calls)
    assert [options["references"][0].speaker for _text, options in engine.calls] == [0, 1, 0]
    assert engine.calls[0][0].count("<|speaker:0|>") == 1
    assert engine.calls[1][0] == "<|speaker:1|> Hi."
    assert engine.calls[2][0] == "<|speaker:0|> Again."
    assert session.report["segments"] == [] and session.report["chunks"]
    assert not session.report["native_audio_separators_used"]


def test_joint_dialogue_ignores_control_markup_in_character_target(tmp_path):
    engine = Native()
    lines = [
        SceneLine("A", "[frustrated]" + "甲" * 37),
        SceneLine("B", "[panting][guilty]" + "乙" * 28),
        SceneLine("A", "[holding back anger]" + "丙" * 15),
        SceneLine("B", "<soft>" + "丁" * 38 + "</soft>"),
        SceneLine("A", "[sighing][softening]" + "戊" * 19),
    ]
    session = SceneSession(
        engine, tmp_path / "joint.wav", scene="A room.",
        characters=[Character("A", "Calm."), Character("B", "Bright.", 1)],
        lines=lines, strategy="joint", max_chunk_chars=160)
    assert list(session.run()) == []
    assert len(engine.calls) == 1
    assert all(line.text in engine.calls[0][0] for line in lines)


def test_joint_dialogue_merges_small_tail_within_soft_overflow(tmp_path):
    engine = Native()
    session = SceneSession(
        engine, tmp_path / "joint.wav", scene="A room.",
        characters=[Character("A", "Calm.")],
        lines=[SceneLine("A", "甲" * 80), SceneLine("A", "乙" * 70),
               SceneLine("A", "丙" * 20)],
        strategy="joint", max_chunk_chars=160)
    assert list(session.run()) == []
    assert len(engine.calls) == 1
    assert "甲" * 80 in engine.calls[0][0] and "丙" * 20 in engine.calls[0][0]


def test_joint_dialogue_tail_merge_never_adds_a_fourth_speaker(tmp_path):
    engine = Native()
    session = SceneSession(
        engine, tmp_path / "joint.wav", scene="A room.",
        characters=[Character(str(index), "Natural.", index) for index in range(4)],
        lines=[SceneLine(str(index), f"Line {index}.") for index in range(4)],
        strategy="joint", max_chunk_chars=160)
    assert list(session.run()) == []
    assert len(engine.calls) == 2
    assert engine.calls[1][0] == "<|speaker:3|> Line 3."


def test_design_uses_character_system_not_spoken_description(tmp_path):
    engine = Native()
    result = GGUFWorkflows(engine).run(
        "design", "Hello.", tmp_path / "designed.wav",
        voice_description="A warm adult voice.", character_name="Ada", direction="Softly.")
    text, options = engine.calls[0]
    assert text == "[Softly.] Hello." and "warm adult" not in text
    assert options["task_mode"] == "design" and options["design"]["name"] == "Ada"
    take, metadata, _ = load_native_take(result["take_file"])
    assert metadata["spoken_text"] == "Hello." and metadata["voice_description"] == "A warm adult voice."


def test_python_design_prompt_places_description_in_system_only():
    tokenizer = Tokenizer()
    config = SimpleNamespace(audio_decoder=SimpleNamespace(num_codebooks=3))
    prompt = build_design_prompt("Hello.", tokenizer, config, name="Ada", description="A warm voice.")
    assert prompt.ndim == 2
    # The fixture tokenizer records all literal segments in order.
    assert prompt.shape[1] == 4


@pytest.mark.parametrize("payload", [
    {"text": "Hello.", "temperature": float("inf")},
    {"text": "Hello.", "seed": True},
    {"text": "Hello.", "max_new_tokens": 999999},
    {"text": "Hello.", "top_p": 0},
    {"text": "Hello.", "candidates": 4},
])
def test_bad_sampling_never_enters_queue(state, payload):
    with pytest.raises(ValueError):
        state.create_job(payload)
    assert not state.jobs() and state._engine is None


def test_automatic_generation_budget_is_per_chunk_not_total_job_length(state):
    text = ("A full sentence. " * 80).strip()
    identifier = state.create_job({
        "text": text, "automatic_length": True, "max_new_tokens": 1, "max_chunk_chars": 80})
    recipe = state.job(identifier)["recipe"]
    assert recipe["length_policy"] == "automatic-v1"
    assert 768 <= recipe["max_new_tokens"] < 2048
    assert "automatic_length" not in recipe


def test_automatic_joint_budget_counts_speech_and_leaves_room_beyond_old_limit(state, monkeypatch):
    monkeypatch.setattr(state, "capabilities", lambda: {"joint_dialogue": True})
    payload = dialogue()
    for line, width in zip(payload["lines"], (50, 50, 37)):
        line["text"] = "[calm]" + "\u7532" * width
    payload["automatic_length"] = True
    identifier = state.create_job(payload)
    assert state.job(identifier)["recipe"]["max_new_tokens"] == 137 * 8 + 128


def test_automatic_budget_honors_duration_hint_and_remains_bounded(state):
    identifier = state.create_job({
        "text": "Short.", "automatic_length": True, "duration_frames": 1500})
    assert state.job(identifier)["recipe"]["max_new_tokens"] == 1500
    identifier = state.create_job({
        "text": "\u7532" * 10000, "automatic_length": True, "single_chunk": True})
    assert state.job(identifier)["recipe"]["max_new_tokens"] == 2048
    with pytest.raises(ValueError, match="boolean"):
        state.create_job({"text": "Short.", "automatic_length": "true"})


def test_retry_preserves_frozen_limit_but_new_edit_can_request_automatic_length(state):
    first = state.create_job({"text": "Original.", "max_new_tokens": 80, "studio": context()})
    finish(state, first)
    retried = state.create_job({"source_version": {"job_id": first}, "operation": "retry"})
    assert state.job(retried)["recipe"]["max_new_tokens"] == 80
    revised = state.create_job({
        "source_version": {"job_id": first}, "operation": "revision",
        "automatic_length": True})
    assert state.job(revised)["recipe"]["max_new_tokens"] == 768


def dialogue(strategy="joint"):
    return {
        "mode": "scene", "scene": "A quiet room.", "scene_strategy": strategy,
        "characters": [{"name": "Ada", "description": "Calm.", "speaker": 0},
                       {"name": "Bo", "description": "Warm.", "speaker": 1}],
        "character_ids": ["ada", "bo"],
        "lines": [{"character": "Ada", "text": "First."}, {"character": "Bo", "text": "Second."},
                  {"character": "Ada", "text": "Third."}],
        "line_ids": ["first", "second", "third"], "studio": {**context(), "layout": "scene"},
    }


def run_latest(state):
    function, args = state._executor.calls[-1]
    function(*args)


class DistinctNative(Native):
    def generate(self, text, output, **kwargs):
        result = super().generate(text, output, **kwargs)
        from drama_low_vram.codec import write_wav
        write_wav(torch.full((result["audio"]["samples"],), len(self.calls) / 10), output)
        return result


def test_joint_reword_updates_script_and_metadata_without_mutating_source(state, monkeypatch):
    monkeypatch.setattr(state, "capabilities", lambda: {"joint_dialogue": True})
    engine = DistinctNative()
    state._get_engine = lambda _: engine
    first = state.create_job(dialogue())
    run_latest(state)
    changed = state.create_job({
        "source_version": {"job_id": first}, "operation": "reword",
        "lines": [{"character": "Bo", "text": "<whisper>New words.</whisper>"}], "line_ids": ["rewritten"],
        "direction": "More relaxed.",
    })
    run_latest(state)
    recipe = state.job(changed)["recipe"]
    assert recipe["text"] == "<|speaker:1|> <whisper>New words.</whisper>"
    assert recipe["lines"][0]["character"] == "Bo" and recipe["line_ids"] == ["rewritten"]
    assert recipe["character_ids"] == ["ada", "bo"] and recipe["single_chunk"]
    assert recipe["conditioning"] == "source_history"
    assert state.job(first)["recipe"]["lines"] == dialogue()["lines"]
    assert engine.calls[-1][1]["previous_takes"][0].codes.shape[1] == 4
    assert "Speaker 0: Calm." in engine.calls[-1][1]["direction"]
    assert "More relaxed." in engine.calls[-1][1]["direction"]


def test_reword_with_references_regenerates_new_words_without_old_performance_history(state, monkeypatch):
    monkeypatch.setattr(state, "capabilities", lambda: {"joint_dialogue": True})
    engine = DistinctNative()
    state._get_engine = lambda _: engine
    voice = state.add_voice("Ada", "Reference.", 0, "reference.wav", b"RIFF" * 32)
    payload = dialogue()
    payload["voice_bindings"] = [{"voice_id": voice["id"], "speaker": 0}]
    first = state.create_job(payload)
    run_latest(state)
    rewritten = state.create_job({
        "source_version": {"job_id": first}, "operation": "reword",
        "lines": [{"character": "Ada", "text": "Different words."}], "line_ids": ["different"],
    })
    run_latest(state)
    recipe = state.job(rewritten)["recipe"]
    assert recipe["mode"] == "tts" and recipe["operation"] == "reword"
    assert recipe["conditioning"] == "frozen_reference_regeneration"
    assert not recipe.get("previous_take") and not engine.calls[-1][1]["previous_takes"]
    assert recipe["reference_snapshot"] == state.job(first)["recipe"]["reference_snapshot"]
    assert engine.calls[-1][0] == "<|speaker:0|> Different words."
    assert "Scene: A quiet room." in engine.calls[-1][1]["direction"]
    revised = state.create_job({"source_version": {"job_id": first}, "operation": "revision", "direction": "Softer."})
    assert state.job(revised)["recipe"]["previous_take"] == state.job(first)["take_file"]
    partial = state.create_job({
        "source_version": {"job_id": first}, "operation": "reword",
        "lines": [{"character": "Bo", "text": "Bo has no separate reference."}], "line_ids": ["bo-only"],
    })
    assert state.job(partial)["recipe"]["previous_take"] == state.job(first)["take_file"]
    assert state.job(partial)["recipe"]["conditioning"] == "source_history"


def test_joint_infill_keeps_scene_script_consistent_for_later_edits(state, monkeypatch):
    monkeypatch.setattr(state, "capabilities", lambda: {"joint_dialogue": True})
    state._get_engine = lambda _: DistinctNative()
    first = state.create_job(dialogue())
    run_latest(state)
    second = state.create_job({
        "source_version": {"job_id": first}, "operation": "infill",
        "text": "<|speaker:0|> First changed.\n<|speaker:1|> Second.",
        "start_frame": 1, "end_frame": 2,
    })
    assert [line["text"] for line in state.job(second)["recipe"]["lines"]] == ["First changed.", "Second."]
    assert len(state.job(second)["recipe"]["line_ids"]) == 2
    with pytest.raises(ValueError, match="speaker labels"):
        state.create_job({"source_version": {"job_id": first}, "operation": "reword", "text": "Missing labels."})


@pytest.mark.parametrize("changes", [
    {"duration_frames": 100, "max_new_tokens": 80},
    {"duration_frames": True}, {"duration_frames": 2.5}, {"duration_frames": -1},
])
def test_invalid_duration_is_rejected_before_queueing(state, changes):
    with pytest.raises(ValueError):
        state.create_job({"text": "Hello.", **changes})
    assert not state.jobs()


def test_conflicting_scene_speaker_tag_is_rejected_before_queueing(state):
    payload = dialogue("per_line")
    payload["lines"][0]["text"] = "<|speaker:1|> Wrong character."
    with pytest.raises(ValueError, match="conflicts"):
        state.create_job(payload)
    assert not state.jobs()


def test_line_retry_restores_real_history_and_adoption_is_explicit(state):
    state._get_engine = lambda _: DistinctNative()
    first = state.create_job(dialogue("per_line"))
    run_latest(state)
    version = state.version({"job_id": first, "part_index": 2})
    recipe = version["recipe"]
    assert recipe["seed"] == 125 and recipe["previous_sources"][0]["edge"] == "first"
    assert version["studio"]["line_id"] == "third"
    retried = state.create_job({"source_version": {"job_id": first, "part_index": 2}, "operation": "retry"})
    run_latest(state)
    work = state.studio.snapshot()["works"]["work"]
    assert not work["segments"][0].get("line_versions")
    state.adopt_version({"work_id": "work", "segment_id": "part-1", "line_id": "third",
                         "version": {"job_id": retried}})
    assert state.studio.snapshot()["works"]["work"]["segments"][0]["line_versions"]["third"]["job_id"] == retried
    with pytest.raises(ValueError, match="dialogue line"):
        state.adopt_version({"work_id": "work", "segment_id": "part-1", "line_id": "second",
                             "version": {"job_id": retried}})
    with pytest.raises(ValueError, match="entire scene"):
        state.adopt_version({"work_id": "work", "segment_id": "part-1", "version": {"job_id": retried}})


def test_composition_has_exact_pcm_order_and_frozen_resolved_manifest(state):
    engine = DistinctNative()
    state._get_engine = lambda _: engine
    first = state.create_job(dialogue("per_line"))
    run_latest(state)
    second = state.create_job({"source_version": {"job_id": first, "part_index": 1}, "operation": "retry"})
    run_latest(state)
    state.adopt_version({"work_id": "work", "segment_id": "part-1", "line_id": "second",
                         "version": {"job_id": second}})
    result = state.export_work({"versions": [{"job_id": first}]})
    rows = state.job(first)["result"]["segments"]
    paths = [rows[0]["path"], state.job(second)["output"], rows[2]["path"]]
    pcm, rate = sf.read(result["output"], dtype="int16")
    assert rate == 44100 and np.array_equal(pcm, np.concatenate([sf.read(p, dtype="int16")[0] for p in paths]))
    manifest = json.loads(Path(result["output"]).with_name("composition.json").read_text(encoding="utf-8"))
    assert [part["line_id"] for part in manifest["parts"]] == ["first", "second", "third"]
    assert manifest["parts"][1]["source_version"] == {"job_id": second}
    assert manifest["parts"][1]["recipe"]["source_version"] == {"job_id": first, "part_index": 1}
    state.adopt_version({"work_id": "work", "segment_id": "part-1", "line_id": "second",
                         "version": {"job_id": first, "part_index": 1}})
    assert sf.read(result["output"], dtype="int16")[0].tolist() == pcm.tolist()
    assert json.loads(Path(result["output"]).with_name("composition.json").read_text()) == manifest


def test_failed_scene_keeps_completed_takes_and_can_export_after_line_repair(state):
    class FailingNative(DistinctNative):
        def generate(self, text, output, **kwargs):
            if "Third." in text:
                raise RuntimeError("Fixture failure on third line")
            return super().generate(text, output, **kwargs)
    state._get_engine = lambda _: FailingNative()
    first = state.create_job(dialogue("per_line"))
    run_latest(state)
    failed = state.job(first)
    assert failed["status"] == "error" and len(failed["result"]["segments"]) == 2
    assert state.version({"job_id": first, "part_index": 1})["status"] == "complete"
    with pytest.raises(ValueError, match="Every dialogue line"):
        state.export_work({"versions": [{"job_id": first}]})
    state._get_engine = lambda _: DistinctNative()
    repair = state.create_job({"text": "<|speaker:0|> Third.", "studio": {**context(), "line_id": "third"}})
    run_latest(state)
    assert state.studio.snapshot()["works"]["work"]["segments"][0]["adopted"] is None
    result = state.export_work({"versions": [{"job_id": first}]})
    assert Path(result["output"]).is_file()
    manifest = json.loads(Path(result["output"]).with_name("composition.json").read_text())
    assert manifest["parts"][-1]["source_version"] == {"job_id": repair}


def test_unrelated_line_version_cannot_be_adopted(state):
    first = state.create_job({"text": "Hello.", "studio": context()})
    finish(state, first)
    with pytest.raises(ValueError, match="belong"):
        state.adopt_version({"work_id": "other", "segment_id": "part-1",
                             "line_id": "line", "version": {"job_id": first}})


def test_individually_generated_lines_export_in_script_order_without_a_full_scene_job(state):
    engine = DistinctNative()
    state._get_engine = lambda _: engine
    jobs = {}
    for index in (2, 0, 1):
        payload = dialogue("per_line")
        payload["lines"] = [payload["lines"][index]]
        payload["line_ids"] = [payload["line_ids"][index]]
        payload["scene_order"] = ["first", "second", "third"]
        payload["studio"]["line_id"] = payload["line_ids"][0]
        jobs[index] = state.create_job(payload)
        run_latest(state)
        if index == 2:
            with pytest.raises(ValueError, match="Every dialogue line"):
                state.export_work({"work_id": "work"})
    output = state.export_work({"work_id": "work"})["output"]
    pcm, _ = sf.read(output, dtype="int16")
    expected = np.concatenate([sf.read(state.job(jobs[index])["output"], dtype="int16")[0]
                               for index in range(3)])
    assert np.array_equal(pcm, expected)
    manifest = json.loads(Path(output).with_name("composition.json").read_text())
    assert [part["line_id"] for part in manifest["parts"]] == ["first", "second", "third"]
    assert [part["source_version"]["job_id"] for part in manifest["parts"]] == [jobs[index] for index in range(3)]


def test_invalid_scene_order_is_rejected(state):
    payload = dialogue("per_line")
    payload["scene_order"] = ["first", "second"]
    with pytest.raises(ValueError, match="line order"):
        state.create_job(payload)
    assert not state.jobs()


def test_reference_group_reuses_every_clip_and_can_remap_or_override_one_clip(state):
    first = state.add_reference_upload("first.wav", b"RIFF" * 32)
    second = state.add_reference_upload("second.wav", b"WAVE" * 32)
    voice = state.save_reference_group({"name": "Two clips", "references": [
        {"id": first["id"], "text": "First reference."}, {"id": second["id"], "text": "Second reference."},
    ]})
    assert voice["schema"] == 2 and len(state.list_voices()[0]["references"]) == 2
    job = state.create_job({"text": "<|speaker:4|> Hello.", "voice_bindings": [
        {"voice_id": voice["id"], "speaker": 4}]})
    references = state.job(job)["recipe"]["reference_snapshot"]
    assert [row["speaker"] for row in references] == [4, 4]
    assert [row["text"] for row in references] == ["First reference.", "Second reference."]
    one = state.create_job({"text": "Hello.", "voice_bindings": [
        {"voice_id": voice["id"], "reference_index": 1, "text": "Corrected.", "speaker": 0}]})
    assert [row["text"] for row in state.job(one)["recipe"]["reference_snapshot"]] == ["Corrected."]
    assert state._voice(voice["id"])["references"][1]["transcript"] == "Second reference."


def test_reference_group_deduplicates_identical_material_and_preserves_old_single_voice(state):
    legacy = state.add_voice("Legacy", "<|speaker:2|> Original.", 2, "old.wav", b"RIFF" * 32)
    assert len(state._voice(legacy["id"])["references"]) == 1
    voice = state.save_reference_group({"name": "Group", "references": [
        {"voice_id": legacy["id"]}, {"voice_id": legacy["id"]},
    ]})
    assert len(voice["references"]) == 1 and voice["speaker"] == 0
    assert voice["transcript"] == "Original."
    assert state._voice(legacy["id"])["speaker"] == 2
    assert Path(voice["audio_path"]).read_bytes() == Path(state._voice(legacy["id"])["audio_path"]).read_bytes()


def test_reference_group_copy_fallback_does_not_modify_uploaded_material(state, monkeypatch):
    source = state.add_reference_upload("source.wav", b"RIFF" * 32)
    def unavailable(*_args):
        raise OSError("No hard links on this filesystem")
    monkeypatch.setattr(ui.os, "link", unavailable)
    voice = state.save_reference_group({"name": "Copied", "references": [{"id": source["id"], "text": "Words."}]})
    assert Path(voice["audio_path"]).read_bytes() == b"RIFF" * 32
    assert Path(state.reference_upload(source["id"])["audio_path"]).read_bytes() == b"RIFF" * 32


def test_expanded_group_clip_limit_is_enforced_before_queue(state):
    source = state.add_reference_upload("source.wav", b"RIFF" * 32)
    voice = state.save_reference_group({"name": "Group", "references": [
        {"id": source["id"], "text": "First."}, {"id": source["id"], "text": "Second."}]})
    with pytest.raises(ValueError, match="32 reference"):
        state.create_job({"text": "Hello.", "voice_bindings": [{"voice_id": voice["id"]}] * 17})
    assert not state.jobs()


@pytest.mark.parametrize("index", [-1, True, 1.5, 100])
def test_invalid_group_clip_index_is_rejected_before_queue(state, index):
    voice = state.add_voice("Single", "Words.", 0, "source.wav", b"RIFF" * 32)
    with pytest.raises(ValueError, match="reference index"):
        state.create_job({"text": "Hello.", "voice_bindings": [{"voice_id": voice["id"], "reference_index": index}]})
    assert not state.jobs()


def test_preparing_restore_uses_frozen_text_and_deduplicated_assets_without_loading_models(state):
    voice = state.add_voice("Source", "Original words.", 0, "source.wav", b"RIFF" * 32)
    job = state.create_job({"text": "Target.", "voice_ids": [voice["id"]], "temperature": .75,
                            "studio": context()})
    finish(state, job, "Target.")
    metadata = state.voices_root / voice["id"] / "voice.json"
    data = json.loads(metadata.read_text(encoding="utf-8"))
    data["transcript"] = "Changed later."
    metadata.write_text(json.dumps(data), encoding="utf-8")
    first = state.prepare_version_restore({"job_id": job})
    again = state.prepare_version_restore({"job_id": job})
    assert first["references"][0]["id"] == again["references"][0]["id"]
    assert first["references"][0]["text"] == "Original words."
    assert first["version"]["recipe"]["temperature"] == .75
    assert first["version"]["take"]["prompt_text"]
    assert state._engine is None
    restored = state.create_job({
        "text": "New target.", "condition_source": {"job_id": job}, "model": str(state.checkpoint),
        "voice_bindings": [{"id": first["references"][0]["id"], "text": first["references"][0]["text"]}],
        "studio": context(), "seed": 123,
    })
    assert state.job(restored)["recipe"]["reference_snapshot"][0]["text"] == "Original words."
    assert not state.job(restored)["recipe"].get("previous_take")


def test_work_history_can_recover_more_than_recent_100_jobs_in_stable_pages(state):
    identifiers = [state.create_job({"text": f"Version {index}.", "studio": context()}) for index in range(112)]
    assert identifiers[0] not in {row["id"] for row in state.jobs()}
    first = state.work_jobs("work", limit=100)
    second = state.work_jobs("work", offset=first["next_offset"], limit=100)
    assert first["total"] == 112 and second["next_offset"] is None
    assert [job["id"] for job in first["jobs"] + second["jobs"]] == identifiers
    assert "events" not in first["jobs"][0]
    assert first["jobs"][0]["recipe"]["text"] == "Version 0."
    with pytest.raises(ValueError):
        state.work_jobs("work", limit=101)


def test_joint_restored_conditions_compile_edited_lines_and_keep_explicit_history(state, monkeypatch):
    monkeypatch.setattr(state, "capabilities", lambda: {"joint_dialogue": True})
    source = state.create_job({"text": "Saved settings."})
    payload = dialogue()
    payload.update(mode="tts", text="Stale text.", previous_take=None, condition_source={"job_id": source},
                   single_chunk=True)
    identifier = state.create_job(payload)
    assert state.job(identifier)["recipe"]["text"].startswith("<|speaker:0|> First.")
    assert state.job(identifier)["recipe"]["text"] != "Stale text."


def test_live_stream_records_accepted_text_separately_from_immutable_recipe(state):
    identifier = state.create_job({"mode": "stream", "chunks": ["First."], "live_stream": True, "studio": context()})
    state.stream_input(identifier, text="Second.")
    state.stream_input(identifier, finish=True)
    assert state.job(identifier)["recipe"]["chunks"] == ["First."]
    assert state.job(identifier)["stream_input"]["chunks"] == ["First.", "Second."]
    assert state.version({"job_id": identifier})["text"] == "First.\nSecond."


def test_invalid_restore_provenance_is_not_recorded(state):
    with pytest.raises(ValueError, match="does not exist"):
        state.create_job({"text": "Hi.", "condition_source": {"job_id": "missing"}})
    assert not state.jobs()


def test_retry_of_finished_live_stream_replays_received_chunks_without_open_input(state):
    identifier = state.create_job({"mode": "stream", "chunks": ["First."], "live_stream": True})
    state.stream_input(identifier, text="Second.")
    state.stream_input(identifier, finish=True)
    finish(state, identifier, "First.\nSecond.")
    with state._jobs_lock:
        state._jobs[identifier]["take_file"] = None
    retried = state.create_job({"source_version": {"job_id": identifier}, "operation": "retry", "seed": 123})
    recipe = state.job(retried)["recipe"]
    assert recipe["chunks"] == ["First.", "Second."] and not recipe["live_stream"]
    assert recipe["replay_stream"] and recipe["text"] == "First.\nSecond."
    assert state.job(identifier)["recipe"]["chunks"] == ["First."]


def test_reword_without_original_references_can_use_checked_source_audio_without_history(state):
    first = state.create_job({"text": "Original words.", "studio": context()})
    finish(state, first, "Original words.")
    upload = state.reference_from_version({"source_version": {"job_id": first}})
    payload = {"source_version": {"job_id": first}, "operation": "reword", "text": "New words.",
               "reword_references": [{"id": upload["id"], "text": "Original words.", "speaker": 0}]}
    with pytest.raises(ValueError, match="transcript"):
        state.create_job(payload)
    rewritten = state.create_job({**payload, "reference_transcripts_checked": True})
    recipe = state.job(rewritten)["recipe"]
    assert recipe["mode"] == "tts" and not recipe.get("previous_take")
    assert recipe["reference_snapshot"][0]["text"] == "Original words."
    assert recipe["conditioning"] == "frozen_reference_regeneration"
    assert not state.job(first)["recipe"]["reference_snapshot"]


def test_prepared_reword_reference_cannot_replace_a_frozen_source_voice(state):
    voice = state.add_voice("Source", "Reference.", 0, "reference.wav", b"RIFF" * 32)
    first = state.create_job({"text": "Original.", "voice_ids": [voice["id"]]})
    finish(state, first)
    with pytest.raises(ValueError, match="missing source voices"):
        state.create_job({
            "source_version": {"job_id": first}, "operation": "reword", "text": "New words.",
            "reference_transcripts_checked": True,
            "reword_references": [{"voice_id": voice["id"], "speaker": 0}],
        })


def test_joint_context_lower_bound_rejects_impossible_reference_budget_before_loading(state, monkeypatch):
    monkeypatch.setattr(state, "capabilities", lambda: {"joint_dialogue": True})
    from drama_low_vram.codec import write_wav
    source = state.data_root / "long-reference.wav"
    write_wav(torch.zeros(44100 * 120), source)
    upload = state.add_reference_upload(source.name, source.read_bytes())
    payload = dialogue()
    payload["voice_bindings"] = [{"id": upload["id"], "text": f"Reference {index}.", "speaker": 0}
                                 for index in range(4)]
    with pytest.raises(ValueError, match="context budget"):
        state.create_job(payload)
    assert not state.jobs() and state._engine is None


def test_runtime_context_failure_has_actionable_joint_error_and_keeps_technical_detail(state, monkeypatch):
    monkeypatch.setattr(state, "capabilities", lambda: {"joint_dialogue": True})
    class TooLong(Native):
        def generate(self, *_args, **_kwargs):
            raise RuntimeError("capacity exceeded; required_context=9000")
    state._get_engine = lambda _: TooLong()
    identifier = state.create_job(dialogue())
    run_latest(state)
    job = state.job(identifier)
    assert job["status"] == "error" and "choose per-line" in job["error"]
    assert job["recipe"]["scene_strategy"] == "joint"
    assert any("required_context=9000" in event.get("technical_error", "") for event in job["events"])


def test_partial_legacy_recipe_is_not_advertised_as_complete(state):
    identifier = state.create_job({"text": "Hello."})
    finish(state, identifier)
    state._jobs[identifier]["recipe"] = {"text": "Hello.", "model": str(state.checkpoint)}
    assert not state.version({"job_id": identifier})["complete_recipe"]
    with pytest.raises(ValueError, match="no complete recipe"):
        state.create_job({"source_version": {"job_id": identifier}, "operation": "retry"})


def test_missing_result_take_does_not_hide_saved_conditions(state):
    identifier = state.create_job({"text": "Hello."})
    take = finish(state, identifier)
    Path(take).unlink()
    version = state.version({"job_id": identifier})
    assert version["complete_recipe"] and version["take_missing"] and version["take_file"] is None
    assert version["text"] == "Hello."


def test_stream_fragment_uses_its_own_text_without_claiming_full_replay_conditions(state):
    identifier = state.create_job({"mode": "stream", "chunks": ["One.", "Two."]})
    finish(state, identifier, "Two.")
    job = state.job(identifier)
    state._jobs[identifier]["result"]["segments"] = [{
        "path": job["output"], "take_file": job["take_file"], "text": "Two.", "index": 0}]
    version = state.version({"job_id": identifier, "part_index": 0})
    assert version["text"] == "Two." and not version["complete_recipe"]


def test_cancel_during_later_stream_chunk_preserves_already_delivered_audio_and_takes(state):
    class CancelSecond(Native):
        def generate(self, text, output, **kwargs):
            if text == "Second.":
                state.cancel(identifier)
                raise ui.GenerationCancelled()
            return super().generate(text, output, **kwargs)
    state._get_engine = lambda _: CancelSecond()
    identifier = state.create_job({"mode": "stream", "chunks": ["First.", "Second."]})
    run_latest(state)
    job = state.job(identifier)
    assert job["status"] == "cancelled" and job["output"] is None
    assert len(job["result"]["segments"]) == 1
    assert job["result"]["segments"][0]["text"] == "First."
    assert Path(job["result"]["segments"][0]["path"]).is_file()
    assert Path(job["result"]["segments"][0]["take_file"]).is_file()


def test_older_completed_stream_recovers_text_from_delivered_rows_without_mutating_recipe(state):
    state._get_engine = lambda _: Native()
    identifier = state.create_job({"mode": "stream", "chunks": ["First.", "Second."]})
    run_latest(state)
    state._jobs[identifier]["stream_input"].pop("chunks")
    state._jobs[identifier]["recipe"]["chunks"] = ["First."]
    version = state.version({"job_id": identifier})
    assert version["text"] == "First.\nSecond."
    assert version["stream_input"]["chunks"] == ["First.", "Second."]
    assert state.job(identifier)["recipe"]["chunks"] == ["First."]


def test_prepare_restore_can_fill_partial_voice_coverage_without_replacing_frozen_voice(state, monkeypatch):
    monkeypatch.setattr(state, "capabilities", lambda: {"joint_dialogue": True})
    state._get_engine = lambda _: Native()
    original = state.add_voice("Ada", "Frozen words.", 0, "ada.wav", b"RIFF" * 32)
    missing = state.add_reference_upload("bo.wav", b"WAVE" * 32)
    payload = dialogue()
    payload["voice_bindings"] = [{"voice_id": original["id"], "speaker": 0}]
    job = state.create_job(payload)
    run_latest(state)
    prepared = state.prepare_version_restore({
        "job_id": job, "prepared_references": [{"id": missing["id"], "text": "Bo's words.", "speaker": 1}],
        "reference_transcripts_checked": True,
    })
    assert [(clip["speaker"], clip["text"]) for clip in prepared["references"]] == [
        (0, "Frozen words."), (1, "Bo's words.")]
    assert len(state.job(job)["recipe"]["reference_snapshot"]) == 1
    with pytest.raises(ValueError, match="transcript"):
        state.prepare_version_restore({
            "job_id": job, "prepared_references": [{"id": missing["id"], "text": "Bo's words.", "speaker": 1}],
        })
    with pytest.raises(ValueError, match="missing source voices"):
        state.prepare_version_restore({
            "job_id": job, "prepared_references": [{"id": missing["id"], "text": "Wrong voice.", "speaker": 0}],
            "reference_transcripts_checked": True,
        })


def test_revision_freezes_text_voice_and_mode_even_when_client_changes_them(state):
    first_voice = state.add_voice("First", "Frozen.", 0, "first.wav", b"RIFF" * 32)
    other_voice = state.add_voice("Other", "Unrelated.", 0, "other.wav", b"WAVE" * 32)
    first = state.create_job({"text": "Original.", "voice_ids": [first_voice["id"]], "studio": context()})
    finish(state, first, "Original.")
    revised = state.create_job({
        "source_version": {"job_id": first}, "operation": "revision", "mode": "tts",
        "text": "Not the source.", "voice_bindings": [{"voice_id": other_voice["id"]}],
        "direction": "Softer.",
    })
    recipe = state.job(revised)["recipe"]
    assert recipe["mode"] == "revision" and recipe["text"] == "Original."
    assert recipe["reference_snapshot"] == state.job(first)["recipe"]["reference_snapshot"]
    assert recipe["direction"] == "Softer."


def test_context_update_uses_adopted_anchors_but_retry_keeps_original_history(state):
    engine = Native()
    state._get_engine = lambda _: engine
    payload = dialogue("per_line")
    payload["lines"] += [{"character": "Bo", "text": "Fourth."}, {"character": "Ada", "text": "Fifth."}]
    payload["line_ids"] += ["fourth", "fifth"]
    base = state.create_job(payload)
    run_latest(state)
    old = state.version({"job_id": base, "part_index": 4})["recipe"]
    assert len(old["previous_sources"]) == 2
    replacements = {}
    for index, line in ((0, "first"), (2, "third")):
        candidate = state.create_job({
            "source_version": {"job_id": base, "part_index": index}, "operation": "retry", "seed": 456,
        })
        run_latest(state)
        state.adopt_version({"work_id": "work", "segment_id": "part-1", "line_id": line,
                             "version": {"job_id": candidate}})
        replacements[line] = state.job(candidate)["take_file"]
    statuses = {row["line_id"]: row for row in state.scene_context_status({"job_id": base})["lines"]}
    assert statuses["fifth"]["changed"] and statuses["fifth"]["available"]
    retried = state.create_job({"source_version": {"job_id": base, "part_index": 4}, "operation": "retry"})
    assert state.job(retried)["recipe"]["previous_sources"] == old["previous_sources"]
    updated = state.create_job({
        "source_version": {"job_id": base, "part_index": 4}, "operation": "update_context", "seed": 789,
        "text": "Ignored.", "temperature": .1,
    })
    run_latest(state)
    recipe = state.job(updated)["recipe"]
    assert recipe["mode"] == "tts" and not recipe.get("previous_take")
    assert recipe["text"] == old["text"] and recipe["temperature"] == old["temperature"]
    assert recipe["seed"] == 789 and recipe["conditioning"] == "adopted_scene_history"
    assert [source["take_file"] for source in recipe["previous_sources"]] == [
        replacements["first"], replacements["third"]]
    for actual, path in zip(engine.calls[-1][1]["previous_takes"], replacements.values()):
        take, _, chunks = load_native_take(path)
        anchor = chunks[0] if chunks else take
        assert actual.text == anchor.text and torch.equal(actual.codes, anchor.codes)
    segment = state.studio.snapshot()["works"]["work"]["segments"][0]
    assert "fifth" not in segment["line_versions"] and segment["adopted"] == {"job_id": base}
    assert state.scene_context_status({"job_id": base})["lines"][4]["changed"]
    state.adopt_version({"work_id": "work", "segment_id": "part-1", "line_id": "fifth",
                         "version": {"job_id": updated}})
    assert not state.scene_context_status({"job_id": base})["lines"][4]["changed"]
    changed_again = state.create_job({"source_version": {"job_id": base, "part_index": 0}, "operation": "retry"})
    run_latest(state)
    state.adopt_version({"work_id": "work", "segment_id": "part-1", "line_id": "first",
                         "version": {"job_id": changed_again}})
    assert state.scene_context_status({"job_id": base})["lines"][4]["changed"]
    repeated_update = state.create_job({"source_version": {"job_id": updated}, "operation": "retry"})
    assert state.job(repeated_update)["recipe"]["previous_sources"] == recipe["previous_sources"]


def test_context_update_preserves_distinct_first_last_chunks_and_lookahead(state):
    state._get_engine = lambda _: Native()
    base = state.create_job(dialogue("per_line"))
    run_latest(state)
    path = state.job(base)["result"]["segments"][0]["path"]
    first = PreviousTake("<|speaker:0|> Echo.", torch.full((10, 4), 3), "Next words.")
    last = PreviousTake("<|speaker:0|> Echo.", torch.full((10, 4), 7), "Other words.")
    saved = save_native_take(path, PreviousTake("Full original.", torch.cat((first.codes, last.codes), dim=1)),
                             chunks=[first, last])
    updated = state.create_job({
        "source_version": {"job_id": base, "part_index": 2}, "operation": "update_context"})
    recipe = state.job(updated)["recipe"]
    assert [source["take_file"] for source in recipe["previous_sources"]] == [str(saved), str(saved)]
    assert [source["edge"] for source in recipe["previous_sources"]] == ["first", "last"]
    previous = state._previous(recipe, Native())
    assert [take.lookahead for take in previous] == ["Next words.", "Other words."]
    assert torch.equal(previous[0].codes, first.codes) and torch.equal(previous[1].codes, last.codes)
    assert all(take.text == "<|speaker:0|> Echo." for take in previous)


@pytest.mark.parametrize("operation", ["revision", "infill"])
def test_context_status_traverses_retried_derivatives_without_loading_codec_codes(state, monkeypatch, operation):
    state._get_engine = lambda _: Native()
    base = state.create_job(dialogue("per_line"))
    run_latest(state)
    parent = state.create_job({
        "source_version": {"job_id": base, "part_index": 2}, "operation": operation,
        "text": "<|speaker:0|> Third changed.", "start_frame": 0, "end_frame": 1,
    })
    run_latest(state)
    for _ in range(2):
        parent = state.create_job({"source_version": {"job_id": parent}, "operation": "retry"})
        run_latest(state)
    state.adopt_version({"work_id": "work", "segment_id": "part-1", "line_id": "third",
                         "version": {"job_id": parent}})
    assert not state.scene_context_status({"job_id": base})["lines"][2]["changed"]
    changed = state.create_job({"source_version": {"job_id": base, "part_index": 0}, "operation": "retry"})
    run_latest(state)
    state.adopt_version({"work_id": "work", "segment_id": "part-1", "line_id": "first",
                         "version": {"job_id": changed}})

    def no_codes(*_args):
        raise AssertionError("Context polling must only inspect metadata")
    monkeypatch.setattr(ui, "load_conditioning_take", no_codes)
    status = state.scene_context_status({"job_id": base})["lines"][2]
    assert status["changed"] and status["available"]


def test_context_update_after_partial_scene_uses_full_scene_order_not_single_line_repair(state):
    class FailingNative(Native):
        def generate(self, text, output, **kwargs):
            if "Third." in text:
                raise RuntimeError("Stopped before third line")
            return super().generate(text, output, **kwargs)
    state._get_engine = lambda _: FailingNative()
    base = state.create_job(dialogue("per_line"))
    run_latest(state)
    state._get_engine = lambda _: Native()
    payload = dialogue("per_line")
    payload["lines"], payload["line_ids"] = payload["lines"][2:], payload["line_ids"][2:]
    payload["scene_order"] = ["first", "second", "third"]
    payload["studio"]["line_id"] = "third"
    repaired = state.create_job(payload)
    run_latest(state)
    updated = state.create_job({"source_version": {"job_id": repaired}, "operation": "update_context"})
    history = state.job(updated)["recipe"]["previous_sources"]
    assert history[0]["take_file"] == state.job(base)["result"]["segments"][0]["take_file"]
    assert state.job(updated)["studio"]["line_id"] == "third"


def test_context_update_rejects_first_line_joint_dialogue_and_changed_adopted_strategy(state, monkeypatch):
    monkeypatch.setattr(state, "capabilities", lambda: {"joint_dialogue": True})
    state._get_engine = lambda _: Native()
    base = state.create_job(dialogue("per_line"))
    run_latest(state)
    with pytest.raises(ValueError, match="no completed preceding"):
        state.create_job({"source_version": {"job_id": base, "part_index": 0}, "operation": "update_context"})
    joint = state.create_job(dialogue())
    run_latest(state)
    with pytest.raises(ValueError, match="no recorded dialogue"):
        state.create_job({"source_version": {"job_id": joint}, "operation": "update_context"})
    state.adopt_version({"work_id": "work", "segment_id": "part-1", "version": {"job_id": joint}})
    with pytest.raises(ValueError, match="no per-line context"):
        state.create_job({"source_version": {"job_id": base, "part_index": 2}, "operation": "update_context"})
    assert not state.scene_context_status({"job_id": base})["lines"][2]["available"]


def test_running_scene_persists_completed_rows_before_generating_later_lines(state):
    inspected = []

    class InspectingNative(Native):
        def generate(self, text, output, **kwargs):
            if "Third." in text:
                current = state.job(identifier)
                disk = json.loads(state._job_file(identifier).read_text(encoding="utf-8"))
                inspected.append(current["status"] == "running"
                                 and len(disk["result"]["segments"]) == 2)
                assert state.version({"job_id": identifier, "part_index": 1})["status"] == "complete"
                assert current["result"]["segments"][1]["line_id"] == "second"
            return super().generate(text, output, **kwargs)
    state._get_engine = lambda _: InspectingNative()
    identifier = state.create_job(dialogue("per_line"))
    run_latest(state)
    assert inspected == [True] and state.job(identifier)["status"] == "complete"


def test_sparse_event_recovery_keeps_recorded_part_indexes_without_retargeting(state):
    state._get_engine = lambda _: Native()
    identifier = state.create_job(dialogue("per_line"))
    run_latest(state)
    rows = state.job(identifier)["result"]["segments"]
    with state._jobs_lock:
        state._jobs[identifier]["result"]["segments"] = []
        state._jobs[identifier]["events"] = [{"stage": "segment", **rows[2]}]
    version = state.version({"job_id": identifier, "part_index": 2})
    assert version["line_id"] == "third" and version["take_file"] == rows[2]["take_file"]
    assert state._source_metadata({"job_id": identifier, "part_index": 2})[1] == version["take_file"]
    with pytest.raises(ValueError, match="unavailable"):
        state.version({"job_id": identifier, "part_index": 0})


def test_short_reference_from_long_result_reads_and_stores_only_selected_samples(state, monkeypatch):
    identifier = state.create_job({"text": "The complete long transcript.", "studio": context()})
    finish(state, identifier, "The complete long transcript.")
    output = Path(state.job(identifier)["output"])
    rate = 8000
    samples = np.stack((np.sin(np.arange(180 * rate) * .03),
                        np.cos(np.arange(180 * rate) * .04)), axis=1).astype("float32") * .5
    sf.write(output, samples, rate, subtype="FLOAT")
    original_size, original_stamp = output.stat().st_size, output.stat().st_mtime_ns
    read_bytes = Path.read_bytes

    def no_full_source_read(path):
        assert path != output, "Extracting a clip must not read the whole long result"
        return read_bytes(path)
    monkeypatch.setattr(Path, "read_bytes", no_full_source_read)
    result = state.reference_from_version({
        "source_version": {"job_id": identifier}, "start": 135.125, "end": 137.625,
    })
    actual, actual_rate = sf.read(result["audio_path"], dtype="float32", always_2d=True)
    assert actual_rate == rate and np.array_equal(actual, samples[135125 * 8:137625 * 8])
    assert result["suggested_text"] == "" and result["duration_seconds"] == 2.5
    assert len(list(state.uploads_root.iterdir())) == 1
    assert output.stat().st_size == original_size and output.stat().st_mtime_ns == original_stamp
    with pytest.raises(ValueError, match="Select up to 120 seconds"):
        state.reference_from_version({"source_version": {"job_id": identifier}})
    assert len(list(state.uploads_root.iterdir())) == 1 and state._engine is None


@pytest.mark.parametrize("bounds", [
    {"start": -1, "end": 1}, {"start": True, "end": 1}, {"start": 1}, {"end": 1},
    {"start": 1, "end": None}, {"start": float("nan"), "end": 1},
    {"start": 0, "end": 121},
])
def test_invalid_source_reference_range_never_creates_an_upload(state, bounds):
    identifier = state.create_job({"text": "Original.", "studio": context()})
    finish(state, identifier)
    with pytest.raises(ValueError):
        state.reference_from_version({"source_version": {"job_id": identifier}, **bounds})
    assert not list(state.uploads_root.iterdir())


@pytest.mark.parametrize("mode", ["revision", "edit", "infill", "scene", "stream", "audio_stream"])
def test_unsupported_native_workflow_is_rejected_before_queue_without_changing_model(state, monkeypatch, mode):
    monkeypatch.setattr(ui, "load_native_runtime", lambda _root: {"protocol": 2})
    if mode == "scene":
        payload = dialogue("per_line")
    elif mode == "stream":
        payload = {"mode": mode, "chunks": ["First."]}
    elif mode == "infill":
        payload = {"mode": mode, "text": "New words.", "source_take": str(state.data_root / "source.take.json"),
                   "start_frame": 0, "end_frame": 1}
    else:
        payload = {"mode": mode, "text": "New words."}
        if mode in ("revision", "edit"):
            payload["previous_take"] = str(state.data_root / "source.take.json")
    selected = state.checkpoint
    with pytest.raises(ValueError, match="protocol 3"):
        state.create_job(payload)
    assert not state.jobs() and state._engine is None and state.checkpoint == selected


@pytest.mark.parametrize("changes", [{"single_chunk": True}, {"duration_frames": 100}])
def test_legacy_native_runtime_does_not_silently_change_whole_text_request(state, monkeypatch, changes):
    monkeypatch.setattr(ui, "load_native_runtime", lambda _root: {"protocol": 3})
    with pytest.raises(ValueError, match="whole-text"):
        state.create_job({"text": "New words.", **changes})
    assert not state.jobs()


def test_saved_version_with_incompatible_codec_is_not_derived(state):
    source = state.create_job({"text": "Original."})
    finish(state, source)
    state._jobs[source]["recipe"]["codec"]["family"] = "other-codec"
    with pytest.raises(ValueError, match="incompatible audio codec"):
        state.create_job({"source_version": {"job_id": source}, "operation": "retry"})
    assert len(state.jobs()) == 1


def test_model_selection_waits_for_recipe_queueing_then_refuses_active_work(state, monkeypatch):
    selected = state.checkpoint
    other = selected.with_name("other.gguf")
    other.write_bytes(b"GGUF fixture")
    monkeypatch.setattr(ui, "inspect_model", lambda _path: {"ready": True, "storage_format": "synthetic"})
    entered, proceed = Event(), Event()
    original = state._references

    def prepare(payload):
        entered.set()
        assert proceed.wait(5)
        return original(payload)
    monkeypatch.setattr(state, "_references", prepare)
    with ThreadPoolExecutor(max_workers=2) as workers:
        submitting = workers.submit(state.create_job, {"text": "Frozen model."})
        assert entered.wait(5)
        switching = workers.submit(state.select_checkpoint, other)
        time.sleep(.05)
        assert not switching.done()
        proceed.set()
        identifier = submitting.result(timeout=5)
        with pytest.raises(RuntimeError, match="Finish or cancel"):
            switching.result(timeout=5)
    assert state.checkpoint == selected and state.job(identifier)["recipe"]["model"] == str(selected)


@pytest.mark.parametrize("mode", ["stream", "audio_stream"])
def test_delivered_audio_survives_event_log_rotation_and_cancelled_result(state, mode):
    payload = {"mode": mode, "text": "First.", "chunks": ["First."]}
    identifier = state.create_job(payload)
    path = state.jobs_root / identifier / "first.wav"
    sf.write(path, np.zeros(8000), 8000)
    state._event(identifier, {"stage": "segment" if mode == "stream" else "audio",
                             "index": 0, "path": str(path), "samples": 8000, "sample_rate": 8000})
    for index in range(300):
        state._event(identifier, {"stage": "generate", "frames": index + 1})
    assert not any(event["stage"] in ("segment", "audio") for event in state.job(identifier)["events"])
    persisted = json.loads(state._job_file(identifier).read_text(encoding="utf-8"))
    assert persisted["result"]["segments"][0]["path"] == str(path)
    state._record_result(identifier, {"status": "cancelled", "output": None, "take_file": None})
    assert state.job(identifier)["result"]["segments"][0]["path"] == str(path)


def test_empty_interrupted_text_stream_does_not_reopen_an_unfinishable_retry(state):
    identifier = state.create_job({"mode": "stream", "chunks": [], "live_stream": True})
    state._jobs[identifier]["status"] = "interrupted"
    with pytest.raises(ValueError, match="no received text"):
        state.create_job({"source_version": {"job_id": identifier}, "operation": "retry"})
    assert len(state.jobs()) == 1


def test_missing_target_or_unresolvable_history_is_not_reported_as_current_context(state):
    state._get_engine = lambda _: Native()
    identifier = state.create_job(dialogue("per_line"))
    run_latest(state)
    row = state._jobs[identifier]["result"]["segments"][2]
    saved = row["history_sources"][0]["take_file"]
    row["history_sources"][0]["take_file"] = str(state.jobs_root / "unknown.take.json")
    status = state.scene_context_status({"job_id": identifier})["lines"][2]
    assert not status["available"] and not status["changed"]
    row["history_sources"][0]["take_file"] = saved
    Path(row["take_file"]).unlink()
    assert not state.scene_context_status({"job_id": identifier})["lines"][2]["available"]


def test_protocol_three_keeps_supported_infill_length_condition(state, monkeypatch):
    monkeypatch.setattr(ui, "load_native_runtime", lambda _root: {"protocol": 3})
    identifier = state.create_job({
        "mode": "infill", "text": "Corrected.", "source_take": str(state.data_root / "source.take.json"),
        "start_frame": 0, "end_frame": 1, "duration_frames": 100,
    })
    assert state.job(identifier)["recipe"]["duration_frames"] == 100


def test_continuing_long_audio_uses_exact_bounded_first_and_recent_chunks(state):
    source = state.create_job({"text": "The whole story.", "studio": context()})
    finish(state, source, "The whole story.")
    path = state.job(source)["output"]
    chunks = [
        PreviousTake("First words.", torch.full((10, 900), 1)),
        PreviousTake("The middle.", torch.full((10, 900), 2)),
        PreviousTake("Last words.", torch.full((10, 900), 3)),
    ]
    save_native_take(path, PreviousTake("The whole story.", torch.cat([chunk.codes for chunk in chunks], dim=1)),
                     chunks=chunks)
    continued = state.create_job({
        "source_version": {"job_id": source}, "operation": "continue", "text": "Only the new words.",
        "studio": {"work_id": "work", "segment_id": "part-2", "layout": "single"},
    })
    recipe = state.job(continued)["recipe"]
    assert recipe["conditioning"] == "bounded_continuation_history"
    assert recipe["text"] == "Only the new words."
    assert [part["frames"] for part in recipe["previous_sources"]] == [900, 900]
    history = state._previous(recipe, Native())
    assert [part.text for part in history] == ["First words.", "Last words."]
    assert torch.equal(history[0].codes, chunks[0].codes) and torch.equal(history[1].codes, chunks[-1].codes)
    retried = state.create_job({"source_version": {"job_id": source}, "operation": "retry"})
    assert not state.job(retried)["recipe"].get("previous_sources")


def test_continuing_old_long_take_without_reliable_boundaries_is_actionably_rejected(state):
    source = state.create_job({"text": "The whole story."})
    finish(state, source, "The whole story.")
    save_native_take(state.job(source)["output"],
                     PreviousTake("The whole story.", torch.zeros(10, 2200, dtype=torch.int64)))
    with pytest.raises(ValueError, match="no recorded chunk boundaries"):
        state.create_job({"source_version": {"job_id": source}, "operation": "continue", "text": "Next."})
    assert len(state.jobs()) == 1


def test_continuation_keeps_single_exact_chunk_and_lookahead(state):
    source = state.create_job({"text": "Original.", "studio": context()})
    finish(state, source, "Original.")
    take = PreviousTake("Original.", torch.ones(10, 4, dtype=torch.int64), "Next.")
    save_native_take(state.job(source)["output"], take, chunks=[take])
    continued = state.create_job({
        "source_version": {"job_id": source}, "operation": "continue", "text": "New words.",
        "studio": {"work_id": "work", "segment_id": "next", "layout": "single"},
    })
    history = state._previous(state.job(continued)["recipe"], Native())
    assert len(history) == 1 and history[0].lookahead == "Next." and history[0].text == "Original."
