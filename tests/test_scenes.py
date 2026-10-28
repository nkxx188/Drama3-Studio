"""Real-take voice anchors, line boundaries, cancellation and PCM exports."""

from threading import Event, RLock
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from drama_low_vram.codec import write_wav
from drama_low_vram.protocols import Character, SceneLine, VoiceReference, directed_text
from drama_low_vram.scenes import SceneSession
from drama_low_vram.workflows import save_take


@pytest.fixture
def scene_runtime(tmp_path, monkeypatch):
    config = SimpleNamespace(
        audio_decoder=SimpleNamespace(num_codebooks=3, vocab_size=8),
        semantic_start_token_id=100, semantic_end_token_id=107)
    engine = SimpleNamespace(
        _lock=RLock(), options=SimpleNamespace(codec_max_frames=1024),
        model=SimpleNamespace(config=config))
    calls = []

    class Workflows:
        def __init__(self, _engine):
            pass

        def run(self, _mode, text, output, **kwargs):
            calls.append((text, kwargs))
            if "fail" in text:
                return {"status": "incomplete", "chunks": [
                    {"frames": 32, "finish_reason": "length", "decode_seconds": 1.0}]}
            multiple = "Longline" in text
            count = 5 if multiple else 3
            codes = torch.arange(count * 3).reshape(3, count) % 8
            frames = torch.cat((codes[:1] + 100, codes)).T
            actual = directed_text(text, kwargs["direction"])
            write_wav(torch.ones(count * 2048) * 0.1, output)
            take = save_take(output, actual, codes, frames)
            chunks = [
                {"text": directed_text("<|speaker:0|> First chunk.", kwargs["direction"]), "frames": 3},
                {"text": directed_text("<|speaker:0|> Last chunk.", kwargs["direction"]), "frames": 2},
            ] if multiple else [{"text": actual, "frames": count}]
            return {
                "status": "complete", "output": str(output), "take_file": str(take),
                "chunks": chunks, "audio": {"samples": count * 2048}}

    monkeypatch.setattr("drama_low_vram.scenes.DramaWorkflows", Workflows)
    return SimpleNamespace(engine=engine, calls=calls, output=tmp_path / "scene.wav")


def session(h, texts=("Hello.", "Hi.", "Again."), references=()):
    roles = ("Ada", "Bo", "Ada")
    return SceneSession(
        h.engine, h.output, scene="A quiet room.",
        characters=[Character("Ada", "steady voice"), Character("Bo", "warm voice", 1)],
        lines=[SceneLine(role, text) for role, text in zip(roles, texts)],
        references=references, max_new_tokens=32)


def test_scheduler_preserves_character_references_real_history_and_exact_pcm(scene_runtime):
    h = scene_runtime
    scene = session(h, references=[
        VoiceReference("a.wav", "Reference A.", 0),
        VoiceReference("b.wav", "Reference B.", 1)])
    rows = list(scene.run())
    assert scene.status == "complete"
    assert [row["speaker"] for row in rows] == [0, 1, 0]
    assert [row["history_chunks"] for row in rows] == [0, 0, 1]
    assert [row["start_sample"] for row in rows] == [0, 6144, 12288]
    assert [kwargs["references"][0].speaker for _text, kwargs in h.calls] == [0, 1, 0]
    assert h.calls[2][1]["previous_takes"][0].codes.shape[1] == 3
    assert not scene.report["native_audio_separators_used"]
    actual, rate = sf.read(h.output, dtype="int16")
    expected = np.concatenate([sf.read(row["path"], dtype="int16")[0] for row in rows])
    assert rate == 44100 and np.array_equal(actual, expected)


def test_long_line_anchors_keep_chunk_transcripts_with_their_own_exact_codes(scene_runtime):
    h = scene_runtime
    scene = session(h, texts=("Longline.", "Hi.", "Again."))
    list(scene.run())
    history = h.calls[2][1]["previous_takes"]
    assert len(history) == 2
    assert history[0].text.endswith("First chunk.") and history[0].codes.shape[1] == 3
    assert history[1].text.endswith("Last chunk.") and history[1].codes.shape[1] == 2


def test_cancellation_keeps_completed_lines_without_publishing_a_complete_scene(scene_runtime):
    h = scene_runtime
    scene, cancel = session(h), Event()
    iterator = scene.run(cancelled=cancel.is_set)
    assert next(iterator)["index"] == 0
    cancel.set()
    assert list(iterator) == []
    assert scene.status == "cancelled" and scene.report["output"] is None
    assert not h.output.exists() and len(h.calls) == 1


def test_incomplete_line_is_never_used_to_publish_a_successful_scene(scene_runtime):
    h = scene_runtime
    scene = session(h, texts=("Hello.", "fail", "Again."))
    assert len(list(scene.run())) == 1
    assert scene.status == "incomplete"
    assert scene.report["output"] is None and not h.output.exists()


def test_incomplete_joint_retains_frame_limit_and_timing_diagnostics(scene_runtime):
    h = scene_runtime
    scene = SceneSession(
        h.engine, h.output, scene="A quiet room.",
        characters=[Character("Ada", "A steady voice.")],
        lines=[SceneLine("Ada", "fail")], strategy="joint", max_new_tokens=32)
    assert list(scene.run()) == []
    assert scene.status == "incomplete" and scene.report["output"] is None
    assert scene.report["chunks"] == [
        {"frames": 32, "finish_reason": "length", "decode_seconds": 1.0}]
    assert scene.report["total_seconds"] >= 0
    assert not h.output.exists()


def test_wrong_character_reference_or_speaker_tag_fails_before_any_generation(scene_runtime):
    h = scene_runtime
    with pytest.raises(ValueError, match="Reference voice"):
        session(h, references=[VoiceReference("a.wav", "A.", 2)])
    with pytest.raises(ValueError, match="conflicts"):
        session(h, texts=("<|speaker:1|> Hello.", "Hi.", "Again."))
    assert not h.calls
