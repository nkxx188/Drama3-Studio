"""Native take integrity and backend-neutral scene/stream workflow integration."""

from pathlib import Path
from threading import RLock
from types import SimpleNamespace
import json

import pytest
import torch

from drama_low_vram.codec import write_wav
from drama_low_vram.gguf_takes import load_conditioning_take, load_native_take, save_native_take
from drama_low_vram.gguf_workflows import GGUFWorkflows
from drama_low_vram.protocols import Character, InfillSpan, PreviousTake, SceneLine
from drama_low_vram.scenes import SceneSession
from drama_low_vram.text_stream import TextStreamSession


class Native:
    is_gguf = True

    def __init__(self):
        self._lock = RLock()
        self.options = SimpleNamespace(codec_max_frames=2048)
        self.model_path = Path("fixture.gguf")
        self.calls = []

    def generate(self, text, output, **kwargs):
        self.calls.append((text, kwargs))
        codes = torch.full((10, 4), len(self.calls), dtype=torch.int64)
        if kwargs["infill"]:
            span = kwargs["infill"]
            codes = torch.cat((span.source[:, :span.start], codes, span.source[:, span.end:]), dim=1)
        take = save_native_take(output, PreviousTake(text, codes,
                                kwargs["lookahead"] if kwargs["task_mode"] == "stream" else None))
        write_wav(torch.zeros(codes.shape[1] * 2048), output)
        return {"status": "complete", "output": str(output), "take_file": str(take),
                "audio": {"samples": codes.shape[1] * 2048, "sample_rate": 44100}}


def test_long_tts_keeps_real_first_and_recent_chunk_anchors(tmp_path):
    engine = Native()
    result = GGUFWorkflows(engine).run(
        "tts", "One. Two. Three.", tmp_path / "all.wav", max_chunk_chars=6)
    assert len(engine.calls) == 3 and result["status"] == "complete"
    history = engine.calls[-1][1]["previous_takes"]
    assert [take.text for take in history] == ["One. ", "<|speaker:0|>Two. "]
    assert torch.equal(history[0].codes, torch.ones(10, 4, dtype=torch.int64))
    take, metadata, chunks = load_native_take(result["take_file"])
    assert take.codes.shape == (10, 12) and len(chunks) == 3
    assert metadata["spoken_text"] == "One. Two. Three."


def test_native_silence_retry_keeps_good_chunks_and_never_conditions_on_bad_audio(tmp_path):
    class SilentNative(Native):
        def generate(self, text, output, **kwargs):
            self.calls.append((text, kwargs))
            bad = len(self.calls) == 2
            marker, count = (7, 120) if bad else (len(self.calls), 4)
            codes = torch.full((10, count), marker, dtype=torch.int64)
            take = save_native_take(output, PreviousTake(text, codes))
            write_wav(torch.full((count * 2048,), 0.0 if bad else .1), output)
            return {"status": "incomplete" if bad else "complete", "output": str(output),
                    "take_file": str(take), "audio": {"samples": count * 2048, "sample_rate": 44100}}

    engine, events = SilentNative(), []
    result = GGUFWorkflows(engine).run(
        "tts", "One. Two. Three.", tmp_path / "retry.wav", max_chunk_chars=6,
        seed=123, silence_retry=True, progress=events.append)
    assert result["status"] == "complete"
    assert result["silence_retry"] == {"retries": 1, "exhausted": False}
    assert len(engine.calls) == 4 and [text for text, _ in engine.calls][1:3] == ["<|speaker:0|>Two. "] * 2
    assert engine.calls[1][1]["seed"] != engine.calls[2][1]["seed"]
    assert torch.equal(engine.calls[3][1]["previous_takes"][-1].codes,
                       torch.full((10, 4), 3, dtype=torch.int64))
    take, _, chunks = load_native_take(result["take_file"])
    assert len(chunks) == 3 and take.codes.shape == (10, 12)
    assert not bool((take.codes == 7).any())
    assert [event["retry"] for event in events if event["stage"] == "silence_retry"] == [1]


def test_native_silent_eos_cannot_publish_success_after_exhausted_retries(tmp_path):
    class SilentNative(Native):
        def generate(self, text, output, **kwargs):
            self.calls.append((text, kwargs))
            codes = torch.ones(10, 120, dtype=torch.int64)
            take = save_native_take(output, PreviousTake(text, codes))
            write_wav(torch.zeros(120 * 2048), output)
            return {"status": "complete", "output": str(output), "take_file": str(take),
                    "audio": {"samples": 120 * 2048, "sample_rate": 44100}}
    engine = SilentNative()
    result = GGUFWorkflows(engine).run("tts", "Hello.", tmp_path / "silent.wav", silence_retry=True)
    assert len(engine.calls) == 3 and result["status"] == "incomplete"
    assert result["silence_retry"] == {"retries": 2, "exhausted": True}
    assert result["take_file"] is None and Path(result["output"]).is_file()


def test_native_infill_checks_only_the_generated_middle(tmp_path):
    class InfillNative(Native):
        def generate(self, text, output, **kwargs):
            result = super().generate(text, output, **kwargs)
            span = kwargs["infill"]
            samples = result["audio"]["samples"]
            audio = torch.zeros(samples)
            audio[span.start * 2048:span.start * 2048 + 4 * 2048] = .1
            write_wav(audio, output)
            return result
    engine = InfillNative()
    result = GGUFWorkflows(engine).run(
        "infill", "Revised.", tmp_path / "infill.wav",
        infill=InfillSpan(torch.zeros(10, 280, dtype=torch.int64), 140, 141),
        silence_retry=True)
    assert result["status"] == "complete" and len(engine.calls) == 1
    assert result["silence_retry"] == {"retries": 0, "exhausted": False}


def test_lookahead_session_and_scene_share_native_takes_without_python_model(tmp_path):
    engine = Native()
    stream = TextStreamSession(GGUFWorkflows(engine), tmp_path / "stream")
    stream.push("One.")
    stream.push("Two.")
    stream.finish()
    assert len(list(stream.run())) == 2
    history = engine.calls[-1][1]["previous_takes"]
    assert history[0].lookahead == "Two."
    assert engine.calls[-1][1]["task_mode"] == "stream"
    assert stream.export(tmp_path / "stream.wav").is_file()
    scene = SceneSession(
        engine, tmp_path / "scene.wav", scene="A room.",
        characters=[Character("Ada", "Calm."), Character("Bo", "Warm.", 1)],
        lines=[SceneLine("Ada", "Hello."), SceneLine("Bo", "Hi."), SceneLine("Ada", "Again.")])
    rows = list(scene.run())
    assert scene.status == "complete" and [row["speaker"] for row in rows] == [0, 1, 0]
    assert rows[-1]["history_chunks"] == 1 and rows[1]["history_chunks"] == 0


def test_native_infill_preserves_codes_outside_window_and_keeps_full_transcript(tmp_path):
    source = torch.arange(100).reshape(10, 10)
    result = GGUFWorkflows(Native()).run(
        "infill", "The full revised transcript.", tmp_path / "filled.wav",
        infill=InfillSpan(source, 3, 5, duration_frames=4))
    assert result["infill"]["prefix_codes_equal"] and result["infill"]["suffix_codes_equal"]
    take, metadata, _ = load_native_take(result["take_file"])
    assert take.codes.shape[1] == 12
    assert metadata["spoken_text"] == "The full revised transcript."


@pytest.mark.parametrize("bad", [True, 1.5, -1, 4096])
def test_saved_take_rejects_invalid_codes_without_loading_gpu(tmp_path, bad):
    data = {"text": "Hello.", "vq_codes": [[0] for _ in range(10)]}
    data["vq_codes"][0][0] = bad
    path = tmp_path / "bad.take.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="integer rows"):
        load_native_take(path)


def test_take_loader_detects_corrupt_chunk_history_and_reads_python_codes(tmp_path):
    from drama_low_vram.workflows import save_take
    codes = torch.ones(10, 4, dtype=torch.int64)
    path = save_native_take(
        tmp_path / "wrong.wav", PreviousTake("Hello.", codes),
        chunks=[PreviousTake("Hello.", codes + 1)])
    with pytest.raises(ValueError, match="do not match"):
        load_native_take(path)
    frames = torch.cat((codes[:1] + 100, codes)).T
    path = save_take(tmp_path / "python.wav", "Hello.", codes, frames)
    take, _, _ = load_conditioning_take(path)
    assert torch.equal(take.codes, codes)


@pytest.mark.parametrize("legacy_report", [False, True])
def test_python_take_recovers_exact_chunk_transcripts_and_codes_for_native_history(tmp_path, legacy_report):
    from drama_low_vram.workflows import save_take
    codes = torch.arange(80, dtype=torch.int64).reshape(10, 8)
    frames = torch.cat((codes[:1] + 100, codes)).T
    records = [{"text": "First chunk.", "frames": 3, "lookahead": "Following."},
               {"text": "Last chunk.", "frames": 5}]
    output = tmp_path / "python.wav"
    saved = save_take(output, "Whole transcript.", codes, frames,
                      metadata={"mode": "tts", **({} if legacy_report else {"chunks": records})})
    if legacy_report:
        output.with_suffix(".json").write_text(json.dumps({"mode": "tts", "chunks": records}), encoding="utf-8")
    take, _, chunks = load_conditioning_take(saved)
    assert torch.equal(take.codes, codes) and len(chunks) == 2
    assert [chunk.text for chunk in chunks] == ["First chunk.", "Last chunk."]
    assert chunks[0].lookahead == "Following."
    assert torch.equal(chunks[0].codes, codes[:, :3]) and torch.equal(chunks[1].codes, codes[:, 3:])


@pytest.mark.parametrize("frames", [True, -1, 2.5, 0, 7, 9])
def test_python_take_rejects_inconsistent_chunk_boundaries(tmp_path, frames):
    from drama_low_vram.workflows import save_take
    codes = torch.zeros(10, 8, dtype=torch.int64)
    saved = save_take(tmp_path / "bad.wav", "Whole.", codes, torch.zeros(8, 11, dtype=torch.int64),
                      metadata={"mode": "tts", "chunks": [{"text": "Part.", "frames": frames}]})
    with pytest.raises(ValueError, match="chunk codes"):
        load_conditioning_take(saved)
