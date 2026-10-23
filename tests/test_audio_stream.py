"""Partial delivery, EOS-only success, cancellation and low-RAM deferral."""

from contextlib import nullcontext
from threading import RLock
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf
import torch

from drama_low_vram.audio_stream import AudioStreamSession
from drama_low_vram.engine import CodecCoexistenceUnavailable
from drama_low_vram.stream_codec import CausalCodecStream


class Cursor:
    def __init__(self, prompt):
        self.frames, self.finished = prompt.clone(), False
        self.steps = 0
        self.prefix_report = {"enabled": False, "hit": False}

    def step(self):
        if self.steps == 5:
            self.finished = True
            return
        self.frames = torch.cat((self.frames, torch.tensor([[101, 1, 2, 3]])))
        self.steps += 1


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    engine = SimpleNamespace(
        options=SimpleNamespace(codec_max_frames=1024, prefill_chunk=None), _lock=RLock(),
        _lifecycle={"park_calls": 0}, graph=object(),
        backbone_graph=SimpleNamespace(enabled=True), codec=object(), prefix_cache=None,
        model=SimpleNamespace(tokenizer=object(), config=object()),
        load=Mock(), _prepare=Mock(return_value=(SimpleNamespace(prefill_chunk=32), None)),
        _remember_prefill_chunk=Mock(),
    )
    monkeypatch.setattr("drama_low_vram.audio_stream.seed_sampling", Mock())
    monkeypatch.setattr("drama_low_vram.audio_stream.build_tts_prompt",
                        lambda *_args, **_kwargs: torch.zeros(2, 4, dtype=torch.long))
    monkeypatch.setattr("drama_low_vram.audio_stream.DecodeCursor.prefill",
                        lambda _model, prompt, **_kwargs: Cursor(prompt))
    codec_state = SimpleNamespace(state_bytes=100, close=Mock())
    monkeypatch.setattr("drama_low_vram.audio_stream.CausalCodecStream",
                        lambda *_args, **_kwargs: codec_state)
    session = AudioStreamSession(
        engine, "Hello.", tmp_path / "stream.wav", max_new_tokens=8, chunk_frames=2)
    session._prepare_codec = Mock()

    def decode(_state, codes, _cursor, _capacity, *, continued):
        return torch.ones(codes.shape[1] * 2048) * 0.2, {"codec_seconds": 0.0}

    session._decode = Mock(side_effect=decode)
    return session, engine, codec_state


def test_frames_arrive_before_eos_and_final_pcm_matches_all_deliveries(runtime):
    session, _engine, codec = runtime
    rows = list(session.run())
    assert session.status == "complete"
    assert [row["end_frame"] for row in rows] == [2, 4, 5]
    assert [row["before_eos"] for row in rows] == [True, True, False]
    actual, _rate = sf.read(session.output, dtype="int16")
    expected = np.concatenate([sf.read(row["path"], dtype="int16")[0] for row in rows])
    assert np.array_equal(actual, expected)
    assert session.report["take_file"] is not None
    codec.close.assert_called_once()
    with pytest.raises(RuntimeError):
        list(session.run())


def test_cancel_before_initialization_does_not_load_or_publish(runtime):
    session, engine, _codec = runtime
    assert list(session.run(cancelled=lambda: True)) == []
    assert session.status == "cancelled" and session.report["output"] is None
    engine.load.assert_not_called()
    session._prepare_codec.assert_not_called()


def test_a_truncated_stream_retains_diagnostics_without_publishing_success(runtime):
    session, _engine, _codec = runtime
    session.maximum = 4
    assert len(list(session.run())) == 2
    assert session.status == "incomplete"
    assert session.report["take_file"] is None and not session.output.exists()
    assert np.load(session.report["diagnostic_frames"], allow_pickle=False).shape == (4, 4)


def test_consumer_closure_cancels_after_partial_audio(runtime):
    session, _engine, codec = runtime
    iterator = session.run()
    assert next(iterator)["before_eos"]
    iterator.close()
    assert session.status == "cancelled"
    assert session.report["output"] is None and not session.output.exists()
    codec.close.assert_called_once()


def test_snapshot_budget_failure_defers_audio_but_preserves_the_sequence(runtime):
    session, _engine, _codec = runtime
    original = session._decode.side_effect

    def decode(*args, **kwargs):
        if kwargs["continued"]:
            return None
        return original(*args, **kwargs)

    session._decode.side_effect = decode
    rows = list(session.run())
    assert len(rows) == 1 and rows[0]["end_frame"] == 5
    assert not rows[0]["before_eos"] and session.status == "complete"
    assert session.report["deferred_audio_for_host_budget"]


def decode_runtime(runtime, monkeypatch):
    session, engine, _ = runtime
    codec = object.__new__(CausalCodecStream)
    codec._state = {}
    codec.latent_frames = codec.decoded_frames = codec._pending = 0
    codec.closed = False
    engine._codec_preserving_sequence = Mock(side_effect=nullcontext)
    engine._admit_prefix_snapshot = Mock(return_value=False)
    engine._run_codec = lambda _count, operation, **_kwargs: operation(engine.codec)
    cursor = SimpleNamespace(
        position=10, tower_rows=11, capture=Mock(return_value=object()), restore=Mock())

    def latents(codes):
        assert not codec._pending
        codec._state["latent"] = torch.ones(4)
        codec.latent_frames += codes.shape[1]
        codec._pending = codes.shape[1]
        return torch.ones(1, 4, codes.shape[1])

    def waveform(_latents):
        audio = torch.ones(codec._pending * 2048)
        codec.decoded_frames += codec._pending
        codec._pending = 0
        return audio

    codec.decode_latents, codec.decode_waveform = latents, waveform
    return session, engine, codec, cursor


def test_resident_stream_does_not_require_a_host_snapshot(runtime, monkeypatch):
    session, engine, codec, cursor = decode_runtime(runtime, monkeypatch)
    estimate = Mock(side_effect=AssertionError("Resident state needs no snapshot"))
    monkeypatch.setattr("drama_low_vram.audio_stream.SequenceSnapshot.estimate_bytes", estimate)
    audio, timing = AudioStreamSession._decode(
        session, codec, torch.zeros(3, 2, dtype=torch.long), cursor, 100, continued=True)
    assert audio.numel() == 4096
    assert timing["sequence_preservation"] == "resident" and timing["checkpoint_seconds"] == 0
    assert (codec.latent_frames, codec.decoded_frames, codec._pending) == (2, 2, 0)
    engine._admit_prefix_snapshot.assert_not_called()
    cursor.capture.assert_not_called()
    cursor.restore.assert_not_called()
    engine._prepare.assert_called_once_with(100)


def test_late_coexistence_failure_rolls_codec_back_before_host_deferral(runtime, monkeypatch):
    session, engine, codec, cursor = decode_runtime(runtime, monkeypatch)
    monkeypatch.setattr("drama_low_vram.audio_stream.SequenceSnapshot.estimate_bytes",
                        Mock(return_value=256))
    waveform = codec.decode_waveform
    codec.decode_waveform = Mock(side_effect=CodecCoexistenceUnavailable("Cannot fit waveform"))
    codes = torch.zeros(3, 2, dtype=torch.long)
    assert AudioStreamSession._decode(
        session, codec, codes, cursor, 100, continued=True) is None
    assert codec._state == {}
    assert (codec.latent_frames, codec.decoded_frames, codec._pending) == (0, 0, 0)
    cursor.capture.assert_not_called()
    engine._prepare.assert_not_called()
    codec.decode_waveform = waveform
    audio, timing = AudioStreamSession._decode(
        session, codec, codes, cursor, 100, continued=False)
    assert audio.numel() == 4096 and timing["sequence_preservation"] == "eos"
    assert (codec.latent_frames, codec.decoded_frames, codec._pending) == (2, 2, 0)


def test_refused_coexistence_can_still_snapshot_and_restore_for_parking(runtime, monkeypatch):
    session, engine, codec, cursor = decode_runtime(runtime, monkeypatch)
    engine._codec_preserving_sequence.side_effect = CodecCoexistenceUnavailable("Requires parking")
    engine._admit_prefix_snapshot.return_value = True
    monkeypatch.setattr("drama_low_vram.audio_stream.SequenceSnapshot.estimate_bytes",
                        Mock(return_value=256))
    audio, timing = AudioStreamSession._decode(
        session, codec, torch.zeros(3, 2, dtype=torch.long), cursor, 100, continued=True)
    assert audio.numel() == 4096 and timing["sequence_preservation"] == "snapshot"
    engine._admit_prefix_snapshot.assert_called_once_with(256)
    cursor.capture.assert_called_once()
    cursor.restore.assert_called_once_with(cursor.capture.return_value)
    engine._prepare.assert_called_once_with(100)
