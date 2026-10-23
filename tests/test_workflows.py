"""Task lifetime, exact code assembly, progress and data-only take persistence."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from threading import RLock
from unittest.mock import Mock
import zipfile

import numpy as np
import pytest
import torch

from drama_low_vram.generation import GenerationResult
from drama_low_vram.protocols import Character, InfillSpan, PreviousTake, SceneLine, VoiceReference
from drama_low_vram.workflows import DramaWorkflows, audio_segments, load_take, save_take
from test_protocols import Tokenizer, codes


@pytest.fixture
def runtime(monkeypatch):
    config = SimpleNamespace(
        audio_decoder=SimpleNamespace(num_codebooks=3, vocab_size=8),
        semantic_start_token_id=100, semantic_end_token_id=107, eos_token_id=3,
    )
    tokenizer = Tokenizer()
    engine = SimpleNamespace(
        options=SimpleNamespace(codec_max_frames=1024, prefill_chunk=None), _lock=RLock(),
        _closed=False, _transition_failure=None, _lifecycle={"park_calls": 0},
        model=SimpleNamespace(config=config, tokenizer=tokenizer,
                              backbone=SimpleNamespace()),
        graph=None, backbone_graph=None, prefix_cache=None, _task_candidates=None,
        _last_prepare_timing={"reused": True},
        _ready_plan=None, _remember_prefill_chunk=Mock(),
    )
    engine.load = Mock(return_value=engine)
    engine.prepare_reference = Mock(return_value=(codes(2), False))
    engine._prepare = Mock(return_value=(SimpleNamespace(
        prefill_chunk=32, as_dict=lambda: {"cache_experts": 0}), None))
    engine._decode_audio_segments = Mock(side_effect=lambda segments: torch.ones(
        sum(segment.shape[1] for segment in segments) * 2048))
    frames = torch.cat((codes(3)[:1] + 100, codes(3))).T
    generated = GenerationResult(frames, "eos", 0.1, 0.2, 0.1)
    generator = Mock(return_value=generated)
    monkeypatch.setattr("drama_low_vram.workflows.generate", generator)
    monkeypatch.setattr("drama_low_vram.workflows.seed_sampling", Mock())
    return SimpleNamespace(engine=engine, workflows=DramaWorkflows(engine),
                           config=config, frames=frames, generator=generator)


def test_multiple_references_and_previous_takes_are_real_prompt_rows(runtime, tmp_path):
    references = [VoiceReference("one.wav", "First.", 0), VoiceReference("two.wav", "Second.", 1)]
    previous = PreviousTake("Previous.", codes(4))
    report = runtime.workflows.run(
        "revision", "Hello.", tmp_path / "voice.wav",
        references=references, previous_takes=[previous], direction="softer",
    )
    assert report["status"] == "complete"
    assert runtime.engine.prepare_reference.call_args_list[1].args[1] == "<|speaker:1|> Second."
    prompt = runtime.generator.call_args.args[1]
    semantic = prompt[(prompt[:, 0] >= 100) & (prompt[:, 0] <= 107)]
    assert len(semantic) == 2 + 2 + 4
    loaded, metadata, frames = load_take(report["take_file"], runtime.config)
    assert "Direction: softer" in loaded.text
    assert metadata["references"][1]["speaker"] == 1
    assert torch.equal(frames, runtime.frames)
    assert runtime.generator.call_args.kwargs["allow_prefill_fallback"]
    runtime.engine._remember_prefill_chunk.assert_called_once()


def test_infill_keeps_original_prefix_suffix_and_saves_whole_take(runtime, tmp_path):
    original = codes(8)
    report = runtime.workflows.run(
        "infill", "A corrected line.", tmp_path / "edit.wav",
        infill=InfillSpan(original, 2, 5, 3))
    assert report["status"] == "complete"
    expected = torch.cat((original[:, :2], runtime.frames[:, 1:].T, original[:, 5:]), dim=1)
    actual = runtime.engine._decode_audio_segments.call_args.args[0][0]
    assert torch.equal(actual, expected)
    take, metadata, frames = load_take(report["take_file"], runtime.config)
    assert torch.equal(take.codes, expected)
    assert torch.equal(frames[:, 1:].T, expected)
    assert report["chunks"][0]["infill"]["prefix_codes_equal"]
    assert report["chunks"][0]["infill"]["suffix_codes_equal"]


def test_stream_take_persists_the_lookahead_used_for_its_real_history(runtime, tmp_path):
    report = runtime.workflows.run(
        "stream", "One.", tmp_path / "stream.wav", lookahead="Two.",
        previous_takes=[PreviousTake("Before.", codes(2), "One.")])
    take, metadata, _ = load_take(report["take_file"], runtime.config)
    assert take.text == "One." and take.lookahead == "Two."
    assert metadata["mode"] == "stream"
    prompt = runtime.generator.call_args.args[1]
    assert (prompt[:, 0] == 93).sum() == 3


def test_empty_middle_eos_is_valid_for_infill_deletion_but_not_empty_speech(runtime, tmp_path):
    runtime.generator.return_value = GenerationResult(runtime.frames[:0], "eos", 0.1, 0.2, None)
    original = codes(8)
    result = runtime.workflows.run(
        "infill", "The shorter line.", tmp_path / "deleted.wav",
        infill=InfillSpan(original, 2, 5))
    assert result["status"] == "complete"
    decoded = runtime.engine._decode_audio_segments.call_args.args[0][0]
    assert torch.equal(decoded, torch.cat((original[:, :2], original[:, 5:]), dim=1))
    assert result["chunks"][0]["infill"]["replacement_frames"] == 0
    result = runtime.workflows.run("tts", "Hello.", tmp_path / "empty.wav")
    assert result["status"] == "incomplete" and result["output"] is None
    result = runtime.workflows.run(
        "infill", "A nonempty line.", tmp_path / "all-empty.wav",
        infill=InfillSpan(original, 0, 8))
    assert result["status"] == "incomplete" and result["output"] is None


@pytest.mark.parametrize("mode,kwargs", [
    ("tts", {"lookahead": "Later."}),
    ("tts", {"scene": "A room."}),
    ("stream", {"direction": "quietly"}),
    ("infill", {"infill": InfillSpan(codes(), 1, 2),
                "references": [VoiceReference("voice.wav", "Hello.")]}),
    ("scene", {"language": "en"}),
])
def test_task_options_are_never_silently_ignored(runtime, tmp_path, mode, kwargs):
    with pytest.raises(ValueError):
        runtime.workflows.run(mode, "" if mode == "scene" else "Hello.",
                              tmp_path / "invalid.wav", **kwargs)
    runtime.engine.load.assert_not_called()
    runtime.engine.prepare_reference.assert_not_called()


def test_stream_references_enter_the_actual_prompt_alongside_lookahead_and_history(runtime, tmp_path):
    cache = runtime.engine.prefix_cache = object()
    result = runtime.workflows.run(
        "stream", "Now.", tmp_path / "clone-stream.wav", lookahead="Next.",
        references=[VoiceReference("voice.wav", "Reference.", 2)],
        previous_takes=[PreviousTake("Before.", codes(4), "Now.")])
    prompt = runtime.generator.call_args.args[1]
    semantic = prompt[(prompt[:, 0] >= 100) & (prompt[:, 0] <= 107)]
    assert torch.equal(semantic[:, 1:], torch.cat((codes(2), codes(4)), dim=1).T)
    assert result["stream_reference_conditioning"] == "system_reference_blocks"
    assert result["reference_cache_hits"] == [False]
    assert (prompt[:, 0] == 93).sum() == 3
    assert runtime.generator.call_args.kwargs["prefix_cache"] is cache


def test_scene_controls_are_never_sent_to_codec(runtime, tmp_path):
    separator = torch.zeros(1, 4, dtype=torch.long)
    separator[0, 0] = 92
    joined = torch.cat((runtime.frames, separator, runtime.frames))
    runtime.generator.return_value = GenerationResult(joined, "eos", 0.1, 0.2, 0.1)
    report = runtime.workflows.run(
        "scene", "", tmp_path / "scene.wav", scene="A room.",
        characters=[Character("Ada", "steady voice"), Character("Bo", "warm voice", 1)],
        lines=[SceneLine("Ada", "Hello."), SceneLine("Bo", "Hi.")])
    assert report["status"] == "complete"
    assert report["chunks"][0]["separator_rows"] == [3]
    codes_for_codec = runtime.engine._decode_audio_segments.call_args.args[0][0]
    assert codes_for_codec.shape == (3, 6)
    assert runtime.engine._prepare.call_args.kwargs["output_controls"] == ("<|audio_separator|>",)


def test_scene_missing_separator_is_incomplete_and_keeps_diagnostic_rows(runtime, tmp_path):
    output = tmp_path / "scene.wav"
    report = runtime.workflows.run(
        "scene", "", output, scene="A room.",
        characters=[Character("Ada", "steady voice")],
        lines=[SceneLine("Ada", "Hello."), SceneLine("Ada", "Again.")])
    assert report["status"] == "segment_mismatch" and report["output"] is None
    assert not output.exists()
    assert np.array_equal(np.load(report["diagnostic_frames"], allow_pickle=False),
                          runtime.frames.numpy())
    runtime.engine._decode_audio_segments.assert_not_called()


def test_cancelled_or_truncated_task_does_not_publish_audio(runtime, tmp_path):
    output = tmp_path / "cancel.wav"
    report = runtime.workflows.run("tts", "Hello.", output, cancelled=lambda: True)
    assert report["status"] == "cancelled" and report["output"] is None
    runtime.generator.assert_not_called()
    runtime.generator.return_value = GenerationResult(runtime.frames, "length", 0.1, 0.2, 0.1)
    report = runtime.workflows.run("tts", "Hello.", output)
    assert report["status"] == "incomplete" and report["output"] is None
    assert not output.exists()
    runtime.engine._decode_audio_segments.assert_not_called()


def test_chunk_preview_is_delivered_before_the_next_generation(runtime, tmp_path):
    events = []
    def generate(*_args, **_kwargs):
        events.append("generate")
        return GenerationResult(runtime.frames, "eos", 0.1, 0.2, 0.1)
    runtime.generator.side_effect = generate
    report = runtime.workflows.run(
        "tts", "First sentence. Second sentence. Third sentence.",
        tmp_path / "long.wav", max_chunk_chars=18,
        segment_ready=lambda value: events.append(("preview", value["index"])))
    assert report["status"] == "complete"
    assert events == ["generate", ("preview", 0), "generate", ("preview", 1),
                      "generate", ("preview", 2)]
    assert report["audio"]["samples"] == 3 * 3 * 2048
    assert runtime.engine._decode_audio_segments.call_count == 3
    # Last chunk sees first and immediately previous chunks, not a fake transcript.
    last_prompt = runtime.generator.call_args.args[1]
    semantic = last_prompt[(last_prompt[:, 0] >= 100) & (last_prompt[:, 0] <= 107)]
    assert len(semantic) == 6


def test_completed_chunk_previews_persist_their_actual_transcripts_and_vq(runtime, tmp_path):
    delivered = []
    report = runtime.workflows.run(
        "tts", "First sentence. Second sentence.", tmp_path / "chunks.wav",
        max_chunk_chars=18, segment_ready=delivered.append)
    assert report["status"] == "complete" and len(delivered) == 2
    for row, chunk in zip(delivered, report["chunks"]):
        take, metadata, frames = load_take(row["take_file"], runtime.config)
        assert metadata["completed_chunk"] and take.text == chunk["text"]
        assert torch.equal(take.codes, runtime.frames[:, 1:].T)
        assert torch.equal(frames, runtime.frames)


def test_invalid_output_rows_fail_instead_of_silently_decoding_controls(runtime):
    rows = runtime.frames.clone()
    rows[0, 0] = 92
    with pytest.raises(ValueError, match="empty"):
        audio_segments(rows, runtime.config, 92)
    rows = runtime.frames.clone()
    rows[1, 0] = 99
    with pytest.raises(ValueError, match="non-audio"):
        audio_segments(rows, runtime.config)
    rows = runtime.frames.clone()
    rows[0, 1] = 7
    with pytest.raises(ValueError, match="disagree"):
        audio_segments(rows, runtime.config)


def test_take_rejects_unknown_fields_and_expanded_zip_payload(runtime, tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, codes=codes().numpy())
    with pytest.raises(ValueError):
        load_take(path, runtime.config)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("text.npy", "codes.npy", "frames.npy", "metadata.npy"):
            archive.writestr(name, b"\0" * (34 * 1024**2))
    with pytest.raises(ValueError, match="expanded"):
        load_take(path, runtime.config)
