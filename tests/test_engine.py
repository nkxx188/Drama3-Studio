"""Request lifecycle checks with deterministic CPU generation and codec fakes."""

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from drama_low_vram.engine import DramaEngine, EngineOptions
from drama_low_vram.generation import GenerationResult
from drama_low_vram.offload import TritonExpertBackend


class FakeStats:
    def as_dict(self):
        return {"forward_calls": 0, "h2d_bytes": 0, "cpu_expert_bytes": 0}


def test_unknown_expert_decode_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="expert_decode"):
        DramaEngine(EngineOptions(checkpoint=tmp_path, expert_decode="unknown"))


def test_unknown_host_gather_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="host_gather"):
        DramaEngine(EngineOptions(checkpoint=tmp_path, host_gather="unknown"))


def test_unknown_codec_placement_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="codec_placement"):
        DramaEngine(EngineOptions(checkpoint=tmp_path, codec_placement="unknown"))


@pytest.mark.parametrize("options", [
    {"cpu_experts": True}, {"cpu_experts": -1}, {"cpu_experts": 9},
    {"native_threads": False}, {"native_threads": 0}, {"native_threads": 65},
    {"prefix_cache_mib": True}, {"prefix_cache_mib": -1},
    {"context_bucket_size": True}, {"context_bucket_size": -1},
    {"context_bucket_size": 4097}, {"compute_policy": "unknown"},
])
def test_invalid_hybrid_decode_configuration_is_rejected(tmp_path, options):
    with pytest.raises(ValueError, match="cpu_experts|native_threads|prefix_cache_mib|context_bucket_size|compute_policy"):
        DramaEngine(EngineOptions(checkpoint=tmp_path, **options))


def test_prefix_admission_counts_only_future_mirrors_and_codec(tmp_path, monkeypatch):
    engine = DramaEngine(EngineOptions(checkpoint=tmp_path, prefix_cache_mib=1))
    engine.host_plan = SimpleNamespace(
        buffer_bytes=10, requirements=SimpleNamespace(parking_bytes=400, codec_build_bytes=200))
    hardware = SimpleNamespace(ram_available_bytes=500, commit_available_bytes=1000)
    monkeypatch.setattr("drama_low_vram.engine.inspect_hardware", lambda *_: hardware)
    assert not engine._admit_prefix_snapshot(50)
    engine._cpu_mirrors[1] = torch.empty(100, dtype=torch.float32)
    assert engine._admit_prefix_snapshot(50)
    hardware.commit_available_bytes = 250
    assert not engine._admit_prefix_snapshot(50)
    engine.codec = object()
    assert engine._admit_prefix_snapshot(50)
    hardware.ram_available_bytes = 59
    assert not engine._admit_prefix_snapshot(50)


def test_official_decode_mode_selects_reference_backend(tmp_path, monkeypatch):
    engine = DramaEngine(EngineOptions(
        checkpoint=tmp_path, expert_storage="ram", expert_decode="official"))
    monkeypatch.setattr(
        "drama_low_vram.engine.inspect_hardware",
        lambda *_: SimpleNamespace(ram_available_bytes=64 * 1024**3, compute_capability=(8, 9)))
    monkeypatch.setattr(
        "drama_low_vram.engine.autotune",
        lambda *_args, **_kwargs: SimpleNamespace(
            can_run=True, backend="triton", pin_memory=True,
            max_staged_experts=4, max_pair_rows=32, prefill_chunk=8))
    loader = Mock(return_value=(object(), object()))
    monkeypatch.setattr("drama_low_vram.engine.load_offloaded_drama3", loader)
    engine.load()
    backend = loader.call_args.kwargs["expert_backend"]
    assert isinstance(backend, TritonExpertBackend)
    assert not backend.decode_specialization


def frame_result(count=3, finish="eos"):
    frames = torch.zeros(count, 5, dtype=torch.long)
    if count:
        frames[:, 0] = 8
        frames[:, 1:] = torch.arange(count).unsqueeze(1)
    return GenerationResult(frames, finish, 0.1, 0.2, 0.15)


@pytest.fixture
def engine_harness(tmp_path, monkeypatch):
    engine = DramaEngine(EngineOptions(checkpoint=tmp_path, tower_graph=False))
    model = SimpleNamespace(
        config=SimpleNamespace(
            text=SimpleNamespace(max_position_embeddings=8192),
            audio_decoder=SimpleNamespace(num_codebooks=4),
        ),
        tokenizer=object(),
    )
    manager = SimpleNamespace(stats=FakeStats())
    # Keep fake requests independent of the real first-use codec loader.
    engine.codec = SimpleNamespace(sample_rate=44100)
    loads, prompts, decodes, writes, seeds, codec_limits = [], [], [], [], [], []

    def fake_load():
        loads.append(engine.model is None)
        engine.model, engine.manager = model, manager
        engine.storage = "ram"
        engine.load_seconds = 0.4
        return engine

    def fake_prompt(text, tokenizer, config, **kwargs):
        prompts.append({"text": text, **kwargs})
        return torch.zeros(4, 5, dtype=torch.long)

    @contextmanager
    def fake_codec(max_frames, *, operation="all"):
        codec_limits.append(max_frames)
        yield SimpleNamespace(sample_rate=44100)

    def fake_decode(codes, codec, device):
        decodes.append(codes.clone())
        return torch.ones(int(codes.shape[-1]) * 2048)

    def fake_write(audio, output, **kwargs):
        writes.append((audio.clone(), output, kwargs))
        Path(output).write_bytes(b"fake-wave")

    plan = SimpleNamespace(prefill_chunk=8, as_dict=lambda: {"cache_experts": 0})
    monkeypatch.setattr(engine, "load", fake_load)
    monkeypatch.setattr(engine, "_prepare", lambda capacity: (plan, None))
    monkeypatch.setattr(engine, "_codec_on_gpu", fake_codec)
    monkeypatch.setattr(engine, "_full_codec_decode_fits", lambda _frames: False)
    monkeypatch.setattr("drama_low_vram.engine.build_drama3_prompt", fake_prompt)
    monkeypatch.setattr("drama_low_vram.engine.generate", lambda *args, **kwargs: frame_result())
    monkeypatch.setattr("drama_low_vram.engine.decode_to_audio", fake_decode)
    monkeypatch.setattr("drama_low_vram.engine.write_wav", fake_write)
    monkeypatch.setattr("drama_low_vram.engine.seed_sampling", seeds.append)
    monkeypatch.setattr("torch.cuda.reset_peak_memory_stats", lambda *args: None)
    monkeypatch.setattr("torch.cuda.max_memory_allocated", lambda *args: 0)
    monkeypatch.setattr("torch.cuda.max_memory_reserved", lambda *args: 0)
    return SimpleNamespace(
        engine=engine, loads=loads, prompts=prompts, decodes=decodes, writes=writes,
        seeds=seeds, codec_limits=codec_limits, tmp_path=tmp_path, monkeypatch=monkeypatch,
    )


def test_complete_request_writes_audio_and_model_is_reused(engine_harness):
    h = engine_harness
    first = h.engine.synthesize("First sentence.", h.tmp_path / "first.wav")
    second = h.engine.synthesize("Second sentence.", h.tmp_path / "second.wav")
    assert first["status"] == second["status"] == "complete"
    assert first["engine"]["model_reused"] is False
    assert second["engine"]["model_reused"] is True
    assert first["cold_model_load_seconds"] == 0.4
    assert second["cold_model_load_seconds"] == 0
    assert first["audio"]["samples"] == 3 * 2048
    assert len(first["chunks"][0]["semantic_codes_sha256"]) == 64
    assert first["chunks"][0]["semantic_codes_sha256"] == second["chunks"][0]["semantic_codes_sha256"]
    assert first["request_index"] == 1
    assert second["request_index"] == 2
    assert h.loads == [True, False]
    assert len(h.writes) == 2


def test_zero_top_k_is_accepted_and_forwarded(engine_harness):
    h = engine_harness
    calls = []
    h.monkeypatch.setattr(
        "drama_low_vram.engine.generate",
        lambda *args, **kwargs: (calls.append(kwargs) or frame_result()),
    )
    report = h.engine.synthesize(
        "No top-k cutoff.", h.tmp_path / "top-k-zero.wav", top_k=0
    )
    assert report["status"] == "complete"
    assert calls[0]["top_k"] == 0


def test_chunk_history_and_bounded_codec_decode(engine_harness):
    h = engine_harness
    report = h.engine.synthesize(
        "First sentence. Second sentence. Third sentence.",
        h.tmp_path / "chunks.wav", max_chunk_chars=18,
    )
    assert report["status"] == "complete"
    assert len(report["chunks"]) == 3
    assert len(h.prompts) == 3
    assert h.prompts[0]["history"] == []
    assert len(h.prompts[1]["history"]) == 1
    assert h.prompts[1]["history"][0][0] == "First sentence. "
    assert all(codes.shape[-1] == 3 for codes in h.decodes)
    assert len(h.decodes) == 3
    assert h.codec_limits == [3]
    assert h.writes[0][0].numel() == 9 * 2048
    assert h.seeds == [123, 124, 125]


@pytest.mark.parametrize("allow_partial", [False, True])
def test_zero_frame_eos_never_publishes_an_empty_complete_clip(
    engine_harness, allow_partial
):
    h = engine_harness
    h.monkeypatch.setattr(
        "drama_low_vram.engine.generate", lambda *args, **kwargs: frame_result(0)
    )
    report = h.engine.synthesize(
        "Speech text.", h.tmp_path / "zero.wav", allow_partial=allow_partial
    )
    assert report["status"] != "complete"
    assert report["output"] is None
    assert report["audio"] is None
    assert not h.writes


def test_later_zero_frame_eos_does_not_reuse_previous_codes_or_continue(engine_harness):
    h = engine_harness
    fake_generate = Mock(side_effect=[frame_result(3), frame_result(0), frame_result(9)])
    h.monkeypatch.setattr("drama_low_vram.engine.generate", fake_generate)
    report = h.engine.synthesize(
        "First sentence. Second sentence. Third sentence.",
        h.tmp_path / "zero-later.wav", max_chunk_chars=18, allow_partial=True,
    )
    assert report["status"] != "complete"
    assert fake_generate.call_count == 2
    assert len(h.decodes) == 1
    assert h.writes[0][0].numel() == 3 * 2048


@pytest.mark.parametrize("reason", ["length", "cancelled"])
def test_incomplete_request_preserves_existing_audio(engine_harness, reason):
    h = engine_harness
    h.monkeypatch.setattr(
        "drama_low_vram.engine.generate", lambda *args, **kwargs: frame_result(3, reason)
    )
    output = h.tmp_path / "existing.wav"
    output.write_bytes(b"previous-good-clip")
    report = h.engine.synthesize("Speech text.", output)
    assert report["status"] != "complete"
    assert report["output"] is None
    assert output.read_bytes() == b"previous-good-clip"
    assert not h.writes
    assert not h.decodes


def test_allow_partial_publishes_only_generated_frames(engine_harness):
    h = engine_harness
    h.monkeypatch.setattr(
        "drama_low_vram.engine.generate", lambda *args, **kwargs: frame_result(2, "length")
    )
    report = h.engine.synthesize(
        "Speech text.", h.tmp_path / "partial.wav", allow_partial=True
    )
    assert report["status"] != "complete"
    assert report["audio"]["samples"] == 2 * 2048
    assert len(h.writes) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"text": ""},
        {"text": " \n"},
        {"seed": -1},
        {"seed": 2**64},
        {"seed": True},
        {"max_new_tokens": 0},
        {"max_chunk_chars": 0},
        {"reference_audio": Path("missing.wav")},
        {"reference_text": "Reference words"},
        {"temperature": 0},
        {"top_p": 2},
    ],
)
def test_invalid_request_fails_before_model_loading(engine_harness, changes):
    h = engine_harness
    arguments = {"text": "Speech text.", "output": h.tmp_path / "invalid.wav", **changes}
    with pytest.raises((ValueError, TypeError)):
        h.engine.synthesize(**arguments)
    assert not h.loads
    assert not h.writes


def test_nonfinite_codec_audio_is_rejected_before_publication(engine_harness):
    h = engine_harness
    h.monkeypatch.setattr(
        "drama_low_vram.engine.decode_to_audio",
        lambda *args, **kwargs: torch.tensor([float("nan")]),
    )
    with pytest.raises(FloatingPointError):
        h.engine.synthesize("Speech text.", h.tmp_path / "nan.wav")
    assert not h.writes


def test_closed_engine_rejects_requests(engine_harness):
    h = engine_harness
    h.engine._closed = True
    with pytest.raises(RuntimeError, match="closed"):
        h.engine.synthesize("Speech text.", h.tmp_path / "closed.wav")
    assert not h.loads


def test_reference_cache_reuses_codes_and_invalidates_changed_contents(
    tmp_path, monkeypatch
):
    engine = DramaEngine(EngineOptions(checkpoint=tmp_path, reference_cache_entries=1))
    audio = tmp_path / "reference.wav"
    audio.write_bytes(b"audio-content-a")
    encoded = []

    @contextmanager
    def codec(max_frames, *, operation="all"):
        yield object()

    def fake_encode(path, codec, device):
        result = torch.full((4, 3), len(encoded), dtype=torch.long)
        encoded.append(result)
        return result

    monkeypatch.setattr(engine, "_codec_on_gpu", codec)
    monkeypatch.setattr("drama_low_vram.engine.encode_audio", fake_encode)
    monkeypatch.setattr(
        "soundfile.info", lambda _: SimpleNamespace(frames=4410, samplerate=44100)
    )
    first, hit = engine.prepare_reference(audio, "Reference words")
    assert not hit
    second, hit = engine.prepare_reference(audio, "Reference words")
    assert hit
    assert torch.equal(first, second)
    audio.write_bytes(b"audio-content-b")
    third, hit = engine.prepare_reference(audio, "Reference words")
    assert not hit
    assert not torch.equal(first, third)
    assert len(encoded) == 2
    assert len(engine._references) == 1


def test_reference_cache_disabled_encodes_each_request(tmp_path, monkeypatch):
    engine = DramaEngine(EngineOptions(checkpoint=tmp_path, reference_cache_entries=0))
    audio = tmp_path / "reference.wav"
    audio.write_bytes(b"unchanged-audio")

    @contextmanager
    def codec(max_frames, *, operation="all"):
        yield object()

    encode = Mock(return_value=torch.zeros(4, 3, dtype=torch.long))
    monkeypatch.setattr(engine, "_codec_on_gpu", codec)
    monkeypatch.setattr("drama_low_vram.engine.encode_audio", encode)
    monkeypatch.setattr(
        "soundfile.info", lambda _: SimpleNamespace(frames=4410, samplerate=44100)
    )
    assert engine.prepare_reference(audio, "Reference words")[1] is False
    assert engine.prepare_reference(audio, "Reference words")[1] is False
    assert encode.call_count == 2
    assert not engine._references


def test_voice_reference_is_forwarded_unchanged_across_every_chunk(engine_harness):
    h = engine_harness
    codes = torch.arange(12).view(4, 3)
    prepare = Mock(return_value=(codes, True))
    h.monkeypatch.setattr(h.engine, "prepare_reference", prepare)
    report = h.engine.synthesize(
        "First sentence. Second sentence. Third sentence.",
        h.tmp_path / "cloned.wav", max_chunk_chars=18,
        reference_audio=h.tmp_path / "reference.wav",
        reference_text="<|speaker:0|>Reference words.",
    )
    assert report["status"] == "complete"
    assert report["reference_cache_hit"]
    assert prepare.call_count == 1
    for prompt in h.prompts:
        assert prompt["prompt_texts"] == ["<|speaker:0|>Reference words."]
        assert len(prompt["prompt_tokens"]) == 1
        assert torch.equal(prompt["prompt_tokens"][0], codes)


@pytest.mark.parametrize(
    "available_gib,expected_storage",
    [(64, "ram"), (44, "ram"), (40, "ram"), (36, "mmap"), (12, "mmap")],
)
def test_auto_storage_selects_ram_or_file_mapping_without_loading_twice(
    tmp_path, monkeypatch, available_gib, expected_storage
):
    cache_path = tmp_path / "cache"
    cache_path.mkdir()
    engine = DramaEngine(
        EngineOptions(checkpoint=tmp_path, expert_cache_dir=cache_path)
    )
    hardware = SimpleNamespace(ram_available_bytes=available_gib * 1024**3, compute_capability=(8, 9))
    runtime = SimpleNamespace(
        can_run=True, backend="triton", pin_memory=True,
        max_staged_experts=4, max_pair_rows=32, prefill_chunk=8,
    )
    loader = Mock(return_value=(object(), object()))
    monkeypatch.setattr("drama_low_vram.engine.inspect_hardware", lambda *_: hardware)
    policy = Mock(return_value=runtime)
    monkeypatch.setattr("drama_low_vram.engine.autotune", policy)
    monkeypatch.setattr("drama_low_vram.engine.load_offloaded_drama3", loader)
    assert engine.load() is engine
    assert engine.load() is engine
    assert loader.call_count == 1
    assert engine.storage == expected_storage
    assert policy.call_args.kwargs["expert_storage"] == expected_storage
    actual_cache_path = loader.call_args.kwargs["expert_cache_dir"]
    assert actual_cache_path == (cache_path if expected_storage == "mmap" else None)
    assert loader.call_args.kwargs["max_staged_experts"] == 4
    assert loader.call_args.kwargs["max_pair_rows"] == 32
    assert engine._batching() == (4, 32, 8)


def test_explicit_batching_overrides_automatic_policy(tmp_path):
    engine = DramaEngine(EngineOptions(
        checkpoint=tmp_path, staged_experts=8, pair_rows=64, prefill_chunk=32
    ))
    engine.runtime = SimpleNamespace(
        max_staged_experts=4, max_pair_rows=32, prefill_chunk=8,
    )
    assert engine._batching() == (8, 64, 32)


def test_completed_codec_scope_is_restored_when_decoding_fails(engine_harness):
    h = engine_harness

    def fail(_segments):
        assert h.engine._codec_completed_depth == 1
        raise ValueError("codec diagnostic failure")

    h.monkeypatch.setattr(h.engine, "_decode_audio_segments", fail)
    with pytest.raises(ValueError, match="codec diagnostic failure"):
        h.engine.synthesize("Completed text.", h.tmp_path / "failed-codec.wav")
    assert h.engine._codec_completed_depth == 0


def test_codec_parking_restores_original_cpu_storage_without_reallocating(tmp_path, monkeypatch):
    class TinyCodec(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
            self.register_buffer("position", torch.arange(3, dtype=torch.float32))
            self._drama_low_vram_max_frames = 16

        def to(self, *_args, **_kwargs):
            # Module.to mutates Parameter.data but replaces registered buffers.
            # A dtype move reproduces that ownership behavior without a GPU.
            return super().to(dtype=torch.float64)

    engine = DramaEngine(EngineOptions(checkpoint=tmp_path))
    engine.codec = TinyCodec()
    pointers = (engine.codec.weight.data_ptr(), engine.codec.position.data_ptr())
    monkeypatch.setattr(engine, "_park_model", lambda: None)
    monkeypatch.setattr("torch.cuda.empty_cache", lambda: None)
    monkeypatch.setattr("torch.cuda.synchronize", lambda *_: None)
    for _ in range(3):
        with engine._codec_on_gpu(8):
            assert engine.codec.weight.dtype == torch.float64
        assert engine.codec.weight.dtype == torch.float32
        assert engine.codec.position.dtype == torch.float32
        assert pointers == (
            engine.codec.weight.data_ptr(), engine.codec.position.data_ptr()
        )
        assert torch.equal(engine.codec.weight, torch.arange(4, dtype=torch.float32))


def test_insufficient_resources_fail_before_loading_or_cache_building(tmp_path, monkeypatch):
    engine = DramaEngine(EngineOptions(checkpoint=tmp_path))
    monkeypatch.setattr(
        "drama_low_vram.engine.inspect_hardware",
        lambda *_: SimpleNamespace(ram_available_bytes=12 * 1024**3, compute_capability=(8, 9)),
    )
    monkeypatch.setattr(
        "drama_low_vram.engine.autotune",
        lambda *_args, **_kwargs: SimpleNamespace(
            can_run=False, warnings=["GPU is currently busy"]
        ),
    )
    loader = Mock()
    monkeypatch.setattr("drama_low_vram.engine.load_offloaded_drama3", loader)
    with pytest.raises(RuntimeError, match="GPU is currently busy"):
        engine.load()
    assert not loader.called


def test_two_requests_cannot_mutate_the_same_model_state_concurrently(engine_harness):
    h = engine_harness
    entered, release, second_started = Event(), Event(), Event()
    count = 0

    def fake_generate(*_args, **_kwargs):
        nonlocal count
        count += 1
        if count == 1:
            entered.set()
            assert release.wait(5), "first request was never released"
        return frame_result()

    h.monkeypatch.setattr("drama_low_vram.engine.generate", fake_generate)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            h.engine.synthesize, "First text.", h.tmp_path / "first.wav"
        )
        assert entered.wait(5)

        def second_request():
            second_started.set()
            return h.engine.synthesize("Second text.", h.tmp_path / "second.wav")

        second = executor.submit(second_request)
        assert second_started.wait(5)
        assert count == 1
        assert not second.done()
        release.set()
        assert first.result(timeout=5)["request_index"] == 1
        assert second.result(timeout=5)["request_index"] == 2
    assert count == 2


def test_close_is_idempotent_and_releases_all_resources(tmp_path, monkeypatch):
    engine = DramaEngine(EngineOptions(checkpoint=tmp_path))
    cache = SimpleNamespace(close=Mock())
    model = SimpleNamespace(_drama_low_vram_expert_cache=cache)
    engine.model, engine.manager, engine.codec = model, object(), object()
    graph = SimpleNamespace(close=Mock())
    engine.graph = graph
    release_model, release_codec = Mock(), Mock()
    monkeypatch.setattr("drama_low_vram.engine.release_drama3", release_model)
    monkeypatch.setattr("drama_low_vram.engine.release_codec", release_codec)
    monkeypatch.setattr("torch.cuda.empty_cache", lambda: None)
    engine.close()
    engine.close()
    assert graph.close.call_count == 1
    assert cache.close.call_count == 1
    assert release_model.call_count == release_codec.call_count == 1
    assert engine.model is engine.manager is engine.codec is None
