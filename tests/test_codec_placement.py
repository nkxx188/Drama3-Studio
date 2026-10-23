"""Selective codec branches, original CPU ownership, and serial decode failures."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from drama_low_vram.codec import (
    codec_operation_modules, codec_operation_roots, codec_supports_split_decode,
    codec_workspace_bytes,
)
from drama_low_vram.engine import DramaEngine, EngineOptions
from fish_speech.models.dac.modded_dac import DAC
from fish_speech.models.dac.rvq import DownsampleResidualVectorQuantize


class Branch(nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(4, dtype=torch.float32))
        self.register_buffer("positions", torch.arange(3))
        self.transfers = []
        self.fail = fail

    def to(self, device):
        self.transfers.append(str(device))
        # Replace the CPU tensor to simulate transfer ownership without CUDA.
        self.weight = nn.Parameter(self.weight.detach().clone(), requires_grad=False)
        if self.fail:
            raise RuntimeError("Injected branch transfer failure")
        return self


def skeleton(codec_type=DAC):
    codec = codec_type.__new__(codec_type)
    nn.Module.__init__(codec)
    codec.encoder = Branch()
    codec.decoder = Branch()
    quantizer = DownsampleResidualVectorQuantize.__new__(DownsampleResidualVectorQuantize)
    nn.Module.__init__(quantizer)
    for name in ("semantic_quantizer", "quantizer", "post_module", "upsample",
                 "pre_module", "downsample"):
        setattr(quantizer, name, Branch())
    def transfer_quantizer(device):
        for branch in quantizer.children():
            branch.to(device)
        return quantizer
    quantizer.to = transfer_quantizer
    codec.quantizer = quantizer
    codec._drama_low_vram_max_frames = 16
    return codec


def test_only_audited_official_branches_are_selected():
    codec = skeleton()
    q = codec.quantizer
    latents = q.semantic_quantizer, q.quantizer, q.post_module, q.upsample
    assert codec_operation_roots(codec, "encode") == (codec.encoder, q)
    assert codec_operation_roots(codec, "decode") == (*latents, codec.decoder)
    assert codec_operation_roots(codec, "decode_latents") == latents
    assert codec_operation_roots(codec, "decode_waveform") == (codec.decoder,)
    assert codec_supports_split_decode(codec)
    assert codec_operation_roots(codec, "all") == (codec,)
    for operation in ("decode", "decode_latents", "decode_waveform"):
        modules = codec_operation_modules(codec, operation)
        assert len({id(module) for module in modules}) == len(modules)
        assert codec.encoder not in modules and q.pre_module not in modules


def test_unknown_implementations_and_subclasses_use_full_placement():
    class CustomDAC(DAC):
        pass

    for codec in (skeleton(CustomDAC), nn.Linear(4, 4)):
        for operation in ("encode", "decode", "decode_latents", "decode_waveform"):
            assert codec_operation_roots(codec, operation) == (codec,)
        assert not codec_supports_split_decode(codec)
    codec = skeleton()
    codec.quantizer = nn.Linear(4, 4)
    assert codec_operation_roots(codec, "decode") == (codec,)
    assert not codec_supports_split_decode(codec)
    with pytest.raises(ValueError, match="operation"):
        codec_operation_roots(codec, "invalid")


@pytest.mark.parametrize("direction", ["encode", "decode_latents", "decode_waveform"])
@pytest.mark.parametrize("failure", [False, True])
def test_branch_staging_restores_original_cpu_storage_after_partial_transfer(
    tmp_path, monkeypatch, direction, failure,
):
    codec = skeleton()
    engine = DramaEngine(EngineOptions(checkpoint=tmp_path))
    engine.codec = codec
    monkeypatch.setattr(engine, "_codec_can_coexist", Mock(return_value=True))
    monkeypatch.setattr(torch.cuda, "empty_cache", Mock())
    monkeypatch.setattr(torch.cuda, "synchronize", Mock())
    selected = codec_operation_roots(codec, direction)
    selected[0].fail = failure
    original = {id(module): module.weight for module in codec.modules() if isinstance(module, Branch)}
    if failure:
        with pytest.raises(RuntimeError, match="branch transfer"):
            with engine._codec_on_gpu(8, operation=direction):
                pytest.fail("A failed transfer must not enter the operation")
    else:
        with engine._codec_on_gpu(8, operation=direction):
            assert engine.codec is codec
    expected_modules = set(codec_operation_modules(codec, direction))
    for module in codec.modules():
        if isinstance(module, Branch):
            assert module.weight.device.type == "cpu"
            assert module.weight.data_ptr() == original[id(module)].data_ptr()
            assert torch.equal(module.weight, original[id(module)])
            if not failure:
                assert bool(module.transfers) == (module in expected_modules)
    assert engine._lifecycle["codec_selective_calls"] == 1
    assert engine._transition_failure is None


def test_split_decode_orders_branches_and_preserves_codes(tmp_path, monkeypatch):
    codec = skeleton()
    engine = DramaEngine(EngineOptions(checkpoint=tmp_path))
    engine.codec, engine.device = codec, torch.device("cpu")
    monkeypatch.setattr(engine, "_full_codec_decode_fits", lambda _frames: False)
    codes = torch.arange(12, dtype=torch.int64).reshape(3, 4)
    expected_codes = codes.clone()
    events = []
    latents = torch.ones(1, 4, 4)

    def decode(indices):
        events.append("quantizer")
        assert indices.shape == (1, 3, 4)
        indices.zero_()  # The official quantizer clamps its private index copy.
        return latents

    codec.quantizer.decode = decode
    codec.decoder.forward = lambda value: (
        events.append("waveform") or value[:, :1]
    )

    def run(frames, operation, *, direction):
        events.append(direction)
        assert frames == 4
        return operation(codec)

    monkeypatch.setattr(engine, "_run_codec", run)
    output = engine._decode_audio_segments([codes, codes])
    assert events == ["decode_latents", "quantizer", "decode_waveform", "waveform"] * 2
    assert output.shape == (8,) and bool((output == 1).all())
    assert torch.equal(codes, expected_codes)
    assert engine._lifecycle["codec_split_decode_calls"] == 1


def test_first_text_only_decode_loads_cpu_codec_before_selecting_branches(tmp_path, monkeypatch):
    codec = skeleton()
    engine = DramaEngine(EngineOptions(checkpoint=tmp_path))
    engine.device = torch.device("cpu")
    monkeypatch.setattr(engine, "_full_codec_decode_fits", lambda _frames: False)
    loader = Mock(return_value=codec)
    monkeypatch.setattr("drama_low_vram.engine.load_codec", loader)
    monkeypatch.setattr("drama_low_vram.engine.resolve_codec_path", lambda _path: tmp_path / "codec.pth")
    codec.quantizer.decode = lambda _indices: torch.ones(1, 4, 4)
    codec.decoder.forward = lambda value: value[:, :1]
    calls = []

    def run(_frames, operation, *, direction):
        calls.append(direction)
        return operation(codec)

    monkeypatch.setattr(engine, "_run_codec", run)
    engine._decode_audio_segments([torch.zeros(3, 4, dtype=torch.int64)])
    loader.assert_called_once_with(
        tmp_path / "codec.pth", device="cpu", dtype=torch.bfloat16, max_frames=4)
    assert calls == ["decode_latents", "decode_waveform"]


@pytest.mark.parametrize("admissions,expected", [
    ([True], True), ([False, True], True), ([False, False], False),
])
def test_full_codec_preference_uses_current_and_reclaimable_memory(
    tmp_path, monkeypatch, admissions, expected,
):
    engine = DramaEngine(EngineOptions(checkpoint=tmp_path))
    engine.manager = SimpleNamespace(stats=SimpleNamespace(hot_cache_bytes=100, device_buffer_bytes=200))
    probe = Mock(side_effect=admissions)
    monkeypatch.setattr(engine, "_codec_can_coexist", probe)
    assert engine._full_codec_decode_fits(100) == expected
    calls = [((100,), {"operation": "all"})]
    if not admissions[0]:
        calls.append(((100,), {"operation": "all", "reclaimable_bytes": 300}))
    assert [(call.args, call.kwargs) for call in probe.call_args_list] == calls


def test_auto_retains_original_whole_codec_call_when_it_already_coexists(tmp_path, monkeypatch):
    engine = DramaEngine(EngineOptions(checkpoint=tmp_path))
    engine.codec = skeleton()
    monkeypatch.setattr(engine, "_full_codec_decode_fits", lambda _frames: True)
    decoder = Mock(return_value=torch.ones(4))
    monkeypatch.setattr("drama_low_vram.engine.decode_to_audio", decoder)
    directions = []

    def run(_frames, operation, *, direction):
        directions.append(direction)
        return operation(engine.codec)

    monkeypatch.setattr(engine, "_run_codec", run)
    engine._decode_audio_segments([torch.zeros(3, 4, dtype=torch.int64)])
    assert directions == ["all"]
    assert engine._lifecycle["codec_split_decode_calls"] == 0
    assert decoder.call_count == 1


@pytest.mark.parametrize("stage", ["decode_latents", "decode_waveform"])
def test_split_decode_failure_does_not_run_later_segments(tmp_path, monkeypatch, stage):
    codec = skeleton()
    engine = DramaEngine(EngineOptions(checkpoint=tmp_path))
    engine.codec = codec
    monkeypatch.setattr(engine, "_full_codec_decode_fits", lambda _frames: False)
    calls = []
    codec.quantizer.decode = lambda _indices: torch.zeros(1, 4, 4)

    def run(_frames, _operation, *, direction):
        calls.append(direction)
        if direction == stage:
            raise RuntimeError("Injected decode operation failure")
        return torch.zeros(1, 4, 4)

    monkeypatch.setattr(engine, "_run_codec", run)
    with pytest.raises(RuntimeError, match="decode operation"):
        engine._decode_audio_segments([torch.zeros(3, 4, dtype=torch.int64)] * 2)
    assert calls == (["decode_latents"] if stage == "decode_latents"
                     else ["decode_latents", "decode_waveform"])


def test_smaller_latent_workspace_is_specific_to_the_audited_stage():
    assert codec_workspace_bytes(375, "decode_latents") == 256 * 1024**2 + 375 * 256 * 1024
    assert codec_workspace_bytes(375, "decode_waveform") == codec_workspace_bytes(375)
    for invalid in (True, 0, -1, 2.5):
        with pytest.raises((ValueError, TypeError)):
            codec_workspace_bytes(invalid, "decode_latents")


def test_codec_loaded_from_inference_scope_keeps_normal_cpu_parameters(tmp_path, monkeypatch):
    from drama_low_vram.codec import load_codec

    source = tmp_path / "codec.pth"
    torch.save({"weight": torch.arange(4).float()}, source)
    observed = []

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            observed.append(torch.is_inference_mode_enabled())
            self.weight = nn.Parameter(torch.zeros(4))

    monkeypatch.setattr("hydra.utils.instantiate", lambda _config: Tiny())
    with torch.inference_mode():
        codec = load_codec(source, device="cpu", dtype=torch.bfloat16, max_frames=8)
    assert observed == [False]
    assert not torch.is_inference(codec.weight)
    assert isinstance(codec.weight._version, int)
    # This is the cuDNN/weight-norm compatibility boundary exercised by a later
    # encode request; ordinary version-tracking tensors must survive first-use.
    with torch.no_grad():
        assert torch.equal(codec.weight, torch.arange(4).bfloat16())
