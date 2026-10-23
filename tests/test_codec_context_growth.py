"""Codec context growth uses small CPU buffers and simulated device staging."""

from __future__ import annotations

import weakref
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import drama_low_vram.engine as engine_module
from drama_low_vram.engine import DramaEngine, EngineOptions


class SimulatedTransformer(torch.nn.Module):
    def __init__(self, *, pos_embed_type="rope", head_dim=4):
        super().__init__()
        self.config = SimpleNamespace(
            pos_embed_type=pos_embed_type, head_dim=head_dim, rope_base=10000
        )
        self.weight = torch.nn.Parameter(torch.arange(head_dim, dtype=torch.float32))
        self.register_buffer("freqs_cis", torch.zeros(8, head_dim // 2, 2))
        self.register_buffer("causal_mask", torch.ones(8, 8, dtype=torch.bool))


class SimulatedCodec(torch.nn.Module):
    def __init__(self, transformers):
        super().__init__()
        self.transformers = torch.nn.ModuleList(transformers)
        self._drama_low_vram_max_frames = 2
        self.transfers = []

    def to(self, device):
        self.transfers.append(device)
        return self


@pytest.fixture(autouse=True)
def prevent_cuda_calls(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Codec context tests must not access CUDA")

    for name in (
        "is_available", "device_count", "current_device", "get_device_name",
        "get_device_capability", "mem_get_info", "memory_allocated",
        "memory_reserved", "manual_seed_all", "reset_peak_memory_stats",
        "max_memory_allocated", "max_memory_reserved", "init", "_lazy_init",
    ):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(torch.cuda, "empty_cache", Mock())
    monkeypatch.setattr(torch.cuda, "synchronize", Mock())


def simulated_engine(tmp_path, monkeypatch, transformers):
    from fish_speech.models import dac

    modded_dac = ModuleType("fish_speech.models.dac.modded_dac")
    modded_dac.Transformer = SimulatedTransformer
    modded_dac.precompute_freqs_cis = Mock()
    monkeypatch.setitem(sys.modules, modded_dac.__name__, modded_dac)
    monkeypatch.setattr(dac, "modded_dac", modded_dac, raising=False)
    engine = DramaEngine(EngineOptions(checkpoint=tmp_path))
    engine.codec = SimulatedCodec(transformers)
    engine._codec_can_coexist = Mock(return_value=True)
    engine._park_model = Mock()
    snapshot = {
        (id(module), "buffer", name): value
        for module in transformers
        for name, value in module._buffers.items()
    }
    engine._codec_cpu_tensors = snapshot
    return engine, modded_dac, snapshot


def test_context_growth_drops_old_buffers_before_allocating_and_keeps_weights(
    tmp_path, monkeypatch
):
    transformers = [
        SimulatedTransformer(),
        SimulatedTransformer(pos_embed_type="conformer", head_dim=6),
    ]
    engine, modded_dac, previous_snapshot = simulated_engine(
        tmp_path, monkeypatch, transformers
    )
    old_buffers = [
        {name: weakref.ref(value) for name, value in module._buffers.items()}
        for module in transformers
    ]
    weights = [(module.weight.data_ptr(), module.weight.detach().clone()) for module in transformers]
    new_frequencies = torch.arange(64, dtype=torch.float32).reshape(16, 2, 2)
    events = []
    original_ones = torch.ones
    original_tril = torch.tril

    def assert_old_storage_released(index):
        assert engine._codec_cpu_tensors is previous_snapshot
        assert not previous_snapshot
        assert all(reference() is None for reference in old_buffers[index].values())

    def precompute(context, head_dim, rope_base):
        assert_old_storage_released(0)
        assert transformers[0].freqs_cis is None
        assert transformers[0].causal_mask is None
        assert engine.codec._drama_low_vram_max_frames == 2
        assert (context, head_dim, rope_base) == (16, 4, 10000)
        events.append("frequencies")
        return new_frequencies

    def ones(*args, **kwargs):
        index = events.count("mask")
        assert_old_storage_released(index)
        assert transformers[index].causal_mask is None
        return original_ones(*args, **kwargs)

    def tril(mask):
        index = events.count("mask")
        assert_old_storage_released(index)
        assert transformers[index].causal_mask is None
        if index == 0:
            assert transformers[index].freqs_cis is new_frequencies
        else:
            assert transformers[index].freqs_cis is None
        assert engine.codec._drama_low_vram_max_frames == 2
        assert mask.shape == (16, 16)
        assert mask.dtype is torch.bool and mask.device.type == "cpu"
        events.append("mask")
        return original_tril(mask)

    precompute_mock = Mock(side_effect=precompute)
    tril_mock = Mock(side_effect=tril)
    monkeypatch.setattr(modded_dac, "precompute_freqs_cis", precompute_mock)
    monkeypatch.setattr(torch, "ones", ones)
    monkeypatch.setattr(torch, "tril", tril_mock)

    with engine._codec_on_gpu(4) as codec:
        assert codec is engine.codec
        assert codec._drama_low_vram_max_frames == 4
        assert events == ["frequencies", "mask", "mask"]
    assert engine._transition_failure is None
    assert engine.codec.transfers == [engine.device]
    assert engine._codec_cpu_tensors is not previous_snapshot
    assert engine._codec_cpu_tensors[(id(transformers[0]), "buffer", "freqs_cis")] is new_frequencies
    for module, (pointer, values) in zip(transformers, weights):
        assert module.weight.data_ptr() == pointer
        torch.testing.assert_close(module.weight, values)
        assert module.causal_mask.shape == (16, 16)
        assert module.causal_mask.dtype is torch.bool
    engine._park_model.assert_not_called()

    buffers = [(module.freqs_cis, module.causal_mask) for module in transformers]
    with engine._codec_on_gpu(4):
        pass
    precompute_mock.assert_called_once()
    assert tril_mock.call_count == 2
    for module, (frequencies, mask) in zip(transformers, buffers):
        assert module.freqs_cis is frequencies
        assert module.causal_mask is mask


@pytest.mark.parametrize("failure_stage", ["frequencies", "mask-allocation", "mask-tril"])
@pytest.mark.parametrize("model_loaded", [False, True])
def test_context_growth_failure_blocks_engine_reuse(
    tmp_path, monkeypatch, failure_stage, model_loaded
):
    transformer = SimulatedTransformer()
    engine, modded_dac, previous_snapshot = simulated_engine(
        tmp_path, monkeypatch, [transformer]
    )
    if model_loaded:
        engine.model = object()
    message = f"Simulated {failure_stage} allocation failure"
    attempts = []
    original_ones = torch.ones
    original_tril = torch.tril

    def fail(stage):
        assert engine._codec_cpu_tensors is previous_snapshot
        assert not previous_snapshot
        assert transformer.causal_mask is None
        attempts.append(stage)
        if stage == failure_stage:
            raise RuntimeError(message)

    def precompute(*_args):
        assert transformer.freqs_cis is None
        fail("frequencies")
        return torch.zeros(16, 2, 2)

    def ones(*args, **kwargs):
        fail("mask-allocation")
        return original_ones(*args, **kwargs)

    def tril(mask):
        fail("mask-tril")
        return original_tril(mask)

    monkeypatch.setattr(modded_dac, "precompute_freqs_cis", precompute)
    monkeypatch.setattr(torch, "ones", ones)
    monkeypatch.setattr(torch, "tril", tril)
    inspect_hardware = Mock()
    load_model = Mock()
    generate = Mock()
    monkeypatch.setattr(engine_module, "inspect_hardware", inspect_hardware)
    monkeypatch.setattr(engine_module, "load_offloaded_drama3", load_model)
    monkeypatch.setattr(engine_module, "generate", generate)

    with pytest.raises(RuntimeError, match=message):
        with engine._codec_on_gpu(4):
            pytest.fail("Failed context growth must not stage the codec")

    assert engine.codec._drama_low_vram_max_frames == 2
    assert engine._transition_failure == (
        "Codec context growth failed; close this engine before retrying."
    )
    assert engine.codec.transfers == []
    engine._codec_can_coexist.assert_not_called()
    engine._park_model.assert_not_called()
    previous_attempts = list(attempts)
    with pytest.raises(RuntimeError, match="close this engine before retrying"):
        engine.load()
    with pytest.raises(RuntimeError, match="close this engine before retrying"):
        engine.synthesize("A valid sentence.", tmp_path / "unused.wav", max_new_tokens=4)
    assert attempts == previous_attempts
    assert engine._request_index == 0
    inspect_hardware.assert_not_called()
    load_model.assert_not_called()
    generate.assert_not_called()
    torch.cuda.empty_cache.assert_not_called()
    torch.cuda.synchronize.assert_not_called()
