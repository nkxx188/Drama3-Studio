"""Cache ownership, in-place restoration and stochastic graph continuation."""

from types import SimpleNamespace
import random

import numpy as np
import pytest
import torch

from drama_low_vram.sequence_state import SequenceSnapshot, _bindings
from drama_low_vram.tower_graph import TowerDecodeGraph
from test_backbone_graph import make_backbone
from test_tower_graph import tiny_tower


def model(device="cpu"):
    result = torch.nn.Module()
    result.backbone = make_backbone(device)
    result.backbone.setup_caches(1, 8)
    result.audio_decoder = tiny_tower(device)
    result.config = SimpleNamespace(eos_token_id=3)
    return result


def same_bytes(a, b):
    return torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                       b.contiguous().reshape(-1).view(torch.uint8))


@torch.inference_mode()
def test_restore_preserves_addresses_prefix_and_zeros_rejected_tail():
    value = model()
    for _, target, _, _ in _bindings(value, 3, 2):
        target.uniform_(-1, 1)
    snapshot = SequenceSnapshot.capture(value, backbone_rows=3, tower_rows=2)
    assert snapshot.nbytes == SequenceSnapshot.estimate_bytes(value, 3, 2)
    addresses = [target.data_ptr() for _, target, _, _ in _bindings(value, 3, 2)]
    for _, target, _, _ in _bindings(value, 3, 2):
        target.fill_(float("nan"))
    snapshot.restore(value)
    assert addresses == [target.data_ptr() for _, target, _, _ in _bindings(value, 3, 2)]
    for name, target, view, dimension in _bindings(value, 3, 2):
        assert same_bytes(view, snapshot.buffers[name])
        if dimension is not None:
            assert not bool(target[:, :, view.shape[2]:].any())


@torch.inference_mode()
def test_snapshot_survives_reallocation_and_validates_every_buffer_before_writing():
    value = model()
    snapshot = SequenceSnapshot.capture(value, backbone_rows=3, tower_rows=2)
    value.backbone.setup_caches(1, 16)
    value.audio_decoder.setup_caches(64)
    snapshot.restore(value)
    with pytest.raises(ValueError, match="another model"):
        snapshot.restore(model())
    before = {name: target.clone() for name, target, _, _ in _bindings(value, 3, 2)}
    last = list(snapshot.buffers)[-1]
    snapshot.buffers[last] = snapshot.buffers[last].double()
    with pytest.raises(ValueError, match="shape/dtype"):
        snapshot.restore(value)
    assert all(same_bytes(target, before[name]) for name, target, _, _ in _bindings(value, 3, 2))


@torch.inference_mode()
def test_gdn_and_attention_continue_identically_after_dirty_branch():
    value = model()
    backbone = value.backbone
    generator = torch.Generator().manual_seed(82)
    prompt = torch.randn(1, 3, 16, generator=generator)
    next_rows = torch.randn(1, 3, 16, generator=generator)
    backbone(inputs_embeds=prompt, input_pos=torch.arange(3))
    snapshot = SequenceSnapshot.capture(value, backbone_rows=3, tower_rows=0)

    def continue_rows():
        return torch.cat([
            backbone(inputs_embeds=row.reshape(1, 1, 16), input_pos=torch.tensor([3 + step]))
            for step, row in enumerate(next_rows[0])
        ], dim=1)

    expected = continue_rows()
    backbone(inputs_embeds=torch.randn(1, 1, 16), input_pos=torch.tensor([6]))
    snapshot.restore(value)
    assert same_bytes(continue_rows(), expected)


def test_random_state_and_budget_contract():
    value = model()
    snapshot = SequenceSnapshot.capture(value, backbone_rows=0, tower_rows=0, include_rng=True)
    expected = (random.random(), np.random.rand(), torch.rand(5))
    snapshot.restore(value)
    actual = (random.random(), np.random.rand(), torch.rand(5))
    assert actual[:2] == expected[:2] and same_bytes(actual[2], expected[2])
    with pytest.raises(MemoryError, match="budget"):
        SequenceSnapshot.capture(value, backbone_rows=0, tower_rows=0, max_bytes=1)
    with pytest.raises(ValueError, match="non-negative"):
        SequenceSnapshot.capture(value, backbone_rows=-1, tower_rows=0)
    with pytest.raises(ValueError, match="sufficient"):
        SequenceSnapshot.capture(value, backbone_rows=9, tower_rows=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_graph_sampling_rng_history_and_kv_restore_together():
    value = model("cuda")
    tower = value.audio_decoder
    graph = TowerDecodeGraph(tower, 32, logits_size=30, buckets=(8, 16, 32))
    try:
        torch.cuda.manual_seed_all(123)
        frame = torch.zeros(1, 5, dtype=torch.long, device="cuda")
        hidden = torch.randn(1, 32, device="cuda")
        logits = torch.zeros(1, 30, device="cuda")
        graph.start(frame.repeat(4, 1), 0.8, 5, 0.9)
        snapshot = SequenceSnapshot.capture(
            value, backbone_rows=0, tower_rows=4, tower_graph=graph, include_rng=True)

        def continue_frames():
            previous = frame
            codes = []
            for position in (4, 5, 6):
                previous, _ = graph.step(logits, hidden, position, previous)
                codes.append(previous.cpu())
            return torch.cat(codes)

        expected = continue_frames()
        snapshot.restore(value, tower_graph=graph)
        assert same_bytes(continue_frames(), expected)
    finally:
        graph.close()
