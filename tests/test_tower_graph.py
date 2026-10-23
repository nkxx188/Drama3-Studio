"""Parity checks use the official tower, not a mocked sampling implementation."""

from types import SimpleNamespace

import pytest
import torch

from drama_low_vram.tower_graph import (
    TowerDecodeGraph,
    _decode_candidates,
    main_token_candidates,
    tower_graph_buckets,
)
from fish_speech.models.text2semantic.fish_omni.config import AudioDecoderConfig
from fish_speech.models.text2semantic.fish_omni.tower import (
    FishOmniAudioDecoder,
    sample_top_k_top_p,
)


def tiny_tower(device="cpu"):
    config = SimpleNamespace(
        audio_decoder=AudioDecoderConfig(
            dim=32,
            intermediate_size=64,
            n_head=4,
            n_local_heads=2,
            head_dim=8,
            norm_eps=1e-6,
            rope_base=10000,
            rotary_percent=0.5,
            vocab_size=16,
            num_codebooks=4,
            shared_enrichment_layers=2,
            vq_backbone_codebooks=2,
        ),
        semantic_start_token_id=7,
        semantic_end_token_id=22,
        eos_token_id=3,
    )
    tower = FishOmniAudioDecoder(config, dtype=torch.float32, device=device).eval()
    tower.setup_caches(32)
    return tower


def reset_caches(tower):
    for layer in (*tower.shared_layers, *tower.residual_layers):
        attention = layer.transformer_layer.attention
        attention.k_cache.zero_()
        attention.v_cache.zero_()


@pytest.mark.parametrize("top_k,top_p", [(0, 1.0), (0, 0.5), (1, 1.0), (5, 0.9), (30, 0.95)])
@pytest.mark.parametrize("repeat_main", [False, True])
@torch.inference_mode()
def test_compact_sampler_matches_official_with_identical_rng(top_k, top_p, repeat_main):
    torch.manual_seed(51)
    tower = tiny_tower()
    hidden = torch.randn(1, 32)
    logits = torch.randn(1, 30)
    if repeat_main:
        logits[:, 10] = 100
    frame = torch.zeros(1, 5, dtype=torch.long)
    positions = torch.tensor([4])
    history = torch.tensor([10, 10, 10] if repeat_main else [1, 2, 3])
    torch.manual_seed(123)
    expected = tower.decode_step(
        logits,
        hidden,
        positions,
        frame,
        history,
        temperature=0.8,
        top_k=top_k,
        top_p=top_p,
    )
    reset_caches(tower)
    torch.manual_seed(123)
    actual = _decode_candidates(
        tower,
        logits[:, main_token_candidates(tower)],
        hidden,
        positions,
        frame,
        history,
        temperature=0.8,
        top_k=top_k,
        top_p=top_p,
    )
    assert torch.equal(actual, expected)


@torch.inference_mode()
def test_nonfinite_residual_logits_are_recorded_on_device():
    tower = tiny_tower()
    tower.codebook_heads[-1].weight.fill_(float("nan"))
    finite = torch.ones((), dtype=torch.bool)
    _decode_candidates(
        tower,
        torch.zeros(1, 17),
        torch.zeros(1, 32),
        torch.tensor([0]),
        torch.zeros(1, 5, dtype=torch.long),
        torch.tensor([-1]),
        temperature=0.8,
        top_k=5,
        top_p=0.9,
        finite=finite,
    )
    assert not finite.item()


def test_zero_top_k_disables_only_the_top_k_filter():
    logits = torch.zeros(1, 4)
    choices = set()
    for seed in range(8):
        torch.manual_seed(seed)
        choices.add(sample_top_k_top_p(logits, 1.0, 0, 1.0).item())
    assert len(choices) > 1

    torch.manual_seed(0)
    nucleus_choice = sample_top_k_top_p(torch.tensor([[3.0, 2.0, 1.0]]), 1.0, 0, 0.6)
    assert nucleus_choice.item() == 0


def test_tower_graph_buckets_respect_actual_cache_capacity():
    assert tower_graph_buckets(25) == (25,)
    assert tower_graph_buckets(513) == (256, 512, 513)
    assert tower_graph_buckets(1024) == (256, 512, 1024)
    assert tower_graph_buckets(1500) == (256, 512, 1024, 1500)
    with pytest.raises(ValueError):
        tower_graph_buckets(0)


def test_cpu_tower_is_rejected_before_cuda_allocation():
    with pytest.raises(ValueError, match="require a CUDA"):
        TowerDecodeGraph(tiny_tower(), 32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@torch.inference_mode()
def test_cuda_graph_greedy_matches_official_across_buckets_and_repeated_requests():
    torch.manual_seed(51)
    tower = tiny_tower("cuda")
    graph = TowerDecodeGraph(
        tower, 32, logits_size=30, buckets=(8, 16, 32), record_logits=True
    )
    hidden = torch.randn(1, 32, device="cuda")
    logits = torch.zeros(1, 30, device="cuda")
    logits[:, 10] = 100
    frame = torch.zeros(1, 5, dtype=torch.long, device="cuda")
    history = torch.zeros(4, dtype=torch.long, device="cuda")
    for position in (0, 7, 8, 15, 16, 31):
        reset_caches(tower)
        expected = tower.decode_step(
            logits,
            hidden,
            torch.tensor([position], device="cuda"),
            frame,
            history,
            temperature=0.8,
            top_k=1,
            top_p=0.9,
        )
        reset_caches(tower)
        graph.start(torch.zeros(4, 5, dtype=torch.long, device="cuda"), 0.8, 1, 0.9)
        actual, main = graph.step(logits, hidden, position, frame)
        assert torch.equal(actual, expected)
        assert main == actual[0, 0].item()
        assert graph.history[-1].item() == main
        assert graph.codebook_logits.isfinite().all()
        assert all(a.static_kv_len is None for a in graph.tower_attentions())
    tower.setup_caches(32)
    with pytest.raises(RuntimeError, match="KV caches changed"):
        graph.start(frame, 0.8, 1, 0.9)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@torch.inference_mode()
def test_capture_preserves_rng_and_replay_advances_sampling():
    tower = tiny_tower("cuda")
    torch.manual_seed(567)
    state = torch.cuda.get_rng_state().clone()
    graph = TowerDecodeGraph(tower, 32, compact_logits=True)
    assert torch.equal(torch.cuda.get_rng_state(), state)
    frame = torch.zeros(1, 5, dtype=torch.long, device="cuda")
    hidden = torch.zeros(1, 32, device="cuda")
    logits = torch.zeros(1, 17, device="cuda")
    graph.start(frame, 1.0, 17, 1.0)
    values = [
        tuple(graph.step(logits, hidden, position, frame)[0].flatten().tolist())
        for position in range(12)
    ]
    assert len(set(values)) > 1
    reset_caches(tower)
    graph.start(frame, 1.0, 17, 1.0)
    torch.manual_seed(567)
    repeated = [
        tuple(graph.step(logits, hidden, position, frame)[0].flatten().tolist())
        for position in range(12)
    ]
    assert repeated == values


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@torch.inference_mode()
def test_cuda_graph_rejects_nonfinite_codebook_logits():
    tower = tiny_tower("cuda")
    tower.codebook_heads[-1].weight.fill_(float("nan"))
    graph = TowerDecodeGraph(tower, 32, compact_logits=True)
    frame = torch.zeros(1, 5, dtype=torch.long, device="cuda")
    graph.start(frame, 0.8, 1, 0.9)
    with pytest.raises(FloatingPointError, match="non-finite logits"):
        graph.step(
            torch.zeros(1, 17, device="cuda"),
            torch.zeros(1, 32, device="cuda"),
            0,
            frame,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@torch.inference_mode()
def test_graph_input_precision_and_close_are_explicit():
    tower = tiny_tower("cuda")
    graph = TowerDecodeGraph(
        tower, 32, compact_logits=True, logits_dtype=torch.float64
    )
    frame = torch.zeros(1, 5, dtype=torch.long, device="cuda")
    hidden = torch.zeros(1, 32, device="cuda")
    graph.start(frame, 0.8, 1, 0.9)
    with pytest.raises(ValueError, match="preserving input precision"):
        graph.step(torch.zeros(1, 17, device="cuda"), hidden, 0, frame)
    graph.step(torch.zeros(1, 17, dtype=torch.float64, device="cuda"), hidden, 0, frame)
    identity = graph._current_cache_identity()
    graph.close()
    graph.close()
    assert not graph.graphs
    assert graph.description()["closed"]
    assert graph._current_cache_identity() == identity
    with pytest.raises(RuntimeError, match="closed"):
        graph.start(frame, 0.8, 1, 0.9)
