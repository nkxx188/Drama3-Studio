"""Parity tests for the decode-only backbone graph prototype."""

from __future__ import annotations

import copy
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from drama_low_vram.backbone_graph import BackboneDecodeGraph
from fish_speech.models.text2semantic.fish_omni.backbone import FishOmniBackbone
from fish_speech.models.text2semantic.fish_omni.config import FishOmniTextConfig


def tiny_config() -> FishOmniTextConfig:
    return FishOmniTextConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        rope_theta=10000.0,
        partial_rotary_factor=0.5,
        rms_norm_eps=1e-6,
        max_position_embeddings=16,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
    )


def make_backbone(device="cpu"):
    torch.manual_seed(411)
    backbone = FishOmniBackbone(
        tiny_config(), fp8_experts=False, dtype=torch.float32, device=device
    ).eval()
    with torch.no_grad():
        for layer in backbone.model.layers:
            nn.init.normal_(layer.mlp.experts.w13_weight, std=0.04)
            nn.init.normal_(layer.mlp.experts.w2_weight, std=0.04)
    return backbone


def prime(backbone, hidden):
    backbone.setup_caches(1, 8)
    return backbone.run_layers(
        hidden,
        torch.tensor([0], device=hidden.device),
        kv_len=1,
        continue_state=False,
        use_cache=True,
    )


def test_disabled_graph_delegates_to_upstream_and_preserves_shape():
    first = make_backbone()
    second = copy.deepcopy(first)
    hidden = torch.randn(1, 1, 16)
    first_prompt = prime(first, hidden)
    second_prompt = prime(second, hidden)
    graph = BackboneDecodeGraph(
        first, sequence_capacity=8, capture=False, device="cpu"
    )
    actual = graph(first_prompt, 1)
    expected = second.run_layers(
        second_prompt,
        torch.tensor([1]),
        kv_len=2,
        continue_state=True,
        use_cache=True,
    )
    assert not graph.enabled
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (1, 1, 16)


def test_capture_failure_falls_back_to_eager_without_closing_runtime():
    first = make_backbone()
    second = copy.deepcopy(first)
    hidden = torch.randn(1, 1, 16)
    first_prompt = prime(first, hidden)
    second_prompt = prime(second, hidden)
    graph = BackboneDecodeGraph(first, sequence_capacity=8, capture=True)
    assert not graph.enabled
    assert "require a CUDA" in graph.capture_error
    assert graph.capture_recovered is True
    assert graph.description["capture_stage"] == "validate"
    actual = graph(first_prompt, 1)
    expected = second.run_layers(
        second_prompt,
        torch.tensor([1]),
        kv_len=2,
        continue_state=True,
        use_cache=True,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("capacity", [0, 1, 17])
def test_invalid_capacity_fails_before_cache_allocation(capacity):
    with pytest.raises(ValueError):
        BackboneDecodeGraph(make_backbone(), sequence_capacity=capacity, capture=False)


def test_decode_shape_position_and_lifecycle_checks():
    backbone = make_backbone()
    graph = BackboneDecodeGraph(backbone, sequence_capacity=8, capture=False)
    hidden = torch.randn(1, 1, 16)
    with pytest.raises(ValueError, match="shape"):
        graph(hidden.squeeze(0), 1)
    with pytest.raises(ValueError, match="hidden width"):
        graph(hidden[:, :, :8], 1)
    with pytest.raises(ValueError, match="within capacity"):
        graph(hidden, 0)
    with pytest.raises(ValueError, match="within capacity"):
        graph(hidden, 8)
    with pytest.raises(TypeError, match="integer"):
        graph(hidden, True)
    graph.close()
    with pytest.raises(RuntimeError, match="closed"):
        graph(hidden, 1)


def test_partial_capture_exception_clears_every_graph_reference():
    backbone = make_backbone()

    def failed_capture(graph):
        graph.capture_stage = "capture_pre"
        graph.capture_layer_index = 7
        graph.capture_layers_completed = 7
        graph._layers.extend([object(), object()])
        graph._cache_addresses[1] = (2,)
        graph._capture_pool = object()
        raise RuntimeError("simulated capture failure")

    with patch.object(BackboneDecodeGraph, "_capture", failed_capture):
        graph = BackboneDecodeGraph(backbone, sequence_capacity=8)
    assert not graph.enabled
    assert not graph._layers
    assert not graph._cache_addresses
    assert graph._capture_pool is None
    assert graph.description["capture_layer_index"] == 7
    assert graph.description["capture_layers_completed"] == 7
    assert "simulated capture failure" in graph.capture_error


def test_capture_with_unrecoverable_rng_raises_even_when_strict_is_false():
    with (
        patch.object(BackboneDecodeGraph, "_capture", side_effect=RuntimeError("capture error")),
        patch.object(
            BackboneDecodeGraph,
            "_check_cuda_capture_recovery",
            side_effect=RuntimeError("Offset increment outside graph capture"),
        ),
    ):
        with pytest.raises(RuntimeError, match="did not recover"):
            BackboneDecodeGraph(make_backbone(), sequence_capacity=8, strict=False)


def test_segment_residual_order_preserves_low_precision_mlp_addition():
    # (x + expert) + shared rounds to 0; x + (expert + shared) rounds to 1.
    residual = torch.tensor([[[1.0]]], dtype=torch.bfloat16)
    expert = torch.tensor([[256.0]], dtype=torch.bfloat16)
    shared = torch.tensor([[[-256.0]]], dtype=torch.bfloat16)
    expected = residual + (expert.view_as(residual) + shared)
    actual = BackboneDecodeGraph._post_impl(residual, expert, shared)
    assert torch.equal(actual, expected)
    assert not torch.equal(actual, (residual + expert.view_as(residual)) + shared)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_graph_replay_matches_eager_decode_and_updates_state():
    eager = make_backbone("cuda").eval()
    graph_backbone = copy.deepcopy(eager).eval()
    hidden = torch.randn(1, 1, 16, device="cuda", dtype=torch.float32)
    eager_first = prime(eager, hidden)
    graph_first = prime(graph_backbone, hidden)
    torch.testing.assert_close(graph_first, eager_first, rtol=0, atol=0)
    graph = BackboneDecodeGraph(
        graph_backbone,
        sequence_capacity=8,
        device="cuda",
        capture=True,
        strict=True,
    )
    assert graph.enabled
    eager_second = eager.run_layers(
        eager_first,
        torch.tensor([1], device="cuda"),
        kv_len=2,
        continue_state=True,
        use_cache=True,
    )
    graph_second = graph(graph_first, 1)
    torch.testing.assert_close(graph_second, eager_second, rtol=1e-4, atol=1e-4)
    eager_third = eager.run_layers(
        eager_second,
        torch.tensor([2], device="cuda"),
        kv_len=3,
        continue_state=True,
        use_cache=True,
    )
    graph_third = graph(graph_second, 2)
    torch.testing.assert_close(graph_third, eager_third, rtol=1e-4, atol=1e-4)
    graph.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_bf16_graph_top8_routes_cache_restore_and_multi_step_bitwise_parity():
    config = replace(
        tiny_config(),
        hidden_size=128,
        num_experts=8,
        num_experts_per_tok=8,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
    )
    torch.manual_seed(216)
    eager = FishOmniBackbone(
        config, fp8_experts=False, dtype=torch.bfloat16, device="cuda"
    ).eval()
    with torch.no_grad():
        for layer in eager.model.layers:
            nn.init.normal_(layer.mlp.experts.w13_weight, std=0.01)
            nn.init.normal_(layer.mlp.experts.w2_weight, std=0.01)
    graphed = copy.deepcopy(eager)
    hidden = torch.randn(1, 1, 128, device="cuda", dtype=torch.bfloat16)
    prime(eager, hidden)
    prime(graphed, hidden)
    state_before = {
        name: getattr(graphed.model.layers[0].linear_attn, name).clone()
        for name in ("conv_state", "recurrent_state")
    }
    graph = BackboneDecodeGraph(
        graphed, sequence_capacity=8, capture=True, strict=True
    )
    assert graph.captured_layer_count == 1
    for name, state in state_before.items():
        assert torch.equal(state, getattr(graphed.model.layers[0].linear_attn, name))
    for position in range(1, 7):
        # Identical teacher-forced input isolates implementation error from
        # autoregressive divergence.
        hidden = torch.randn_like(hidden)
        expected = eager.run_layers(
            hidden,
            torch.tensor([position], device="cuda"),
            kv_len=position + 1,
            continue_state=True,
            use_cache=True,
        )
        actual = graph(hidden, position)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert graph._layers[0].topk_ids.shape[-1] == 8
        assert graph._layers[0].topk_ids.unique().numel() == 8
    assert graph.graph_replay_count == 12
    graphed.setup_caches(1, 8)
    with pytest.raises(RuntimeError, match="reallocated"):
        graph(hidden, 7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@torch.inference_mode()
def test_thirty_gdn_graph_segments_and_stochastic_tower_replay_share_rng_safely():
    from types import SimpleNamespace

    from drama_low_vram.tower_graph import TowerDecodeGraph
    from fish_speech.models.text2semantic.fish_omni.config import AudioDecoderConfig
    from fish_speech.models.text2semantic.fish_omni.tower import FishOmniAudioDecoder

    config = replace(
        tiny_config(), num_hidden_layers=30, layer_types=["linear_attention"] * 30
    )
    torch.manual_seed(418)
    backbone = FishOmniBackbone(
        config, fp8_experts=False, dtype=torch.float32, device="cuda"
    ).eval()
    for layer in backbone.model.layers:
        nn.init.normal_(layer.mlp.experts.w13_weight, std=0.01)
        nn.init.normal_(layer.mlp.experts.w2_weight, std=0.01)
    backbone.setup_caches(1, 8)
    tower_config = SimpleNamespace(
        audio_decoder=AudioDecoderConfig(
            dim=16,
            intermediate_size=32,
            n_head=2,
            n_local_heads=1,
            head_dim=8,
            norm_eps=1e-6,
            rope_base=10000,
            rotary_percent=0.5,
            vocab_size=8,
            num_codebooks=3,
            shared_enrichment_layers=1,
            vq_backbone_codebooks=1,
        ),
        semantic_start_token_id=7,
        semantic_end_token_id=14,
        eos_token_id=3,
    )
    tower = FishOmniAudioDecoder(tower_config, dtype=torch.float32, device="cuda").eval()
    tower.setup_caches(8)
    rng_before = torch.cuda.get_rng_state().clone()
    graph = BackboneDecodeGraph(backbone, sequence_capacity=8, strict=True)
    assert graph.captured_layer_count == 30
    assert graph.description["capture_stream_status"] == "idle"
    assert torch.equal(torch.cuda.get_rng_state(), rng_before)
    tower_graph = TowerDecodeGraph(tower, 8, compact_logits=True)
    assert torch.equal(torch.cuda.get_rng_state(), rng_before)
    prompt = torch.zeros(1, 4, dtype=torch.long, device="cuda")
    hidden = torch.zeros(1, 1, 16, device="cuda")
    logits = torch.zeros(1, 9, device="cuda")
    for request in range(2):
        tower_graph.start(prompt, 0.8, 5, 0.95)
        for position in range(1, 4):
            output = graph(hidden, position)
            sampled, _ = tower_graph.step(logits, output.reshape(1, -1), position, prompt)
            assert torch.isfinite(output).all()
            assert sampled.shape == prompt.shape
    assert graph.graph_replay_count == 30 * 2 * 6
    tower_graph.close()
    graph.close()
