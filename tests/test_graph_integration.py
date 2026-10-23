"""CUDA-graph integration checks for independent tower/backbone captures."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn

from drama_low_vram.backbone_graph import BackboneDecodeGraph
from drama_low_vram.tower_graph import TowerDecodeGraph
from fish_speech.models.text2semantic.fish_omni.backbone import FishOmniBackbone
from fish_speech.models.text2semantic.fish_omni.config import (
    AudioDecoderConfig,
    FishOmniTextConfig,
)
from fish_speech.models.text2semantic.fish_omni.tower import FishOmniAudioDecoder


def backbone_config() -> FishOmniTextConfig:
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


def tower_config() -> object:
    return type(
        "Config",
        (),
        {
            "audio_decoder": AudioDecoderConfig(
                dim=16,
                intermediate_size=32,
                n_head=2,
                n_local_heads=1,
                head_dim=8,
                norm_eps=1e-6,
                rope_base=10000,
                rotary_percent=0.5,
                vocab_size=16,
                num_codebooks=4,
                shared_enrichment_layers=2,
                vq_backbone_codebooks=2,
            ),
            "semantic_start_token_id": 7,
            "semantic_end_token_id": 22,
            "eos_token_id": 3,
        },
    )()


def make_pair(*, layers=2, capacity=8, dtype=torch.bfloat16) -> tuple[FishOmniBackbone, FishOmniAudioDecoder]:
    torch.manual_seed(803)
    config = backbone_config()
    if layers != 2:
        config = replace(
            config,
            num_hidden_layers=layers,
            layer_types=[
                "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
                for index in range(layers)
            ],
        )
    config = replace(config, max_position_embeddings=max(16, capacity))
    backbone = FishOmniBackbone(
        config, fp8_experts=False, dtype=dtype, device="cuda"
    ).eval()
    tower = FishOmniAudioDecoder(
        tower_config(), dtype=dtype, device="cuda"
    ).eval()
    with torch.no_grad():
        for layer in backbone.model.layers:
            nn.init.normal_(layer.mlp.experts.w13_weight, std=0.02)
            nn.init.normal_(layer.mlp.experts.w2_weight, std=0.02)
    return backbone, tower


def prime(backbone: FishOmniBackbone, tower: FishOmniAudioDecoder):
    backbone.setup_caches(1, 8)
    tower.setup_caches(8)
    frame = torch.zeros((1, 5), dtype=torch.long, device="cuda")
    embeddings = tower.embed_backbone(
        frame, backbone.embed(frame[:, 0])
    )
    hidden = backbone(
        inputs_embeds=embeddings[None],
        input_pos=torch.zeros(1, dtype=torch.long, device="cuda"),
    )[0].to("cuda")
    return frame, hidden


def _capture_pair(*, tower_first: bool):
    backbone, tower = make_pair()
    frame, hidden = prime(backbone, tower)
    if tower_first:
        tower_graph = TowerDecodeGraph(tower, 8, compact_logits=True, buckets=(8,))
        backbone_graph = BackboneDecodeGraph(
            backbone, sequence_capacity=8, capture=True, strict=True
        )
    else:
        backbone_graph = BackboneDecodeGraph(
            backbone, sequence_capacity=8, capture=True, strict=True
        )
        tower_graph = TowerDecodeGraph(tower, 8, compact_logits=True, buckets=(8,))
    return backbone, tower, frame, hidden, tower_graph, backbone_graph


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("tower_first", [True, False])
@torch.no_grad()
def test_tower_and_backbone_graphs_capture_independently_and_replay(
    tower_first: bool,
):
    backbone, tower, frame, hidden, tower_graph, backbone_graph = _capture_pair(
        tower_first=tower_first
    )
    logits = torch.zeros((1, 17), dtype=torch.bfloat16, device="cuda")
    logits[:, 8] = 100
    tower_graph.start(frame, 0.8, 1, 1.0)
    sampled, main = tower_graph.step(logits, hidden, 0, frame)
    assert main == int(sampled[0, 0])
    next_embeddings = tower.embed_backbone(
        sampled, backbone.embed(sampled[:, 0])
    )
    next_hidden = backbone_graph(next_embeddings[None], 1)[0].to("cuda")
    assert torch.isfinite(next_hidden).all()
    # A second replay proves the capture remains valid after the first graph
    # has consumed its RNG and the backbone graph updated recurrent state.
    sampled2, _ = tower_graph.step(logits, next_hidden, 1, sampled)
    assert sampled2.shape == sampled.shape
    assert torch.isfinite(sampled2).all()
    tower_graph.close()
    backbone_graph.close()


def _reset_sequence(backbone, tower):
    for layer in backbone.model.layers:
        mixer = layer.self_attn if layer.layer_type == "full_attention" else layer.linear_attn
        names = (
            ("k_cache", "v_cache")
            if layer.layer_type == "full_attention"
            else ("conv_state", "recurrent_state")
        )
        for name in names:
            getattr(mixer, name).zero_()
    for layer in (*tower.shared_layers, *tower.residual_layers):
        layer.transformer_layer.attention.k_cache.zero_()
        layer.transformer_layer.attention.v_cache.zero_()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@torch.no_grad()
def test_thirty_captured_gdn_layers_and_three_sampling_buckets_repeat_after_reseed():
    backbone, tower = make_pair(layers=40, capacity=256)
    backbone.setup_caches(1, 256)
    tower.setup_caches(256)
    backbone_graph = BackboneDecodeGraph(
        backbone, sequence_capacity=256, capture=True, strict=True
    )
    tower_graph = TowerDecodeGraph(
        tower, 256, compact_logits=True, buckets=(64, 128, 256)
    )
    assert backbone_graph.captured_layer_count == 30
    frame = torch.zeros((1, 5), dtype=torch.long, device="cuda")
    logits = torch.zeros((1, 17), dtype=torch.bfloat16, device="cuda")

    def request():
        _reset_sequence(backbone, tower)
        hidden = backbone(
            inputs_embeds=tower.embed_backbone(frame, backbone.embed(frame[:, 0]))[None],
            input_pos=torch.zeros(1, dtype=torch.long, device="cuda"),
        )[0]
        torch.manual_seed(123)
        tower_graph.start(frame, 1.0, 17, 1.0)
        current, outputs = frame, []
        for position in range(200):
            current, _ = tower_graph.step(logits, hidden, position, current)
            outputs.append(current)
            if position < 199:
                embeddings = tower.embed_backbone(
                    current, backbone.embed(current[:, 0])
                )
                hidden = backbone_graph(embeddings[None], position + 1)[0]
        assert torch.isfinite(hidden).all()
        return torch.cat(outputs)

    first = request()
    second = request()
    assert torch.equal(first, second)
    assert first.unique().numel() > 4
    assert backbone_graph.graph_replay_count == 2 * 199 * 30 * 2
    tower_graph.close()
    backbone_graph.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_first_cuda_rng_graph_capture_is_safe_in_a_fresh_process():
    # The generator initializes its bookkeeping only on the process's first
    # graph. Other CUDA tests can mask this bug by creating it in normal mode.
    script = (
        "import runpy; "
        f"tests = runpy.run_path({str(Path(__file__).resolve())!r}); "
        "tests['test_tower_and_backbone_graphs_capture_independently_and_replay'](True)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@torch.no_grad()
def test_fp16_compatibility_backbone_and_audio_tower_execute_and_capture():
    """Tests the compatible arithmetic on this GPU, not a physical RTX20."""
    backbone, tower = make_pair(dtype=torch.float16)
    frame, hidden = prime(backbone, tower)
    assert hidden.dtype == torch.float16 and torch.isfinite(hidden).all()
    bg = BackboneDecodeGraph(backbone, sequence_capacity=8, capture=True, strict=True)
    tg = TowerDecodeGraph(tower, 8, compact_logits=True, buckets=(8,))
    try:
        logits = torch.zeros((1, 17), dtype=torch.float16, device="cuda")
        logits[:, 8] = 100
        tg.start(frame, 0.8, 1, 1.0)
        sampled, _ = tg.step(logits, hidden, 0, frame)
        value = bg(tower.embed_backbone(sampled, backbone.embed(sampled[:, 0]))[None], 1)[0]
        assert value.dtype == torch.float16 and torch.isfinite(value).all()
    finally:
        tg.close()
        bg.close()
