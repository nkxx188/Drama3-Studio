"""Small real-model cursor replay and oracle validation contracts."""

from types import SimpleNamespace

import pytest
import torch

from drama_low_vram.backbone_graph import BackboneDecodeGraph
from drama_low_vram.verification import DecodeCursor, backbone_block
from drama_low_vram.tower_graph import TowerDecodeGraph
from drama_low_vram.sequence_state import SequenceSnapshot
from fish_speech.models.text2semantic.fish_omni.backbone import FishOmniBackbone
from test_backbone_graph import tiny_config
from test_tower_graph import tiny_tower
from test_sequence_state import same_bytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_cursor_stochastic_replay_and_target_window_restoration():
    torch.manual_seed(411)
    cfg = tiny_config()
    cfg.hidden_size = 32
    cfg.max_position_embeddings = 32
    backbone = FishOmniBackbone(cfg, fp8_experts=False, dtype=torch.float32, device="cuda").eval()
    for layer in backbone.model.layers:
        torch.nn.init.normal_(layer.mlp.experts.w13_weight, std=0.04)
        torch.nn.init.normal_(layer.mlp.experts.w2_weight, std=0.04)
    backbone.setup_caches(1, 32)
    model = torch.nn.Module()
    model.backbone, model.audio_decoder = backbone, tiny_tower("cuda")
    model.config = SimpleNamespace(eos_token_id=3)
    graph = BackboneDecodeGraph(backbone, sequence_capacity=32, strict=True)
    tower = TowerDecodeGraph(model.audio_decoder, 32, logits_size=32, record_logits=True)
    try:
        prompt = torch.zeros(4, 5, dtype=torch.int64)
        prompt[:, 0] = 1
        cursor = DecodeCursor.prefill(
            model, prompt, chunk_size=2, tower_graph=tower, backbone_graph=graph, top_k=1)
        base = cursor.capture()
        records = [cursor.step() for _ in range(4)]
        if cursor.finished:
            pytest.skip("Tiny random model produced an early EOS")
        final = cursor.capture()
        cursor.restore(base)
        again = [cursor.step() for _ in range(4)]
        assert all(same_bytes(a["frame"], b["frame"]) for a, b in zip(records, again))
        assert same_bytes(cursor.hidden.cpu(), final.hidden)
        proposals = torch.cat([row["frame"] for row in records])
        cursor.restore(base)
        actual = cursor.force(proposals, block=False)
        assert same_bytes(actual["semantic_logits"], torch.cat([row["semantic_logits"] for row in records]))
        assert same_bytes(actual["acoustic_logits"], torch.stack([row["acoustic_logits"] for row in records]))
        assert same_bytes(cursor.hidden.cpu(), final.hidden)
        sequential = cursor.capture()
        cursor.restore(base)
        actual = cursor.force(proposals, block=True, batch_experts=False)
        assert same_bytes(actual["semantic_logits"], torch.cat([row["semantic_logits"] for row in records]))
        assert same_bytes(cursor.hidden.cpu(), final.hidden)
        state = SequenceSnapshot.capture(model, backbone_rows=8, tower_rows=7)
        assert all(same_bytes(state.buffers[name], sequential.sequence.buffers[name])
                   for name in state.buffers)
        for bad in (proposals[:3], torch.zeros_like(proposals), proposals.double()):
            cursor.restore(base)
            with pytest.raises(ValueError, match="valid complete"):
                cursor.force(bad)
        with pytest.raises(ValueError, match="one, two or four"):
            backbone_block(graph, torch.zeros(1, 3, 32, device="cuda"), 4)
    finally:
        tower.close()
        graph.close()
