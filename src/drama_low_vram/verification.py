"""Research target verification and rollback; no draft or speculative sampler."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
import time

import torch

from .backbone_graph import BackboneDecodeGraph
from .semantic_head import SemanticHead
from .sequence_state import SequenceSnapshot
from fish_speech.models.text2semantic.sampling import (
    DEFAULT_TEMPERATURE, DEFAULT_TOP_P,
)
from .sampling import DEFAULT_TOP_K


def _hybrid_scope(model):
    hybrid = getattr(getattr(model, "_drama_low_vram_manager", None), "hybrid_decode", None)
    return hybrid.decode() if hybrid is not None else nullcontext()


def _logits(backbone, hidden):
    if isinstance(backbone.lm_head, SemanticHead):
        return backbone.lm_head.compact_logits(hidden)
    return backbone.compute_logits(hidden)


@torch.inference_mode()
def backbone_block(graph, embeddings, start, *, batch_experts=True, state_trace=None):
    """Serial recurrent mixers and dense ops; batch independent expert work.

    This changes traversal order to layer -> time. Each mixer still performs
    the same single-row operation as ordinary graph decode. No chunk-GDN
    approximation is introduced. Requires the existing captured graph.
    """
    if not graph.enabled or graph._closed:
        raise ValueError("Block verification requires an active backbone graph")
    if getattr(graph, "shared_stream", False):
        raise ValueError("Block verification does not support the experimental shared stream")
    if embeddings.ndim != 3 or embeddings.shape[0] != 1 or embeddings.shape[1] not in (1, 2, 4):
        raise ValueError("Block verification requires one, two or four embedded rows")
    count = embeddings.shape[1]
    if not 1 <= start or start + count > graph.sequence_capacity:
        raise ValueError("Verification block exceeds the sequence capacity")
    hidden = [row.reshape(1, 1, -1).to(graph.backbone.residual_dtype)
              for row in embeddings[0]]
    for layer_index, layer_graph in enumerate(graph._layers):
        layer = layer_graph.layer
        if graph._cache_addresses_for(layer) != graph._cache_addresses[id(layer)]:
            raise RuntimeError("Backbone caches changed; rebuild the graph")
        pending = []
        for offset, row in enumerate(hidden):
            position = start + offset
            graph._set_position(position)
            if layer_graph.pre_graph is None:
                context = replace(layer_graph.context, kv_len=position + 1)
                values = graph._pre_impl(layer, row, context)
            else:
                layer_graph.input.copy_(row)
                layer_graph.pre_graph.replay()
                values = (
                    layer_graph.after_mixer, layer_graph.mlp_input,
                    layer_graph.topk_ids, layer_graph.topk_weights, layer_graph.shared_output,
                )
            if state_trace is not None:
                state_trace(layer_index, offset, layer)
            # Captured outputs are overwritten by the next time step.
            pending.append(tuple(value.clone() for value in values))
        inputs = torch.cat([values[1].reshape(1, -1) for values in pending])
        ids = torch.cat([values[2] for values in pending])
        weights = torch.cat([values[3] for values in pending])
        if batch_experts:
            outputs = layer.mlp.experts(inputs, ids, weights)
        else:
            outputs = torch.cat([
                layer.mlp.experts(inputs[index:index + 1], ids[index:index + 1],
                                  weights[index:index + 1])
                for index in range(count)
            ])
        next_hidden = []
        for index, values in enumerate(pending):
            after_mixer, _, _, _, shared = values
            if layer_graph.post_graph is None:
                result = graph._post_impl(after_mixer, outputs[index:index + 1], shared)
            else:
                layer_graph.after_mixer.copy_(after_mixer)
                layer_graph.shared_output.copy_(shared)
                layer_graph.post_expert.copy_(
                    outputs[index:index + 1].view_as(layer_graph.post_expert))
                layer_graph.post_graph.replay()
                result = layer_graph.output
            next_hidden.append(result.clone())
        hidden = next_hidden
    norm = graph.backbone.model.norm
    # Preserve the one-row final norm used by ordinary decode.
    return torch.cat([norm(row).to(graph.backbone.dtype) for row in hidden], dim=1)


@dataclass
class DecodeCheckpoint:
    sequence: SequenceSnapshot
    hidden: torch.Tensor
    frames: torch.Tensor
    tower_rows: int
    finished: bool

    @property
    def nbytes(self):
        return self.sequence.nbytes + sum(
            value.numel() * value.element_size() for value in (self.hidden, self.frames))


class DecodeCursor:
    """Own the last hidden/input row and use the production sampling graphs."""

    def __init__(self, model, hidden, frames, tower_graph, backbone_graph):
        self.model, self.hidden, self.frames = model, hidden, frames
        self.tower_graph, self.backbone_graph = tower_graph, backbone_graph
        self.tower_rows = len(frames) - 1
        self.finished = False
        self.prefix_report = {"enabled": False, "hit": False, "reused_rows": 0}

    @property
    def position(self):
        return len(self.frames) - 1

    @classmethod
    @torch.inference_mode()
    def prefill(cls, model, prompt, *, chunk_size=32, tower_graph, backbone_graph,
                temperature=DEFAULT_TEMPERATURE, top_k=DEFAULT_TOP_K, top_p=DEFAULT_TOP_P,
                prefix_cache=None, allow_prefill_fallback=False):
        if chunk_size < 1 or len(prompt) < 1:
            raise ValueError("Prefill requires positive chunks and a nonempty prompt")
        from .generation import prefill_prompt

        tower = model.audio_decoder
        rows = prompt.to(tower.device)
        positions = torch.arange(len(rows), device=tower.device)
        hidden, prefix_report = prefill_prompt(
            model, prompt, rows, positions, chunk_size=chunk_size,
            prefix_cache=prefix_cache, allow_oom_fallback=allow_prefill_fallback,
        )
        tower_graph.start(rows, temperature, top_k, top_p)
        cursor = cls(model, hidden[-1:].clone(), rows.clone(), tower_graph, backbone_graph)
        cursor.prefix_report = prefix_report
        return cursor

    @torch.inference_mode()
    def capture(self, *, storage_device="cpu", include_rng=True):
        return DecodeCheckpoint(
            SequenceSnapshot.capture(
                self.model, backbone_rows=self.position + 1, tower_rows=self.tower_rows,
                tower_graph=self.tower_graph, include_rng=include_rng,
                storage_device=storage_device),
            self.hidden.detach().to(storage_device, copy=True),
            self.frames.detach().to(storage_device, copy=True),
            self.tower_rows, self.finished,
        )

    @torch.inference_mode()
    def restore(self, checkpoint):
        checkpoint.sequence.restore(self.model, tower_graph=self.tower_graph)
        device = self.model.audio_decoder.device
        self.hidden = checkpoint.hidden.to(device, copy=True)
        self.frames = checkpoint.frames.to(device, copy=True)
        self.tower_rows, self.finished = checkpoint.tower_rows, checkpoint.finished

    @torch.inference_mode()
    def step(self):
        if self.finished:
            raise RuntimeError("Cannot advance a finished sequence")
        if self.position + 1 >= self.backbone_graph.sequence_capacity:
            raise ValueError("Decode cursor has no room for another frame")
        position = self.position
        logits = _logits(self.model.backbone, self.hidden)
        sampled, main = self.tower_graph.step(
            logits, self.hidden, position, self.frames[-1:].contiguous())
        self.tower_rows = position + 1
        acoustic = (
            None if self.tower_graph.codebook_logits is None
            else self.tower_graph.codebook_logits.clone())
        if main == self.model.config.eos_token_id:
            self.finished = True
        else:
            self._advance_backbone(sampled)
        return {"frame": sampled, "semantic_logits": logits.clone(), "acoustic_logits": acoustic}

    def _advance_backbone(self, frame):
        tower, backbone = self.model.audio_decoder, self.model.backbone
        embedded = tower.embed_backbone(frame, backbone.embed(frame[:, 0]))
        with _hybrid_scope(self.model):
            self.hidden = self.backbone_graph.forward(
                inputs_embeds=embedded[None], input_pos=self.position + 1)[0].clone()
        self.frames = torch.cat((self.frames, frame))

    @torch.inference_mode()
    def force(self, proposals, *, block=False, batch_experts=True):
        """Teacher-force target logits; intentionally consumes no sampling RNG.

        All proposals must be complete non-EOS audio frames. This is an oracle
        verification primitive, not an acceptance test or an unbiased sampler.
        """
        tower, backbone = self.model.audio_decoder, self.model.backbone
        proposals = proposals.to(tower.device)
        if (self.finished or proposals.ndim != 2 or proposals.shape[1] != self.frames.shape[1]
                or len(proposals) not in (1, 2, 4)
                or proposals.dtype != torch.int64
                or not bool(tower.is_semantic(proposals[:, 0]).all())
                or not torch.equal(proposals[:, 1], proposals[:, 0] - tower.semantic_begin)
                or int(proposals[:, 1:].min()) < 0
                or int(proposals[:, 1:].max()) >= tower.config.vocab_size):
            raise ValueError("Oracle verification requires one/two/four valid complete audio frames")
        start = self.position
        if start + len(proposals) >= self.backbone_graph.sequence_capacity:
            raise ValueError("Oracle block exceeds the sequence capacity")
        if block:
            # Embed each row independently, matching ordinary decode.
            embedded = torch.stack([
                tower.embed_backbone(row[None], backbone.embed(row[None, 0]))[0]
                for row in proposals
            ])[None]
            with _hybrid_scope(self.model):
                next_hidden = backbone_block(
                    self.backbone_graph, embedded, start + 1, batch_experts=batch_experts)[0]
            contexts = [self.hidden] + [row[None] for row in next_hidden[:-1]]
        else:
            contexts = []
        outputs = []
        previous_lengths = [attention.static_kv_len for attention in self.tower_graph.tower_attentions()]
        try:
            for offset, proposed in enumerate(proposals):
                position = start + offset
                hidden = contexts[offset] if block else self.hidden
                logits = _logits(backbone, hidden)
                length = next(n for n in self.tower_graph.buckets if n >= position + 1)
                for attention in self.tower_graph.tower_attentions():
                    attention.static_kv_len = length
                pos = torch.tensor([position], dtype=torch.int64, device=tower.device)
                main, c0 = proposed[None, 0], proposed[None, 1]
                semantic = tower.is_semantic(main)
                current = tower.run_shared(
                    hidden, c0, semantic, pos, self.frames[-1:])
                heads = []
                for index, (layer, head) in enumerate(zip(tower.residual_layers, tower.codebook_heads)):
                    current = layer(current, proposed[None, index + 1], semantic, pos)
                    heads.append(head(current)[0])
                outputs.append({"semantic_logits": logits, "acoustic_logits": torch.stack(heads)})
                self.tower_graph.history.copy_(torch.cat(
                    (self.tower_graph.history[1:], proposed[:1])))
                self.tower_rows = position + 1
                if block:
                    self.frames = torch.cat((self.frames, proposed[None]))
                else:
                    self._advance_backbone(proposed[None])
            if block:
                self.hidden = next_hidden[-1:].clone()
        finally:
            for attention, length in zip(self.tower_graph.tower_attentions(), previous_lengths):
                attention.static_kv_len = length
        return {
            "semantic_logits": torch.cat([row["semantic_logits"] for row in outputs]),
            "acoustic_logits": torch.stack([row["acoustic_logits"] for row in outputs]),
        }
