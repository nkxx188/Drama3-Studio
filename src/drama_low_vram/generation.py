"""Drama 3 request execution with a resident backbone and optional tower graph."""

from __future__ import annotations

import time
import gc
from contextlib import nullcontext
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import Tensor

from fish_speech.models.text2semantic.fish_omni.generate import ensure_drama3_caches
from fish_speech.models.text2semantic.sampling import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)

from .semantic_head import SemanticHead
from .sampling import DEFAULT_TOP_K


@dataclass
class GenerationResult:
    frames: Tensor
    finish_reason: str
    prefill_seconds: float
    decode_seconds: float
    first_frame_seconds: float | None
    component_host_seconds: dict[str, float] = field(default_factory=dict)
    prefix_cache: dict = field(default_factory=dict)
    control_events: list[dict] = field(default_factory=list)

    def as_dict(self):
        count = len(self.frames)
        total = self.prefill_seconds + self.decode_seconds
        return {
            "frames": count,
            "finish_reason": self.finish_reason,
            "complete": self.finish_reason == "eos",
            "prefill_seconds": self.prefill_seconds,
            "decode_seconds": self.decode_seconds,
            "generation_seconds": total,
            "first_frame_seconds": self.first_frame_seconds,
            "decode_frames_per_second": count / max(self.decode_seconds, 1e-9),
            "component_host_seconds": self.component_host_seconds,
            "prefix_cache": self.prefix_cache,
            "control_events": self.control_events,
            "timing_scope": "Synchronized prefill/decode; component timers are host wall time.",
        }


def _prefill_once(model, prompt, frames, positions, chunk_size, prefix_cache, cancelled,
                  *, limit_rows=None):
    """Restart at row zero or a complete prefix snapshot, never partial state."""
    backbone, tower = model.backbone, model.audio_decoder
    prompt_rows = len(prompt)
    limit_rows = prompt_rows if limit_rows is None else limit_rows
    if not 1 <= limit_rows <= prompt_rows:
        raise ValueError("Prefill limit must fit within the prompt")
    key, boundary, resumed_rows = None, 0, 0
    report = {
        "enabled": prefix_cache is not None and prefix_cache.enabled, "hit": False,
        "reused_rows": 0, "cached_rows": 0, "restore_seconds": 0.0,
        "capture_seconds": 0.0,
    }
    if prefix_cache is not None:
        from .prefix_cache import prefix_execution_key

        key, boundary = prefix_cache.key(
            model, prompt, chunk_size, prefix_execution_key(model))
        snapshot = prefix_cache.get(key) if key is not None else None
        if snapshot is not None:
            started = time.perf_counter()
            snapshot.restore(model)
            report.update(
                hit=True, reused_rows=boundary,
                restore_seconds=time.perf_counter() - started)
            resumed_rows = boundary
    hidden = None
    for start in range(resumed_rows, limit_rows, chunk_size):
        if cancelled is not None and cancelled():
            return None, report
        end = min(start + chunk_size, limit_rows)
        chunk = frames[start:end]
        embeddings = tower.embed_backbone(chunk, backbone.embed(chunk[:, 0]))
        hidden = backbone(
            inputs_embeds=embeddings[None], input_pos=positions[start:end]
        )[0].to(tower.device)
        paired_end = min(end, prompt_rows - 1)
        if paired_end > start:
            tower.prefill(
                hidden[:paired_end - start], positions[start:paired_end],
                frames[start:paired_end], frames[start + 1:paired_end + 1],
            )
        if key is not None and not resumed_rows and end == boundary:
            started = time.perf_counter()
            stored = prefix_cache.capture(key, model, boundary)
            report["capture_seconds"] = time.perf_counter() - started
            report["cached_rows"] = boundary if stored else 0
    # Include asynchronous prefill allocation failures in this bounded retry.
    torch.cuda.synchronize(tower.device)
    return None if hidden is None else hidden[-1:], report


@torch.inference_mode()
def prefill_reference_prefix(model, prompt, *, chunk_size, prefix_cache,
                             cancelled=None, allow_oom_fallback=False):
    """Populate only a reusable system/reference prefix, without decoding speech."""
    if chunk_size < 1 or len(prompt) < 1:
        raise ValueError("Prefill requires positive chunks and a nonempty prompt")
    if prefix_cache is None or not prefix_cache.enabled:
        return {"status": "disabled", "cached_rows": 0}

    from .prefix_cache import prefix_execution_key

    ensure_drama3_caches(model, len(prompt))
    tower = model.audio_decoder
    device = tower.device
    capacity = len(prompt)
    frames = torch.zeros((capacity, prompt.shape[1]), dtype=torch.long, device=device)
    frames[:] = prompt.to(device)
    positions = torch.arange(capacity, device=device)
    torch.cuda.synchronize(device)

    requested, retries = chunk_size, 0
    while True:
        if cancelled is not None and cancelled():
            return {"status": "cancelled", "cached_rows": 0}
        key, boundary = prefix_cache.key(
            model, prompt, chunk_size, prefix_execution_key(model))
        if key is None:
            return {"status": "ineligible", "cached_rows": 0}
        if prefix_cache.peek(key) is not None:
            return {"status": "ready", "hit": True, "reused_rows": boundary,
                    "cached_rows": 0, "boundary": boundary,
                    "chunk_size": chunk_size, "requested_chunk_size": requested,
                    "oom_retries": retries, "_cache_key": key}
        if not prefix_cache.can_capture(model, boundary):
            return {"status": "skipped_memory", "cached_rows": 0, "boundary": boundary}
        try:
            _hidden, report = _prefill_once(
                model, prompt, frames, positions, chunk_size, prefix_cache, cancelled,
                limit_rows=boundary)
        except torch.cuda.OutOfMemoryError:
            if not allow_oom_fallback or chunk_size == 1:
                raise
        else:
            return {
                **report,
                "status": ("ready" if report["hit"] or report["cached_rows"] else
                           "cancelled" if _hidden is None else "skipped_memory"),
                "boundary": boundary,
                "_cache_key": key,
                "chunk_size": chunk_size,
                "requested_chunk_size": requested,
                "oom_retries": retries,
            }
        gc.collect()
        torch.cuda.empty_cache()
        chunk_size = max(1, chunk_size // 2)
        retries += 1


def prefill_prompt(model, prompt, frames, positions, *, chunk_size,
                   prefix_cache=None, cancelled=None, allow_oom_fallback=False):
    """Try the actual prompt; only automatic prefill may halve its row count.

    A prefill from row zero resets the GDN recurrent history and overwrites all
    attended KV rows. Sampling has not started. Replaying a partial failed
    chunk as a continuation would be incorrect.
    """
    if chunk_size < 1 or len(prompt) < 1:
        raise ValueError("Prefill requires positive chunks and a nonempty prompt")
    requested, retries = chunk_size, 0
    while True:
        try:
            hidden, report = _prefill_once(
                model, prompt, frames, positions, chunk_size, prefix_cache, cancelled)
        except torch.cuda.OutOfMemoryError:
            if not allow_oom_fallback or chunk_size == 1:
                raise
        else:
            return hidden, {
                **report, "chunk_size": chunk_size,
                "requested_chunk_size": requested, "oom_retries": retries,
            }
        # Leave the exception scope before reclaiming its temporary tensors.
        gc.collect()
        torch.cuda.empty_cache()
        chunk_size = max(1, chunk_size // 2)
        retries += 1


@torch.inference_mode()
def generate(
    model,
    prompt: Tensor,
    *,
    max_new_tokens: int,
    prefill_chunk_size: int = 32,
    temperature: float = DEFAULT_TEMPERATURE,
    top_k: int = DEFAULT_TOP_K,
    top_p: float = DEFAULT_TOP_P,
    tower_graph=None,
    backbone_graph=None,
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    prefix_cache=None,
    logit_head=None,
    control_token_ids=(),
    candidate_ids=None,
    allow_prefill_fallback: bool = False,
) -> GenerationResult:
    """Preserve the official frame contract, with explicit phase measurements."""
    if max_new_tokens < 1 or prefill_chunk_size < 1 or top_k < 0:
        raise ValueError("Generation and batching limits must be positive")
    if not 0 < temperature < 2 or not 0 < top_p <= 1:
        raise ValueError("Invalid sampling parameters")
    backbone, tower = model.backbone, model.audio_decoder
    prompt_rows = len(prompt)
    capacity = prompt_rows + max_new_tokens
    if prompt_rows < 1 or capacity > model.config.text.max_position_embeddings:
        raise ValueError("Prompt and generation budget exceed the model context")
    ensure_drama3_caches(model, capacity)
    frames = torch.zeros(
        (capacity, prompt.shape[1]), dtype=torch.long, device=tower.device
    )
    frames[:prompt_rows] = prompt.to(tower.device)
    positions = torch.arange(capacity, device=tower.device)
    torch.cuda.synchronize(tower.device)
    started = time.perf_counter()
    hidden, prefix_report = prefill_prompt(
        model, prompt, frames, positions, chunk_size=prefill_chunk_size,
        prefix_cache=prefix_cache, cancelled=cancelled,
        allow_oom_fallback=allow_prefill_fallback,
    )
    prefill_seconds = time.perf_counter() - started
    if hidden is None:
        return GenerationResult(
            frames[:0].cpu(), "cancelled", prefill_seconds, 0.0, None,
            prefix_cache=prefix_report,
        )
    if tower_graph is not None:
        tower_graph.start(frames[:prompt_rows], temperature, top_k, top_p)
    produced, reason, first_frame = 0, "length", None
    control_events = []
    control_ids = frozenset(control_token_ids)
    timings = {"head": 0.0, "tower": 0.0, "backbone": 0.0}
    started = time.perf_counter()
    for step in range(max_new_tokens):
        if cancelled is not None and cancelled():
            reason = "cancelled"
            break
        position = prompt_rows - 1 + step
        stage = time.perf_counter()
        if logit_head is not None:
            logits = logit_head(hidden)
        elif tower_graph is not None and isinstance(backbone.lm_head, SemanticHead):
            logits = backbone.lm_head.compact_logits(hidden)
        else:
            logits = backbone.compute_logits(hidden).to(tower.device)
        timings["head"] += time.perf_counter() - stage
        stage = time.perf_counter()
        if tower_graph is not None:
            sampled, main = tower_graph.step(
                logits, hidden, position, frames[position:position + 1]
            )
        elif candidate_ids is not None:
            from .tower_graph import _decode_candidates

            sampled = _decode_candidates(
                tower, logits, hidden, positions[position:position + 1],
                frames[position:position + 1], frames[:position + 1, 0],
                temperature=temperature, top_k=top_k, top_p=top_p,
                candidates=candidate_ids,
            )
            main = int(sampled[0, 0])
        else:
            sampled = tower.decode_step(
                logits, hidden, positions[position:position + 1],
                frames[position:position + 1], frames[:position + 1, 0],
                temperature=temperature, top_k=top_k, top_p=top_p,
            )
            main = int(sampled[0, 0])
        timings["tower"] += time.perf_counter() - stage
        if main == model.config.eos_token_id:
            reason = "eos"
            break
        if main in control_ids:
            sampled[:, 1:].zero_()
            control_events.append({"row": produced, "token_id": main})
        frames[position + 1] = sampled[0]
        produced += 1
        if first_frame is None:
            first_frame = prefill_seconds + time.perf_counter() - started
        if progress is not None:
            progress(produced, max_new_tokens)
        if step + 1 < max_new_tokens:
            stage = time.perf_counter()
            embeddings = tower.embed_backbone(sampled, backbone.embed(sampled[:, 0]))
            manager = getattr(model, "_drama_low_vram_manager", None)
            hybrid = getattr(manager, "hybrid_decode", None)
            with hybrid.decode() if hybrid is not None else nullcontext():
                if backbone_graph is not None and backbone_graph.enabled:
                    hidden = backbone_graph.forward(
                        inputs_embeds=embeddings[None],
                        input_pos=position + 1,
                    )[0].to(tower.device)
                else:
                    hidden = backbone(
                        inputs_embeds=embeddings[None],
                        input_pos=positions[position + 1:position + 2],
                    )[0].to(tower.device)
            timings["backbone"] += time.perf_counter() - stage
    torch.cuda.synchronize(tower.device)
    decode_seconds = time.perf_counter() - started
    return GenerationResult(
        frames[prompt_rows:prompt_rows + produced].cpu(), reason,
        prefill_seconds, decode_seconds, first_frame, timings, prefix_report, control_events,
    )
