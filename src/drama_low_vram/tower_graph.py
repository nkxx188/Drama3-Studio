"""Capture the official audio tower without capturing the offloaded backbone.

The sampling equations, codebook order and repetition window follow Fish
Speech's ``FishOmniAudioDecoder.decode_step``. The graph removes per-layer
position reads and per-codebook finite-check synchronizations. It keeps one
finite status check after the entire frame, including all codebook heads.
"""

from __future__ import annotations

import traceback
from collections.abc import Iterator, Sequence
from typing import Any

import torch
from torch import Tensor

from fish_speech.models.text2semantic.fish_omni.tower import (
    RAS_TEMPERATURE,
    RAS_TOP_K,
    RAS_TOP_P,
    RAS_WINDOW_FRAMES,
    FishOmniAudioDecoder,
    TowerAttention,
    sample_top_k_top_p,
)


class TowerGraphRecoveryError(RuntimeError):
    """Failed capture left CUDA execution or its random generator unusable."""


def main_token_candidates(tower: FishOmniAudioDecoder) -> Tensor:
    """The official sorted EOS + semantic vocabulary, initialized outside capture."""
    candidates = tower.main_candidates
    if candidates is None or candidates.device != tower.device:
        eos = torch.tensor([tower.eos_id], device=tower.device)
        band = torch.arange(
            tower.semantic_begin, tower.semantic_end + 1, device=tower.device
        )
        candidates = torch.cat((eos, band)).sort().values
        tower.main_candidates = candidates
    return candidates


def tower_graph_buckets(capacity: int) -> tuple[int, ...]:
    """Use a small set of attended lengths, never extending beyond the KV cache."""
    if capacity < 1:
        raise ValueError("tower graph capacity must be positive")
    return tuple(sorted({min(length, capacity) for length in (256, 512, 1024, capacity)}))


def _decode_candidates(
    tower: FishOmniAudioDecoder,
    candidate_logits: Tensor,
    hidden: Tensor,
    positions: Tensor,
    frames: Tensor,
    history: Tensor,
    *,
    temperature: float | Tensor,
    top_k: int | Tensor,
    top_p: float | Tensor,
    finite: Tensor | None = None,
    codebook_logits: Tensor | None = None,
    candidates: Tensor | None = None,
) -> Tensor:
    """Official frame sampler, with optional device-side finite accumulation."""
    candidates = main_token_candidates(tower) if candidates is None else candidates

    def sample(
        logits: Tensor, temperature: float | Tensor, top_k: int | Tensor, top_p: float | Tensor
    ) -> Tensor:
        if finite is not None:
            finite.logical_and_(torch.isfinite(logits).all())
        return sample_top_k_top_p(
            logits, temperature, top_k, top_p, check_finite=finite is None
        )

    first = candidates[sample(candidate_logits, temperature, top_k, top_p)]
    retry = candidates[
        sample_top_k_top_p(
            candidate_logits,
            RAS_TEMPERATURE,
            RAS_TOP_K,
            RAS_TOP_P,
            check_finite=False,
        )
    ]
    repeated = (history[-RAS_WINDOW_FRAMES:] == first[:, None]).any(-1)
    repeated &= tower.is_semantic(first)
    main = torch.where(repeated, retry, first)
    c0 = (main - tower.semantic_begin).clamp(0, tower.config.vocab_size - 1)
    semantic = tower.is_semantic(main)
    hidden = tower.run_shared(hidden, c0, semantic, positions, frames)
    codes = [c0]
    for index, (layer, head) in enumerate(
        zip(tower.residual_layers, tower.codebook_heads)
    ):
        hidden = layer(hidden, codes[-1], semantic, positions)
        logits = head(hidden)
        if codebook_logits is not None:
            codebook_logits[index].copy_(logits[0])
        codes.append(sample(logits, temperature, top_k, top_p))
    return torch.stack((main, *codes), dim=-1)


class TowerDecodeGraph:
    """CUDA graphs for a batch-one official tower and its stochastic sampler.

    Construct before prompt prefill. Capture writes dummy row zero in each KV
    cache; prefill subsequently overwrites all valid prompt rows. Later cache
    rows are masked by the actual position.

    ``compact_logits=True`` accepts logits ordered by ``candidates``. Otherwise
    ``logits_size`` is the backbone vocabulary size. Neither mode moves logits
    to the CPU. ``start`` resets request history; ``step`` takes the current
    backbone hidden state and returns an independent frame plus its main ID.

    Graph and eager random-number streams have the same sampling distribution
    but are not promised to produce the same audio for the same seed.
    """

    def __init__(
        self,
        tower: FishOmniAudioDecoder,
        capacity: int,
        *,
        logits_size: int | None = None,
        logits_dtype: torch.dtype | None = None,
        compact_logits: bool = False,
        buckets: Sequence[int] | None = None,
        record_logits: bool = False,
        candidate_ids: Tensor | None = None,
    ) -> None:
        if tower.device.type != "cuda":
            raise ValueError("tower CUDA graphs require a CUDA audio decoder")
        if capacity < 1:
            raise ValueError("tower graph capacity must be positive")
        selected = tuple(
            sorted(set(tower_graph_buckets(capacity) if buckets is None else buckets))
        )
        if not selected or selected[0] < 1 or selected[-1] != capacity:
            raise ValueError("graph buckets must be positive and end at capacity")
        if any(value > capacity for value in selected):
            raise ValueError("graph bucket exceeds capacity")
        self.tower = tower
        self.capacity = int(capacity)
        self.buckets = selected
        self.compact_logits = compact_logits
        self.closed = False
        self.candidates = main_token_candidates(tower) if candidate_ids is None else candidate_ids
        if (self.candidates.ndim != 1 or len(self.candidates) < 1 or self.candidates.dtype != torch.int64
                or self.candidates.device != tower.device
                or not torch.equal(self.candidates, self.candidates.unique(sorted=True))
                or int(self.candidates.min()) < 0):
            raise ValueError("Tower candidate IDs must be unique sorted int64 on its device")
        base_candidates = main_token_candidates(tower)
        if not bool(torch.isin(base_candidates, self.candidates).all()):
            raise ValueError("Task candidates must retain EOS and every semantic candidate")
        self._cache_identity: tuple[tuple[int, int], ...] = ()
        minimum_logits_size = int(self.candidates.max()) + 1
        if compact_logits:
            if logits_size is not None and logits_size != len(self.candidates):
                raise ValueError("compact logits_size must equal the candidate count")
            columns = len(self.candidates)
        else:
            columns = minimum_logits_size if logits_size is None else logits_size
            if columns < minimum_logits_size:
                raise ValueError("logits_size does not include all semantic and EOS IDs")
        device, dtype = tower.device, tower.codebook_embeddings.weight.dtype
        self.logits = torch.zeros(
            (1, columns), dtype=logits_dtype or dtype, device=device
        )
        self.hidden = torch.zeros((1, tower.config.dim), dtype=dtype, device=device)
        self.position = torch.zeros(1, dtype=torch.long, device=device)
        self.frame = torch.zeros(
            (1, 1 + tower.config.num_codebooks), dtype=torch.long, device=device
        )
        self.sampled = torch.zeros_like(self.frame)
        self.history = torch.full(
            (RAS_WINDOW_FRAMES,), -1, dtype=torch.long, device=device
        )
        self.temperature = torch.ones((), dtype=torch.float32, device=device)
        self.top_k = torch.full((), 0, dtype=torch.long, device=device)
        self.top_p = torch.full((), 0.95, dtype=torch.float32, device=device)
        self.finite = torch.ones((), dtype=torch.bool, device=device)
        self.status = torch.zeros(2, dtype=torch.long, device=device)
        self.codebook_logits = (
            torch.empty(
                (tower.config.num_codebooks - 1, tower.config.vocab_size),
                dtype=dtype,
                device=device,
            )
            if record_logits
            else None
        )
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.started = False
        self._prepare_caches()
        self._capture_with_recovery()

    def _capture_with_recovery(self) -> None:
        try:
            self._capture()
        except Exception as capture_error:
            # Keep partial graphs out of the eager fallback. A failed
            # capture_end may also leave CUDA or its RNG in capture mode.
            self.close()
            traceback.clear_frames(capture_error.__traceback__)
            try:
                self._check_cuda_capture_recovery()
            except Exception as recovery_error:
                raise TowerGraphRecoveryError(
                    "Tower CUDA graph capture failed and CUDA/RNG state did "
                    "not recover. Restart the engine process before retrying. "
                    f"Capture: {type(capture_error).__name__}: {capture_error}. "
                    f"Recovery: {type(recovery_error).__name__}: {recovery_error}"
                ) from capture_error
            # Preserve OutOfMemoryError so callers may select eager execution
            # only after both CUDA execution and actual RNG use were verified.
            raise

    def _check_cuda_capture_recovery(self) -> None:
        device = self.tower.device
        if device.type != "cuda":
            return
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Current CUDA stream remains in capture mode")
            torch.cuda.synchronize(device)
            device_index = (
                device.index if device.index is not None else torch.cuda.current_device()
            )
            with torch.random.fork_rng(devices=[device_index]):
                torch.empty(1, device=device).uniform_()
                torch.cuda.synchronize(device)

    def tower_attentions(self) -> Iterator[TowerAttention]:
        for layer in (*self.tower.shared_layers, *self.tower.residual_layers):
            yield layer.transformer_layer.attention

    def _current_cache_identity(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (id(attention.k_cache), id(attention.v_cache))
            for attention in self.tower_attentions()
        )

    def _prepare_caches(self) -> None:
        for attention in self.tower_attentions():
            cache = attention.k_cache
            if cache is None or cache.shape[2] < self.capacity:
                attention.setup_cache(self.capacity)
        self._cache_identity = self._current_cache_identity()

    def _run(self) -> None:
        logits = self.logits if self.compact_logits else self.logits[:, self.candidates]
        self.finite.fill_(True)
        sampled = _decode_candidates(
            self.tower,
            logits,
            self.hidden,
            self.position,
            self.frame,
            self.history,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            finite=self.finite,
            codebook_logits=self.codebook_logits,
            candidates=self.candidates,
        )
        self.sampled.copy_(sampled)
        self.status[0].copy_(sampled[0, 0])
        self.status[1].copy_(self.finite)

    @torch.inference_mode()
    def _capture(self) -> None:
        device = self.tower.device
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        attentions = list(self.tower_attentions())
        previous_lengths = [attention.static_kv_len for attention in attentions]
        # Warmup and capture must not consume the caller's sampling RNG stream.
        with torch.cuda.device(device), torch.random.fork_rng(devices=[device_index]):
            try:
                for length in self.buckets:
                    for attention in attentions:
                        attention.static_kv_len = length
                    stream = torch.cuda.Stream(device=device)
                    stream.wait_stream(torch.cuda.current_stream(device))
                    with torch.cuda.stream(stream):
                        for _ in range(2):
                            self._run()
                    torch.cuda.current_stream(device).wait_stream(stream)
                    # PyTorch creates the shared CUDA-generator bookkeeping at
                    # capture_begin. It must be an ordinary tensor: a later
                    # independent graph may begin outside inference mode.
                    # Keep tower work in inference mode for existing KV buffers.
                    with torch.inference_mode(False):
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, pool=self.pool):
                            with torch.inference_mode():
                                self._run()
                    self.graphs[length] = graph
            finally:
                for attention, previous in zip(attentions, previous_lengths):
                    attention.static_kv_len = previous

    @torch.inference_mode()
    def start(
        self,
        prompt_frames: Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> None:
        """Reset the RAS history after this request's teacher-forced prefill."""
        if self.closed:
            raise RuntimeError("tower CUDA graphs have been closed")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if prompt_frames.ndim != 2 or len(prompt_frames) < 1:
            raise ValueError("prompt_frames must contain at least one frame row")
        if prompt_frames.shape[1] != self.frame.shape[1]:
            raise ValueError("prompt frame width does not match the audio decoder")
        if self._current_cache_identity() != self._cache_identity:
            raise RuntimeError("tower KV caches changed; rebuild the CUDA graphs")
        self.history.fill_(-1)
        window = prompt_frames[-RAS_WINDOW_FRAMES:, 0].to(self.tower.device)
        self.history[-len(window) :].copy_(window)
        self.temperature.fill_(temperature)
        self.top_k.fill_(top_k)
        self.top_p.fill_(top_p)
        self.started = True

    @torch.inference_mode()
    def step(
        self,
        logits: Tensor,
        hidden: Tensor,
        position: int,
        frame: Tensor,
    ) -> tuple[Tensor, int]:
        """Sample one frame, with one host check for EOS and numerical validity."""
        if self.closed:
            raise RuntimeError("tower CUDA graphs have been closed")
        if not self.started:
            raise RuntimeError("call start after prompt prefill before decoding")
        if not 0 <= position < self.capacity:
            raise ValueError(f"tower position {position} exceeds capacity {self.capacity}")
        if logits.shape != self.logits.shape or hidden.shape != self.hidden.shape:
            raise ValueError("logits or hidden shape does not match the captured inputs")
        if frame.shape != self.frame.shape:
            raise ValueError("frame shape does not match the captured inputs")
        if logits.device != self.tower.device or hidden.device != self.tower.device:
            raise ValueError("tower graph logits and hidden inputs must already be on CUDA")
        if logits.dtype != self.logits.dtype or hidden.dtype != self.hidden.dtype:
            raise ValueError(
                "logits or hidden dtype differs from capture; preserving input precision "
                "requires rebuilding the graph with the correct logits_dtype"
            )
        self.logits.copy_(logits)
        self.hidden.copy_(hidden)
        self.position.fill_(position)
        self.frame.copy_(frame)
        length = next(length for length in self.buckets if length >= position + 1)
        self.graphs[length].replay()
        sampled = self.sampled.clone()
        self.history.copy_(torch.cat((self.history[1:], sampled[0, :1])))
        main, finite = self.status.tolist()
        if not finite:
            raise FloatingPointError("non-finite logits; refusing to sample invalid audio")
        return sampled, main

    def close(self) -> None:
        """Release graphs and request buffers; leave the tower's KV caches intact."""
        self.graphs.clear()
        self.pool = None
        for name in (
            "logits",
            "hidden",
            "position",
            "frame",
            "sampled",
            "history",
            "temperature",
            "top_k",
            "top_p",
            "finite",
            "status",
            "codebook_logits",
        ):
            setattr(self, name, None)
        self.started = False
        self.closed = True

    def description(self) -> dict[str, Any]:
        """Small JSON-compatible runtime diagnostics."""
        return {
            "backend": "cuda_graph",
            "capacity": self.capacity,
            "buckets": list(self.buckets),
            "compact_logits": self.compact_logits,
            "candidate_count": len(self.candidates),
            "finite_check": "main_and_all_codebook_logits",
            "same_seed_matches_eager": False,
            "closed": self.closed,
        }
