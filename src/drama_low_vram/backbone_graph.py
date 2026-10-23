"""Decode-only CUDA-graph segments for the Drama 3 text backbone.

The routed expert call remains outside CUDA graphs because expert residency and
CPU-to-GPU transfers are intentionally dynamic. Each layer is split into:

``input norm -> mixer/cache update -> residual -> post norm/router/shared MLP``
``expert offload -> residual add``

The two static pieces can be replayed without changing the model's official
top-k router or residual ordering. Capture is deliberately opt-in and falls
back to the upstream ``run_layers`` path when CUDA graph capture is unavailable.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn

from fish_speech.models.text2semantic.fish_omni.layers import LayerContext


def _rope_inputs(
    config: Any, capacity: int, *, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Build the same half-split RoPE values as ``FishOmniBackbone.run_layers``."""
    rotary_dim = config.rotary_dim
    exponents = torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
    frequencies = 1.0 / (config.rope_theta ** (exponents / rotary_dim))
    positions = torch.arange(capacity, dtype=torch.float32, device=device)
    angles = positions[:, None] * frequencies[None, :]
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos(), angles.sin()


def _cache_state(layer: nn.Module) -> tuple[tuple[str, Tensor], ...]:
    """Return cloneable recurrent/cache state for capture warmup restoration."""
    if layer.layer_type == "full_attention":
        module = layer.self_attn
        return tuple(
            (name, getattr(module, name).clone())
            for name in ("k_cache", "v_cache")
            if getattr(module, name) is not None
        )
    module = layer.linear_attn
    return tuple(
        (name, getattr(module, name).clone())
        for name in ("conv_state", "recurrent_state")
        if getattr(module, name) is not None
    )


@torch.inference_mode()
def _restore_cache_state(layer: nn.Module, saved: Iterable[tuple[str, Tensor]]) -> None:
    if layer.layer_type == "full_attention":
        module = layer.self_attn
    else:
        module = layer.linear_attn
    for name, value in saved:
        target = getattr(module, name)
        target.copy_(value)


@dataclasses.dataclass
class _LayerGraph:
    layer: nn.Module
    device: torch.device
    context: LayerContext
    input: Tensor | None = None
    after_mixer: Tensor | None = None
    mlp_input: Tensor | None = None
    topk_ids: Tensor | None = None
    topk_weights: Tensor | None = None
    shared_output: Tensor | None = None
    post_expert: Tensor | None = None
    output: Tensor | None = None
    pre_graph: torch.cuda.CUDAGraph | None = None
    post_graph: torch.cuda.CUDAGraph | None = None
    shared_graph: torch.cuda.CUDAGraph | None = None


class BackboneDecodeGraph:
    """Replay graph-captured backbone segments for one-token decode.

    Parameters
    ----------
    backbone:
        A loaded ``FishOmniBackbone`` with decode caches allocated.
    manager:
        Optional expert offload manager. The manager's patched expert module is
        called outside graph replay, so it may freely stage RAM/SSD weights.
    sequence_capacity:
        Fixed attention-cache capacity used by graph replay. Future cache rows
        remain masked by the current position.
    capture:
        When false, ``forward`` delegates to the upstream implementation. This
        is useful for parity tests and for CPU systems.
    strict:
        Raise when capture fails instead of falling back to upstream execution.
    capture_attention:
        Also capture full-attention mixers using a fixed masked cache capacity.
        This is off by default: static SDPA shape can alter floating-point
        rounding compared with the upstream variable-length attention.
    """

    @torch.inference_mode(False)
    @torch.no_grad()
    def __init__(
        self,
        backbone: nn.Module,
        manager: Any | None = None,
        *,
        sequence_capacity: int,
        device: str | torch.device | None = None,
        capture: bool = True,
        strict: bool = False,
        capture_attention: bool = False,
        shared_stream: bool = False,
    ) -> None:
        if isinstance(sequence_capacity, bool) or not isinstance(sequence_capacity, int):
            raise TypeError("sequence_capacity must be an integer")
        if sequence_capacity < 2:
            raise ValueError("sequence_capacity must allow a prompt and a decode position")
        if sequence_capacity > backbone.config.max_position_embeddings:
            raise ValueError("sequence_capacity exceeds the model context limit")
        if getattr(backbone, "max_batch_size", 0) not in (0, 1):
            raise ValueError("BackboneDecodeGraph only supports batch size one")
        self.backbone = backbone
        self.manager = manager
        self.sequence_capacity = int(sequence_capacity)
        self.strict = bool(strict)
        self.capture_attention = bool(capture_attention)
        self.shared_stream = bool(shared_stream)
        self.capture_error: str | None = None
        self.capture_stage = "init"
        self.capture_layer_index: int | None = None
        self.capture_layers_completed = 0
        self.capture_stream_status = "unknown"
        self.capture_cleanup_error: str | None = None
        self.capture_recovered: bool | None = None
        self._closed = False
        self._layers: list[_LayerGraph] = []
        self._capture_pool = None
        self._shared_pool = None
        self._shared_stream = None
        self._shared_ready = None
        self._shared_done = None
        self.device = self._resolve_device(backbone, device)
        self._replay_count = 0
        self._cache_addresses: dict[int, tuple[int, ...]] = {}
        self._input_pos = torch.tensor([1], dtype=torch.long, device=self.device)
        self._rope_cos, self._rope_sin = _rope_inputs(
            backbone.config, self.sequence_capacity, device=self.device
        )
        self._cos = self._rope_cos[1:2].clone()
        self._sin = self._rope_sin[1:2].clone()
        if getattr(backbone, "max_seq_len", 0) == 0:
            backbone.setup_caches(1, self.sequence_capacity)
        elif backbone.max_seq_len < self.sequence_capacity:
            raise ValueError(
                "Backbone caches are smaller than sequence_capacity: "
                f"{backbone.max_seq_len} < {self.sequence_capacity}"
            )
        if capture:
            try:
                self._capture()
            except Exception as exc:
                self.capture_stream_status = self._stream_status()
                self.capture_error = (
                    f"{type(exc).__name__}: {exc}; stage={self.capture_stage}; "
                    f"layer={self.capture_layer_index}; completed={self.capture_layers_completed}; "
                    f"stream={self.capture_stream_status}"
                )
                # CUDAGraph objects retain their private pools until their last
                # Python reference disappears. Drop every partial layer before
                # returning the eager fallback; never leave a half-captured
                # backbone attached to the runtime.
                self._layers.clear()
                self._cache_addresses.clear()
                self._capture_pool = None
                self._shared_pool = None
                self._shared_stream = self._shared_ready = self._shared_done = None
                # Non-strict mode is a valid eager fallback.
                self._closed = False
                try:
                    self._check_cuda_capture_recovery()
                    self.capture_recovered = True
                except Exception as recovery:
                    self.capture_recovered = False
                    self.capture_cleanup_error = f"{type(recovery).__name__}: {recovery}"
                    self._closed = True
                    raise RuntimeError(
                        "Backbone CUDA graph capture failed and CUDA/RNG state did "
                        "not recover. Restart the engine process before retrying. "
                        f"Capture: {self.capture_error}. Recovery: {self.capture_cleanup_error}"
                    ) from exc
                if self.strict:
                    raise

    @staticmethod
    def _resolve_device(
        backbone: nn.Module, requested: str | torch.device | None
    ) -> torch.device:
        if requested is not None:
            device = torch.device(requested)
        else:
            layers = list(backbone.model.layers)
            if not layers:
                raise ValueError("Backbone has no decoder layers")
            device = layers[0].input_layernorm.weight.device
        if device.type != "cuda":
            return device
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA graph decode requested but CUDA is unavailable")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        return device

    @staticmethod
    def _stream_status() -> str:
        try:
            return (
                "capturing"
                if torch.cuda.is_current_stream_capturing()
                else "idle"
            )
        except Exception as exc:
            return f"unknown:{type(exc).__name__}"

    def _check_cuda_capture_recovery(self) -> None:
        if self.device.type != "cuda":
            return
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Current CUDA stream remains in capture mode")
            torch.cuda.synchronize(self.device)
            # A failed capture_end can leave PyTorch's default Philox generator
            # in capture mode even after the CUDA stream itself became idle.
            # Probe an actual RNG operation, preserving seed and offset.
            with torch.random.fork_rng(devices=[self.device.index]):
                torch.empty(1, device=self.device).uniform_()
                torch.cuda.synchronize(self.device)

    @property
    def enabled(self) -> bool:
        return any(graph.pre_graph is not None for graph in self._layers)

    @property
    def layer_count(self) -> int:
        return len(self._layers)

    @property
    def graph_replay_count(self) -> int:
        return self._replay_count

    @property
    def captured_layer_count(self) -> int:
        return sum(graph.pre_graph is not None for graph in self._layers)

    @property
    def description(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "capture_attention": self.capture_attention,
            "captured_layer_count": self.captured_layer_count,
            "total_layer_count": self.layer_count,
            "sequence_capacity": self.sequence_capacity,
            "device": str(self.device),
            "capture_error": self.capture_error,
            "capture_stage": self.capture_stage,
            "capture_layer_index": self.capture_layer_index,
            "capture_layers_completed": self.capture_layers_completed,
            "capture_stream_status": self.capture_stream_status,
            "capture_recovered": self.capture_recovered,
            "capture_cleanup_error": self.capture_cleanup_error,
            "graph_replay_count": self.graph_replay_count,
        }

    def describe(self) -> dict[str, object]:
        """Return a JSON-serializable status snapshot."""
        return self.description

    def _make_context(self, layer: nn.Module) -> LayerContext:
        return LayerContext(
            input_pos=self._input_pos,
            cos=self._cos,
            sin=self._sin,
            kv_len=self.sequence_capacity,
            continue_state=True,
            use_cache=True,
        )

    @staticmethod
    def _pre_impl(
        layer: nn.Module, hidden: Tensor, context: LayerContext
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        after_mixer, mlp_input, topk_ids, topk_weights = BackboneDecodeGraph._route_impl(
            layer, hidden, context)
        shared_output = BackboneDecodeGraph._shared_impl(layer, mlp_input, after_mixer)
        return after_mixer, mlp_input, topk_ids, topk_weights, shared_output

    @staticmethod
    def _route_impl(layer, hidden, context):
        compute_dtype = layer.input_layernorm.weight.dtype
        mixer = (
            layer.self_attn
            if layer.layer_type == "full_attention"
            else layer.linear_attn
        )
        normalized = layer.input_layernorm(hidden)
        mixed = mixer(normalized.to(compute_dtype), context)
        after_mixer = hidden + mixed
        mlp_input = layer.post_attention_layernorm(after_mixer).to(compute_dtype)
        flat = mlp_input.reshape(-1, mlp_input.shape[-1])
        moe = layer.mlp
        probabilities = torch.softmax(moe.gate(flat).float(), dim=-1)
        topk_weights, topk_ids = probabilities.topk(moe.top_k, dim=-1)
        if moe.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(flat.dtype)
        return after_mixer, mlp_input, topk_ids, topk_weights

    @staticmethod
    def _shared_impl(layer, mlp_input, after_mixer):
        flat = mlp_input.reshape(-1, mlp_input.shape[-1])
        shared_gate = torch.sigmoid(layer.mlp.shared_expert_gate(flat))
        return (shared_gate * layer.mlp.shared_expert(flat)).view_as(after_mixer)

    @staticmethod
    def _post_impl(
        after_mixer: Tensor, expert_output: Tensor, shared_output: Tensor
    ) -> Tensor:
        # Match DecoderLayer(x + SparseMoeBlock(experts + shared)) exactly.
        return after_mixer + (expert_output.view_as(after_mixer) + shared_output)

    @staticmethod
    def _cache_addresses_for(layer: nn.Module) -> tuple[int, ...]:
        if layer.layer_type == "full_attention":
            module = layer.self_attn
            names = ("k_cache", "v_cache")
        else:
            module = layer.linear_attn
            names = ("conv_state", "recurrent_state")
        state = [getattr(module, name) for name in names]
        if any(tensor is None for tensor in state):
            raise ValueError("Allocate backbone decode caches before graph capture")
        return tuple(tensor.data_ptr() for tensor in state)

    @torch.inference_mode(False)
    @torch.no_grad()
    def _capture(self) -> None:
        self.capture_stage = "validate"
        if self.device.type != "cuda":
            raise RuntimeError("CUDA graphs require a CUDA backbone device")
        layers = list(self.backbone.model.layers)
        if not layers:
            raise ValueError("Backbone has no decoder layers")
        if any(
            layer.input_layernorm.weight.device != self.device for layer in layers
        ):
            raise ValueError(
                "BackboneDecodeGraph requires every decoder layer on the same CUDA device"
            )
        if self.backbone.residual_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"Unsupported residual dtype {self.backbone.residual_dtype}")
        with torch.cuda.device(self.device):
            device_index = (
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            )
            # Some CUDA operators reserve Philox state even when this segment
            # currently contains no stochastic operation. Keep graph setup from
            # advancing the caller's sampling stream, matching TowerDecodeGraph.
            self._capture_pool = torch.cuda.graph_pool_handle()
            if self.shared_stream:
                # Concurrent graphs cannot alias the same capture pool.
                self._shared_pool = torch.cuda.graph_pool_handle()
                self._shared_stream = torch.cuda.Stream(device=self.device)
                self._shared_ready, self._shared_done = torch.cuda.Event(), torch.cuda.Event()
            rng_context = torch.random.fork_rng(devices=[device_index])
            rng_context.__enter__()
            try:
                self.capture_stage = "synchronize"
                torch.cuda.synchronize(self.device)
                for layer_index, layer in enumerate(layers):
                    self.capture_layer_index = layer_index
                    self.capture_stage = "prepare_layer"
                    context = self._make_context(layer)
                    self._cache_addresses[id(layer)] = self._cache_addresses_for(layer)
                    if layer.layer_type == "full_attention" and not self.capture_attention:
                        self._layers.append(
                            _LayerGraph(layer=layer, device=self.device, context=context)
                        )
                        self.capture_layers_completed = layer_index + 1
                        continue
                    hidden = torch.zeros(
                        1,
                        1,
                        self.backbone.config.hidden_size,
                        dtype=self.backbone.residual_dtype,
                        device=self.device,
                    )
                    saved = _cache_state(layer)
                    self.capture_stage = "warmup"
                    warmup = torch.cuda.Stream(device=self.device)
                    warmup.wait_stream(torch.cuda.current_stream(self.device))
                    try:
                        with torch.cuda.stream(warmup), torch.inference_mode():
                            for _ in range(3):
                                self._pre_impl(layer, hidden, context)
                        torch.cuda.current_stream(self.device).wait_stream(warmup)
                    finally:
                        _restore_cache_state(layer, saved)
                    torch.cuda.synchronize(self.device)
                    captured = _LayerGraph(
                        layer=layer,
                        device=self.device,
                        context=context,
                        input=hidden,
                        after_mixer=torch.empty_like(hidden),
                        mlp_input=torch.empty_like(hidden),
                        topk_ids=torch.empty(
                            (1, layer.mlp.top_k), dtype=torch.long, device=self.device
                        ),
                        topk_weights=torch.empty(
                            (1, layer.mlp.top_k), dtype=hidden.dtype, device=self.device
                        ),
                        shared_output=torch.empty_like(hidden),
                        post_expert=torch.empty(
                            hidden.shape,
                            dtype=layer.input_layernorm.weight.dtype,
                            device=self.device,
                        ),
                        output=torch.empty_like(hidden),
                    )
                    self.capture_stage = "capture_pre"
                    graph = torch.cuda.CUDAGraph()
                    saved = _cache_state(layer)
                    try:
                        with torch.cuda.graph(graph, pool=self._capture_pool):
                            with torch.inference_mode():
                                if self.shared_stream:
                                    (captured.after_mixer, captured.mlp_input,
                                     captured.topk_ids, captured.topk_weights) = self._route_impl(
                                        layer, captured.input, context)
                                else:
                                    (
                                        captured.after_mixer,
                                        captured.mlp_input,
                                        captured.topk_ids,
                                        captured.topk_weights,
                                        captured.shared_output,
                                    ) = self._pre_impl(layer, captured.input, context)
                    finally:
                        _restore_cache_state(layer, saved)
                    captured.pre_graph = graph
                    torch.cuda.synchronize(self.device)
                    if self.shared_stream:
                        shared_graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(shared_graph, pool=self._shared_pool):
                            with torch.inference_mode():
                                captured.shared_output = self._shared_impl(
                                    layer, captured.mlp_input, captured.after_mixer)
                        captured.shared_graph = shared_graph
                        torch.cuda.synchronize(self.device)
                    self.capture_stage = "capture_post"
                    post_graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(post_graph, pool=self._capture_pool):
                        with torch.inference_mode():
                            captured.output = self._post_impl(
                                captured.after_mixer,
                                captured.post_expert,
                                captured.shared_output,
                            )
                    captured.post_graph = post_graph
                    self._layers.append(captured)
                    self.capture_layers_completed = layer_index + 1
                self.capture_stage = "complete"
                torch.cuda.synchronize(self.device)
            finally:
                # ``fork_rng`` restores the caller state even when capture
                # fails. Avoid any CUDA synchronizing calls while a failed
                # capture still owns the stream.
                try:
                    rng_context.__exit__(None, None, None)
                finally:
                    self.capture_stream_status = self._stream_status()

    def _set_position(self, position: int) -> None:
        if not 1 <= position < self.sequence_capacity:
            raise ValueError(
                f"decode position {position} exceeds sequence capacity "
                f"{self.sequence_capacity}"
            )
        self._input_pos.fill_(position)
        self._cos.copy_(self._rope_cos[position : position + 1])
        self._sin.copy_(self._rope_sin[position : position + 1])

    def _fallback(self, hidden: Tensor, position: int) -> Tensor:
        input_pos = torch.tensor([position], dtype=torch.long, device=hidden.device)
        return self.backbone.run_layers(
            hidden,
            input_pos,
            kv_len=position + 1,
            continue_state=position > 0,
            use_cache=True,
        )

    @torch.inference_mode()
    def forward(
        self, inputs_embeds: Tensor, input_pos: int | Tensor
    ) -> Tensor:
        if self._closed:
            raise RuntimeError("BackboneDecodeGraph is closed")
        if inputs_embeds.ndim != 3 or inputs_embeds.shape[:2] != (1, 1):
            raise ValueError("decode graph requires inputs_embeds with shape [1, 1, H]")
        if isinstance(input_pos, Tensor):
            if input_pos.numel() != 1:
                raise ValueError("input_pos must contain one decode position")
            position = int(input_pos.item())
        else:
            if isinstance(input_pos, bool) or not isinstance(input_pos, int):
                raise TypeError("input_pos must be an integer")
            position = input_pos
        if not 1 <= position < self.sequence_capacity:
            raise ValueError("Decode must continue a prompt at a position within capacity")
        if inputs_embeds.shape[-1] != self.backbone.config.hidden_size:
            raise ValueError("inputs_embeds hidden width does not match the backbone")
        hidden = inputs_embeds.to(self.backbone.residual_dtype)
        if not self.enabled:
            return self._fallback(hidden, position)
        hidden = hidden.to(self.device)
        self._set_position(position)
        if self.shared_stream:
            return self._forward_shared(hidden, position)
        for graph in self._layers:
            if self._cache_addresses_for(graph.layer) != self._cache_addresses[id(graph.layer)]:
                raise RuntimeError("Backbone caches were reallocated; recreate BackboneDecodeGraph")
            if graph.pre_graph is None:
                context = dataclasses.replace(graph.context, kv_len=position + 1)
                hidden = graph.layer(hidden, context)
                continue
            graph.input.copy_(hidden)
            graph.pre_graph.replay()
            self._replay_count += 1
            expert_input = graph.mlp_input.reshape(-1, graph.mlp_input.shape[-1])
            expert_module = graph.layer.mlp.experts
            expert_output = expert_module(
                expert_input, graph.topk_ids, graph.topk_weights
            )
            graph.post_expert.copy_(expert_output.view_as(graph.post_expert))
            graph.post_graph.replay()
            self._replay_count += 1
            hidden = graph.output
        norm = self.backbone.model.norm
        return norm(hidden.to(norm.weight.device)).to(self.backbone.dtype)

    def _forward_shared(self, hidden, position):
        """Publish real routes before shared expert work finishes."""
        main = torch.cuda.current_stream(self.device)
        for graph in self._layers:
            if self._cache_addresses_for(graph.layer) != self._cache_addresses[id(graph.layer)]:
                raise RuntimeError("Backbone caches changed; rebuild the graphs")
            if graph.pre_graph is None:
                context = dataclasses.replace(graph.context, kv_len=position + 1)
                after, inputs, ids, weights = self._route_impl(graph.layer, hidden, context)
            else:
                graph.input.copy_(hidden)
                graph.pre_graph.replay()
                after, inputs, ids, weights = (
                    graph.after_mixer, graph.mlp_input, graph.topk_ids, graph.topk_weights)
            self._shared_ready.record(main)
            self._shared_stream.wait_event(self._shared_ready)
            with torch.cuda.stream(self._shared_stream):
                inputs.record_stream(self._shared_stream)
                if graph.shared_graph is None:
                    shared = self._shared_impl(graph.layer, inputs, after)
                else:
                    graph.shared_graph.replay()
                    shared = graph.shared_output
                self._shared_done.record(self._shared_stream)
            expert = graph.layer.mlp.experts(inputs.reshape(-1, inputs.shape[-1]), ids, weights)
            main.wait_event(self._shared_done)
            if graph.post_graph is None:
                hidden = self._post_impl(after, expert, shared)
            else:
                graph.post_expert.copy_(expert.view_as(graph.post_expert))
                graph.post_graph.replay()
                hidden = graph.output
            shared.record_stream(main)
        norm = self.backbone.model.norm
        return norm(hidden.to(norm.weight.device)).to(self.backbone.dtype)

    def close(self) -> None:
        if self._layers and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._closed = True
        self._layers.clear()
        self._cache_addresses.clear()
        self._capture_pool = None
        self._shared_pool = self._shared_stream = self._shared_ready = self._shared_done = None

    __call__ = forward


__all__ = ["BackboneDecodeGraph"]
