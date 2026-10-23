"""Bounded sequence snapshots; restore into existing CUDA-graph cache addresses."""

from __future__ import annotations

from dataclasses import dataclass
import random
import weakref

import numpy as np
import torch


def _length(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _bindings(model, backbone_rows, tower_rows):
    """Yield (name, full target, valid view, sequence dimension or None)."""
    _length(backbone_rows, "backbone_rows")
    _length(tower_rows, "tower_rows")
    for index, layer in enumerate(model.backbone.model.layers):
        if layer.layer_type == "full_attention":
            for name in ("k_cache", "v_cache"):
                target = getattr(layer.self_attn, name)
                if target is None or target.shape[2] < backbone_rows:
                    raise ValueError("Allocate sufficient backbone KV caches before snapshot/restore")
                yield f"backbone.{index}.{name}", target, target[:, :, :backbone_rows], 2
        else:
            for name in ("conv_state", "recurrent_state"):
                target = getattr(layer.linear_attn, name)
                if target is None:
                    raise ValueError("Allocate GDN caches before snapshot/restore")
                yield f"backbone.{index}.{name}", target, target, None
    tower = model.audio_decoder
    for index, layer in enumerate((*tower.shared_layers, *tower.residual_layers)):
        attention = layer.transformer_layer.attention
        for name in ("k_cache", "v_cache"):
            target = getattr(attention, name)
            if target is None or target.shape[2] < tower_rows:
                raise ValueError("Allocate sufficient tower KV caches before snapshot/restore")
            yield f"tower.{index}.{name}", target, target[:, :, :tower_rows], 2


def _sampling_bindings(graph):
    if graph is None:
        return []
    if graph.closed:
        raise ValueError("Cannot snapshot a closed tower graph")
    return [(name, getattr(graph, name)) for name in
            ("history", "temperature", "top_k", "top_p")]


@dataclass
class RandomState:
    python: object
    numpy: tuple
    torch_cpu: torch.Tensor
    cuda: dict[int, torch.Tensor]

    @classmethod
    def capture(cls, devices):
        return cls(
            random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
            {index: torch.cuda.get_rng_state(index).clone() for index in devices},
        )

    def restore(self):
        random.setstate(self.python)
        np.random.set_state(self.numpy)
        torch.set_rng_state(self.torch_cpu)
        for index, state in self.cuda.items():
            torch.cuda.set_rng_state(state, index)


@dataclass
class SequenceSnapshot:
    """CPU-backed by default; no model weights or expert caches are copied.

    Snapshots bind to one immutable model instance. Reallocated/larger caches
    are supported when shapes and dtypes match. Prevalidation precedes writes.
    The caller must stop sequence execution while capturing/restoring.
    """

    owner: object
    backbone_rows: int
    tower_rows: int
    buffers: dict[str, torch.Tensor]
    sampling: dict[str, torch.Tensor]
    graph_started: bool | None
    rng: RandomState | None

    @property
    def nbytes(self):
        values = list(self.buffers.values()) + list(self.sampling.values())
        if self.rng is not None:
            values += [self.rng.torch_cpu, *self.rng.cuda.values()]
        return sum(value.numel() * value.element_size() for value in values)

    @staticmethod
    def estimate_bytes(model, backbone_rows, tower_rows):
        return sum(view.numel() * view.element_size()
                   for _, _, view, _ in _bindings(model, backbone_rows, tower_rows))

    @classmethod
    @torch.inference_mode()
    def capture(cls, model, *, backbone_rows, tower_rows, tower_graph=None,
                include_rng=False, storage_device="cpu", max_bytes=None):
        bindings = list(_bindings(model, backbone_rows, tower_rows))
        if tower_graph is not None and tower_graph.tower is not model.audio_decoder:
            raise ValueError("Tower graph belongs to another model")
        sampling = _sampling_bindings(tower_graph)
        devices = {target.device.index for _, target, _, _ in bindings if target.is_cuda}
        for device in devices:
            with torch.cuda.device(device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Cannot snapshot during CUDA graph capture")
                torch.cuda.synchronize(device)
        rng = RandomState.capture(devices) if include_rng else None
        size = sum(view.numel() * view.element_size() for _, _, view, _ in bindings)
        size += sum(value.numel() * value.element_size() for _, value in sampling)
        if rng is not None:
            size += sum(value.numel() * value.element_size()
                        for value in (rng.torch_cpu, *rng.cuda.values()))
        if max_bytes is not None and size > max_bytes:
            raise MemoryError("Sequence snapshot exceeds its byte budget")
        return cls(
            weakref.ref(model), backbone_rows, tower_rows,
            {name: view.detach().to(storage_device, copy=True).contiguous()
             for name, _, view, _ in bindings},
            {name: value.detach().to(storage_device, copy=True).contiguous()
             for name, value in sampling},
            None if tower_graph is None else tower_graph.started,
            rng,
        )

    @torch.inference_mode()
    def restore(self, model, *, tower_graph=None, clear_tail=True, restore_rng=True):
        if self.owner() is not model:
            raise ValueError("Sequence snapshot belongs to another model")
        if tower_graph is not None and tower_graph.tower is not model.audio_decoder:
            raise ValueError("Tower graph belongs to another model")
        bindings = list(_bindings(model, self.backbone_rows, self.tower_rows))
        if [name for name, *_ in bindings] != list(self.buffers):
            raise ValueError("Model cache structure changed")
        for name, _, view, _ in bindings:
            saved = self.buffers[name]
            if saved.dtype != view.dtype or saved.shape != view.shape:
                raise ValueError(f"Snapshot cache shape/dtype mismatch: {name}")
        sampling = _sampling_bindings(tower_graph) if self.graph_started is not None else []
        if self.graph_started is not None and tower_graph is None:
            raise ValueError("This snapshot requires a live tower graph")
        if {name for name, _ in sampling} != set(self.sampling):
            raise ValueError("Tower sampling structure changed")
        for name, target in sampling:
            if target.shape != self.sampling[name].shape or target.dtype != self.sampling[name].dtype:
                raise ValueError("Tower sampling shape/dtype changed")
        devices = {target.device.index for _, target, _, _ in bindings if target.is_cuda}
        for device in devices:
            with torch.cuda.device(device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Cannot restore during CUDA graph capture")
                torch.cuda.synchronize(device)
        for name, target, view, dimension in bindings:
            view.copy_(self.buffers[name])
            if clear_tail and dimension is not None:
                target.narrow(dimension, view.shape[dimension],
                              target.shape[dimension] - view.shape[dimension]).zero_()
        for name, target in sampling:
            target.copy_(self.sampling[name])
        if self.graph_started is not None:
            tower_graph.started = self.graph_started
        for device in devices:
            torch.cuda.synchronize(device)
        if self.rng is not None and restore_rng:
            self.rng.restore()
