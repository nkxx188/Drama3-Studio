"""Bounded RAM-to-GPU expert staging for the upstream Fish Omni backbone.

The checkpoint remains block-FP8 in ordinary CPU memory. By default all layers
share one small staging pool. An optional ``cache_experts`` (or
``hot_cache_experts``) setting adds a bounded FP8 cache owned by each expert
module. Cache hits and cold staged experts can participate in the same
forward, including caches smaller than top-k. The Torch backend
dequantizes one selected expert at a time and does not use the W8A8 activation
quantization of the original Triton backend.
"""

from __future__ import annotations

import threading
import time
import types
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Protocol

import torch
import torch.nn.functional as F
from torch import Tensor, nn

if TYPE_CHECKING:
    from .host_residency import HostResidency

FP8_DTYPE = torch.float8_e4m3fn
FP8_BLOCK_SIZE = 128
EXPERT_BUFFER_NAMES = (
    "w13_weight",
    "w13_scale_inv",
    "w2_weight",
    "w2_scale_inv",
)


def select_fp8(weight: Tensor, index: Tensor | int) -> Tensor:
    """Select FP8 entries through their bytes; Float8 advanced indexing is absent."""
    return weight.view(torch.uint8)[index].view(weight.dtype)


def dequantize_fp8_blocks(
    weight: Tensor, scale_inv: Tensor, dtype: torch.dtype
) -> Tensor:
    """Dequantize one staged projection, without expanding the CPU expert bank."""
    *leading, rows, columns = weight.shape
    row_blocks, column_blocks = scale_inv.shape[-2:]
    if rows != row_blocks * FP8_BLOCK_SIZE or columns != column_blocks * FP8_BLOCK_SIZE:
        raise ValueError("FP8 weight shape does not match its 128x128 block scales")
    blocks = weight.view(
        *leading, row_blocks, FP8_BLOCK_SIZE, column_blocks, FP8_BLOCK_SIZE
    )
    scaled = blocks.float() * scale_inv[..., :, None, :, None]
    return scaled.reshape(*leading, rows, columns).to(dtype)


@dataclass
class OffloadStatistics:
    """Cumulative counts and timings; host and GPU timings can overlap.

    ``host_staging_seconds`` remains the inclusive legacy stage wall time.
    Gather and prior-DMA waits split its main CPU costs; submission includes
    blocking pageable-copy time if pinning is unavailable. ``h2d_seconds`` is
    CUDA-event elapsed time and is collected only with transfer profiling.
    ``h2d_copy_calls`` counts tensor copy submissions, not driver DMA commands.
    """

    forward_calls: int = 0
    combined_decode_calls: int = 0
    staging_calls: int = 0
    packed_staging_calls: int = 0
    h2d_copy_calls: int = 0
    expert_loads: int = 0
    expert_cache_hits: int = 0
    hot_cache_hits: int = 0
    hot_cache_loads: int = 0
    hot_cache_bytes: int = 0
    hot_cache_capacity_experts: int = 0
    staged_bytes: int = 0
    h2d_bytes: int = 0
    host_staging_seconds: float = 0.0
    host_gather_seconds: float = 0.0
    native_gather_calls: int = 0
    torch_gather_calls: int = 0
    dma_source_wait_seconds: float = 0.0
    h2d_submit_seconds: float = 0.0
    routing_host_seconds: float = 0.0
    h2d_seconds: float = 0.0
    cpu_expert_bytes: int = 0
    pinned_bytes: int = 0
    pinned_allocation_fallbacks: int = 0
    device_buffer_bytes: int = 0
    peak_staged_experts: int = 0

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class ExpertStore:
    w13_weight: Tensor
    w13_scale_inv: Tensor
    w2_weight: Tensor
    w2_scale_inv: Tensor

    @classmethod
    def from_module(cls, module: nn.Module) -> ExpertStore:
        quantized = getattr(module, "_drama_quant_store", None)
        if quantized is not None:
            quantized.validate()
            return quantized
        if not getattr(module, "fp8", False):
            raise ValueError("Expert offload requires FP8 buffers, not dense BF16 weights")
        store = cls(**{name: getattr(module, name) for name in EXPERT_BUFFER_NAMES})
        store.validate()
        return store

    @property
    def num_experts(self) -> int:
        return self.w13_weight.shape[0]

    @property
    def hidden(self) -> int:
        return self.w13_weight.shape[2]

    @property
    def intermediate(self) -> int:
        return self.w13_weight.shape[1] // 2

    @property
    def nbytes(self) -> int:
        return sum(
            getattr(self, name).numel() * getattr(self, name).element_size()
            for name in EXPERT_BUFFER_NAMES
        )

    def validate(self) -> None:
        for name in EXPERT_BUFFER_NAMES:
            tensor = getattr(self, name)
            if tensor.device.type != "cpu":
                raise ValueError(f"{name} must stay in CPU memory; got {tensor.device}")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
            expected_dtype = FP8_DTYPE if name.endswith("_weight") else torch.float32
            if tensor.dtype != expected_dtype:
                raise ValueError(f"{name} must be {expected_dtype}; got {tensor.dtype}")
            if tensor.ndim != 3:
                raise ValueError(f"{name} must be a stacked three-dimensional tensor")
        experts, gate_up, hidden = self.w13_weight.shape
        if experts < 1 or gate_up % 2:
            raise ValueError("Expert bank needs at least one expert and even gate/up rows")
        intermediate = gate_up // 2
        if hidden % FP8_BLOCK_SIZE or intermediate % FP8_BLOCK_SIZE:
            raise ValueError("Expert dimensions must be divisible by the FP8 block size")
        expected_shapes = {
            "w2_weight": (experts, hidden, intermediate),
            "w13_scale_inv": (
                experts,
                gate_up // FP8_BLOCK_SIZE,
                hidden // FP8_BLOCK_SIZE,
            ),
            "w2_scale_inv": (
                experts,
                hidden // FP8_BLOCK_SIZE,
                intermediate // FP8_BLOCK_SIZE,
            ),
        }
        for name, shape in expected_shapes.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"{name} has shape {getattr(self, name).shape}, expected {shape}")


@dataclass(frozen=True)
class StagedExperts:
    w13_weight: Tensor
    w13_scale_inv: Tensor
    w2_weight: Tensor
    w2_scale_inv: Tensor

    @property
    def count(self) -> int:
        return self.w13_weight.shape[0]


class ExpertBackend(Protocol):
    def forward(
        self, x: Tensor, ids: Tensor, weights: Tensor, staged: StagedExperts
    ) -> Tensor: ...

    def forward_pairs(
        self, x: Tensor, ids: Tensor, weights: Tensor, staged: StagedExperts
    ) -> Tensor:
        """Return weighted [pairs, hidden] contributions in float32, before summing."""
        ...


class TorchExpertBackend:
    """Portable W16A16 fallback: expand only one staged expert's projections."""

    def forward(
        self, x: Tensor, ids: Tensor, weights: Tensor, staged: StagedExperts
    ) -> Tensor:
        tokens, top_k = ids.shape
        pairs = self.forward_pairs(
            x.repeat_interleave(top_k, dim=0),
            ids.reshape(-1, 1),
            weights.reshape(-1, 1),
            staged,
        )
        return pairs.view(tokens, top_k, x.shape[-1]).sum(dim=1).to(x.dtype)

    def forward_pairs(
        self, x: Tensor, ids: Tensor, weights: Tensor, staged: StagedExperts
    ) -> Tensor:
        if ids.shape != (x.shape[0], 1) or weights.shape != ids.shape:
            raise ValueError("Pair backend expects one expert and weight per input row")
        out = torch.empty(x.shape, device=x.device, dtype=torch.float32)
        host_ids = ids.detach().to("cpu").reshape(-1).tolist()
        groups: dict[int, list[int]] = {}
        for row, expert in enumerate(host_ids):
            groups.setdefault(expert, []).append(row)
        for expert, rows in groups.items():
            if not 0 <= expert < staged.count:
                raise ValueError(f"Local expert slot {expert} is outside the staging pool")
            row_ids = torch.tensor(rows, dtype=torch.int64, device=x.device)
            # Do not gather repeated weight copies for the (token, expert) pairs.
            w13 = dequantize_fp8_blocks(
                select_fp8(staged.w13_weight, expert),
                staged.w13_scale_inv[expert],
                x.dtype,
            )
            w2 = dequantize_fp8_blocks(
                select_fp8(staged.w2_weight, expert),
                staged.w2_scale_inv[expert],
                x.dtype,
            )
            gate, up = F.linear(x.index_select(0, row_ids), w13).chunk(2, dim=-1)
            values = F.linear(F.silu(gate) * up, w2)
            weighted = values.float() * weights.index_select(0, row_ids).float()
            out.index_copy_(0, row_ids, weighted)
            del w13, w2
        return out


class TritonExpertBackend:
    """Reuse Fish's W8A8 kernels, including float32 partial accumulation."""

    def __init__(self, *, decode_specialization: bool = True) -> None:
        from fish_speech.models.text2semantic.fish_omni import fp8_moe

        self.kernel = fp8_moe
        self.decode_specialization = decode_specialization
        self._checked_devices: set[torch.device] = set()
        self._direct_decode_devices: set[torch.device] = set()

    def _check(self, x: Tensor) -> None:
        if x.device not in self._checked_devices:
            if not self.kernel.fp8_gemm_supported([x.device]):
                raise RuntimeError(
                    "The Triton expert backend requires CUDA SM89+ and working Triton. "
                    "Choose backend='torch' explicitly for staged dequantization."
                )
            self._checked_devices.add(x.device)
            if torch.cuda.get_device_capability(x.device) == (8, 9):
                self._direct_decode_devices.add(x.device)

    def forward(
        self, x: Tensor, ids: Tensor, weights: Tensor, staged: StagedExperts
    ) -> Tensor:
        self._check(x)
        if (self.decode_specialization and x.device in self._direct_decode_devices
                and x.shape[0] == 1
                and 1 <= ids.shape[1] <= 8):
            from .fp8_decode import forward

            return forward(self.kernel, x, ids, weights, staged)
        return self.kernel.fp8_moe_forward(
            x,
            ids,
            weights,
            staged.w13_weight,
            staged.w13_scale_inv,
            staged.w2_weight,
            staged.w2_scale_inv,
        )

    def forward_pairs(
        self, x: Tensor, ids: Tensor, weights: Tensor, staged: StagedExperts
    ) -> Tensor:
        self._check(x)
        if ids.shape != (x.shape[0], 1) or weights.shape != ids.shape:
            raise ValueError("Pair backend expects one expert and weight per input row")
        kernel = self.kernel
        pairs, hidden = x.shape
        if pairs == 0:
            return torch.empty_like(x, dtype=torch.float32)
        if pairs <= 64:
            block_m, group_m = 16, 1
        elif pairs <= 1024:
            block_m, group_m = 32, 1
        else:
            block_m, group_m = 64, 8
        with torch.cuda.device(x.device):
            sorted_ids, expert_ids, padded = kernel.moe_align(ids, staged.count, block_m)
            layout = dict(top_k=1, block_m=block_m, group_m=group_m)
            x_q, x_scale = kernel.quantize_groups(
                kernel.quant_kernel, x.contiguous(), hidden
            )
            gate_up = torch.empty(
                (pairs, staged.w13_weight.shape[1]), device=x.device, dtype=x.dtype
            )
            kernel.fused_moe_gemm(
                x_q,
                x_scale,
                staged.w13_weight,
                staged.w13_scale_inv,
                gate_up,
                None,
                sorted_ids,
                expert_ids,
                padded,
                **layout,
            )
            h_q, h_scale = kernel.quantize_groups(
                kernel.silu_mul_quant_kernel, gate_up, gate_up.shape[-1] // 2
            )
            down = torch.empty((pairs, hidden), device=x.device, dtype=torch.float32)
            kernel.fused_moe_gemm(
                h_q,
                h_scale,
                staged.w2_weight,
                staged.w2_scale_inv,
                down,
                weights.reshape(-1).contiguous(),
                sorted_ids,
                expert_ids,
                padded,
                **layout,
            )
        # The upstream forward casts after reducing all top-k contributions.
        # Keeping these partials in FP32 preserves that behavior across groups.
        return down

    def forward_decode_pairs(self, x, ids, weights, staged):
        """Small partition of one original decode row; FP32 partials retained."""
        self._check(x)
        if (self.decode_specialization and x.device in self._direct_decode_devices
                and 1 <= x.shape[0] <= 8):
            from .fp8_decode import quantized_pairs

            q, scales = self.kernel.quantize_groups(
                self.kernel.quant_kernel, x.contiguous(), x.shape[-1])
            return quantized_pairs(self.kernel, q, scales, ids, weights, staged, dtype=x.dtype)
        return self.forward_pairs(x, ids, weights, staged)


class _StagingPool:
    def __init__(
        self,
        capacity: int,
        pin_memory: bool,
        profile_transfers: bool,
        stats: OffloadStatistics,
        *,
        packed_staging: bool = False,
    ) -> None:
        self.capacity = capacity
        self.pin_memory = pin_memory
        self.pin_memory_error: str | None = None
        self.profile_transfers = profile_transfers
        self.packed_staging = packed_staging
        self.stats = stats
        self._lock = threading.Lock()
        self._host: dict[str, Tensor] = {}
        self._device: dict[str, Tensor] = {}
        self._host_arena: Tensor | None = None
        self._device_arena: Tensor | None = None
        self._target: torch.device | None = None
        self._shape: tuple[int, int] | None = None
        self._loaded_ids: Tensor | None = None
        self._loaded_store_key: tuple[int, ...] | None = None
        self._copy_start: torch.cuda.Event | None = None
        self._copy_done: torch.cuda.Event | None = None
        # Pending references clear after DMA; these handles survive for reuse.
        self._copy_start_event: torch.cuda.Event | None = None
        self._copy_done_event: torch.cuda.Event | None = None
        self._compute_done: torch.cuda.Event | None = None
        self._compute_stream_key: tuple[torch.device, int] | None = None
        self.host_residency: HostResidency | None = None
        self.host_store_layers: dict[tuple[int, ...], int] = {}
        self.native_gather = None
        self.native_gather_min_experts = 1

    def _allocate_packed(
        self, store: ExpertStore, device: torch.device, *, pin_memory: bool = False
    ) -> tuple[dict[str, Tensor], Tensor]:
        role_bytes = {
            name: getattr(store, name)[0].numel()
            * getattr(store, name).element_size()
            for name in EXPERT_BUFFER_NAMES
        }
        expert_bytes = sum(role_bytes.values())
        options = {"device": device}
        if device.type == "cpu":
            options["pin_memory"] = pin_memory
        arena = torch.empty(
            (self.capacity, expert_bytes), dtype=torch.uint8, **options
        )
        buffers = {}
        offset = 0
        for name in EXPERT_BUFFER_NAMES:
            source = getattr(store, name)
            buffers[name] = (
                arena.narrow(1, offset, role_bytes[name])
                .view(source.dtype)
                .view(self.capacity, *source.shape[1:])
            )
            offset += role_bytes[name]
        return buffers, arena

    def _allocate(self, store: ExpertStore, device: torch.device) -> None:
        shape = (store.hidden, store.intermediate)
        if self._target is not None:
            if self._target != device or self._shape != shape:
                raise ValueError(
                    "One offload manager supports one compute device and one expert shape"
                )
            return
        pinned = self.pin_memory and device.type == "cuda"

        def allocate_host(pin: bool) -> tuple[dict[str, Tensor], Tensor | None]:
            if self.packed_staging:
                return self._allocate_packed(
                    store, torch.device("cpu"), pin_memory=pin
                )
            return {
                name: torch.empty(
                    (self.capacity, *getattr(store, name).shape[1:]),
                    dtype=getattr(store, name).dtype,
                    device="cpu",
                    pin_memory=pin,
                )
                for name in EXPERT_BUFFER_NAMES
            }, None

        try:
            host_buffers, host_arena = allocate_host(pinned)
        except RuntimeError as exc:
            message = str(exc).lower()
            pin_memory_failure = any(
                marker in message
                for marker in (
                    "pin_memory",
                    "pin memory",
                    "pinned",
                    "page-locked",
                    "page locked",
                    "lock host",
                    "hostalloc",
                    "host allocation",
                    "host-lock",
                    "out of memory",
                    "allocation failed",
                )
            )
            if (
                not pinned
                or not pin_memory_failure
            ):
                raise
            # Windows locking/commit limits may reject pinned host memory while
            # ordinary RAM still works. Keep the same bytes and use blocking DMA.
            host_buffers, host_arena = allocate_host(False)
            self.pin_memory = pinned = False
            self.pin_memory_error = f"{type(exc).__name__}: {exc}"
            self.stats.pinned_allocation_fallbacks += 1
        if self.packed_staging:
            device_buffers, device_arena = self._allocate_packed(store, device)
        else:
            device_buffers = {
                name: torch.empty(
                    (self.capacity, *getattr(store, name).shape[1:]),
                    dtype=getattr(store, name).dtype,
                    device=device,
                )
                for name in EXPERT_BUFFER_NAMES
            }
            device_arena = None
        self._host = host_buffers
        self._device = device_buffers
        self._host_arena = host_arena
        self._device_arena = device_arena
        self._target = device
        self._shape = shape
        buffer_bytes = sum(t.numel() * t.element_size() for t in self._device.values())
        self.stats.device_buffer_bytes = buffer_bytes
        self.stats.pinned_bytes = buffer_bytes if pinned else 0

    def _finish_copy(self) -> None:
        if self._copy_done is None:
            return
        # A pinned source must not be overwritten while a previous DMA reads it.
        started = time.perf_counter()
        self._copy_done.synchronize()
        self.stats.dma_source_wait_seconds += time.perf_counter() - started
        if self._copy_start is not None:
            self.stats.h2d_seconds += self._copy_start.elapsed_time(self._copy_done) / 1000
        self._copy_start = None
        self._copy_done = None

    @contextmanager
    def session(self, store: ExpertStore, device: torch.device) -> Iterator[None]:
        with self._lock:
            self._allocate(store, device)
            if device.type == "cuda":
                stream = torch.cuda.current_stream(device)
                stream_key = (stream.device, stream.cuda_stream)
                if (
                    self._compute_done is not None
                    and self._compute_stream_key != stream_key
                ):
                    stream.wait_event(self._compute_done)
                buffers = (
                    (self._device_arena,)
                    if self._device_arena is not None
                    else self._device.values()
                )
                for tensor in buffers:
                    tensor.record_stream(stream)
            try:
                yield
            finally:
                if device.type == "cuda":
                    with torch.cuda.device(device):
                        if self._compute_done is None:
                            self._compute_done = torch.cuda.Event()
                        stream = torch.cuda.current_stream(device)
                        self._compute_done.record(stream)
                        self._compute_stream_key = (stream.device, stream.cuda_stream)

    def stage(self, store: ExpertStore, expert_ids: Tensor) -> StagedExperts:
        if expert_ids.device.type != "cpu" or expert_ids.dtype != torch.int64:
            raise ValueError("Staging IDs must be an int64 CPU tensor")
        count = expert_ids.numel()
        if not 0 < count <= self.capacity:
            raise ValueError(f"Cannot stage {count} experts in {self.capacity} slots")
        target_device = self._target
        store_key = tuple(id(getattr(store, name)) for name in EXPERT_BUFFER_NAMES)
        if (
            self._loaded_ids is not None
            and self._loaded_store_key == store_key
            and target_device is not None
            and torch.equal(self._loaded_ids, expert_ids)
        ):
            self.stats.expert_cache_hits += count
            return StagedExperts(
                **{name: self._device[name][:count] for name in EXPERT_BUFFER_NAMES}
            )
        started = time.perf_counter()
        self._finish_copy()
        gather_started = time.perf_counter()
        if self.host_residency is not None and self.host_residency.enabled:
            layer = self.host_store_layers.get(store_key)
            if layer is None:
                raise RuntimeError("Expert buffers changed; reconfigure host residency")
            self.host_residency.copy_to(
                layer, expert_ids, self._host, observe=False
            )
        elif (
            self.native_gather is not None
            and count >= self.native_gather_min_experts
            and not self.packed_staging
        ):
            self.native_gather(
                store, expert_ids,
                {name: target[:count] for name, target in self._host.items()},
            )
            self.stats.native_gather_calls += 1
        else:
            for name in EXPERT_BUFFER_NAMES:
                source = getattr(store, name)
                target = self._host[name][:count]
                if source.dtype == FP8_DTYPE:
                    torch.index_select(
                        source.view(torch.uint8),
                        0,
                        expert_ids,
                        out=target.view(torch.uint8),
                    )
                else:
                    torch.index_select(source, 0, expert_ids, out=target)
            self.stats.torch_gather_calls += 1
        self.stats.host_gather_seconds += time.perf_counter() - gather_started
        if target_device is None:
            raise RuntimeError("Use the staging pool inside a session")
        copy_started = time.perf_counter()
        copy_stream = (
            torch.cuda.current_stream(target_device)
            if target_device.type == "cuda"
            else None
        )
        if target_device.type == "cuda" and self.profile_transfers:
            with torch.cuda.device(target_device):
                if self._copy_start_event is None:
                    self._copy_start_event = torch.cuda.Event(enable_timing=True)
                self._copy_start = self._copy_start_event
                self._copy_start.record(copy_stream)
        if self.packed_staging:
            if self._host_arena is None or self._device_arena is None:
                raise RuntimeError("Packed staging arenas were not allocated")
            source = self._host_arena[:count]
            self._device_arena[:count].copy_(
                source,
                non_blocking=target_device.type == "cuda" and source.is_pinned(),
            )
            self.stats.packed_staging_calls += 1
            copy_calls = 1
        else:
            for name in EXPERT_BUFFER_NAMES:
                source = self._host[name][:count]
                target = self._device[name][:count]
                if source.dtype == FP8_DTYPE:
                    source, target = source.view(torch.uint8), target.view(torch.uint8)
                target.copy_(
                    source,
                    non_blocking=target_device.type == "cuda" and source.is_pinned(),
                )
            copy_calls = len(EXPERT_BUFFER_NAMES)
        if target_device.type == "cuda":
            with torch.cuda.device(target_device):
                if self._copy_done_event is None:
                    self._copy_done_event = torch.cuda.Event(
                        enable_timing=self.profile_transfers
                    )
                self._copy_done = self._copy_done_event
                self._copy_done.record(copy_stream)
            self.stats.h2d_submit_seconds += time.perf_counter() - copy_started
            self.stats.h2d_copy_calls += copy_calls
        copied_bytes = count * store.nbytes // store.num_experts
        self.stats.staging_calls += 1
        self.stats.expert_loads += count
        self.stats.staged_bytes += copied_bytes
        if target_device.type == "cuda":
            self.stats.h2d_bytes += copied_bytes
        self.stats.host_staging_seconds += time.perf_counter() - started
        self.stats.peak_staged_experts = max(self.stats.peak_staged_experts, count)
        self._loaded_ids = expert_ids.clone()
        self._loaded_store_key = store_key
        return StagedExperts(**{name: self._device[name][:count] for name in EXPERT_BUFFER_NAMES})

    def combine_resident(
        self,
        cold_count: int,
        resident: StagedExperts,
        resident_slots: list[int],
    ) -> StagedExperts:
        """Append GPU-resident experts after the already-staged cold prefix."""
        count = cold_count + len(resident_slots)
        if cold_count < 1 or not resident_slots or count > self.capacity:
            raise ValueError("Combined expert bank must fit the staging capacity")
        if self._target is None or not self._device:
            raise RuntimeError("Stage cold experts before combining the bank")
        slots = torch.tensor(
            resident_slots, dtype=torch.int64, device=self._target
        )
        for name in EXPERT_BUFFER_NAMES:
            source = getattr(resident, name)
            target = self._device[name][cold_count:count]
            if source.dtype == FP8_DTYPE:
                source, target = source.view(torch.uint8), target.view(torch.uint8)
            torch.index_select(source, 0, slots, out=target)
        self.stats.peak_staged_experts = max(
            self.stats.peak_staged_experts, count
        )
        return StagedExperts(
            **{name: self._device[name][:count] for name in EXPERT_BUFFER_NAMES}
        )

    def synchronize(self) -> None:
        with self._lock:
            self._finish_copy()
            if self._compute_done is not None:
                self._compute_done.synchronize()

    def close(self) -> None:
        self.synchronize()
        with self._lock:
            self._host.clear()
            self._device.clear()
            self._host_arena = None
            self._device_arena = None
            self._target = None
            self._shape = None
            self._loaded_ids = None
            self._loaded_store_key = None
            self._copy_start_event = None
            self._copy_done_event = None
            self._compute_done = None
            self._compute_stream_key = None
            self.stats.pinned_bytes = 0
            self.stats.device_buffer_bytes = 0


class _LayerHotCache:
    """A bounded GPU expert cache owned by one MoE layer.

    The cache stores the original FP8/scales, not dequantized weights. It is
    deliberately separate per module: expert id ``7`` in two layers never
    aliases the same cache slot or weights.
    """

    def __init__(
        self, capacity: int, stats: OffloadStatistics, pool: _StagingPool
    ) -> None:
        self.capacity = capacity
        self.stats = stats
        self.pool = pool
        self._device: dict[str, Tensor] = {}
        self._target: torch.device | None = None
        self._shape: tuple[int, int] | None = None
        self._store_key: tuple[int, ...] | None = None
        self._loaded: dict[int, int] = {}
        self._slot_ids: list[int] = [-1] * capacity
        self._next_slot = 0

    def _allocate(self, store: ExpertStore, device: torch.device) -> None:
        shape = (store.hidden, store.intermediate)
        store_key = tuple(id(getattr(store, name)) for name in EXPERT_BUFFER_NAMES)
        if self._target is not None:
            if (
                self._target != device
                or self._shape != shape
                or self._store_key != store_key
            ):
                self.clear()
            else:
                return
        device_buffers = {
            name: torch.empty(
                (self.capacity, *getattr(store, name).shape[1:]),
                dtype=getattr(store, name).dtype,
                device=device,
            )
            for name in EXPERT_BUFFER_NAMES
        }
        self._device = device_buffers
        self._target = device
        self._shape = shape
        self._store_key = store_key
        self.stats.hot_cache_bytes += sum(
            tensor.numel() * tensor.element_size() for tensor in self._device.values()
        )
        self.stats.hot_cache_capacity_experts += self.capacity

    def _staged(self, device: torch.device) -> StagedExperts:
        if device.type == "cuda":
            for tensor in self._device.values():
                tensor.record_stream(torch.cuda.current_stream(device))
        return StagedExperts(
            **{name: self._device[name] for name in EXPERT_BUFFER_NAMES}
        )

    def lookup(
        self, store: ExpertStore, expert_ids: Tensor, device: torch.device
    ) -> tuple[StagedExperts, dict[int, int]]:
        """Read existing slots without bringing prompt-only experts into the cache."""
        self._allocate(store, device)
        slots = {
            expert: self._loaded[expert]
            for expert in expert_ids.tolist()
            if expert in self._loaded
        }
        self.stats.hot_cache_hits += len(slots)
        return self._staged(device), slots

    def _slot_for(self, expert: int, required: set[int]) -> int:
        existing = self._loaded.get(expert)
        if existing is not None:
            return existing
        for slot, current in enumerate(self._slot_ids):
            if current < 0:
                return slot
        for offset in range(self.capacity):
            slot = (self._next_slot + offset) % self.capacity
            if self._slot_ids[slot] not in required:
                return slot
        raise RuntimeError("Hot cache has no evictable expert slot")

    def prepare(
        self, store: ExpertStore, expert_ids: Tensor, device: torch.device
    ) -> tuple[StagedExperts, dict[int, int]]:
        if expert_ids.device.type != "cpu" or expert_ids.dtype != torch.int64:
            raise ValueError("Hot-cache IDs must be an int64 CPU tensor")
        count = expert_ids.numel()
        if not 0 < count <= self.capacity:
            raise ValueError(f"Cannot cache {count} experts in {self.capacity} slots")
        self._allocate(store, device)
        ordered_ids = list(dict.fromkeys(expert_ids.tolist()))
        required = set(ordered_ids)
        slots: dict[int, int] = {}
        missing: list[int] = []
        for expert in ordered_ids:
            slot = self._loaded.get(expert)
            if slot is None:
                missing.append(expert)
            else:
                slots[expert] = slot
                self.stats.hot_cache_hits += 1
        for start in range(0, len(missing), self.pool.capacity):
            group = missing[start : start + self.pool.capacity]
            group_ids = torch.tensor(group, dtype=torch.int64)
            staged = self.pool.stage(store, group_ids)
            group_slots = []
            for offset, expert in enumerate(group):
                slot = self._slot_for(expert, required)
                previous = self._slot_ids[slot]
                if previous >= 0:
                    self._loaded.pop(previous, None)
                self._slot_ids[slot] = expert
                self._loaded[expert] = slot
                slots[expert] = slot
                group_slots.append(slot)
            slot_indices = torch.tensor(
                group_slots, dtype=torch.int64, device=device
            )
            for name in EXPERT_BUFFER_NAMES:
                source = getattr(staged, name)[: len(group)]
                target = self._device[name]
                if source.dtype == FP8_DTYPE:
                    target.view(torch.uint8).index_copy_(
                        0, slot_indices, source.view(torch.uint8)
                    )
                else:
                    target.index_copy_(0, slot_indices, source)
            self.stats.hot_cache_loads += len(group)
            self._next_slot = (group_slots[-1] + 1) % self.capacity
        return self._staged(device), slots

    def adopt_from_bank(
        self,
        bank: StagedExperts,
        bank_slots: dict[int, int],
        expert_ids: list[int],
        device: torch.device,
    ) -> None:
        """Populate selected cold cache entries from a complete GPU bank."""
        required = set(expert_ids)
        if not required or len(required) > self.capacity:
            raise ValueError("Selected cache experts exceed hot-cache capacity")
        missing = [expert for expert in expert_ids if expert not in self._loaded]
        if not missing:
            return
        target_slots = []
        for expert in missing:
            slot = self._slot_for(expert, required)
            previous = self._slot_ids[slot]
            if previous >= 0:
                self._loaded.pop(previous, None)
            self._slot_ids[slot] = expert
            self._loaded[expert] = slot
            target_slots.append(slot)
        source_slots = torch.tensor(
            [bank_slots[expert] for expert in missing],
            dtype=torch.int64,
            device=device,
        )
        target_slots_tensor = torch.tensor(
            target_slots, dtype=torch.int64, device=device
        )
        for name in EXPERT_BUFFER_NAMES:
            source = getattr(bank, name)
            target = self._device[name]
            if source.dtype == FP8_DTYPE:
                source, target = source.view(torch.uint8), target.view(torch.uint8)
            # Both tensors stay on the compute stream. The pool's session
            # completion event protects these copies before its next reuse.
            target.index_copy_(
                0,
                target_slots_tensor,
                source.index_select(0, source_slots),
            )
        self.stats.hot_cache_loads += len(missing)
        self._next_slot = (target_slots[-1] + 1) % self.capacity

    def clear(self) -> None:
        allocated = bool(self._device)
        if allocated:
            self.stats.hot_cache_bytes -= sum(
                tensor.numel() * tensor.element_size() for tensor in self._device.values()
            )
            self.stats.hot_cache_capacity_experts -= self.capacity
        self._device.clear()
        self._target = None
        self._shape = None
        self._store_key = None
        self._loaded.clear()
        self._slot_ids = [-1] * self.capacity
        self._next_slot = 0


class ExpertOffloadManager:
    """Install one manager per backbone; modules retain their original buffer keys."""

    def __init__(
        self,
        *,
        backend: str | ExpertBackend = "triton",
        max_staged_experts: int = 8,
        max_pair_rows: int = 64,
        cache_experts: int = 0,
        hot_cache_experts: int | None = None,
        pin_memory: bool = True,
        profile_transfers: bool = False,
        combined_decode_bank: bool = False,
        packed_staging: bool = False,
    ) -> None:
        if hot_cache_experts is not None:
            if cache_experts not in (0, hot_cache_experts):
                raise ValueError(
                    "cache_experts and hot_cache_experts must match when both are set"
                )
            cache_experts = hot_cache_experts
        if max_staged_experts < 1 or max_pair_rows < 1 or cache_experts < 0:
            raise ValueError(
                "Staging capacity and pair row limit must be positive; "
                "cache_experts cannot be negative"
            )
        if not isinstance(combined_decode_bank, bool):
            raise TypeError("combined_decode_bank must be a boolean")
        if not isinstance(packed_staging, bool):
            raise TypeError("packed_staging must be a boolean")
        if backend == "triton":
            self.backend: ExpertBackend = TritonExpertBackend()
        elif backend == "torch":
            self.backend = TorchExpertBackend()
        elif isinstance(backend, str):
            raise ValueError(f"Unknown expert backend {backend!r}")
        else:
            self.backend = backend
        self.max_staged_experts = max_staged_experts
        self.max_pair_rows = max_pair_rows
        self.cache_experts = cache_experts
        self.combined_decode_bank = combined_decode_bank
        self.packed_staging = packed_staging
        self.stats = OffloadStatistics()
        self.pool = _StagingPool(
            max_staged_experts,
            pin_memory,
            profile_transfers,
            self.stats,
            packed_staging=packed_staging,
        )
        self.modules: dict[str, nn.Module] = {}
        self._hot_caches: dict[int, _LayerHotCache] = {}
        self._host_residency: HostResidency | None = None
        self._host_module_layers: dict[int, int] = {}
        self._host_store_keys: tuple[tuple[int, ...], ...] = ()
        self._closed = False
        self.hybrid_decode = None
        self.host_gather_mode = "torch"
        self.host_gather_error = None

    def configure_host_gather(self, mode="auto"):
        """Select an optional byte-copy C ABI; retain Torch for small groups."""
        if mode not in ("auto", "torch", "native"):
            raise ValueError("host_gather must be auto, torch or native")
        if self._closed:
            raise RuntimeError("This expert offload manager has been closed")
        gather = None
        error = None
        if mode != "torch":
            from .native_backend import NativeLibrary

            try:
                library = NativeLibrary()
                if library.gather_rows is None:
                    raise RuntimeError("Native gather extension unavailable")
                gather = library.gather_experts
            except (FileNotFoundError, OSError, RuntimeError) as exc:
                if mode == "native":
                    raise
                error = f"{type(exc).__name__}: {exc}"
        with self.pool._lock:
            self.pool.native_gather = gather
            self.pool.native_gather_min_experts = 4 if mode == "auto" else 1
            self.host_gather_mode = mode
            self.host_gather_error = error
        return {
            "requested": mode, "native_available": gather is not None,
            "native_min_experts": self.pool.native_gather_min_experts,
            "fallback": error,
        }

    def attach(self, name: str, module: nn.Module) -> None:
        if self._closed:
            raise RuntimeError("This expert offload manager has been closed")
        if hasattr(module, "_drama_offload_original_forward") or name in self.modules:
            raise ValueError(f"Expert module {name!r} already has an offload manager")
        if self._host_residency is not None:
            raise ValueError("Attach all expert modules before configuring host residency")
        store = ExpertStore.from_module(module)
        self.stats.cpu_expert_bytes += store.nbytes
        module._drama_offload_original_forward = module.forward
        manager = self

        def forward(
            experts: nn.Module, x: Tensor, topk_ids: Tensor, topk_weights: Tensor
        ) -> Tensor:
            return manager.forward(experts, x, topk_ids, topk_weights)

        module.forward = types.MethodType(forward, module)
        self.modules[name] = module

    @torch.no_grad()
    def forward(
        self, module: nn.Module, x: Tensor, topk_ids: Tensor, topk_weights: Tensor
    ) -> Tensor:
        if self._closed:
            raise RuntimeError("This expert offload manager has been closed")
        if x.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Dynamic expert offload cannot run inside a CUDA graph")
        store = ExpertStore.from_module(module)
        if x.ndim != 2 or x.shape[-1] != store.hidden:
            raise ValueError(f"Expected hidden rows [N, {store.hidden}], got {tuple(x.shape)}")
        if topk_ids.ndim != 2 or topk_ids.shape[0] != x.shape[0]:
            raise ValueError("Expert IDs must have shape [tokens, top_k]")
        if topk_weights.shape != topk_ids.shape or topk_ids.shape[1] < 1:
            raise ValueError("Routing weights must match IDs and top_k must be positive")
        if topk_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Expert IDs must have an integer dtype")
        if topk_ids.device != x.device or topk_weights.device != x.device:
            raise ValueError("Hidden states, expert IDs and routing weights must share a device")
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("Expert compute dtype must be float16, bfloat16 or float32")
        if not topk_weights.is_floating_point():
            raise ValueError("Routing weights must be floating point")
        self.stats.forward_calls += 1
        if x.shape[0] == 0:
            return torch.zeros_like(x)
        if self.hybrid_decode is not None and self.hybrid_decode.eligible(module, x, topk_ids):
            return self.hybrid_decode.forward(module, store, x, topk_ids, topk_weights)
        routing_started = time.perf_counter()
        host_ids = topk_ids.detach().to(device="cpu", dtype=torch.int64)
        if int(host_ids.min()) < 0 or int(host_ids.max()) >= store.num_experts:
            raise ValueError(f"Expert IDs must be in [0, {store.num_experts})")
        unique = torch.unique(host_ids, sorted=True)
        self.stats.routing_host_seconds += time.perf_counter() - routing_started
        if self._host_residency is not None and self._host_residency.budget_bytes:
            layer = self._host_module_layers.get(id(module))
            store_key = tuple(id(getattr(store, name)) for name in EXPERT_BUFFER_NAMES)
            if layer is None or self._host_store_keys[layer] != store_key:
                raise RuntimeError("Expert buffers changed; reconfigure host residency")
            self._host_residency.observe(layer, host_ids)
        with self.pool.session(store, x.device):
            if self.cache_experts:
                return self._forward_hot_cached(
                    module, x, host_ids, topk_weights, store, unique
                )
            if unique.numel() <= self.max_staged_experts:
                return self._forward_resident(x, host_ids, topk_weights, store, unique)
            return self._forward_partitioned(x, host_ids, topk_weights, store, unique)

    def _forward_hot_cached(
        self,
        module: nn.Module,
        x: Tensor,
        host_ids: Tensor,
        weights: Tensor,
        store: ExpertStore,
        unique: Tensor,
    ) -> Tensor:
        key = id(module)
        cache = self._hot_caches.get(key)
        if cache is None:
            cache = _LayerHotCache(self.cache_experts, self.stats, self.pool)
            self._hot_caches[key] = cache
        if x.shape[0] == 1:
            cache._allocate(store, x.device)
            routed = list(dict.fromkeys(host_ids[0].tolist()))
            hits = [expert for expert in routed if expert in cache._loaded]
            if (
                self.combined_decode_bank
                and 0 < len(hits) < len(routed)
                and len(routed) <= self.max_staged_experts
                and host_ids.shape[1] <= self.max_pair_rows
            ):
                return self._forward_combined_decode(
                    cache, x, host_ids, weights, store, routed, hits
                )
            # Keep this token's hits; fill remaining slots in router order.
            # Upstream topk() already orders those IDs by routing probability.
            selected = [expert for expert in routed if expert in cache._loaded]
            selected.extend(
                expert
                for expert in routed
                if expert not in cache._loaded
            )
            cache_ids = torch.tensor(
                selected[: self.cache_experts], dtype=torch.int64
            )
            staged, slots = cache.prepare(store, cache_ids, x.device)
        else:
            staged, slots = cache.lookup(store, unique, x.device)
        if not slots:
            if unique.numel() <= self.max_staged_experts:
                return self._forward_resident(x, host_ids, weights, store, unique)
            return self._forward_partitioned(x, host_ids, weights, store, unique)
        if len(slots) != unique.numel() or host_ids.shape[1] > self.max_pair_rows:
            return self._forward_partitioned(
                x,
                host_ids,
                weights,
                store,
                unique,
                resident=staged,
                resident_slots=slots,
            )
        lookup = torch.full((store.num_experts,), -1, dtype=torch.int64)
        for expert, slot in slots.items():
            lookup[expert] = slot
        local_ids = lookup[host_ids].to(x.device)
        top_k = host_ids.shape[1]
        rows_per_chunk = max(1, self.max_pair_rows // top_k)
        out = torch.empty_like(x)
        for start in range(0, x.shape[0], rows_per_chunk):
            end = min(start + rows_per_chunk, x.shape[0])
            out[start:end].copy_(
                self.backend.forward(
                    x[start:end], local_ids[start:end], weights[start:end], staged
                )
            )
        return out

    def _forward_combined_decode(
        self,
        cache: _LayerHotCache,
        x: Tensor,
        host_ids: Tensor,
        weights: Tensor,
        store: ExpertStore,
        routed: list[int],
        hits: list[int],
    ) -> Tensor:
        """Execute mixed residency as one official MoE call in router slot order."""
        cold = [expert for expert in routed if expert not in cache._loaded]
        cold_ids = torch.tensor(cold, dtype=torch.int64)
        resident, resident_slots = cache.lookup(
            store, torch.tensor(hits, dtype=torch.int64), x.device
        )
        self.pool.stage(store, cold_ids)
        bank = self.pool.combine_resident(
            len(cold), resident, [resident_slots[expert] for expert in hits]
        )
        ordered_bank_ids = cold + hits
        bank_slots = {expert: slot for slot, expert in enumerate(ordered_bank_ids)}
        local_ids = torch.tensor(
            [[bank_slots[expert] for expert in host_ids[0].tolist()]],
            dtype=torch.int64,
            device=x.device,
        )
        output = self.backend.forward(x, local_ids, weights, bank)
        selected = (hits + cold)[: self.cache_experts]
        cache.adopt_from_bank(bank, bank_slots, selected, x.device)
        self.stats.combined_decode_calls += 1
        return output

    def _forward_resident(
        self,
        x: Tensor,
        host_ids: Tensor,
        weights: Tensor,
        store: ExpertStore,
        unique: Tensor,
    ) -> Tensor:
        top_k = host_ids.shape[1]
        if top_k > self.max_pair_rows:
            return self._forward_partitioned(x, host_ids, weights, store, unique)
        staged = self.pool.stage(store, unique)
        lookup = torch.full((store.num_experts,), -1, dtype=torch.int64)
        lookup[unique] = torch.arange(unique.numel())
        local_ids = lookup[host_ids].to(x.device)
        rows_per_chunk = max(1, self.max_pair_rows // top_k)
        out = torch.empty_like(x)
        for start in range(0, x.shape[0], rows_per_chunk):
            end = min(start + rows_per_chunk, x.shape[0])
            out[start:end].copy_(
                self.backend.forward(
                    x[start:end], local_ids[start:end], weights[start:end], staged
                )
            )
        return out

    def _forward_partitioned(
        self,
        x: Tensor,
        host_ids: Tensor,
        weights: Tensor,
        store: ExpertStore,
        unique: Tensor,
        *,
        resident: StagedExperts | None = None,
        resident_slots: dict[int, int] | None = None,
    ) -> Tensor:
        top_k = host_ids.shape[1]
        flat_ids = host_ids.reshape(-1)
        order = torch.argsort(flat_ids, stable=True)
        counts = torch.bincount(flat_ids, minlength=store.num_experts)
        starts = (counts.cumsum(0) - counts).tolist()
        host_counts = counts.tolist()
        pair_out = torch.empty(
            (flat_ids.numel(), store.hidden), dtype=torch.float32, device=x.device
        )

        def pair_positions(group: Tensor) -> Tensor:
            return torch.cat(
                [
                    order[starts[expert] : starts[expert] + host_counts[expert]]
                    for expert in group.tolist()
                ]
            )

        if resident is not None and resident_slots:
            lookup = torch.full((store.num_experts,), -1, dtype=torch.int64)
            for expert, slot in resident_slots.items():
                lookup[expert] = slot
            group = torch.tensor(sorted(resident_slots), dtype=torch.int64)
            self._forward_pair_group(
                x, host_ids, weights, resident, lookup, pair_positions(group), pair_out
            )
            unique = unique[lookup[unique] < 0]

        for begin in range(0, unique.numel(), self.max_staged_experts):
            group = unique[begin : begin + self.max_staged_experts]
            staged = self.pool.stage(store, group)
            lookup = torch.full((store.num_experts,), -1, dtype=torch.int64)
            lookup[group] = torch.arange(group.numel())
            self._forward_pair_group(
                x, host_ids, weights, staged, lookup, pair_positions(group), pair_out
            )
        # Restore the router's slot order before its FP32 reduction, just as the
        # upstream FP8 kernel does. Grouping/cache residency must not change it.
        return pair_out.view(x.shape[0], top_k, store.hidden).sum(dim=1).to(x.dtype)

    def _forward_pair_group(
        self,
        x: Tensor,
        host_ids: Tensor,
        weights: Tensor,
        staged: StagedExperts,
        lookup: Tensor,
        pair_positions: Tensor,
        pair_out: Tensor,
    ) -> None:
        top_k = host_ids.shape[1]
        flat_ids = host_ids.reshape(-1)
        for start in range(0, pair_positions.numel(), self.max_pair_rows):
            positions = pair_positions[start : start + self.max_pair_rows]
            token_rows = (positions // top_k).to(x.device)
            local_ids = lookup[flat_ids[positions]].to(x.device).unsqueeze(1)
            device_positions = positions.to(x.device)
            pair_weights = weights.reshape(-1)[device_positions].unsqueeze(1)
            operation = (
                self.backend.forward_decode_pairs
                if x.shape[0] == 1 and isinstance(self.backend, TritonExpertBackend)
                else self.backend.forward_pairs
            )
            values = operation(
                x.index_select(0, token_rows), local_ids, pair_weights, staged
            )
            if values.shape != (positions.numel(), x.shape[-1]):
                raise ValueError("Expert backend returned an invalid pair contribution shape")
            if values.dtype != torch.float32:
                raise ValueError("Partitioned expert contributions must remain float32")
            pair_out.index_copy_(0, device_positions, values)

    def configure_hot_cache(self, capacity: int) -> None:
        """Change capacity between generations, retaining slots when it is unchanged."""
        if capacity < 0:
            raise ValueError("Hot-cache capacity cannot be negative")
        if self._closed:
            raise RuntimeError("This expert offload manager has been closed")
        if capacity == self.cache_experts:
            return
        self.clear_hot_cache()
        self.cache_experts = capacity

    def configure_hybrid_decode(self, cpu_jobs: int = 0, *, threads: int = 4,
                                device: str | torch.device = "cuda:0") -> None:
        """Opt in to one shared native worker pool; zero restores GPU-only decode."""
        if (isinstance(cpu_jobs, bool) or not isinstance(cpu_jobs, int)
                or not 0 <= cpu_jobs <= 8):
            raise ValueError("Hybrid CPU jobs must be an integer from 0 to 8")
        if (isinstance(threads, bool) or not isinstance(threads, int)
                or not 1 <= threads <= 64):
            raise ValueError("Hybrid native threads must be an integer from 1 to 64")
        if self._closed:
            raise RuntimeError("This expert offload manager has been closed")
        replacement = None
        if cpu_jobs:
            from .hybrid_decode import HybridDecodeRuntime

            replacement = HybridDecodeRuntime(
                self, cpu_jobs=cpu_jobs, threads=threads, device=device)
        if self.hybrid_decode is not None:
            self.hybrid_decode.close()
        self.hybrid_decode = replacement

    def configure_host_residency(
        self, budget_bytes: int
    ) -> dict[str, int | float | str | bool | None]:
        """Allocate bounded RAM experts at a serialized request boundary.

        This tier is useful above an mmap source. The engine owns the budget
        and must avoid duplicating an already-resident full RAM bank.
        """
        if self._closed:
            raise RuntimeError("This expert offload manager has been closed")
        if isinstance(budget_bytes, bool) or not isinstance(budget_bytes, int) or budget_bytes < 0:
            raise ValueError("Host resident budget must be an integer >= 0")
        if not self.modules:
            raise ValueError("No expert modules are attached to this manager")
        from .host_residency import HostResidency

        stores = [ExpertStore.from_module(module) for module in self.modules.values()]
        keys = tuple(
            tuple(id(getattr(store, name)) for name in EXPERT_BUFFER_NAMES)
            for store in stores
        )
        self.synchronize()
        if self._host_residency is None or keys != self._host_store_keys:
            replacement = HostResidency(stores)
            if self._host_residency is not None:
                self._host_residency.close()
                self.clear_hot_cache()
                self.pool._loaded_ids = None
                self.pool._loaded_store_key = None
            self._host_residency = replacement
            self._host_store_keys = keys
            self._host_module_layers = {
                id(module): layer for layer, module in enumerate(self.modules.values())
            }
        self.pool.host_residency = self._host_residency
        self.pool.host_store_layers = {key: layer for layer, key in enumerate(keys)}
        self._host_residency.set_budget(budget_bytes)
        return self.host_residency_report()

    def rebalance_host_residency(self) -> dict[str, int | float | str | bool | None]:
        """Apply observed route frequencies at a serialized request boundary."""
        if self._closed:
            raise RuntimeError("This expert offload manager has been closed")
        if self._host_residency is not None:
            self.synchronize()
            self._host_residency.configure(self._host_residency.budget_bytes)
        return self.host_residency_report()

    def host_residency_report(self) -> dict[str, int | float | str | bool | None]:
        """Report logical tier reads, which do not measure physical disk I/O."""
        if self._host_residency is None:
            return {
                "configured": False,
                "budget_bytes": 0,
                "resident_bytes": 0,
                "resident_experts": 0,
            }
        return {"configured": True, **self._host_residency.as_dict()}

    def clear_hot_cache(self) -> None:
        """Release resident experts explicitly, retaining the configured capacity."""
        if self._closed:
            raise RuntimeError("This expert offload manager has been closed")
        self.synchronize()
        for cache in self._hot_caches.values():
            cache.clear()
        self._hot_caches.clear()

    def release_runtime_buffers(self) -> None:
        """Release GPU caches and shared staging for replanning or codec execution.

        Configuration survives so the next ``prepare_buffers`` or forward can
        recreate it. Ordinary request boundaries should retain these buffers.
        """
        if self._closed:
            raise RuntimeError("This expert offload manager has been closed")
        if self.hybrid_decode is not None:
            self.hybrid_decode.release_runtime_buffers()
        self.pool.close()
        for cache in self._hot_caches.values():
            cache.clear()
        self._hot_caches.clear()

    def configure_staging(
        self, max_staged_experts: int, max_pair_rows: int
    ) -> None:
        """Resize shared scratch buffers between generations, retaining expert caches."""
        if max_staged_experts < 1 or max_pair_rows < 1:
            raise ValueError("Staging capacity and pair row limit must be positive")
        if self._closed:
            raise RuntimeError("This expert offload manager has been closed")
        if max_staged_experts == self.max_staged_experts:
            self.max_pair_rows = max_pair_rows
            return
        previous = self.pool
        previous.close()
        self.pool = _StagingPool(
            max_staged_experts,
            previous.pin_memory,
            previous.profile_transfers,
            self.stats,
            packed_staging=previous.packed_staging,
        )
        self.pool.host_residency = previous.host_residency
        self.pool.host_store_layers = previous.host_store_layers
        self.pool.native_gather = previous.native_gather
        self.pool.native_gather_min_experts = previous.native_gather_min_experts
        for cache in self._hot_caches.values():
            cache.pool = self.pool
        self.max_staged_experts = max_staged_experts
        self.max_pair_rows = max_pair_rows

    def prepare_buffers(
        self, device: str | torch.device
    ) -> dict[str, int | float]:
        """Allocate exact staging/cache buffers before generation, without loading experts."""
        if self._closed:
            raise RuntimeError("This expert offload manager has been closed")
        if not self.modules:
            raise ValueError("No expert modules are attached to this manager")
        target = torch.device(device)
        if target.type == "cuda" and target.index is None:
            target = torch.device("cuda", torch.cuda.current_device())
        stores = [
            (module, ExpertStore.from_module(module))
            for module in self.modules.values()
        ]
        with self.pool.session(stores[0][1], target):
            try:
                if self.cache_experts:
                    for module, store in stores:
                        cache = self._hot_caches.setdefault(
                            id(module),
                            _LayerHotCache(self.cache_experts, self.stats, self.pool),
                        )
                        cache._allocate(store, target)
            except Exception:
                for cache in self._hot_caches.values():
                    cache.clear()
                self._hot_caches.clear()
                raise
        return self.stats.as_dict()

    def synchronize(self) -> None:
        self.pool.synchronize()

    def close(self) -> None:
        """Release staging buffers; further forwards fail until a new manager is installed."""
        if self.hybrid_decode is not None:
            self.hybrid_decode.close()
            self.hybrid_decode = None
        self.pool.close()
        for cache in self._hot_caches.values():
            cache.clear()
        self._hot_caches.clear()
        if self._host_residency is not None:
            self._host_residency.close()
            self._host_residency = None
        self._host_module_layers.clear()
        self._host_store_keys = ()
        self.pool.host_residency = None
        self.pool.host_store_layers.clear()
        for module in self.modules.values():
            module.forward = module._drama_offload_original_forward
            del module._drama_offload_original_forward
        self.modules.clear()
        self._closed = True


def _expert_modules(backbone: nn.Module) -> list[tuple[str, nn.Module]]:
    return [
        (name, module)
        for name, module in backbone.named_modules()
        if hasattr(module, "w13_weight") and hasattr(module, "w2_weight")
    ]


def materialize_offloaded_backbone(
    backbone: nn.Module,
    device: str | torch.device = "cuda",
    *,
    embedding_device: str | torch.device = "cpu",
    lm_head_device: str | torch.device | None = None,
    expert_device: str | torch.device = "cpu",
) -> dict[str, str]:
    """Allocate a meta FishOmniBackbone directly into its final device layout.

    Call before ``load_backbone_weights``. The upstream weight loader can still
    fill all original names, including the FP8 CPU expert buffers. Calling
    ``backbone.to(cuda)`` afterward would defeat this placement.
    """
    experts = _expert_modules(backbone)
    if not experts:
        raise ValueError("Backbone contains no routed expert modules")
    for name, module in experts:
        if not getattr(module, "fp8", False):
            raise ValueError(f"{name} must be constructed with fp8_experts=True")
    tensors = [*backbone.parameters(), *backbone.buffers()]
    if any(t.device.type != "meta" for t in tensors):
        raise ValueError("Device placement requires an unloaded, fully meta backbone")
    compute = torch.device(device)
    embedding = torch.device(embedding_device)
    head = compute if lm_head_device is None else torch.device(lm_head_device)
    expert_ids = {id(module) for _, module in experts}
    embed_id = id(backbone.model.embed_tokens)
    head_id = id(backbone.lm_head)
    placement: dict[str, str] = {}
    for name, module in backbone.named_modules():
        if id(module) in expert_ids:
            target = torch.device(expert_device)
        elif id(module) == embed_id:
            target = embedding
        elif id(module) == head_id:
            target = head
        else:
            target = compute
        if list(module.parameters(recurse=False)) or list(module.buffers(recurse=False)):
            module.to_empty(device=target, recurse=False)
            placement[name] = str(target)
    if hasattr(backbone, "inverse_frequencies"):
        backbone.inverse_frequencies.clear()
    return placement


def install_expert_offload(
    backbone: nn.Module,
    *,
    backend: str | ExpertBackend = "triton",
    max_staged_experts: int = 8,
    max_pair_rows: int = 64,
    cache_experts: int = 0,
    hot_cache_experts: int | None = None,
    pin_memory: bool = True,
    profile_transfers: bool = False,
    combined_decode_bank: bool = False,
    packed_staging: bool = False,
) -> ExpertOffloadManager:
    """Replace only expert forwards; checkpoint keys and router/shared experts stay intact."""
    experts = _expert_modules(backbone)
    if not experts:
        raise ValueError("Backbone contains no routed expert modules")
    # Validate all stores before changing any module's forward.
    for _, module in experts:
        ExpertStore.from_module(module)
        if hasattr(module, "_drama_offload_original_forward"):
            raise ValueError("Backbone already has an expert offload manager")
    manager = ExpertOffloadManager(
        backend=backend,
        max_staged_experts=max_staged_experts,
        max_pair_rows=max_pair_rows,
        cache_experts=cache_experts,
        hot_cache_experts=hot_cache_experts,
        pin_memory=pin_memory,
        profile_transfers=profile_transfers,
        combined_decode_bank=combined_decode_bank,
        packed_staging=packed_staging,
    )
    for name, module in experts:
        manager.attach(name, module)
    return manager
