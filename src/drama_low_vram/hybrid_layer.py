"""Opt-in single-MoE-layer CPU/GPU experiment; not a DramaEngine backend."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import time

import torch

from .native_backend import NativeExpertExecutor
from .offload import (
    ExpertStore, OffloadStatistics, TritonExpertBackend, _LayerHotCache, _StagingPool,
)


def partition_routes(ids, resident, cpu_jobs):
    """Prefer low-ranked cold routes for CPU; retain original reduction order."""
    if not 0 <= cpu_jobs <= len(ids):
        raise ValueError("CPU job count must fit the route count")
    cold = [slot for slot in reversed(range(len(ids))) if ids[slot] not in resident]
    hot = [slot for slot in reversed(range(len(ids))) if ids[slot] in resident]
    selected = set((cold + hot)[:cpu_jobs])
    return (
        [slot for slot in range(len(ids)) if slot in selected],
        [slot for slot in range(len(ids)) if slot not in selected],
    )


def gpu_quantized_pairs(backend, xq, scales, ids, weights, staged):
    """The official two-GEMM path, using a previously quantized single row."""
    backend._check(xq)
    if backend.decode_specialization and xq.device in backend._direct_decode_devices:
        from .fp8_decode import quantized_pairs

        return quantized_pairs(backend.kernel, xq, scales, ids, weights, staged)
    k = backend.kernel
    count = ids.numel()
    sorted_ids, expert_ids, padded = k.moe_align(ids.reshape(1, count), staged.count, 16)
    gu = torch.empty(
        (count, staged.w13_weight.shape[1]), device=xq.device, dtype=torch.bfloat16)
    k.fused_moe_gemm(
        xq, scales, staged.w13_weight, staged.w13_scale_inv, gu, None,
        sorted_ids, expert_ids, padded, top_k=count, block_m=16, group_m=1,
    )
    hq, hs = k.quantize_groups(k.silu_mul_quant_kernel, gu, gu.shape[1] // 2)
    partials = torch.empty((count, xq.shape[1]), device=xq.device, dtype=torch.float32)
    k.fused_moe_gemm(
        hq, hs, staged.w2_weight, staged.w2_scale_inv, partials, weights,
        sorted_ids, expert_ids, padded, top_k=1, block_m=16, group_m=1,
    )
    return partials


class HybridLayerExperiment:
    """Synchronous completion, persistent CPU workers, ordered FP32 merging.

    Inputs/routes start on the GPU. CPU inputs use bounded pinned buffers. A
    distinct CUDA stream returns CPU partials; final completion fences every
    source before reuse. The original FP8 expert bank is borrowed, never copied
    in full. Attention, router GEMM and the shared expert are outside this test.
    """

    def __init__(self, store: ExpertStore, *, cpu_jobs=0, threads=4,
                 cache_experts=0, parallel=True, cpu_launch="gather",
                 native_arithmetic="reference", silu_table=None,
                 device="cuda:0", library=None):
        store.validate()
        if (isinstance(cpu_jobs, bool) or not isinstance(cpu_jobs, int)
                or not 0 <= cpu_jobs <= 8
                or isinstance(cache_experts, bool) or not isinstance(cache_experts, int)
                or not 0 <= cache_experts <= store.num_experts
                or cpu_launch not in ("gather", "dma", "compute")):
            raise ValueError("Invalid hybrid job/cache count")
        self.store, self.device = store, torch.device(device)
        if (cpu_jobs and native_arithmetic == "sm89"
                and torch.cuda.get_device_capability(self.device) != (8, 9)):
            raise ValueError("SM89 native arithmetic requires a matching SM89 GPU")
        self.cpu_jobs, self.parallel = cpu_jobs, parallel
        self.cpu_launch = cpu_launch
        self.stats = OffloadStatistics()
        self.backend = TritonExpertBackend()
        self.pool = _StagingPool(8, True, False, self.stats)
        self.cache = _LayerHotCache(cache_experts, self.stats, self.pool) if cache_experts else None
        self.executor = NativeExpertExecutor(
            store, capacity=8, threads=threads, arithmetic=native_arithmetic,
            silu_table=silu_table, library=library) if cpu_jobs else None
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="drama-hybrid") if cpu_jobs else None
        self.lock = threading.Lock()
        self.closed = False
        self.return_stream = torch.cuda.Stream(device=self.device)
        self.host_ids = torch.empty(8, dtype=torch.int64, pin_memory=True)
        self.host_weights = torch.empty(8, dtype=torch.float32, pin_memory=True)
        self.host_q = torch.empty((1, store.hidden), dtype=torch.uint8, pin_memory=True)
        self.host_scales = torch.empty((1, store.hidden // 128), dtype=torch.float32, pin_memory=True)
        self.host_result = torch.empty((8, store.hidden), dtype=torch.float32, pin_memory=True)
        self.device_result = torch.empty_like(self.host_result, device=self.device)
        self.ordered = torch.empty_like(self.host_result, device=self.device)
        self.events = {
            name: torch.cuda.Event(enable_timing=True)
            for name in ("start", "handoff", "gpu_start", "gpu_done",
                         "return_start", "return_done", "merge_start", "end")
        }
        with self.pool.session(store, self.device):
            if self.cache is not None:
                self.cache._allocate(store, self.device)

    def _cpu(self, ids, weights, start_event=None):
        if start_event is not None:
            start_event.synchronize()
        started = time.perf_counter_ns()
        wall_started = time.time_ns()
        self.executor.forward_quantized_pairs(
            self.host_q, self.host_scales, ids, weights,
            out=self.host_result[:ids.numel()],
        )
        return {
            "start_ns": started, "end_ns": time.perf_counter_ns(),
            "wall_start_ns": wall_started, "wall_end_ns": time.time_ns(),
            "thread_id": threading.get_native_id(),
        }

    @torch.inference_mode()
    def forward(self, x, ids, weights):
        if (x.device != self.device or x.dtype != torch.bfloat16
                or x.shape != (1, self.store.hidden) or not x.is_contiguous()
                or ids.device != self.device or ids.dtype != torch.int64
                or ids.shape != (1, 8) or weights.device != self.device
                or weights.shape != (1, 8)
                or not weights.is_floating_point()):
            raise ValueError("Hybrid experiment expects GPU BF16 [1,H] and eight routes")
        with self.lock:
            if self.closed:
                raise RuntimeError("Hybrid layer is closed")
            started = time.perf_counter_ns()
            wall_started = time.time_ns()
            stream = torch.cuda.current_stream(self.device)
            future = None
            cpu_times = None
            completed = False
            before_h2d = self.stats.h2d_bytes
            before_gather = self.stats.host_gather_seconds
            try:
                e = self.events
                e["start"].record(stream)
                q, scales = self.backend.kernel.quantize_groups(
                    self.backend.kernel.quant_kernel, x, self.store.hidden)
                self.host_ids.copy_(ids.reshape(-1), non_blocking=True)
                self.host_weights.copy_(weights.reshape(-1).float(), non_blocking=True)
                if self.cpu_jobs:
                    self.host_q.copy_(q.view(torch.uint8), non_blocking=True)
                    self.host_scales.copy_(scales, non_blocking=True)
                e["handoff"].record(stream)
                e["handoff"].synchronize()
                handoff_finished = time.perf_counter_ns()
                routed = self.host_ids.tolist()
                if min(routed) < 0 or max(routed) >= self.store.num_experts:
                    raise ValueError("Hybrid expert ID outside bank")
                resident = self.cache._loaded if self.cache is not None else {}
                cpu_slots, gpu_slots = partition_routes(routed, resident, self.cpu_jobs)
                cpu_index = torch.tensor(cpu_slots, dtype=torch.int64)
                gpu_index = torch.tensor(gpu_slots, dtype=torch.int64, device=self.device)
                if cpu_slots:
                    cpu_ids = self.host_ids.index_select(0, cpu_index)
                    cpu_weights = self.host_weights.index_select(0, cpu_index)
                    if not self.parallel:
                        cpu_times = self._cpu(cpu_ids, cpu_weights)
                    elif self.cpu_launch == "gather":
                        future = self.worker.submit(self._cpu, cpu_ids, cpu_weights)
                gpu_routed = list(dict.fromkeys(routed[slot] for slot in gpu_slots))
                gpu_partials = None
                with self.pool.session(self.store, self.device):
                    if gpu_routed:
                        hits = [expert for expert in gpu_routed if expert in resident]
                        cold = [expert for expert in gpu_routed if expert not in resident]
                        if self.cache is not None:
                            cached, cached_slots = self.cache.lookup(
                                self.store, torch.tensor(hits, dtype=torch.int64), self.device)
                        bank_slots = (
                            {expert: slot for slot, expert in enumerate(cold + hits)}
                            if cold else cached_slots
                        )
                        # Materialize tiny CUDA indices before expert DMA. Creating a
                        # CUDA tensor from a Python list later would synchronize the
                        # stream and finish the transfer before CPU work can start.
                        local_ids = torch.tensor(
                            [bank_slots[routed[slot]] for slot in gpu_slots],
                            dtype=torch.int64, device=self.device)
                        route_weights = weights.reshape(-1).index_select(0, gpu_index).float()
                        if cold:
                            bank = self.pool.stage(self.store, torch.tensor(cold, dtype=torch.int64))
                            if cpu_slots and self.parallel and self.cpu_launch == "dma":
                                future = self.worker.submit(self._cpu, cpu_ids, cpu_weights)
                            if hits:
                                bank = self.pool.combine_resident(
                                    len(cold), cached, [cached_slots[expert] for expert in hits])
                        else:
                            bank = cached
                            if cpu_slots and self.parallel and self.cpu_launch == "dma":
                                future = self.worker.submit(self._cpu, cpu_ids, cpu_weights)
                        e["gpu_start"].record(stream)
                        if cpu_slots and self.parallel and self.cpu_launch == "compute":
                            future = self.worker.submit(
                                self._cpu, cpu_ids, cpu_weights, e["gpu_start"])
                        gpu_partials = gpu_quantized_pairs(
                            self.backend, q, scales, local_ids, route_weights, bank)
                        if self.cache is not None:
                            keep = (hits + cold)[:self.cache.capacity]
                            self.cache.adopt_from_bank(bank, bank_slots, keep, self.device)
                    else:
                        if cpu_slots and self.parallel and future is None:
                            future = self.worker.submit(self._cpu, cpu_ids, cpu_weights)
                        e["gpu_start"].record(stream)
                    e["gpu_done"].record(stream)
                    gpu_submitted = time.perf_counter_ns()
                    if future is not None:
                        cpu_times = future.result()
                    cpu_wait_finished = time.perf_counter_ns()
                    if cpu_slots:
                        with torch.cuda.stream(self.return_stream):
                            e["return_start"].record(self.return_stream)
                            self.device_result[:len(cpu_slots)].copy_(
                                self.host_result[:len(cpu_slots)], non_blocking=True)
                            e["return_done"].record(self.return_stream)
                        stream.wait_event(e["return_done"])
                    e["merge_start"].record(stream)
                    if gpu_partials is not None:
                        self.ordered.index_copy_(0, gpu_index, gpu_partials)
                    if cpu_slots:
                        cpu_device_index = cpu_index.to(self.device)
                        self.ordered.index_copy_(
                            0, cpu_device_index, self.device_result[:len(cpu_slots)])
                    output = self.ordered.sum(0, keepdim=True).to(torch.bfloat16)
                    e["end"].record(stream)
                e["end"].synchronize()
                completed = True
                finished = time.perf_counter_ns()
                return output, {
                    "wall_start_ns": wall_started, "wall_end_ns": time.time_ns(),
                    "total_ms": (finished - started) / 1e6,
                    "handoff_host_ms": (handoff_finished - started) / 1e6,
                    "gpu_submission_host_ms": (gpu_submitted - handoff_finished) / 1e6,
                    "cpu_remaining_wait_host_ms": (cpu_wait_finished - gpu_submitted) / 1e6,
                    "cpu_job_ms": 0 if cpu_times is None else (cpu_times["end_ns"] - cpu_times["start_ns"]) / 1e6,
                    "cpu_interval": cpu_times,
                    "gpu_compute_and_cache_ms": e["gpu_start"].elapsed_time(e["gpu_done"]),
                    "cpu_result_h2d_ms": e["return_start"].elapsed_time(e["return_done"]) if cpu_slots else 0,
                    "merge_ms": e["merge_start"].elapsed_time(e["end"]),
                    "gpu_expert_h2d_bytes": self.stats.h2d_bytes - before_h2d,
                    "host_gather_ms": (self.stats.host_gather_seconds - before_gather) * 1000,
                    "control_d2h_bytes": 8 * 8 + 8 * 4 + (self.store.hidden + 4 * self.store.hidden // 128 if cpu_slots else 0),
                    "cpu_result_h2d_bytes": len(cpu_slots) * self.store.hidden * 4,
                    "cpu_slots": cpu_slots, "gpu_slots": gpu_slots,
                    "cpu_launch": self.cpu_launch,
                }
            finally:
                if not completed:
                    try:
                        if future is not None:
                            future.result()
                    finally:
                        # Fence pinned buffers on validation/dispatch failure.
                        stream.synchronize()
                        self.return_stream.synchronize()

    def close(self):
        with self.lock:
            if not self.closed:
                if self.worker is not None:
                    self.worker.shutdown(wait=True)
                    self.executor.close()
                self.pool.close()
                self.return_stream.synchronize()
                if self.cache is not None:
                    self.cache.clear()
                self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
