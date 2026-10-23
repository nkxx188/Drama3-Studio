"""Experimental SM89 decode cooperation with one workspace shared by all layers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import threading

import torch

from .fp8_decode import build_gpu_silu_table
from .hybrid_layer import gpu_quantized_pairs, partition_routes
from .native_backend import NativeExpertExecutor, NativeQuantizationBoundary
from .offload import EXPERT_BUFFER_NAMES, ExpertStore, TritonExpertBackend, _LayerHotCache


@dataclass
class HybridDecodeStatistics:
    calls: int = 0
    cpu_pairs: int = 0
    gpu_pairs: int = 0
    control_d2h_bytes: int = 0
    partial_h2d_bytes: int = 0
    workspace_allocations: int = 0
    fallbacks: int = 0
    boundary_fallback_calls: int = 0
    boundary_recomputed_pairs: int = 0
    completion_host_waits: int = 0
    completion_stream_waits: int = 0


class HybridDecodeRuntime:
    """Borrow manager staging/cache; own one native pool and small handoff buffers.

    The caller serializes model requests. Stream dependencies fence borrowed
    GPU scratch before reuse; native workers finish before return. Failures
    drain both streams before releasing scratch. Prefill never
    enters this backend: the generation loop explicitly marks decode scopes.
    SM89 arithmetic is an empirical contract, restricted to that GPU family.
    """

    def __init__(self, manager, *, cpu_jobs=4, threads=4, device="cuda:0",
                 silu_table=None, library=None, synchronize_return=True):
        if (isinstance(cpu_jobs, bool) or not isinstance(cpu_jobs, int)
                or not 1 <= cpu_jobs <= 8):
            raise ValueError("Hybrid CPU jobs must be an integer from 1 to 8")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("Hybrid decode requires CUDA")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.get_device_capability(self.device) != (8, 9):
            raise ValueError("Experimental hybrid arithmetic requires SM89")
        if not isinstance(manager.backend, TritonExpertBackend):
            raise ValueError("Hybrid decode requires the Triton expert backend")
        if manager._closed or not manager.modules:
            raise ValueError("Attach expert modules before configuring hybrid decode")
        stores = [ExpertStore.from_module(module) for module in manager.modules.values()]
        if len({(store.hidden, store.intermediate) for store in stores}) != 1:
            raise ValueError("Hybrid layers must have identical expert geometry")
        self.manager = manager
        self.cpu_jobs = cpu_jobs
        self.stats = HybridDecodeStatistics()
        self._lock = threading.Lock()
        self._scope = threading.local()
        self._closed = False
        self._buffers = {}
        self._layer_indices = {
            id(module): index for index, module in enumerate(manager.modules.values())
        }
        self._stream = None
        self._handoff = None
        self._returned = None
        self._done = None
        self._pending = False
        self.synchronize_return = bool(synchronize_return)
        # Research-only, explicitly enabled by the execution benchmark. The
        # default still uses one combined GPU bank and the original scheduling.
        self.pipeline_hits = False
        self.direct_sources = False
        self._copy_stream = None
        self._cold_ready = None
        table = build_gpu_silu_table(self.device) if silu_table is None else silu_table
        self.executor = NativeExpertExecutor(
            stores[0], capacity=8, threads=threads, arithmetic="sm89",
            silu_table=table, guard_fp8_boundaries=True, library=library,
        )
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="drama-decode")

    @contextmanager
    def decode(self, *, layers=None):
        """Mark decode on the calling thread; optional layer subset is diagnostic."""
        previous = getattr(self._scope, "layers", ())
        self._scope.layers = None if layers is None else frozenset(layers)
        try:
            yield
        finally:
            self._scope.layers = previous

    @contextmanager
    def suspend(self):
        """Use the GPU path during a serialized diagnostic request."""
        previous = getattr(self._scope, "suspended", False)
        self._scope.suspended = True
        try:
            yield
        finally:
            self._scope.suspended = previous

    def eligible(self, module, x, ids):
        if getattr(self._scope, "suspended", False):
            return False
        layers = getattr(self._scope, "layers", ())
        if layers is not None and self._layer_indices.get(id(module)) not in layers:
            return False
        if self._closed:
            raise RuntimeError("Hybrid decode runtime is closed")
        eligible = (
            id(module) in self._layer_indices and x.device == self.device
            and x.dtype == torch.bfloat16 and x.shape[0] == 1 and x.is_contiguous()
            and 1 <= ids.shape[1] <= 8
            and self.manager.pool.capacity >= max(1, ids.shape[1] - self.cpu_jobs)
        )
        if not eligible:
            self.stats.fallbacks += 1
        return eligible

    def _allocate(self, store):
        if self._buffers:
            return
        hidden = store.hidden
        # All layers use these same buffers, independent of model depth.
        shapes = {
            "ids": ((8,), torch.int64),
            "weights": ((8,), torch.float32),
            "q": ((1, hidden), torch.uint8),
            "scales": ((1, hidden // 128), torch.float32),
            "result": ((8, hidden), torch.float32),
        }
        buffers = {
            name: torch.empty(shape, dtype=dtype, pin_memory=True)
            for name, (shape, dtype) in shapes.items()
        }
        buffers["device_result"] = torch.empty_like(buffers["result"], device=self.device)
        buffers["ordered"] = torch.empty_like(buffers["result"], device=self.device)
        self._stream = torch.cuda.Stream(device=self.device)
        self._handoff = torch.cuda.Event()
        self._returned = torch.cuda.Event()
        self._done = torch.cuda.Event()
        self._buffers = buffers
        self.stats.workspace_allocations += 1

    def _cpu(self, store, ids, weights):
        self.executor.forward_quantized_pairs(
            self._buffers["q"], self._buffers["scales"], ids, weights, store=store,
            out=self._buffers["result"][:ids.numel()],
        )

    @torch.inference_mode()
    def forward(self, module, store, x, ids, weights):
        with self._lock:
            if self._closed:
                raise RuntimeError("Hybrid decode runtime is closed")
            self._allocate(store)
            manager, b = self.manager, self._buffers
            backend = manager.backend
            backend._check(x)
            stream = torch.cuda.current_stream(self.device)
            if self._pending:
                # A subsequent call may use a different CUDA stream. Order all
                # scratch reuse behind the previous merge without stalling the
                # submitting CPU. The handoff fence below also ensures the
                # previous return DMA has finished before native workers can
                # overwrite the shared pinned result.
                stream.wait_event(self._done)
                self.stats.completion_stream_waits += 1
            count = ids.numel()
            future = None
            completed = False
            try:
                q, scales = backend.kernel.quantize_groups(
                    backend.kernel.quant_kernel, x, store.hidden)
                b["ids"][:count].copy_(ids.reshape(-1), non_blocking=True)
                b["weights"][:count].copy_(weights.reshape(-1).float(), non_blocking=True)
                b["q"].copy_(q.view(torch.uint8), non_blocking=True)
                b["scales"].copy_(scales, non_blocking=True)
                self._handoff.record(stream)
                self._handoff.synchronize()
                routed = b["ids"][:count].tolist()
                if min(routed) < 0 or max(routed) >= store.num_experts:
                    raise ValueError("Hybrid expert ID outside bank")
                if manager._host_residency is not None and manager._host_residency.budget_bytes:
                    layer = manager._host_module_layers.get(id(module))
                    key = tuple(id(getattr(store, name)) for name in EXPERT_BUFFER_NAMES)
                    if layer is None or manager._host_store_keys[layer] != key:
                        raise RuntimeError("Expert buffers changed; reconfigure host residency")
                    manager._host_residency.observe(layer, b["ids"][:count].reshape(1, count))
                # Resolve manager-owned resources every call; codec parking and
                # runtime replanning may replace the pool or invalidate caches.
                with manager.pool.session(store, self.device):
                    cache = None
                    if manager.cache_experts:
                        cache = manager._hot_caches.get(id(module))
                        if cache is None:
                            cache = _LayerHotCache(manager.cache_experts, manager.stats, manager.pool)
                            manager._hot_caches[id(module)] = cache
                        cache._allocate(store, self.device)
                    resident = cache._loaded if cache is not None else {}
                    cpu_slots, gpu_slots = partition_routes(
                        routed, resident, min(self.cpu_jobs, count))
                    cpu_index = torch.tensor(cpu_slots, dtype=torch.int64)
                    cpu_ids = b["ids"][:count].index_select(0, cpu_index)
                    cpu_weights = b["weights"][:count].index_select(0, cpu_index)
                    future = self.worker.submit(self._cpu, store, cpu_ids, cpu_weights)
                    gpu_index = torch.tensor(gpu_slots, dtype=torch.int64, device=self.device)
                    cpu_device_index = cpu_index.to(self.device)
                    gpu_routed = list(dict.fromkeys(routed[slot] for slot in gpu_slots))
                    gpu_partials = None
                    if gpu_routed:
                        hits = [expert for expert in gpu_routed if expert in resident]
                        cold = [expert for expert in gpu_routed if expert not in resident]
                        if cache is not None:
                            cached, cached_slots = cache.lookup(
                                store, torch.tensor(hits, dtype=torch.int64), self.device)
                        bank_slots = (
                            {expert: slot for slot, expert in enumerate(cold + hits)}
                            if cold else cached_slots
                        )
                        local_ids = torch.tensor(
                            [bank_slots[routed[slot]] for slot in gpu_slots],
                            dtype=torch.int64, device=self.device)
                        route_weights = weights.reshape(-1).index_select(0, gpu_index).float()
                        if self.direct_sources and hits and cold:
                            from .fp8_decode import quantized_pairs_sources
                            bank = manager.pool.stage(store, torch.tensor(cold, dtype=torch.int64))
                            bank_slots = {expert: slot for slot, expert in enumerate(cold)}
                            local_ids = torch.tensor(
                                [cached_slots[routed[slot]] if routed[slot] in cached_slots
                                 else bank_slots[routed[slot]] for slot in gpu_slots],
                                dtype=torch.int64, device=self.device)
                            sources = torch.tensor(
                                [routed[slot] in cached_slots for slot in gpu_slots],
                                dtype=torch.int32, device=self.device)
                            gpu_partials = quantized_pairs_sources(
                                backend.kernel, q, scales, local_ids, sources,
                                route_weights, bank, cached)
                        elif self.pipeline_hits and hits and cold:
                            gpu_partials, bank, bank_slots = self._gpu_hit_pipeline(
                                store, q, scales, [routed[slot] for slot in gpu_slots], hits, cold,
                                cached, cached_slots, local_ids, route_weights)
                        elif cold:
                            bank = manager.pool.stage(store, torch.tensor(cold, dtype=torch.int64))
                            if hits:
                                bank = manager.pool.combine_resident(
                                    len(cold), cached, [cached_slots[expert] for expert in hits])
                        else:
                            bank = cached
                        if gpu_partials is None:
                            gpu_partials = gpu_quantized_pairs(
                                backend, q, scales, local_ids, route_weights, bank)
                        if cache is not None:
                            # The split path's bank contains only cold experts.
                            # Existing hits need no copy; future admissions must
                            # refer to cold bank slots, not a logical mixed bank.
                            cache.adopt_from_bank(
                                bank, bank_slots, (hits + cold)[:cache.capacity], self.device)
                    recovered = False
                    try:
                        future.result()
                    except NativeQuantizationBoundary:
                        future = None
                        # Native workers have all joined before returning this
                        # status. Recompute their ordered contributions through
                        # the original GPU arithmetic, using bounded staging.
                        recovered = True
                        for start in range(0, len(cpu_slots), manager.pool.capacity):
                            end = min(start + manager.pool.capacity, len(cpu_slots))
                            group_ids = cpu_ids[start:end]
                            unique = torch.unique(group_ids, sorted=True)
                            bank = manager.pool.stage(store, unique)
                            remap = {expert: slot for slot, expert in enumerate(unique.tolist())}
                            local = torch.tensor(
                                [remap[expert] for expert in group_ids.tolist()],
                                device=self.device, dtype=torch.int64)
                            group_weights = weights.reshape(-1).index_select(
                                0, cpu_device_index[start:end]).float()
                            recovered_partials = gpu_quantized_pairs(
                                backend, q, scales, local, group_weights, bank)
                            b["ordered"].index_copy_(
                                0, cpu_device_index[start:end], recovered_partials)
                    else:
                        future = None
                    if not recovered:
                        with torch.cuda.stream(self._stream):
                            b["device_result"][:len(cpu_slots)].copy_(
                                b["result"][:len(cpu_slots)], non_blocking=True)
                            self._returned.record(self._stream)
                        stream.wait_event(self._returned)
                    if gpu_partials is not None:
                        b["ordered"].index_copy_(0, gpu_index, gpu_partials)
                    if not recovered:
                        b["ordered"].index_copy_(
                            0, cpu_device_index, b["device_result"][:len(cpu_slots)])
                    output = b["ordered"][:count].sum(0, keepdim=True).to(torch.bfloat16)
                    self._done.record(stream)
                    self._pending = True
                if self.synchronize_return:
                    self._done.synchronize()
                    self._pending = False
                    self.stats.completion_host_waits += 1
                completed = True
                self.stats.calls += 1
                self.stats.cpu_pairs += len(cpu_slots)
                self.stats.gpu_pairs += len(gpu_slots) + (len(cpu_slots) if recovered else 0)
                self.stats.control_d2h_bytes += count * 12 + store.hidden + scales.numel() * 4
                if recovered:
                    self.stats.boundary_fallback_calls += 1
                    self.stats.boundary_recomputed_pairs += len(cpu_slots)
                else:
                    self.stats.partial_h2d_bytes += len(cpu_slots) * store.hidden * 4
                return output
            finally:
                if not completed:
                    try:
                        if future is not None:
                            future.result()
                    finally:
                        stream.synchronize()
                        self._stream.synchronize()
                        if self._copy_stream is not None:
                            self._copy_stream.synchronize()
                        self._pending = False

    def _gpu_hit_pipeline(self, store, q, scales, routed, hits, cold,
                          cached, cached_slots, local_ids, weights):
        """Known routes: cached math while CPU gathers and copies cold weights."""
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream(device=self.device)
            self._cold_ready = torch.cuda.Event()
        main = torch.cuda.current_stream(self.device)
        copy = self._copy_stream
        # The pool may have been consumed on the previous main stream. Its
        # session event orders device staging reuse on this independent stream.
        if self.manager.pool._compute_done is not None:
            copy.wait_event(self.manager.pool._compute_done)
        copy.wait_stream(main)
        hit_positions = [index for index, expert in enumerate(routed) if expert in cached_slots]
        cold_positions = [index for index, expert in enumerate(routed) if expert not in cached_slots]
        hp = torch.tensor(hit_positions, device=self.device)
        cp = torch.tensor(cold_positions, device=self.device)
        hit_ids = torch.tensor([cached_slots[routed[index]] for index in hit_positions], device=self.device)
        cold_lookup = {expert: slot for slot, expert in enumerate(cold)}
        cold_ids = torch.tensor([cold_lookup[routed[index]] for index in cold_positions], device=self.device)
        partials = torch.empty((len(routed), store.hidden), dtype=torch.float32, device=self.device)
        hit_values = gpu_quantized_pairs(
            self.manager.backend, q, scales, hit_ids, weights.index_select(0, hp), cached)
        partials.index_copy_(0, hp, hit_values)
        # This CPU gather runs while the GPU executes the submitted hit math.
        with torch.cuda.stream(copy):
            bank = self.manager.pool.stage(store, torch.tensor(cold, dtype=torch.int64))
            for name in EXPERT_BUFFER_NAMES:
                getattr(bank, name).record_stream(copy)
            self._cold_ready.record(copy)
        main.wait_event(self._cold_ready)
        cold_values = gpu_quantized_pairs(
            self.manager.backend, q, scales, cold_ids, weights.index_select(0, cp), bank)
        partials.index_copy_(0, cp, cold_values)
        return partials, bank, cold_lookup

    def report(self):
        return {
            **asdict(self.stats), "cpu_jobs": self.cpu_jobs,
            "enabled": not self._closed and not getattr(self._scope, "suspended", False),
            "native_threads": self.executor.threads,
            "native_implementation": self.executor.implementation,
            "arithmetic": "sm89_gpu_quantization_and_silu_table",
            "synchronize_return": self.synchronize_return,
            "pinned_workspace_bytes": sum(
                value.numel() * value.element_size()
                for value in self._buffers.values() if value.device.type == "cpu"),
            "device_workspace_bytes": sum(
                value.numel() * value.element_size()
                for value in self._buffers.values() if value.device.type == "cuda"),
        }

    def release_runtime_buffers(self):
        with self._lock:
            if self._copy_stream is not None:
                self._copy_stream.synchronize()
            if self._stream is not None:
                self._stream.synchronize()
            if self._done is not None:
                self._done.synchronize()
            self._buffers.clear()
            self._stream = self._handoff = self._returned = self._done = None
            self._copy_stream = self._cold_ready = None
            self._pending = False

    def close(self):
        with self._lock:
            if self._closed:
                return
            self.worker.shutdown(wait=True)
            self.executor.close()
            if self._copy_stream is not None:
                self._copy_stream.synchronize()
            if self._stream is not None:
                self._stream.synchronize()
            if self._done is not None:
                self._done.synchronize()
            self._buffers.clear()
            self._stream = self._handoff = self._returned = self._done = None
            self._copy_stream = self._cold_ready = None
            self._pending = False
            self._closed = True
