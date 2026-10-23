"""A reusable native-Windows execution engine for official Drama 3 weights."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import random
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from fish_speech.models.text2semantic.fish_omni.generate import (
    build_drama3_prompt,
    ensure_drama3_caches,
    validate_drama3_text,
)
from fish_speech.models.text2semantic.inference import resolve_codec_path
from fish_speech.models.text2semantic.sampling import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)

from .autotune import (
    GIB,
    autotune,
    inspect_hardware,
)
from .host_budget import HostAllocationState, estimate_host_requirements, plan_host_memory
from .codec import (
    codec_operation_modules, codec_operation_roots,
    codec_supports_split_decode, codec_workspace_bytes,
    decode_to_audio, encode_audio, load_codec, release_codec, write_wav,
)
from .generation import generate, prefill_reference_prefix
from .sampling import DEFAULT_TOP_K
from .model import configure_model_runtime, load_offloaded_drama3, release_drama3
from .offload import TritonExpertBackend
from .semantic_head import SemanticHead


class CodecCoexistenceUnavailable(RuntimeError):
    """Codec staging was refused before it could park a live sequence."""


@dataclass(frozen=True)
class EngineOptions:
    checkpoint: Path
    device: str = "cuda:0"
    expert_storage: str = "auto"
    expert_cache_dir: Path | None = None
    compact_head: bool = True
    tower_graph: bool = True
    backbone_graph: bool = False
    combined_decode_bank: bool = False
    packed_staging: bool = False
    profile_transfers: bool = False
    host_resident_gib: float = 0.0
    codec_policy: str = "auto"
    cache_experts: int | None = None
    staged_experts: int | None = None
    pair_rows: int | None = None
    prefill_chunk: int | None = None
    codec_max_frames: int = 1024
    reference_cache_entries: int = 8
    expert_decode: str = "auto"
    cpu_experts: int = 0
    native_threads: int = 4
    prefix_cache_mib: int = 0
    host_gather: str = "torch"
    codec_placement: str = "auto"
    context_bucket_size: int = 512
    compute_policy: str = "manual"
    compute_profile_dir: Path | None = None


def seed_sampling(seed: int):
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an integer in [0, 2**64)")
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DramaEngine:
    """Own model resources across serialized requests, including voice references.

    Each instance is tied to one checkpoint and GPU. Model and reference codes
    stay loaded between requests. Disk mode maps immutable original FP8 bytes;
    the OS controls file residency. ``close`` ends the engine's resource lifetime.
    """

    def __init__(self, options: EngineOptions):
        self.options = options
        if options.expert_storage not in ("auto", "ram", "mmap"):
            raise ValueError("expert_storage must be auto, ram or mmap")
        if options.codec_policy not in ("auto", "park"):
            raise ValueError("codec_policy must be auto or park")
        if options.codec_placement not in ("auto", "full"):
            raise ValueError("codec_placement must be auto or full")
        if options.expert_decode not in ("auto", "official"):
            raise ValueError("expert_decode must be auto or official")
        if options.host_gather not in ("auto", "torch", "native"):
            raise ValueError("host_gather must be auto, torch or native")
        if options.compute_policy not in ("manual", "auto"):
            raise ValueError("compute_policy must be manual or auto")
        if (isinstance(options.cpu_experts, bool) or not isinstance(options.cpu_experts, int)
                or not 0 <= options.cpu_experts <= 8):
            raise ValueError("cpu_experts must be an integer from 0 to 8")
        if (isinstance(options.native_threads, bool)
                or not isinstance(options.native_threads, int)
                or not 1 <= options.native_threads <= 64):
            raise ValueError("native_threads must be an integer from 1 to 64")
        if (isinstance(options.prefix_cache_mib, bool)
                or not isinstance(options.prefix_cache_mib, int) or options.prefix_cache_mib < 0):
            raise ValueError("prefix_cache_mib must be a non-negative integer")
        if (isinstance(options.context_bucket_size, bool)
                or not isinstance(options.context_bucket_size, int)
                or not 0 <= options.context_bucket_size <= 4096):
            raise ValueError("context_bucket_size must be an integer from 0 to 4096")
        if (isinstance(options.host_resident_gib, bool)
                or not math.isfinite(options.host_resident_gib)
                or options.host_resident_gib < 0):
            raise ValueError("host_resident_gib must be finite and non-negative")
        for value in (options.staged_experts, options.pair_rows, options.prefill_chunk):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError("Batching overrides must be positive integers")
        if options.codec_max_frames < 1 or options.reference_cache_entries < 0:
            raise ValueError("Invalid codec or reference-cache limit")
        self.checkpoint = Path(options.checkpoint).resolve()
        self.device = torch.device(options.device)
        if self.device.type != "cuda":
            raise ValueError("The current engine requires a CUDA GPU")
        if self.device.index is None:
            self.device = torch.device("cuda:0")
        self.model = None
        self.manager = None
        self.codec = None
        self.graph = None
        self.backbone_graph = None
        self._lock = threading.RLock()
        self._references = OrderedDict()
        self._request_index = 0
        self._closed = False
        self._transition_failure = None
        self.load_seconds = 0.0
        self.runtime = None
        self.storage = None
        self.host_plan = None
        self._host_resident_budget = 0
        self._parked = False
        self._cpu_mirrors = {}
        self._codec_cpu_tensors = {}
        self._ready_plan = None
        self._auto_prefill_chunk_limit = None
        self._graph_error = None
        self._codec_force_park = False
        self._codec_preserve_depth = 0
        self._codec_completed_depth = 0
        self.prefix_cache = None
        if options.prefix_cache_mib:
            from .prefix_cache import ReferencePrefixCache

            self.prefix_cache = ReferencePrefixCache(
                options.prefix_cache_mib * 1024**2, admission=self._admit_prefix_snapshot)
        self._lifecycle = {
            "runtime_reuses": 0, "runtime_rebuilds": 0,
            "park_calls": 0, "park_seconds": 0.0,
            "resume_calls": 0, "resume_seconds": 0.0,
            "codec_coexist_calls": 0, "codec_park_calls": 0,
            "codec_oom_retries": 0,
            "codec_cache_reclaims": 0,
            "codec_sequence_reclaims": 0,
            "codec_selective_calls": 0,
            "codec_staged_bytes": 0,
            "codec_split_decode_calls": 0,
        }
        self._last_prepare_timing = {}
        self._prefix_admission_report = {"file_page_admissions": 0, "last_credit_bytes": 0}
        self._active_output_controls = ()
        self._task_candidates = None
        self._computing_policy = False
        self._compute_initialized = False
        self.compute_report = {
            "mode": options.compute_policy,
            "status": "manual" if options.compute_policy == "manual" else "pending",
        }
        self.status_callback = None

    def _emit_status(self, event):
        if self.status_callback is not None:
            self.status_callback(event)

    def _batching(self):
        return (
            self.options.staged_experts
            if self.options.staged_experts is not None else self.runtime.max_staged_experts,
            self.options.pair_rows
            if self.options.pair_rows is not None else self.runtime.max_pair_rows,
            self.options.prefill_chunk
            if self.options.prefill_chunk is not None else min(
                self.runtime.prefill_chunk,
                self._auto_prefill_chunk_limit or self.runtime.prefill_chunk),
        )

    def _remember_prefill_chunk(self, report):
        """Retain a successful automatic OOM reduction through later requests."""
        actual = report.get("chunk_size")
        if self.options.prefill_chunk is not None or actual is None:
            return
        self._auto_prefill_chunk_limit = min(
            self._auto_prefill_chunk_limit or actual, actual)
        previous = self._ready_plan
        if previous is not None and actual < previous.prefill_chunk:
            self._ready_plan = replace(
                previous, prefill_chunk=actual,
                warnings=previous.warnings + (
                    f"Automatic prefill reduced to {actual} rows after actual prompt allocation pressure.",
                ),
            )
            self.manager.runtime_plan = self._ready_plan
            self.model._drama_low_vram_runtime_plan = self._ready_plan

    def _admit_prefix_snapshot(self, size):
        """Charge only a future snapshot against current available RAM/commit."""
        hardware = inspect_hardware(str(self.device))
        margin = self.host_plan.buffer_bytes if self.host_plan is not None else 256 * 1024**2
        commit = getattr(hardware, "commit_available_bytes", None)
        # Snapshot memory survives parking. Leave room for future model mirrors
        # and codec construction that have not yet been allocated, counted once.
        future = 0
        if self.host_plan is not None:
            requirements = self.host_plan.requirements
            mirrors = sum(value.numel() * value.element_size() for value in self._cpu_mirrors.values())
            future += max(0, requirements.parking_bytes - mirrors)
            if self.codec is None:
                future += requirements.codec_build_bytes
        required = size + margin + future
        if commit is not None and commit < required:
            return False
        if hardware.ram_available_bytes >= required:
            return True
        # A small reference state can replace clean expert file pages. Current
        # "available" RAM excludes mapped pages in this process's working set.
        # Credit only actual valid, unlocked pages; never assume the entire
        # mapping is resident or charge anonymous CPU buffers as reclaimable.
        if (
            self.prefix_cache is None or self.prefix_cache.budget_bytes > 256 * 1024**2
            or size > 128 * 1024**2
            or getattr(self.model, "_drama_low_vram_weight_sources", None) is None
        ):
            return False
        from .weight_sources import reclaimable_file_bytes

        tensors = [
            tensor for module in self.manager.modules.values()
            if getattr(module, "_drama_expert_storage", None) == "mmap"
            for tensor in module._buffers.values()
            if tensor is not None and tensor.device.type == "cpu"
        ]
        if not tensors:
            return False
        try:
            credit = reclaimable_file_bytes(tensors, required - hardware.ram_available_bytes)
        except (MemoryError, OSError, AttributeError):
            # This is an optional cache admission probe, not a prerequisite
            # for continuing the ordinary generation path.
            return False
        self._prefix_admission_report["last_credit_bytes"] = credit
        admitted = hardware.ram_available_bytes + credit >= required
        self._prefix_admission_report["file_page_admissions"] += int(admitted)
        return admitted

    def load(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("This engine is closed")
            if self._transition_failure is not None:
                raise RuntimeError(self._transition_failure)
            if self.model is not None:
                return self
            hardware = inspect_hardware(str(self.device))
            from .nvidia_policy import policy_for
            self.hardware_policy = policy_for(hardware.compute_capability or (0, 0))
            if not self.hardware_policy.supported:
                raise RuntimeError("The Windows NVIDIA runtime requires an RTX20/SM75 or newer CUDA GPU")
            if self.options.cpu_experts and not self.hardware_policy.experimental_hybrid:
                raise ValueError("CPU expert cooperation is currently validated only on SM89; set cpu_experts=0")
            estimated_staging = self.options.staged_experts or 4
            requirements = estimate_host_requirements(
                self.checkpoint if (self.checkpoint / "config.json").is_file() else None,
                compact_head=self.options.compact_head,
                staged_experts=estimated_staging,
                codec_max_frames=self.options.codec_max_frames,
            )
            allocated = HostAllocationState(codec_bytes=_cpu_tensor_bytes(self.codec))
            from .quantized_model import quantization_mode
            from .shared_checkpoint import is_shared
            quant_mode = quantization_mode(self.checkpoint)
            # Packed checkpoints stay file-backed; resident OS pages may still
            # occupy spare RAM. Only GPTQ's repacked bank is private RAM.
            planned_storage = ("ram" if quant_mode == 0 else
                               "mmap" if is_shared(self.checkpoint) else self.options.expert_storage)
            self.host_plan = plan_host_memory(
                ram_available_bytes=hardware.ram_available_bytes,
                commit_available_bytes=getattr(hardware, "commit_available_bytes", None),
                swap_available_bytes=getattr(hardware, "pagefile_available_bytes", 0),
                expert_storage=planned_storage,
                requirements=requirements, allocated=allocated,
            )
            self.storage = self.host_plan.expert_storage
            self.runtime = autotune(
                hardware, expert_storage=self.storage,
                host_requirements=requirements, host_allocated=allocated,
            )
            # Hardware tiers select staging independently of host storage.
            # Reuse the checkpoint metadata with that actual capacity before
            # choosing the final resident bank and admitting allocations.
            staged_experts = (
                self.options.staged_experts
                if self.options.staged_experts is not None
                else getattr(self.runtime, "max_staged_experts", estimated_staging)
            )
            if staged_experts != estimated_staging:
                from dataclasses import replace

                requirements = replace(
                    requirements,
                    staging_bytes=requirements.staging_bytes // estimated_staging * staged_experts,
                )
                self.host_plan = plan_host_memory(
                    ram_available_bytes=hardware.ram_available_bytes,
                    commit_available_bytes=getattr(hardware, "commit_available_bytes", None),
                    swap_available_bytes=getattr(hardware, "pagefile_available_bytes", 0),
                    expert_storage=planned_storage,
                    requirements=requirements, allocated=allocated,
                )
                self.storage = self.host_plan.expert_storage
                self.runtime = autotune(
                    hardware, expert_storage=self.storage,
                    host_requirements=requirements, host_allocated=allocated,
                )
            if not self.runtime.can_run:
                raise RuntimeError("; ".join(self.runtime.warnings))
            staged_experts, pair_rows, _ = self._batching()
            cache_dir = None
            if self.storage == "mmap" and not is_shared(self.checkpoint) and quant_mode is None:
                from .expert_cache_file import build_expert_cache

                cache_dir = self.options.expert_cache_dir
                if cache_dir is None:
                    identity = hashlib.sha256(str(self.checkpoint).encode("utf-8")).hexdigest()[:16]
                    cache_dir = Path(__file__).resolve().parents[2] / ".cache" / f"drama3-fp8-experts-{identity}"
                cache_dir = Path(cache_dir).resolve()
                if not cache_dir.exists():
                    print(f"Preparing original FP8 expert file cache on {cache_dir}", flush=True)
                    build_expert_cache(self.checkpoint, cache_dir)
            started = time.perf_counter()
            self.model, self.manager = load_offloaded_drama3(
                self.checkpoint, device=self.device,
                dtype=(torch.float16 if self.hardware_policy.compute_dtype == "float16" else torch.bfloat16),
                expert_backend=(
                    TritonExpertBackend(decode_specialization=False)
                    if self.runtime.backend == "triton" and self.options.expert_decode == "official"
                    else self.runtime.backend
                ),
                lm_head_device="cpu",
                max_staged_experts=staged_experts,
                max_pair_rows=pair_rows,
                pin_memory=self.runtime.pin_memory,
                semantic_head=self.options.compact_head,
                expert_cache_dir=cache_dir,
                profile_transfers=self.options.profile_transfers,
                combined_decode_bank=self.options.combined_decode_bank,
                packed_staging=self.options.packed_staging,
            )
            self.load_seconds = time.perf_counter() - started
            if self.options.host_gather != "torch":
                try:
                    self.manager.configure_host_gather(self.options.host_gather)
                except Exception:
                    release_drama3(self.model, self.manager)
                    self.model = self.manager = None
                    raise
            if self.options.cpu_experts and quant_mode is None:
                try:
                    self.manager.configure_hybrid_decode(
                        self.options.cpu_experts, threads=self.options.native_threads,
                        device=self.device)
                except Exception:
                    release_drama3(self.model, self.manager)
                    self.model = self.manager = None
                    raise
            if self.storage == "mmap" and self.options.host_resident_gib:
                self._host_resident_budget = min(
                    int(self.options.host_resident_gib * GIB),
                    self.host_plan.expert_ram_budget_bytes,
                )
                self.manager.configure_host_residency(self._host_resident_budget)
            return self

    def rebalance_host_residency(self):
        """Apply observed route ranks at a serialized, explicit request boundary."""
        with self._lock:
            self.load()
            return self.manager.rebalance_host_residency()

    def _release_graphs(self):
        if self.graph is not None:
            self.graph.close()
            self.graph = None
        if self.backbone_graph is not None:
            self.backbone_graph.close()
            self.backbone_graph = None

    def _select_output_controls(self, controls):
        controls = tuple(controls)
        if any(token != "<|audio_separator|>" for token in controls):
            raise ValueError("Unsupported generated task control")
        controls = tuple(sorted(set(controls)))
        if controls == self._active_output_controls and (not controls or self._task_candidates is not None):
            return
        if controls:
            from .control_head import ControlHead

            backbone = self.model.backbone
            if not hasattr(backbone, "protocol_head"):
                backbone.protocol_head = ControlHead.load(
                    self.checkpoint, self.model.tokenizer, backbone.lm_head, controls)
            self._task_candidates = backbone.protocol_head.candidates(backbone.lm_head)
        else:
            self._task_candidates = None
        # Only the tower sampler/candidate shape changes; keep the backbone
        # graph and its cache addresses whenever the context still fits.
        if self.graph is not None:
            self.graph.close()
            self.graph = None
        self._ready_plan = None
        self._active_output_controls = controls

    def _prepare(self, capacity, *, output_controls=()):
        max_capacity = getattr(
            getattr(getattr(self.model, "config", None), "text", None),
            "max_position_embeddings", None,
        )
        if (
            isinstance(capacity, bool)
            or not isinstance(capacity, int)
            or capacity < 1
            or (max_capacity is not None and capacity > max_capacity)
        ):
            raise ValueError("sequence_capacity is outside the model's supported context")
        started = time.perf_counter()
        self._resume_model()
        self._select_output_controls(output_controls)
        timing = {"resume_seconds": time.perf_counter() - started}
        if self._ready_plan is not None and self.model.cache_capacity >= capacity:
            self._lifecycle["runtime_reuses"] += 1
            self._last_prepare_timing = {
                **timing, "total_seconds": time.perf_counter() - started,
                "reused": True,
            }
            self._prepare_compute_policy()
            return self._ready_plan, self._graph_error
        self._lifecycle["runtime_rebuilds"] += 1
        self._ready_plan = None
        requested_capacity = capacity
        quantum = self.options.context_bucket_size
        capacity = (
            (capacity + quantum - 1) // quantum * quantum if quantum else capacity
        )
        if max_capacity is not None:
            capacity = min(capacity, max_capacity)
        if self.model.cache_capacity < capacity:
            self._release_graphs()
        staged_experts, pair_rows, prefill_chunk = self._batching()
        stage_started = time.perf_counter()
        self.manager.release_runtime_buffers()
        self.manager.configure_staging(staged_experts, pair_rows)
        timing["buffer_reset_seconds"] = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        bucket_fallback = False
        try:
            ensure_drama3_caches(self.model, capacity)
        except torch.cuda.OutOfMemoryError:
            if capacity == requested_capacity:
                raise
            # A bucket is a reuse optimization, never a reason to reject an
            # otherwise fitting request. Partial allocations must not survive.
            self._clear_sequence_caches()
            gc.collect()
            torch.cuda.empty_cache()
            capacity = requested_capacity
            bucket_fallback = True
            ensure_drama3_caches(self.model, capacity)
        timing["sequence_cache_seconds"] = time.perf_counter() - stage_started
        timing.update(
            requested_sequence_capacity=requested_capacity,
            allocated_sequence_capacity=self.model.cache_capacity,
            context_bucket_fallback=bucket_fallback,
        )
        graph_error = None
        stage_started = time.perf_counter()
        if self.options.backbone_graph and self.backbone_graph is None:
            from .backbone_graph import BackboneDecodeGraph

            self.backbone_graph = BackboneDecodeGraph(
                self.model.backbone, self.manager,
                sequence_capacity=self.model.cache_capacity,
            )
        timing["backbone_graph_seconds"] = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        if self.options.tower_graph and self.graph is None:
            from .tower_graph import TowerDecodeGraph, TowerGraphRecoveryError

            try:
                self.graph = TowerDecodeGraph(
                    self.model.audio_decoder, self.model.cache_capacity,
                    compact_logits=isinstance(self.model.backbone.lm_head, SemanticHead),
                    logits_size=(
                        None if isinstance(self.model.backbone.lm_head, SemanticHead)
                        else self.model.config.text.vocab_size
                    ),
                    candidate_ids=self._task_candidates,
                )
            except TowerGraphRecoveryError as exc:
                self._transition_failure = str(exc)
                raise
            except torch.cuda.OutOfMemoryError:
                self.graph = None
                gc.collect()
                torch.cuda.empty_cache()
                graph_error = "Tower graph allocation exceeded current VRAM; using eager tower."
        timing["tower_graph_seconds"] = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        plan = configure_model_runtime(
            self.model, self.manager, sequence_capacity=capacity,
            runtime_config=self.runtime,
            cache_experts=self.options.cache_experts,
            prefill_chunk_size=prefill_chunk,
            allow_staging_reduction=all(value is None for value in (
                self.options.staged_experts, self.options.pair_rows, self.options.prefill_chunk)),
        )
        timing["runtime_plan_seconds"] = time.perf_counter() - stage_started
        if self.backbone_graph is not None:
            print(f"Backbone graph: {self.backbone_graph.description}", flush=True)
        self._ready_plan, self._graph_error = plan, graph_error
        self._last_prepare_timing = {
            **timing, "total_seconds": time.perf_counter() - started, "reused": False,
        }
        self._prepare_compute_policy()
        return self._ready_plan, self._graph_error

    def _prepare_compute_policy(self):
        policy = getattr(self, "hardware_policy", None)
        if policy is not None and not policy.experimental_hybrid:
            self._compute_initialized = True
            self.compute_report = {"mode": self.options.compute_policy, "status": "generic_gpu",
                                   "cpu_experts": 0, "hardware_policy": policy.report()}
            return
        if (self.options.compute_policy != "auto" or self._compute_initialized
                or self._computing_policy):
            return
        quantized = getattr(getattr(self.manager, "backend", None), "quantized_mode", None)
        if quantized:
            self.compute_report = {
                "status": "quantized_gpu", "backend": quantized,
                "cpu_experts": 0, "scope": "Compressed-bank GPU path; FP8-only CPU executor is incompatible.",
            }
            self._compute_initialized = True
            return
        from .calibration import ensure_compute_policy

        self._computing_policy = True
        timing = dict(self._last_prepare_timing)
        started = time.perf_counter()
        try:
            self._emit_status({"stage": "compute_policy", "status": "checking"})
            self.compute_report = ensure_compute_policy(self)
            self._compute_initialized = True
            self._emit_status({"stage": "compute_policy", **self.compute_report})
        finally:
            self._computing_policy = False
            elapsed = time.perf_counter() - started
            self._last_prepare_timing = {
                **timing, "compute_policy_seconds": elapsed,
                "total_seconds": timing["total_seconds"] + elapsed,
            }

    def _clear_sequence_caches(self):
        if self.model is None:
            return
        for layer in self.model.backbone.model.layers:
            if layer.layer_type == "full_attention":
                layer.self_attn.k_cache = None
                layer.self_attn.v_cache = None
            else:
                layer.linear_attn.conv_state = None
                layer.linear_attn.recurrent_state = None
        self.model.backbone.inverse_frequencies.clear()
        for layer in (*self.model.audio_decoder.shared_layers, *self.model.audio_decoder.residual_layers):
            layer.transformer_layer.attention.k_cache = None
            layer.transformer_layer.attention.v_cache = None
        self.model.audio_decoder.main_candidates = None
        self.model.cache_capacity = 0

    def _park_model(self):
        if self.model is None or self._parked:
            return
        if self._codec_preserve_depth:
            raise CodecCoexistenceUnavailable(
                "Cannot park the model while its live sequence has no snapshot")
        started = time.perf_counter()
        try:
            self._ready_plan = None
            self._release_graphs()
            self.manager.release_runtime_buffers()
            self._clear_sequence_caches()
            # Traverse direct tensors to leave immutable file-backed expert banks
            # untouched. Moving entire modules would duplicate mapped weights.
            experts = {id(module) for module in self.manager.modules.values()}
            sources = getattr(self.model, "_drama_low_vram_weight_sources", None)
            for module in self.model.modules():
                if id(module) in experts:
                    continue
                for name, parameter in list(module._parameters.items()):
                    if parameter is not None and parameter.is_cuda:
                        key = (id(module), "parameter", name)
                        mirror = self._cpu_mirrors.get(key)
                        if mirror is None and sources is not None:
                            mirror = sources.get(module, "parameter", name)
                        if mirror is None:
                            mirror = torch.nn.Parameter(
                                parameter.detach().cpu(), requires_grad=False
                            )
                        self._cpu_mirrors[key] = mirror
                        module._parameters[name] = mirror
                for name, buffer in list(module._buffers.items()):
                    if buffer is not None and buffer.is_cuda:
                        key = (id(module), "buffer", name)
                        mirror = self._cpu_mirrors.get(key)
                        if mirror is None and sources is not None:
                            mirror = sources.get(module, "buffer", name)
                        if mirror is None:
                            mirror = buffer.cpu()
                        self._cpu_mirrors[key] = mirror
                        module._buffers[name] = mirror
        except Exception:
            self._transition_failure = (
                "Engine weight parking failed; close this engine before retrying."
            )
            raise
        self._parked = True
        gc.collect()
        torch.cuda.empty_cache()
        self._lifecycle["park_calls"] += 1
        self._lifecycle["park_seconds"] += time.perf_counter() - started

    def _resume_model(self):
        if self.model is None or not self._parked:
            return
        started = time.perf_counter()
        experts = {id(module) for module in self.manager.modules.values()}
        embedding = id(self.model.backbone.model.embed_tokens)
        head = id(self.model.backbone.lm_head)
        # A failed transfer can leave a mixed placement. Mark it unparked
        # first so the rollback scans and restores every CUDA tensor.
        self._parked = False
        try:
            for module in self.model.modules():
                if id(module) in experts or id(module) == embedding:
                    continue
                if id(module) == head and not self.options.compact_head:
                    continue
                for name, parameter in list(module._parameters.items()):
                    if parameter is not None:
                        module._parameters[name] = torch.nn.Parameter(
                            parameter.detach().to(self.device), requires_grad=False
                        )
                for name, buffer in list(module._buffers.items()):
                    if buffer is not None:
                        module._buffers[name] = buffer.to(self.device)
        except Exception:
            self._park_model()
            raise
        self._lifecycle["resume_calls"] += 1
        self._lifecycle["resume_seconds"] += time.perf_counter() - started

    def _codec_can_coexist(self, max_frames, *, reclaimable_bytes=0, operation="all"):
        if self.model is None:
            return True
        if self._parked or self._codec_force_park or self.options.codec_policy == "park":
            return False
        free, _ = torch.cuda.mem_get_info(self.device)
        reusable = max(0, torch.cuda.memory_reserved(self.device)
                       - torch.cuda.memory_allocated(self.device))
        weights = sum(
            tensor.numel() * tensor.element_size()
            for module in codec_operation_modules(
                self.codec, operation if self.options.codec_placement == "auto" else "all")
            for tensor in (*module._parameters.values(), *module._buffers.values())
            if tensor is not None
        )
        # Admission estimate only: a real allocation failure retries once with
        # the backbone parked. Codec activation memory grows with clip length.
        workspace_operation = operation if self.options.codec_placement == "auto" else "all"
        if workspace_operation == "decode_latents" and not codec_supports_split_decode(self.codec):
            workspace_operation = "all"
        workspace = codec_workspace_bytes(max_frames, workspace_operation)
        if workspace_operation == "decode_waveform":
            from .codec_decoder import decoder_workspace_bytes

            bounded = decoder_workspace_bytes(self.codec, max_frames)
            if bounded is not None:
                workspace = bounded
        return free + reusable + reclaimable_bytes >= weights + workspace

    @torch.inference_mode()
    def _decode_audio_segments(self, segments):
        """Keep official whole-sequence math, stage its two serial branches."""
        maximum = max(int(codes.shape[-1]) for codes in segments)
        if self.codec is None and self.options.codec_placement == "auto":
            self.codec = load_codec(
                resolve_codec_path(self.checkpoint), device="cpu",
                dtype=(torch.float16 if getattr(getattr(self, "hardware_policy", None), "compute_dtype", "") == "float16"
                       else torch.bfloat16), max_frames=maximum,
            )
        prefer_full = (
            self.options.codec_placement == "auto"
            and self._full_codec_decode_fits(maximum)
        )
        if (not prefer_full and self.options.codec_placement == "auto" and self.codec is not None
                and codec_supports_split_decode(self.codec)):
            self._lifecycle["codec_split_decode_calls"] += 1
            outputs = []
            for codes in segments:
                frames = int(codes.shape[-1])
                # The checkpoint decoder mutates its index argument while
                # clamping. Use a private GPU copy just like decode_to_audio.
                latents = self._run_codec(
                    frames,
                    lambda codec: codec.quantizer.decode(
                        codes.to(device=self.device, dtype=torch.long, copy=True)[None]),
                    direction="decode_latents",
                )
                try:
                    audio = self._run_codec(
                        frames, lambda codec: codec.decoder(latents)[0, 0].cpu(),
                        direction="decode_waveform",
                    )
                finally:
                    del latents
                if not bool(torch.isfinite(audio).all()):
                    raise FloatingPointError("Non-finite codec output")
                outputs.append(audio)
            return torch.cat(outputs)

        def decode_segments(codec):
            audio_segments = []
            for codes in segments:
                audio = decode_to_audio(codes, codec, self.device)
                if not bool(torch.isfinite(audio).all()):
                    raise FloatingPointError("Non-finite codec output")
                audio_segments.append(audio.cpu())
            return torch.cat(audio_segments)

        return self._run_codec(
            maximum, decode_segments, direction="all" if prefer_full else "decode")

    def _full_codec_decode_fits(self, frames):
        """Keep the original codec path when it can already preserve the model.

        Under tight memory cuDNN can choose a different convolution plan.
        Avoid an unnecessary placement change when full placement coexists;
        use serial branches to avoid an otherwise required model parking.
        """
        if self._codec_can_coexist(frames, operation="all"):
            return True
        stats = getattr(self.manager, "stats", None)
        reclaimable = (
            getattr(stats, "hot_cache_bytes", 0) + getattr(stats, "device_buffer_bytes", 0)
        )
        return bool(reclaimable and self._codec_can_coexist(
            frames, reclaimable_bytes=reclaimable, operation="all"))

    @contextmanager
    def _codec_preserving_sequence(self):
        """Permit codec/expert-buffer changes, retaining live KV/GDN and graphs.

        A caller without a sequence snapshot must use this guard across both
        codec stages and runtime re-preparation. Admission failure or a real
        codec OOM returns control without parking or clearing sequence caches.
        """
        with self._lock:
            if self.model is None or self._parked:
                raise CodecCoexistenceUnavailable("A resident model is required")
            self._codec_preserve_depth += 1
            try:
                yield
            finally:
                self._codec_preserve_depth -= 1

    def _run_codec(self, max_frames, operation, *, direction="all"):
        for attempt in range(2):
            try:
                with (
                    self._codec_on_gpu(max_frames) if direction == "all"
                    else self._codec_on_gpu(max_frames, operation=direction)
                ) as codec:
                    return operation(codec)
            except torch.cuda.OutOfMemoryError as exc:
                if self._codec_preserve_depth:
                    raise CodecCoexistenceUnavailable(
                        "Codec exceeded current VRAM while preserving the live sequence"
                    ) from exc
                if attempt or self.model is None or self._parked:
                    raise
            # Leave the exception scope before retrying so failed activations
            # and their traceback no longer retain GPU allocations.
            self._lifecycle["codec_oom_retries"] += 1
            self._codec_force_park = True
            gc.collect()
            self._park_model()

    @contextmanager
    def _codec_on_gpu(self, max_frames, *, operation="all"):
        """Share VRAM when it fits; otherwise park weights without reloading."""
        if self.codec is None:
            self.codec = load_codec(
                resolve_codec_path(self.checkpoint), device="cpu",
                dtype=(torch.float16 if getattr(getattr(self, "hardware_policy", None), "compute_dtype", "") == "float16"
                       else torch.bfloat16), max_frames=max_frames,
            )
        elif self.codec._drama_low_vram_max_frames < max_frames:
            # Increasing only positional buffers preserves all learned weights.
            from fish_speech.models.dac import modded_dac

            context = max_frames * 4
            self._codec_cpu_tensors.clear()
            try:
                for module in self.codec.modules():
                    if isinstance(module, modded_dac.Transformer):
                        # Drop old CPU buffers before constructing replacements;
                        # the previous parking snapshot must not retain them.
                        module.freqs_cis = None
                        module.causal_mask = None
                        if module.config.pos_embed_type == "rope":
                            module.freqs_cis = modded_dac.precompute_freqs_cis(
                                context, module.config.head_dim, module.config.rope_base
                            )
                        module.causal_mask = torch.tril(
                            torch.ones(context, context, dtype=torch.bool)
                        )
            except Exception:
                self._transition_failure = (
                    "Codec context growth failed; close this engine before retrying."
                )
                raise
            self.codec._drama_low_vram_max_frames = max_frames
        direction = operation if self.options.codec_placement == "auto" else "all"
        roots = codec_operation_roots(self.codec, direction)
        modules = codec_operation_modules(self.codec, direction)
        admission = {} if operation == "all" else {"operation": operation}
        coexist = self._codec_can_coexist(max_frames, **admission)
        if not coexist and self.model is not None and not self._parked:
            stats = self.manager.stats
            reclaimable = stats.hot_cache_bytes + stats.device_buffer_bytes
            if reclaimable and self._codec_can_coexist(
                max_frames, reclaimable_bytes=reclaimable, **admission
            ):
                self.manager.release_runtime_buffers()
                self._ready_plan = None
                self._lifecycle["codec_cache_reclaims"] += 1
                gc.collect()
                torch.cuda.empty_cache()
                coexist = self._codec_can_coexist(max_frames, **admission)
        if (
            not coexist and self.model is not None and not self._parked
            and self._codec_completed_depth and not self._codec_preserve_depth
            and not self._codec_force_park and self.options.codec_policy == "auto"
            and getattr(self.model, "cache_capacity", 0) > 0
            and getattr(self.model, "_drama_low_vram_weight_sources", None) is not None
        ):
            # Only an ended ordinary synthesis opts in. Reclaim its KV/GDN
            # caches and dependent graphs before evicting immutable GPU weights.
            # CPU reference snapshots already own independent copies.
            try:
                self._release_graphs()
                self.manager.release_runtime_buffers()
                self._clear_sequence_caches()
                self._ready_plan = None
            except Exception:
                self._transition_failure = (
                    "Completed sequence reclamation failed; close this engine before retrying."
                )
                raise
            self._lifecycle["codec_sequence_reclaims"] += 1
            gc.collect()
            torch.cuda.empty_cache()
            coexist = self._codec_can_coexist(max_frames, **admission)
        if coexist:
            self._lifecycle["codec_coexist_calls"] += 1
        else:
            if self._codec_preserve_depth:
                raise CodecCoexistenceUnavailable(
                    "Codec needs model parking; a live-sequence snapshot is required")
            self._lifecycle["codec_park_calls"] += 1
            self._park_model()
        torch.cuda.empty_cache()
        # Keep the immutable CPU tensors rather than allocating new CPU copies
        # after every request. Windows' allocator may retain freed copies.
        cpu_tensors = {}
        for module in modules:
            for kind, tensors in (("parameter", module._parameters), ("buffer", module._buffers)):
                for name, tensor in tensors.items():
                    if tensor is not None:
                        if tensor.device.type != "cpu":
                            raise RuntimeError("Codec must be parked on CPU before staging")
                        with torch.inference_mode(False):
                            cpu_tensors[(id(module), kind, name)] = (
                                torch.nn.Parameter(tensor.detach(), requires_grad=False)
                                if kind == "parameter" else tensor
                            )
        self._codec_cpu_tensors = cpu_tensors
        self._lifecycle["codec_staged_bytes"] += sum(
            tensor.numel() * tensor.element_size()
            for tensor in cpu_tensors.values()
            if isinstance(tensor, torch.Tensor)
        )
        if roots != (self.codec,):
            self._lifecycle["codec_selective_calls"] += 1
        try:
            for root in roots:
                root.to(self.device)
            yield self.codec
        finally:
            for module in modules:
                for kind, tensors in (("parameter", module._parameters), ("buffer", module._buffers)):
                    for name in list(tensors):
                        key = (id(module), kind, name)
                        if key in self._codec_cpu_tensors:
                            tensors[name] = self._codec_cpu_tensors[key]
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()

    def prepare_reference(self, audio: Path, transcript: str):
        """Cache immutable reference codes; changed audio contents invalidate them."""
        with self._lock:
            if self._closed:
                raise RuntimeError("This engine is closed")
            if self._transition_failure is not None:
                raise RuntimeError(self._transition_failure)
            validate_drama3_text(transcript, "reference")
            audio = Path(audio).resolve()
            digest = hashlib.sha256()
            with audio.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            key = (digest.hexdigest(), transcript)
            if key in self._references:
                codes = self._references.pop(key)
                self._references[key] = codes
                return codes.clone(), True
            # Read duration before allocating codec buffers.
            import soundfile as sf

            info = sf.info(audio)
            frames = max(1, (info.frames * 44100 + info.samplerate * 2048 - 1) // (info.samplerate * 2048))
            if frames > self.options.codec_max_frames:
                raise ValueError(
                    f"Reference needs {frames} frames; codec budget is "
                    f"{self.options.codec_max_frames}. Use a shorter reference."
                )
            codes = self._run_codec(
                frames, lambda codec: encode_audio(audio, codec, self.device).cpu(),
                direction="encode",
            )
            if self.options.reference_cache_entries:
                self._references[key] = codes
                while len(self._references) > self.options.reference_cache_entries:
                    self._references.popitem(last=False)
            return codes.clone(), False

    def reference_prefix_cached(self, key):
        from .prefix_cache import prefix_execution_key

        return bool(
            key and self.model is not None and self.prefix_cache is not None
            and key[3] == prefix_execution_key(self.model)
            and key[1] == self._batching()[2]
            and self.prefix_cache.peek(key) is not None
        )

    @torch.inference_mode()
    def prewarm_reference_prefix(self, references, *, max_new_tokens=768,
                                 cancelled=None, on_stage=None):
        """Load and cache reference conditioning without generating target speech."""
        from .protocols import build_tts_prompt

        with self._lock:
            if self._closed:
                raise RuntimeError("This engine is closed")
            if cancelled is not None and cancelled():
                return {"status": "cancelled"}
            if on_stage:
                on_stage("loading_model")
            self.load()
            transcripts, codes = [], []
            if on_stage:
                on_stage("encoding_reference")
            for reference in references:
                if cancelled is not None and cancelled():
                    return {"status": "cancelled"}
                transcript = reference.transcript()
                encoded, _hit = self.prepare_reference(Path(reference.audio), transcript)
                transcripts.append(transcript)
                codes.append(encoded)
            if cancelled is not None and cancelled():
                return {"status": "cancelled"}
            prompt = build_tts_prompt(
                "Reference warm-up.", self.model.tokenizer, self.model.config,
                transcripts=transcripts, references=codes,
            )
            if on_stage:
                on_stage("preparing_runtime")
            prepare_started = time.perf_counter()
            plan, graph_error = self._prepare(len(prompt) + max_new_tokens)
            prepare_seconds = time.perf_counter() - prepare_started
            if on_stage:
                on_stage("warming_prefix")
            result = prefill_reference_prefix(
                self.model, prompt, chunk_size=plan.prefill_chunk,
                prefix_cache=self.prefix_cache, cancelled=cancelled,
                allow_oom_fallback=self.options.prefill_chunk is None,
            )
            self._remember_prefill_chunk(result)
            return {
                **result,
                "runtime_prepare_seconds": prepare_seconds,
                "graph_warning": graph_error,
            }

    def synthesize(
        self, text: str, output: Path, *, reference_audio: Path | None = None,
        reference_text: str | None = None, seed: int = 123,
        max_new_tokens: int = 768, temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K, top_p: float = DEFAULT_TOP_P,
        max_chunk_chars: int = 160, allow_partial: bool = False,
        progress: Callable[[int, int], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ):
        from .text_chunks import split_text

        if bool(reference_audio) != bool(reference_text):
            raise ValueError("Voice cloning requires both reference audio and transcript")
        if (
            isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int)
            or max_new_tokens < 1
        ):
            raise ValueError("max_new_tokens must be a positive integer")
        if max_new_tokens > self.options.codec_max_frames:
            raise ValueError("Generation frame limit exceeds the engine codec budget")
        if not isinstance(max_chunk_chars, int) or isinstance(max_chunk_chars, bool) or max_chunk_chars < 1:
            raise ValueError("max_chunk_chars must be a positive integer")
        if not 0 < temperature < 2 or not 0 < top_p <= 1:
            raise ValueError("Invalid sampling parameters")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        validate_drama3_text(text, "text")
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**64:
            raise ValueError("seed must be an integer in [0, 2**64)")
        if reference_text is not None:
            validate_drama3_text(reference_text, "reference")
        output = Path(output).resolve()
        with self._lock:
            if self._closed:
                raise RuntimeError("This engine is closed")
            request_started = time.perf_counter()
            lifecycle_before = dict(self._lifecycle)
            reference_started = time.perf_counter()
            reference_codes, reference_hit = None, False
            if reference_audio is not None:
                reference_codes, reference_hit = self.prepare_reference(
                    reference_audio, reference_text
                )
            reference_seconds = time.perf_counter() - reference_started
            was_loaded = self.model is not None
            self.load()
            self._request_index += 1
            chunks = split_text(text, max_chars=max_chunk_chars)
            history, generated_codes, results = [], [], []
            counters_before = self.manager.stats.as_dict()
            for index, chunk in enumerate(chunks):
                prompt = build_drama3_prompt(
                    chunk, self.model.tokenizer, self.model.config,
                    prompt_texts=[reference_text] if reference_codes is not None else [],
                    prompt_tokens=[reference_codes] if reference_codes is not None else [],
                    history=history[-1:],
                )
                prepare_started = time.perf_counter()
                plan, graph_error = self._prepare(len(prompt) + max_new_tokens)
                prepare_seconds = time.perf_counter() - prepare_started
                seed_sampling((seed + index) % 2**64)
                torch.cuda.reset_peak_memory_stats(self.device)
                result = generate(
                    self.model, prompt, max_new_tokens=max_new_tokens,
                    prefill_chunk_size=plan.prefill_chunk, temperature=temperature,
                    top_k=top_k, top_p=top_p, tower_graph=self.graph,
                    backbone_graph=self.backbone_graph,
                    progress=progress, cancelled=cancelled,
                    prefix_cache=self.prefix_cache,
                    allow_prefill_fallback=self.options.prefill_chunk is None,
                )
                self._remember_prefill_chunk(result.prefix_cache)
                plan = self._ready_plan or plan
                item = result.as_dict()
                item.update({
                    "text": chunk,
                    "runtime_prepare_seconds": prepare_seconds,
                    "runtime_prepare_breakdown": dict(self._last_prepare_timing),
                    "runtime_plan": plan.as_dict(),
                    "tower": self.graph.description() if self.graph else {"backend": "eager", "warning": graph_error},
                    "backbone": (
                        self.backbone_graph.description if self.backbone_graph is not None
                        else {"enabled": False}
                    ),
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)),
                })
                results.append(item)
                if len(result.frames):
                    codes = result.frames[:, 1:].T.contiguous()
                    item["semantic_codes_sha256"] = hashlib.sha256(
                        codes.numpy().tobytes()
                    ).hexdigest()
                    generated_codes.append(codes)
                if result.finish_reason != "eos" or not len(result.frames):
                    if not len(result.frames):
                        item["complete"] = False
                        item["finish_reason"] = "empty"
                    break
                history.append((chunk, codes))
            complete = len(results) == len(chunks) and all(
                result["finish_reason"] == "eos" for result in results
            )
            audio_info = None
            codec_seconds = 0.0
            if generated_codes and (complete or allow_partial):
                codec_started = time.perf_counter()
                self._codec_completed_depth += 1
                try:
                    audio = self._decode_audio_segments(generated_codes)
                finally:
                    self._codec_completed_depth -= 1
                write_wav(audio, output, sample_rate=44100)
                codec_seconds = time.perf_counter() - codec_started
                audio_info = {
                    "sample_rate": 44100, "samples": int(audio.numel()),
                    "duration_seconds": audio.numel() / 44100,
                }
            counters_after = self.manager.stats.as_dict()
            hybrid = getattr(self.manager, "hybrid_decode", None)
            hybrid_report = hybrid.report() if hybrid is not None else None
            report = {
                "status": "complete" if complete else "incomplete",
                "output": str(output) if audio_info else None,
                "text": text, "seed": seed, "request_index": self._request_index,
                "cold_model_load_seconds": 0.0 if was_loaded else self.load_seconds,
                "reference_seconds": reference_seconds,
                "reference_cache_hit": reference_hit,
                "codec_seconds": codec_seconds,
                "total_seconds": time.perf_counter() - request_started,
                "audio": audio_info,
                "chunks": results,
                "lifecycle_delta": {
                    name: value - lifecycle_before[name]
                    for name, value in self._lifecycle.items()
                },
                "offload_delta": {
                    name: counters_after[name] - counters_before[name]
                    for name in counters_before
                    if name not in (
                        "cpu_expert_bytes", "pinned_bytes", "device_buffer_bytes",
                        "hot_cache_bytes", "hot_cache_capacity_experts", "peak_staged_experts",
                    )
                },
                "memory": {
                    "allocated_bytes": int(torch.cuda.memory_allocated(self.device)),
                    "reserved_bytes": int(torch.cuda.memory_reserved(self.device)),
                    "driver_free_bytes": int(torch.cuda.mem_get_info(self.device)[0]),
                    "process": _process_memory(),
                    "host_plan": self.host_plan.as_dict() if self.host_plan else None,
                },
                "host_residency": (
                    self.manager.host_residency_report()
                    if self._host_resident_budget else None
                ),
                "hybrid_decode": hybrid_report,
                "reference_prefix_cache": (
                    self.prefix_cache.report() if self.prefix_cache is not None else None
                ),
                "engine": {
                    "hardware_policy": (self.hardware_policy.report()
                                        if getattr(self, "hardware_policy", None) is not None else None),
                    "compute_policy": dict(self.compute_report),
                    "expert_storage": self.storage,
                    "requested_expert_decode": self.options.expert_decode,
                    "requested_cpu_experts": self.options.cpu_experts,
                    "requested_host_gather": self.options.host_gather,
                    "active_host_gather": getattr(
                        self.manager, "host_gather_mode", self.options.host_gather),
                    "native_host_gather_available": getattr(
                        getattr(self.manager, "pool", None), "native_gather", None) is not None,
                    "host_gather_fallback": getattr(self.manager, "host_gather_error", None),
                    "torch_cpu_threads": torch.get_num_threads(),
                    "cpu_experts": (
                        hybrid_report["cpu_jobs"]
                        if hybrid_report is not None and hybrid_report["enabled"] else 0
                    ),
                    "batch_one_expert_decode": (
                        "direct_slots"
                        if isinstance(getattr(self.manager, "backend", None), TritonExpertBackend)
                        and self.manager.backend.decode_specialization
                        and self.device in self.manager.backend._direct_decode_devices
                        else "general"
                    ),
                    "compact_head": self.options.compact_head,
                    "tower_graph": any(
                        item["tower"]["backend"] == "cuda_graph" for item in results
                    ),
                    "model_reused": was_loaded,
                    "combined_decode_bank": self.options.combined_decode_bank,
                    "packed_staging": self.options.packed_staging,
                    "codec_policy": self.options.codec_policy,
                    "codec_placement": self.options.codec_placement,
                    "profile_transfers": self.options.profile_transfers,
                    "host_resident_budget_bytes": self._host_resident_budget,
                    "weights": getattr(self.model, "_drama_quantized_mode",
                                       "official FP8 routed experts / BF16 dense and audio tower"),
                    "all_top8_routes_preserved": True,
                },
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            output.with_suffix(".json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            return report

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._release_graphs()
            if self.model is not None:
                cache = getattr(self.model, "_drama_low_vram_expert_cache", None)
                release_drama3(self.model, self.manager)
                if cache is not None:
                    cache.close()
            self.model, self.manager = None, None
            if self.codec is not None:
                release_codec(self.codec)
                self.codec = None
            self._references.clear()
            if self.prefix_cache is not None:
                self.prefix_cache.clear()
            self._cpu_mirrors.clear()
            self._codec_cpu_tensors.clear()
            self._task_candidates = None
            self._closed = True
            gc.collect()
            torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _process_memory():
    """Keep the engine module independent of CLI code."""
    import ctypes
    import os

    if os.name != "nt":
        return None
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("faults", wintypes.DWORD),
            ("peak_rss", ctypes.c_size_t), ("rss", ctypes.c_size_t),
            ("peak_paged", ctypes.c_size_t), ("paged", ctypes.c_size_t),
            ("peak_nonpaged", ctypes.c_size_t), ("nonpaged", ctypes.c_size_t),
            ("pagefile", ctypes.c_size_t), ("peak_pagefile", ctypes.c_size_t),
            ("private", ctypes.c_size_t),
        ]

    value = Counters()
    value.cb = ctypes.sizeof(value)
    get_process = ctypes.windll.kernel32.GetCurrentProcess
    get_process.restype = wintypes.HANDLE
    read_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    read_memory.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
    if read_memory(get_process(), ctypes.byref(value), value.cb):
        return {"resident_bytes": int(value.rss), "peak_resident_bytes": int(value.peak_rss),
                "private_committed_bytes": int(value.private),
                "page_fault_count": int(value.faults)}
    return None


def _cpu_tensor_bytes(module):
    if module is None:
        return 0
    return sum(
        tensor.numel() * tensor.element_size()
        for child in module.modules()
        for tensor in (*child._parameters.values(), *child._buffers.values())
        if tensor is not None and tensor.device.type == "cpu"
    )
