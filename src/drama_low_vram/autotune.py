"""Hardware inspection and conservative runtime tuning for Drama 3.

The model is large enough that a single hard-coded configuration is a poor
user experience.  This module deliberately contains *policy only*: it does
not load a checkpoint or allocate model tensors.  The returned
``RuntimeConfig`` can be passed to the existing loader by a caller that wants
automatic settings.

The policy follows the same shape as Strata's runtime decisions:

* use the total VRAM tier to choose a baseline;
* use currently free VRAM to avoid competing with another process;
* compare future host allocation peaks with current RAM and commit capacity
  to select expert residency and transfer limits;
* degrade to smaller staging/prefill batches instead of rejecting an 8 GB
  GPU.

Windows is the primary target, but the memory reader has a small Linux/macOS
fallback so that the policy can be unit-tested and inspected elsewhere.
"""

from __future__ import annotations

import ctypes
import importlib.util
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .host_budget import (
    HostAllocationState,
    HostMemoryPlan,
    HostMemoryRequirements,
    estimate_host_requirements,
    plan_host_memory,
)

try:  # Torch is part of the runtime, but keeping import optional helps tooling.
    import torch
except Exception:  # pragma: no cover - only used by environments without torch
    torch = None  # type: ignore[assignment]


GIB = 1024**3
MIB = 1024**2
PROFILE_NAMES = ("8gb", "12gb", "16gb", "24gb", "32gb+")

# Approximate fixed GPU residency for the current Drama 3 placement.  These
# are intentionally safety estimates, not promises about exact allocations.
# The measured prototype uses just under 5 GiB with lm_head on CPU.
BASE_GPU_RESIDENT_GIB = 4.75
LM_HEAD_GPU_GIB = 0.95
GPU_SAFETY_FRACTION = 0.90
# Compatibility constant for callers that used the former fixed reserve.
# Host plans now use explicit phase allocations and a 256-512 MiB buffer.
RAM_RUNTIME_RESERVE_GIB = 0.25
EXPERT_CACHE_GIB_PER_ENTRY = 0.13
POST_LOAD_RESERVE_BYTES = 512 * MIB
POST_LOAD_WORKSPACE_BYTES = 256 * MIB
EXPERT_BANK_HOST_BYTES = 32_220_119_040
# Legacy estimates kept for external import compatibility. Placement-aware
# host requirements replace these constants in every allocation decision.
CPU_EMBEDDING_HEAD_BYTES = 2 * GIB
HOST_LOADING_WORKSPACE_BYTES = 2 * GIB


def _gib(value: int | float) -> float:
    return float(value) / GIB


def _bytes(value: int | float) -> int:
    return max(0, int(value))


@dataclass(frozen=True)
class HardwareInfo:
    """Snapshot of resources relevant to the low-VRAM runtime.

    Byte fields are canonical so callers can preserve exact values in JSON.
    Convenience ``*_gib`` properties and ``from_gib`` make deterministic tests
    and UI callers less verbose.
    """

    platform: str = field(default_factory=platform.system)
    cuda_available: bool = False
    cuda_device_index: int | None = None
    device_name: str | None = None
    compute_capability: tuple[int, int] | None = None
    vram_total_bytes: int = 0
    vram_free_bytes: int = 0
    ram_total_bytes: int = 0
    ram_available_bytes: int = 0
    pagefile_total_bytes: int = 0
    pagefile_available_bytes: int = 0
    commit_limit_bytes: int | None = None
    commit_available_bytes: int | None = None
    pagefile_metrics_estimated: bool = False
    cpu_logical_cores: int = 1
    cpu_physical_cores: int = 1
    triton_available: bool = False
    torch_version: str | None = None
    cuda_version: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def vram_total_gib(self) -> float:
        return _gib(self.vram_total_bytes)

    @property
    def vram_free_gib(self) -> float:
        return _gib(self.vram_free_bytes)

    @property
    def ram_total_gib(self) -> float:
        return _gib(self.ram_total_bytes)

    @property
    def ram_available_gib(self) -> float:
        return _gib(self.ram_available_bytes)

    @property
    def pagefile_total_gib(self) -> float:
        return _gib(self.pagefile_total_bytes)

    @property
    def pagefile_available_gib(self) -> float:
        return _gib(self.pagefile_available_bytes)

    @property
    def gpu_name(self) -> str | None:
        """Alias used by UI/reporting callers."""

        return self.device_name

    @classmethod
    def from_gib(
        cls,
        *,
        vram: float = 0,
        vram_free: float | None = None,
        ram: float = 0,
        ram_available: float | None = None,
        pagefile: float = 0,
        pagefile_available: float | None = None,
        commit_limit: float | None = None,
        commit_available: float | None = None,
        **kwargs: Any,
    ) -> "HardwareInfo":
        """Build a snapshot from GiB values, primarily for tests and UIs."""

        if vram_free is None:
            vram_free = vram
        if ram_available is None:
            ram_available = ram
        if pagefile_available is None:
            pagefile_available = pagefile
        # A non-zero VRAM value is normally supplied by tests/UI callers to
        # represent a CUDA device.  Callers can still override this explicitly
        # with ``cuda_available=False``.
        kwargs.setdefault("cuda_available", bool(vram > 0))
        if commit_limit is not None:
            kwargs["commit_limit_bytes"] = _bytes(commit_limit * GIB)
        if commit_available is not None:
            kwargs["commit_available_bytes"] = _bytes(commit_available * GIB)
        return cls(
            vram_total_bytes=_bytes(vram * GIB),
            vram_free_bytes=_bytes(vram_free * GIB),
            ram_total_bytes=_bytes(ram * GIB),
            ram_available_bytes=_bytes(ram_available * GIB),
            pagefile_total_bytes=_bytes(pagefile * GIB),
            pagefile_available_bytes=_bytes(pagefile_available * GIB),
            **kwargs,
        )

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "platform": self.platform,
            "cuda_available": self.cuda_available,
            "cuda_device_index": self.cuda_device_index,
            "device_name": self.device_name,
            "compute_capability": (
                list(self.compute_capability)
                if self.compute_capability is not None
                else None
            ),
            "vram_total_bytes": self.vram_total_bytes,
            "vram_free_bytes": self.vram_free_bytes,
            "vram_total_gib": round(self.vram_total_gib, 3),
            "vram_free_gib": round(self.vram_free_gib, 3),
            "ram_total_bytes": self.ram_total_bytes,
            "ram_available_bytes": self.ram_available_bytes,
            "ram_total_gib": round(self.ram_total_gib, 3),
            "ram_available_gib": round(self.ram_available_gib, 3),
            "pagefile_total_bytes": self.pagefile_total_bytes,
            "pagefile_available_bytes": self.pagefile_available_bytes,
            "pagefile_total_gib": round(self.pagefile_total_gib, 3),
            "pagefile_available_gib": round(self.pagefile_available_gib, 3),
            "pagefile_metrics_estimated": self.pagefile_metrics_estimated,
            "commit_limit_bytes": self.commit_limit_bytes,
            "commit_available_bytes": self.commit_available_bytes,
            "commit_limit_gib": (
                round(_gib(self.commit_limit_bytes), 3)
                if self.commit_limit_bytes is not None
                else None
            ),
            "commit_available_gib": (
                round(_gib(self.commit_available_bytes), 3)
                if self.commit_available_bytes is not None
                else None
            ),
            "cpu_logical_cores": self.cpu_logical_cores,
            "cpu_physical_cores": self.cpu_physical_cores,
            "triton_available": self.triton_available,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "notes": list(self.notes),
        }
        return result

    to_dict = as_dict


@dataclass(frozen=True)
class RuntimeConfig:
    """A complete, serializable recommendation for the existing runtime."""

    profile: str
    device: str
    lm_head_device: str
    cache_experts: int
    max_staged_experts: int
    max_pair_rows: int
    pin_memory: bool
    backend: str
    prefill_chunk: int
    codec_max_frames: int
    safe_vram_budget_bytes: int
    safe_ram_budget_bytes: int
    fixed_gpu_bytes: int
    gpu_headroom_bytes: int
    expert_storage: str = "ram"
    status: str = "ready"
    warnings: tuple[str, ...] = ()
    explanations: tuple[str, ...] = ()
    hardware: HardwareInfo | None = None
    required_host_commit_bytes: int = 0
    available_commit_bytes: int = 0
    commit_budget_estimated: bool = False
    host_plan: HostMemoryPlan | None = None
    max_auto_cache_experts: int | None = None

    @property
    def max_frames(self) -> int:
        """Compatibility alias for the codec loader's ``max_frames`` option."""

        return self.codec_max_frames

    @property
    def prefill_chunk_size(self) -> int:
        """Compatibility alias used by Fish's generation helper."""

        return self.prefill_chunk

    @property
    def can_run(self) -> bool:
        """Whether the recommendation expects sufficient resources now."""

        return self.status == "ready"

    def as_dict(self, *, include_hardware: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "profile": self.profile,
            "device": self.device,
            "lm_head_device": self.lm_head_device,
            "cache_experts": self.cache_experts,
            "max_auto_cache_experts": self.max_auto_cache_experts,
            "max_staged_experts": self.max_staged_experts,
            "max_pair_rows": self.max_pair_rows,
            "pin_memory": self.pin_memory,
            "backend": self.backend,
            "prefill_chunk": self.prefill_chunk,
            "prefill_chunk_size": self.prefill_chunk,
            "codec_max_frames": self.codec_max_frames,
            "max_frames": self.codec_max_frames,
            "safe_vram_budget_bytes": self.safe_vram_budget_bytes,
            "safe_ram_budget_bytes": self.safe_ram_budget_bytes,
            "fixed_gpu_bytes": self.fixed_gpu_bytes,
            "gpu_headroom_bytes": self.gpu_headroom_bytes,
            "expert_storage": self.expert_storage,
            "required_host_commit_bytes": self.required_host_commit_bytes,
            "available_commit_bytes": self.available_commit_bytes,
            "commit_budget_estimated": self.commit_budget_estimated,
            "host_plan": self.host_plan.as_dict() if self.host_plan is not None else None,
            "status": self.status,
            "can_run": self.can_run,
            "warnings": list(self.warnings),
            "explanations": list(self.explanations),
        }
        if include_hardware and self.hardware is not None:
            result["hardware"] = self.hardware.as_dict()
        return result

    to_dict = as_dict

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, ensure_ascii=False)

    def as_json(self, *, indent: int = 2) -> str:
        return self.to_json(indent=indent)


@dataclass(frozen=True)
class PostLoadPlan:
    """GPU budget measured after fixed weights and sequence caches are loaded.

    ``cache_experts`` is a per-layer capacity. ``cache_slot_bytes`` is the
    exact checkpoint cost of increasing that capacity by one across all
    routed-expert layers; staging is one shared pool rather than per-layer.
    """

    fixed_gpu_bytes: int
    vram_total_bytes: int
    vram_free_bytes: int
    reusable_reserved_bytes: int
    available_gpu_bytes: int
    reserve_bytes: int
    runtime_workspace_bytes: int
    expert_total_bytes: int
    num_layers: int
    num_experts_per_layer: int
    staged_expert_bytes: int
    cache_slot_bytes: int
    max_staged_experts: int
    max_pair_rows: int
    prefill_chunk: int
    staging_bytes: int
    cache_experts: int
    cache_bytes: int
    available_cache_bytes: int
    gpu_headroom_bytes: int
    requested_cache_experts: int | None = None
    status: str = "ready"
    warnings: tuple[str, ...] = ()
    explanations: tuple[str, ...] = ()

    @property
    def can_run(self) -> bool:
        return self.status == "ready"

    @property
    def planned_extra_gpu_bytes(self) -> int:
        return self.runtime_workspace_bytes + self.staging_bytes + self.cache_bytes

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["warnings"] = list(self.warnings)
        result["explanations"] = list(self.explanations)
        result["can_run"] = self.can_run
        result["planned_extra_gpu_bytes"] = self.planned_extra_gpu_bytes
        return result

    to_dict = as_dict

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, ensure_ascii=False)


@dataclass(frozen=True)
class AutotuneResult:
    """Optional wrapper when callers want both the raw snapshot and config."""

    hardware: HardwareInfo
    config: RuntimeConfig

    def as_dict(self) -> dict[str, Any]:
        return {
            "hardware": self.hardware.as_dict(),
            "config": self.config.as_dict(include_hardware=False),
        }

    to_dict = as_dict

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, ensure_ascii=False)


def _read_memory_status() -> tuple[
    int, int, int, int, tuple[str, ...], int | None, int | None
]:
    """Return physical memory, legacy swap estimates, and actual commit bytes."""

    notes: list[str] = []
    if os.name == "nt":
        try:
            from ctypes import wintypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", wintypes.ULARGE_INTEGER),
                    ("ullAvailPhys", wintypes.ULARGE_INTEGER),
                    ("ullTotalPageFile", wintypes.ULARGE_INTEGER),
                    ("ullAvailPageFile", wintypes.ULARGE_INTEGER),
                    ("ullTotalVirtual", wintypes.ULARGE_INTEGER),
                    ("ullAvailVirtual", wintypes.ULARGE_INTEGER),
                    ("ullAvailExtendedVirtual", wintypes.ULARGE_INTEGER),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                total_phys = int(status.ullTotalPhys)
                available_phys = int(status.ullAvailPhys)
                # MEMORYSTATUSEX pagefile fields are commit capacity. The
                # derived legacy fields are estimates, not pagefile disk space.
                commit_limit = int(status.ullTotalPageFile)
                commit_available = int(status.ullAvailPageFile)
                page_total = max(0, commit_limit - total_phys)
                page_available = max(0, commit_available - available_phys)
                return (
                    total_phys,
                    available_phys,
                    page_total,
                    page_available,
                    tuple(notes),
                    commit_limit,
                    commit_available,
                )
        except Exception as exc:  # pragma: no cover - depends on host API
            notes.append(f"Windows memory query failed: {type(exc).__name__}")

    # Portable fallback for Linux and test environments.
    try:
        meminfo = Path("/proc/meminfo")
        if meminfo.is_file():
            values: dict[str, int] = {}
            for line in meminfo.read_text(encoding="ascii").splitlines():
                key, _, value = line.partition(":")
                number = value.strip().split(" ", 1)[0]
                if number.isdigit():
                    values[key] = int(number) * 1024
            total = values.get("MemTotal", 0)
            available = values.get("MemAvailable", values.get("MemFree", 0))
            page_total = values.get("SwapTotal", 0)
            page_available = values.get("SwapFree", 0)
            if total:
                return total, available, page_total, page_available, tuple(notes), None, None
    except Exception as exc:  # pragma: no cover - unusual /proc failures
        notes.append(f"/proc/meminfo query failed: {type(exc).__name__}")

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        return (
            page_size * pages,
            page_size * available_pages,
            0,
            0,
            tuple(notes),
            None,
            None,
        )
    except Exception as exc:  # pragma: no cover - platform-specific fallback
        notes.append(f"system memory query unavailable: {type(exc).__name__}")
        return 0, 0, 0, 0, tuple(notes), None, None


def _triton_is_installed() -> bool:
    try:
        return importlib.util.find_spec("triton") is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _cpu_core_counts() -> tuple[int, int, tuple[str, ...]]:
    """Return logical and physical core counts without making psutil required."""

    logical = max(1, int(os.cpu_count() or 1))
    try:
        psutil_spec = importlib.util.find_spec("psutil")
        if psutil_spec is not None:
            import psutil  # type: ignore[import-not-found]

            physical = int(psutil.cpu_count(logical=False) or 0)
            if physical > 0:
                return logical, min(logical, physical), ()
    except Exception as exc:  # pragma: no cover - optional dependency/platform API
        return logical, logical, (f"Physical CPU core query failed: {type(exc).__name__}",)
    return logical, logical, (
        "Physical CPU core count unavailable; using logical count as a conservative proxy.",
    )


def _parse_device_index(device: int | str | None) -> int:
    if isinstance(device, int):
        if device < 0:
            raise ValueError("CUDA device index must be non-negative")
        return device
    if device is None or device == "cuda":
        return 0
    if isinstance(device, str) and device.startswith("cuda:"):
        try:
            index = int(device.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device {device!r}") from exc
        if index < 0:
            raise ValueError("CUDA device index must be non-negative")
        return index
    raise ValueError(f"Expected a CUDA device, got {device!r}")


def inspect_hardware(device: int | str | None = None) -> HardwareInfo:
    """Inspect CUDA, memory, pagefile, and CPU resources without model loading."""

    (
        ram_total,
        ram_available,
        pagefile_total,
        pagefile_available,
        memory_notes,
        commit_limit,
        commit_available,
    ) = _read_memory_status()
    logical, physical, cpu_notes = _cpu_core_counts()
    notes = list(memory_notes) + list(cpu_notes)
    triton_available = _triton_is_installed()
    cuda_available = bool(torch is not None and torch.cuda.is_available())
    requested_index = _parse_device_index(device)

    if not cuda_available:
        notes.append("CUDA is unavailable; runtime recommendations are inspection-only.")
        return HardwareInfo(
            platform=platform.system(),
            cuda_available=False,
            cuda_device_index=None,
            ram_total_bytes=ram_total,
            ram_available_bytes=ram_available,
            pagefile_total_bytes=pagefile_total,
            pagefile_available_bytes=pagefile_available,
            commit_limit_bytes=commit_limit,
            commit_available_bytes=commit_available,
            pagefile_metrics_estimated=os.name == "nt" and commit_limit is not None,
            cpu_logical_cores=logical,
            cpu_physical_cores=physical,
            triton_available=triton_available,
            torch_version=getattr(torch, "__version__", None),
            cuda_version=getattr(getattr(torch, "version", None), "cuda", None),
            notes=tuple(notes),
        )

    count = int(torch.cuda.device_count())
    if count <= 0:
        notes.append("CUDA reported available but no CUDA devices were enumerated.")
        return HardwareInfo(
            platform=platform.system(),
            cuda_available=False,
            cuda_device_index=None,
            ram_total_bytes=ram_total,
            ram_available_bytes=ram_available,
            pagefile_total_bytes=pagefile_total,
            pagefile_available_bytes=pagefile_available,
            commit_limit_bytes=commit_limit,
            commit_available_bytes=commit_available,
            pagefile_metrics_estimated=os.name == "nt" and commit_limit is not None,
            cpu_logical_cores=logical,
            cpu_physical_cores=physical,
            triton_available=triton_available,
            torch_version=getattr(torch, "__version__", None),
            cuda_version=getattr(getattr(torch, "version", None), "cuda", None),
            notes=tuple(notes),
        )
    if requested_index >= count:
        raise ValueError(
            f"Requested cuda:{requested_index} is unavailable; "
            f"this host has {count} CUDA device(s)."
        )
    index = requested_index

    properties = torch.cuda.get_device_properties(index)
    total_vram = int(getattr(properties, "total_memory", 0))
    free_vram = 0
    try:
        free_vram, reported_total = torch.cuda.mem_get_info(index)
        free_vram = int(free_vram)
        if not total_vram:
            total_vram = int(reported_total)
    except Exception as exc:  # pragma: no cover - driver-specific
        notes.append(f"CUDA free-memory query failed: {type(exc).__name__}")
    major = getattr(properties, "major", None)
    minor = getattr(properties, "minor", None)
    capability = (
        (int(major), int(minor)) if major is not None and minor is not None else None
    )
    usable_triton = bool(
        triton_available and capability is not None and capability >= (8, 9)
    )
    if triton_available and not usable_triton:
        notes.append("Triton is installed but this GPU is below the FP8 SM89 target.")
    return HardwareInfo(
        platform=platform.system(),
        cuda_available=True,
        cuda_device_index=index,
        device_name=str(getattr(properties, "name", None) or f"cuda:{index}"),
        compute_capability=capability,
        vram_total_bytes=total_vram,
        vram_free_bytes=free_vram,
        ram_total_bytes=ram_total,
        ram_available_bytes=ram_available,
        pagefile_total_bytes=pagefile_total,
        pagefile_available_bytes=pagefile_available,
        commit_limit_bytes=commit_limit,
        commit_available_bytes=commit_available,
        pagefile_metrics_estimated=os.name == "nt" and commit_limit is not None,
        cpu_logical_cores=logical,
        cpu_physical_cores=physical,
        triton_available=usable_triton,
        torch_version=getattr(torch, "__version__", None),
        cuda_version=getattr(getattr(torch, "version", None), "cuda", None),
        notes=tuple(notes),
    )


get_hardware_info = inspect_hardware
detect_hardware = inspect_hardware


def _profile_for_vram(vram_gib: float) -> str:
    if vram_gib < 10:
        return "8gb"
    if vram_gib < 14:
        return "12gb"
    if vram_gib < 20:
        return "16gb"
    if vram_gib < 28:
        return "24gb"
    return "32gb+"


def _baseline(profile: str) -> dict[str, int]:
    return {
        "8gb": {
            "cache_experts": 0,
            "max_staged_experts": 4,
            "max_pair_rows": 32,
            "prefill_chunk": 8,
            "codec_max_frames": 2048,
        },
        "12gb": {
            "cache_experts": 4,
            "max_staged_experts": 8,
            "max_pair_rows": 64,
            "prefill_chunk": 16,
            "codec_max_frames": 4096,
        },
        "16gb": {
            "cache_experts": 8,
            "max_staged_experts": 16,
            "max_pair_rows": 128,
            "prefill_chunk": 32,
            "codec_max_frames": 4096,
        },
        "24gb": {
            "cache_experts": 16,
            "max_staged_experts": 32,
            "max_pair_rows": 256,
            "prefill_chunk": 48,
            "codec_max_frames": 8192,
        },
        "32gb+": {
            "cache_experts": 32,
            "max_staged_experts": 64,
            "max_pair_rows": 512,
            "prefill_chunk": 64,
            "codec_max_frames": 8192,
        },
    }[profile].copy()


def select_profile(
    hardware: HardwareInfo,
    *,
    expert_storage: str = "ram",
    host_requirements: HostMemoryRequirements | None = None,
    host_allocated: HostAllocationState | None = None,
) -> RuntimeConfig:
    """Select a safe profile from a hardware snapshot.

    An 8 GB GPU is intentionally a valid profile. CUDA absence is ``blocked``;
    an occupied GPU that cannot fit the fixed weights is ``resource_busy``.
    Smaller staging limits apply when the weights still fit.
    """

    if expert_storage not in ("auto", "ram", "mmap"):
        raise ValueError("expert_storage must be auto, ram or mmap")
    profile = _profile_for_vram(hardware.vram_total_gib)
    values = _baseline(profile)
    max_auto_cache_experts = None
    warnings: list[str] = list(hardware.notes)
    explanations: list[str] = [
        f"Selected {profile} from {hardware.vram_total_gib:.1f} GiB total VRAM.",
    ]
    total_free_vram = hardware.vram_free_gib
    safe_vram_gib = min(
        hardware.vram_total_gib * GPU_SAFETY_FRACTION,
        total_free_vram * GPU_SAFETY_FRACTION,
    )
    if hardware.vram_total_bytes <= 0:
        safe_vram_gib = 0.0
    # 8 GB stays on CPU lm_head; larger cards use it only if current free VRAM
    # leaves enough room for the dense backbone and runtime allocations.
    lm_head_device = "cuda" if profile != "8gb" and total_free_vram >= 10.0 else "cpu"
    fixed_gpu_gib = BASE_GPU_RESIDENT_GIB + (
        LM_HEAD_GPU_GIB if lm_head_device == "cuda" else 0.0
    )
    if host_requirements is not None and host_requirements.fixed_gpu_bytes:
        fixed_gpu_gib = _gib(host_requirements.fixed_gpu_bytes)
        explanations.append("Estimate resident GPU weights from this checkpoint's actual tensor format and placement.")
    headroom_gib = max(0.0, safe_vram_gib - fixed_gpu_gib)
    if lm_head_device == "cpu":
        explanations.append("Keep lm_head on CPU to preserve headroom for the dense tower and staging.")
    else:
        explanations.append("Free VRAM is sufficient to place lm_head on CUDA and reduce logits transfers.")

    # Clamp the persistent hot cache to actual headroom. Each entry occupies
    # roughly 0.13 GiB across the MoE layers in the current checkpoint.
    slot_gib = EXPERT_CACHE_GIB_PER_ENTRY
    if (host_requirements is not None and host_requirements.expert_bytes
            and host_requirements.num_experts_per_layer):
        slot_gib = _gib(
            (host_requirements.expert_bytes + host_requirements.num_experts_per_layer - 1)
            // host_requirements.num_experts_per_layer)
    cache_limit = max(0, int(max(0.0, headroom_gib - 0.20) / slot_gib))
    values["cache_experts"] = min(values["cache_experts"], cache_limit)
    if values["cache_experts"] == 0 and profile != "8gb":
        explanations.append("Hot expert cache disabled because current free VRAM is too tight.")

    # If another process has consumed VRAM, shrink staging before doing any
    # model work. The model still runs by partitioning routed pairs.
    if headroom_gib < 0.75:
        values["cache_experts"] = 0
        values["max_staged_experts"] = min(values["max_staged_experts"], 2)
        values["max_pair_rows"] = min(values["max_pair_rows"], 16)
        values["prefill_chunk"] = min(values["prefill_chunk"], 4)
        values["codec_max_frames"] = min(values["codec_max_frames"], 1024)
        warnings.append("Very little free VRAM is visible; using minimum staging and prefill limits.")
    elif headroom_gib < 1.50:
        values["max_staged_experts"] = min(values["max_staged_experts"], 4)
        values["max_pair_rows"] = min(values["max_pair_rows"], 32)
        values["prefill_chunk"] = min(values["prefill_chunk"], 8)
        values["codec_max_frames"] = min(values["codec_max_frames"], 2048)
        warnings.append("Free VRAM is below the profile target; staging is conservatively capped.")

    # RAM is where the 30+ GiB FP8 expert bank lives. Available RAM matters
    # more than installed RAM while another application is running.
    ram_total = hardware.ram_total_gib
    ram_available = hardware.ram_available_gib
    bounded_fp8_pool = bool(
        host_requirements is not None
        and host_requirements.file_backed_parking_bytes
        and hardware.cuda_available and hardware.triton_available
        and ram_total >= 20 and ram_available >= 12
    )
    if bounded_fp8_pool:
        if ram_total < 40 or ram_available < 20:
            values["cache_experts"] = min(values["cache_experts"], 2)
        # Eight official FP8 experts occupy about 24MiB per shared pool.
        # Avoid shrinking this small transfer workspace because the complete
        # 30GiB source bank is file-backed. Post-load VRAM planning still clamps
        # the actual capacity before allocation.
        values["max_staged_experts"] = max(values["max_staged_experts"], 8)
        values["max_pair_rows"] = max(values["max_pair_rows"], 64)
        pin_memory = True
        explanations.append(
            "File-backed FP8 weights permit a bounded eight-expert pinned transfer pool; "
            "recheck staging and prefill against actual VRAM after loading."
        )
        if profile == "8gb" and ram_total >= 40 and ram_available >= 28:
            values["prefill_chunk"] = 32
            max_auto_cache_experts = 1
            explanations.append(
                "With ample host RAM, request the measured 32-row FP8 prefill; "
                "post-load allocation and prefill OOM recovery determine what fits. Cap "
                "automatic expert residency at one slot per layer to preserve codec "
                "and graph headroom; explicit cache overrides remain unchanged."
            )
    elif ram_total < 24 or ram_available < 12:
        values["cache_experts"] = 0
        values["max_staged_experts"] = min(values["max_staged_experts"], 2)
        values["max_pair_rows"] = min(values["max_pair_rows"], 16)
        values["prefill_chunk"] = min(values["prefill_chunk"], 8)
        values["codec_max_frames"] = min(values["codec_max_frames"], 1024)
        pin_memory = False
        warnings.append(
            "RAM headroom is tight; avoid pinned transfer buffers and use "
            "smaller temporary batches."
        )
    elif ram_total < 40 or ram_available < 20:
        values["cache_experts"] = min(values["cache_experts"], 2)
        values["max_staged_experts"] = min(values["max_staged_experts"], 4)
        values["max_pair_rows"] = min(values["max_pair_rows"], 64)
        pin_memory = hardware.cuda_available
        warnings.append(
            "RAM headroom is moderate; smaller transfer buffers reduce host "
            "memory pressure."
        )
    else:
        pin_memory = bool(hardware.cuda_available and ram_available >= 16)
    if host_requirements is not None and host_requirements.num_experts_per_layer:
        values["cache_experts"] = min(values["cache_experts"], host_requirements.num_experts_per_layer)
        values["max_staged_experts"] = min(
            values["max_staged_experts"], host_requirements.num_experts_per_layer)
    host_plan = plan_host_memory(
        ram_available_bytes=hardware.ram_available_bytes,
        commit_available_bytes=hardware.commit_available_bytes,
        swap_available_bytes=hardware.pagefile_available_bytes,
        expert_storage=expert_storage,
        requirements=host_requirements or estimate_host_requirements(
            staged_experts=values["max_staged_experts"],
            codec_max_frames=values["codec_max_frames"],
            compact_head=False, lm_head_device=lm_head_device,
        ),
        allocated=host_allocated,
    )
    expert_storage = host_plan.expert_storage
    required_host_commit = host_plan.required_commit_bytes
    commit_estimated = host_plan.commit_budget_estimated
    available_commit = host_plan.available_commit_bytes
    warnings.extend(host_plan.warnings)
    explanations.extend(host_plan.explanations)

    if hardware.cpu_logical_cores <= 4:
        values["prefill_chunk"] = min(values["prefill_chunk"], 8)
        values["max_pair_rows"] = min(values["max_pair_rows"], 32)
        warnings.append("Low logical-core count limits prefill and pair batching.")
    elif hardware.cpu_logical_cores <= 8:
        values["prefill_chunk"] = min(values["prefill_chunk"], 16)

    backend = "triton" if hardware.cuda_available and hardware.triton_available else "torch"
    if backend == "torch":
        warnings.append("Triton FP8 backend is unavailable; using the slower staged Torch backend.")

    status = "ready" if hardware.cuda_available else "blocked"
    device = (
        f"cuda:{hardware.cuda_device_index or 0}" if hardware.cuda_available else "cpu"
    )
    if not hardware.cuda_available:
        warnings.append("A CUDA GPU is required by the current Drama 3 loader.")
    elif safe_vram_gib < fixed_gpu_gib + _gib(POST_LOAD_WORKSPACE_BYTES):
        status = "resource_busy"
        warnings.append(
            f"Only {total_free_vram:.2f} GiB VRAM is free; fixed weights and "
            "runtime workspace do not currently fit. Free GPU memory and retry."
        )
    if hardware.cuda_available and not host_plan.can_run:
        status = "resource_busy"
        warnings.append(
            f"Host commit headroom is {_gib(available_commit):.2f} GiB; "
            f"the selected {expert_storage} loader needs about {_gib(required_host_commit):.2f} GiB. "
            "Free host memory or provide more OS commit capacity before loading."
        )
    if hardware.cuda_available and profile == "8gb":
        explanations.append(
            "8 GB mode remains runnable: CPU lm_head and bounded staging; "
            "persistent expert cache is planned from actual post-load headroom."
        )
    if expert_storage == "mmap":
        explanations.append(
            "Use the verified read-only FP8 expert cache. OS file pages can be "
            "reclaimed; cold pages need disk reads before GPU staging."
        )
    else:
        explanations.append(
            "Keep every FP8 expert weight in process RAM and stage only routed experts."
        )

    return RuntimeConfig(
        profile=profile,
        device=device,
        lm_head_device=lm_head_device,
        cache_experts=max(0, int(values["cache_experts"])),
        max_staged_experts=max(1, int(values["max_staged_experts"])),
        max_pair_rows=max(1, int(values["max_pair_rows"])),
        pin_memory=pin_memory,
        backend=backend,
        prefill_chunk=max(1, int(values["prefill_chunk"])),
        codec_max_frames=max(256, int(values["codec_max_frames"])),
        safe_vram_budget_bytes=_bytes(safe_vram_gib * GIB),
        safe_ram_budget_bytes=max(0, hardware.ram_available_bytes - host_plan.buffer_bytes),
        fixed_gpu_bytes=_bytes(fixed_gpu_gib * GIB),
        gpu_headroom_bytes=_bytes(headroom_gib * GIB),
        expert_storage=expert_storage,
        status=status,
        warnings=tuple(dict.fromkeys(warnings)),
        explanations=tuple(explanations),
        hardware=hardware,
        required_host_commit_bytes=required_host_commit,
        available_commit_bytes=available_commit,
        commit_budget_estimated=commit_estimated,
        host_plan=host_plan,
        max_auto_cache_experts=max_auto_cache_experts,
    )


def plan_post_load(
    *,
    fixed_gpu_bytes: int,
    vram_free_bytes: int,
    vram_total_bytes: int,
    expert_total_bytes: int,
    num_layers: int,
    num_experts_per_layer: int,
    vram_allocated_bytes: int = 0,
    vram_reserved_bytes: int = 0,
    max_staged_experts: int = 8,
    max_pair_rows: int = 64,
    prefill_chunk: int = 8,
    cache_experts: int | None = None,
    max_auto_cache_experts: int | None = None,
    backend: str = "triton",
    reserve_bytes: int | None = None,
    runtime_workspace_bytes: int | None = None,
    allow_staging_reduction: bool = True,
) -> PostLoadPlan:
    """Plan an FP8 hot cache using real memory observations without allocating.

    Call after dense weights, the audio tower, and sequence caches are placed.
    ``expert_total_bytes`` includes both FP8 weights and their scale tensors.
    A manual ``cache_experts`` is never silently reduced: an unsafe request
    returns ``resource_busy`` with its exact requested capacity preserved.
    Set ``allow_staging_reduction=False`` to preserve explicit batch limits.
    """
    values = {
        "fixed_gpu_bytes": fixed_gpu_bytes,
        "vram_free_bytes": vram_free_bytes,
        "vram_total_bytes": vram_total_bytes,
        "expert_total_bytes": expert_total_bytes,
        "num_layers": num_layers,
        "num_experts_per_layer": num_experts_per_layer,
        "vram_allocated_bytes": vram_allocated_bytes,
        "vram_reserved_bytes": vram_reserved_bytes,
        "max_staged_experts": max_staged_experts,
        "max_pair_rows": max_pair_rows,
        "prefill_chunk": prefill_chunk,
    }
    for name, value in values.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    for name in (
        "vram_total_bytes",
        "expert_total_bytes",
        "num_layers",
        "num_experts_per_layer",
        "max_staged_experts",
        "max_pair_rows",
        "prefill_chunk",
    ):
        if values[name] == 0:
            raise ValueError(f"{name} must be positive")
    if backend not in ("triton", "torch"):
        raise ValueError(f"Unknown expert backend {backend!r}")
    if cache_experts is not None and (
        not isinstance(cache_experts, int)
        or isinstance(cache_experts, bool)
        or not 0 <= cache_experts <= num_experts_per_layer
    ):
        raise ValueError(
            f"cache_experts must be between 0 and {num_experts_per_layer}, or None"
        )
    if max_auto_cache_experts is not None and (
        isinstance(max_auto_cache_experts, bool)
        or not isinstance(max_auto_cache_experts, int)
        or not 0 <= max_auto_cache_experts <= num_experts_per_layer
    ):
        raise ValueError("max_auto_cache_experts must be a valid expert capacity or None")
    reserve = POST_LOAD_RESERVE_BYTES if reserve_bytes is None else reserve_bytes
    workspace = (
        POST_LOAD_WORKSPACE_BYTES
        if runtime_workspace_bytes is None
        else runtime_workspace_bytes
    )
    for name, value in (("reserve_bytes", reserve), ("runtime_workspace_bytes", workspace)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")

    expert_count = num_layers * num_experts_per_layer
    staged_expert_bytes = (expert_total_bytes + expert_count - 1) // expert_count
    # Count each layer separately to retain a conservative bound when the
    # checkpoint total is not perfectly divisible by the expert count.
    cache_slot_bytes = staged_expert_bytes * num_layers
    free = min(vram_free_bytes, vram_total_bytes)
    reusable = max(0, vram_reserved_bytes - vram_allocated_bytes)
    reusable = min(reusable, max(0, vram_total_bytes - free))
    available = min(
        free + reusable,
        max(0, vram_total_bytes - max(fixed_gpu_bytes, vram_allocated_bytes)),
    )
    warnings: list[str] = []
    explanations = [
        f"Fixed GPU weights and sequence caches occupy {_gib(fixed_gpu_bytes):.3f} GiB.",
        f"Each per-layer FP8 cache slot costs {_gib(cache_slot_bytes):.4f} GiB across {num_layers} layers.",
        f"Reserve {_gib(reserve):.3f} GiB VRAM and {_gib(workspace):.3f} GiB for runtime workspace.",
    ]
    staged_capacity = min(max_staged_experts, num_experts_per_layer)
    pair_limit = max_pair_rows
    prefill_limit = prefill_chunk
    requested_cache_bytes = (cache_experts or 0) * cache_slot_bytes
    # A pool is shared by all layers. Reduce only temporary batch sizes when
    # needed; the requested FP8 cache capacity remains intact in manual mode.
    while (
        allow_staging_reduction
        and staged_capacity > 1
        and reserve
        + workspace
        + staged_capacity * staged_expert_bytes
        + requested_cache_bytes
        > available
    ):
        staged_capacity = max(1, staged_capacity // 2)
        pair_limit = max(1, pair_limit // 2)
        prefill_limit = max(1, prefill_limit // 2)
    if staged_capacity < min(max_staged_experts, num_experts_per_layer):
        warnings.append(
            "Post-load VRAM pressure reduced staging, pair batching, and prefill limits."
        )
    staging_bytes = staged_capacity * staged_expert_bytes
    cache_budget = max(0, available - reserve - workspace - staging_bytes)
    selected_cache = (
        min(num_experts_per_layer, cache_budget // cache_slot_bytes)
        if cache_experts is None
        else cache_experts
    )
    if cache_experts is None and max_auto_cache_experts is not None:
        selected_cache = min(selected_cache, max_auto_cache_experts)
        explanations.append(
            f"Limit automatic residency to {max_auto_cache_experts} experts per layer "
            "to retain codec and graph headroom."
        )
    cache_bytes = selected_cache * cache_slot_bytes
    required = reserve + workspace + staging_bytes + cache_bytes
    status = "ready" if required <= available else "resource_busy"
    if cache_experts is None:
        explanations.append(
            f"Measured headroom permits {selected_cache} cached FP8 experts per layer."
        )
    else:
        explanations.append(f"Preserved manual cache capacity of {cache_experts} experts per layer.")
    if status == "resource_busy":
        deficit = required - available
        warnings.append(
            f"Post-load GPU budget is short by {deficit / 1024**2:.2f} MiB "
            f"({deficit:,} bytes). "
            "Free GPU memory or lower the explicit cache request before generation."
        )

    return PostLoadPlan(
        fixed_gpu_bytes=fixed_gpu_bytes,
        vram_total_bytes=vram_total_bytes,
        vram_free_bytes=free,
        reusable_reserved_bytes=reusable,
        available_gpu_bytes=available,
        reserve_bytes=reserve,
        runtime_workspace_bytes=workspace,
        expert_total_bytes=expert_total_bytes,
        num_layers=num_layers,
        num_experts_per_layer=num_experts_per_layer,
        staged_expert_bytes=staged_expert_bytes,
        cache_slot_bytes=cache_slot_bytes,
        max_staged_experts=staged_capacity,
        max_pair_rows=pair_limit,
        prefill_chunk=prefill_limit,
        staging_bytes=staging_bytes,
        cache_experts=selected_cache,
        cache_bytes=cache_bytes,
        available_cache_bytes=cache_budget,
        gpu_headroom_bytes=max(0, available - workspace - staging_bytes - cache_bytes),
        requested_cache_experts=cache_experts,
        status=status,
        warnings=tuple(warnings),
        explanations=tuple(explanations),
    )


def autotune(
    hardware: HardwareInfo | None = None, *, expert_storage: str = "ram",
    host_requirements: HostMemoryRequirements | None = None,
    host_allocated: HostAllocationState | None = None,
    **kwargs: Any,
) -> RuntimeConfig:
    """Inspect hardware (unless supplied) and return a runtime recommendation."""

    snapshot = hardware if hardware is not None else inspect_hardware(**kwargs)
    return select_profile(
        snapshot, expert_storage=expert_storage,
        host_requirements=host_requirements, host_allocated=host_allocated,
    )


auto_tune = autotune
recommend_config = autotune
build_runtime_config = autotune
choose_profile = select_profile

# Friendly aliases for callers that use "snapshot" or "config" terminology.
HardwareSnapshot = HardwareInfo
AutotuneConfig = RuntimeConfig


def inspect_and_autotune(
    hardware: HardwareInfo | None = None, *, expert_storage: str = "ram",
    host_requirements: HostMemoryRequirements | None = None,
    host_allocated: HostAllocationState | None = None,
    **kwargs: Any,
) -> AutotuneResult:
    snapshot = hardware if hardware is not None else inspect_hardware(**kwargs)
    return AutotuneResult(snapshot, select_profile(
        snapshot, expert_storage=expert_storage,
        host_requirements=host_requirements, host_allocated=host_allocated,
    ))


def config_to_json(config: RuntimeConfig, *, indent: int = 2) -> str:
    return config.to_json(indent=indent)


__all__ = [
    "AutotuneResult",
    "AutotuneConfig",
    "HardwareInfo",
    "HardwareSnapshot",
    "HostAllocationState",
    "HostMemoryPlan",
    "HostMemoryRequirements",
    "PostLoadPlan",
    "RuntimeConfig",
    "PROFILE_NAMES",
    "autotune",
    "auto_tune",
    "build_runtime_config",
    "choose_profile",
    "config_to_json",
    "detect_hardware",
    "get_hardware_info",
    "inspect_and_autotune",
    "inspect_hardware",
    "estimate_host_requirements",
    "plan_host_memory",
    "plan_post_load",
    "recommend_config",
    "select_profile",
]
