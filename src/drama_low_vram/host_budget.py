"""Plan future host allocations without reserving the OS's used RAM twice.

Physical availability and Windows commit capacity are independent constraints.
Read-only file windows consume reclaimable physical pages but no private commit;
the expert RAM bank and model/codec tensors consume both. Phase peaks are compared
with ``max`` because checkpoint loading and codec construction do not overlap.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


GIB = 1024**3
MIB = 1024**2


def _integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class HostMemoryRequirements:
    """Bytes for the actual placement and loader, excluding existing process RAM.

    Defaults describe the official checkpoint at BF16 compute, a compact CUDA
    head, four staged experts, and a 1024-frame codec. ``codec_build_bytes`` is
    the FP32 constructor; it replaces, rather than adds to, BF16 codec residency.
    ``ram_load_temporary_bytes`` covers the legacy RAM loader's full output head.
    """

    expert_bytes: int = 32_220_119_040
    cpu_weight_bytes: int = 1_033_371_648
    parking_bytes: int = 5_324_847_368
    staging_bytes: int = 12_585_984
    ram_load_temporary_bytes: int = 1_050_185_736
    mmap_load_temporary_bytes: int = 16_814_088
    load_file_window_bytes: int = 1_033_371_648
    codec_resident_bytes: int = 834_701_282
    codec_build_bytes: int = 1_617_626_632
    codec_load_temporary_bytes: int = 92_274_688
    codec_checkpoint_commit_bytes: int = 1_871_099_728
    codec_file_window_bytes: int = 75_497_472
    file_backed_cpu_weight_bytes: int = 0
    file_backed_parking_bytes: int = 0
    fixed_gpu_bytes: int = 5_324_847_368
    num_experts_per_layer: int = 256
    source: str = "official-checkpoint-defaults"

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name != "source":
                _integer(name, value)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HostAllocationState:
    """Allocations already charged before the availability snapshot was taken.

    Only include tensors represented by ``HostMemoryRequirements``. The engine
    can pass an already loaded CPU codec here; its RAM must not be deducted again.
    Clean mmap pages are excluded: available RAM already incorporates reclaimable
    file pages according to the OS, and these pages are not private allocations.
    """

    experts_bytes: int = 0
    cpu_weights_bytes: int = 0
    parking_bytes: int = 0
    staging_bytes: int = 0
    codec_bytes: int = 0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _integer(name, value)

    @property
    def total_bytes(self) -> int:
        return sum(asdict(self).values())


@dataclass(frozen=True)
class HostMemoryPlan:
    requested_storage: str
    expert_storage: str
    ram_available_bytes: int
    available_commit_bytes: int
    commit_budget_estimated: bool
    buffer_bytes: int
    already_allocated_bytes: int
    required_physical_bytes: int
    required_commit_bytes: int
    expert_ram_budget_bytes: int
    physical_fits: bool
    commit_fits: bool
    phase_physical_bytes: dict[str, int]
    phase_commit_bytes: dict[str, int]
    requirements: HostMemoryRequirements
    status: str = "ready"
    warnings: tuple[str, ...] = ()
    explanations: tuple[str, ...] = ()

    @property
    def can_run(self) -> bool:
        return self.status == "ready"

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["warnings"] = list(self.warnings)
        result["explanations"] = list(self.explanations)
        result["can_run"] = self.can_run
        return result

    to_dict = as_dict

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, ensure_ascii=False)


def estimate_host_requirements(
    checkpoint_dir: str | Path | None = None,
    *,
    compact_head: bool = True,
    staged_experts: int = 4,
    codec_max_frames: int = 1024,
    park_model: bool = True,
    include_codec: bool = True,
    direct_compact_head: bool = False,
    lm_head_device: str = "cpu",
    embedding_device: str = "cpu",
) -> HostMemoryRequirements:
    """Read tensor metadata only; no GPU allocation or checkpoint payload read.

    Safetensors headers are validated by the same reader as model loading.
    ``FakeTensorMode`` inspects codec metadata without reading or mapping tensor
    payloads. Without a path the official checkpoint's measured dimensions
    are used. Runtime BF16 tensors are counted at two bytes per element.
    """
    _integer("staged_experts", staged_experts)
    _integer("codec_max_frames", codec_max_frames)
    if staged_experts < 1 or codec_max_frames < 1:
        raise ValueError("staged_experts and codec_max_frames must be positive")
    if lm_head_device not in ("cpu", "cuda") or embedding_device not in ("cpu", "cuda"):
        raise ValueError("Host weight placement must be cpu or cuda")
    expert = 32_220_119_040
    embedding = head = 1_033_371_648
    fixed = 5_308_033_280
    compact = 4_097 * 2_048 * 2 + 4_097 * 8
    expert_count = 40 * 256
    largest = 1_033_371_648
    codec_build = 1_565_722_120
    codec_resident = 782_796_770
    codec_window = 75_497_472
    codec_commit = 1_871_099_728
    codec_transformers = 3
    codec_rope_head_dim = 64
    auxiliary_cpu = 0
    mapped_embedding = mapped_fixed = mapped_head = 0
    experts_per_layer = 256
    source = "official-checkpoint-defaults"
    if checkpoint_dir is not None:
        import torch

        from .checkpoint_reader import _checkpoint_headers, _read_header
        from .shared_checkpoint import descriptor

        checkpoint = Path(checkpoint_dir).resolve()
        config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        shared = descriptor(checkpoint)
        headers = [_read_header(Path(shared["model"]))] if shared else _checkpoint_headers(checkpoint)
        entries = {
            name: entry
            for header in headers
            for name, entry in header.entries.items()
        }
        expert_entries = [entry for name, entry in entries.items() if ".experts." in name]
        if not expert_entries:
            raise ValueError("Checkpoint contains no routed experts")
        embedding_key = "model.embed_tokens.weight" if shared else "model.language_model.embed_tokens.weight"
        head_key = "audio_head.weight" if shared else "lm_head.weight"
        embedding_entry = entries[embedding_key]
        head_entry = entries[head_key]
        quantization = (
            shared.get("quantization") if shared else
            "gptq" if config.get("quantization_config", {}).get("quant_method") == "drama3-int4"
            else None
        )
        expert = sum(entry.nbytes for entry in expert_entries)
        if quantization == "gptq":
            expert = sum(entry.nbytes for name, entry in entries.items()
                         if ".experts." in name and name.endswith((".qweight", ".scales")))
        elif quantization == "int8_convrot":
            expert = sum(entry.nbytes for name, entry in entries.items()
                         if ".experts." in name and name.endswith((".weight", ".weight_scale")))
        embedding = embedding_entry.nbytes
        head = head_entry.nbytes
        if shared and shared.get("quantization") in ("w6a8", "w4a8"):
            # W6 payload weights remain mapped, while the staging interface
            # packs relative/channel scales into private byte rows.
            auxiliary_cpu = sum(entry.nbytes for name, entry in entries.items()
                                if ".experts." in name
                                and name.endswith((".weight_s_rel", ".weight_s_channel")))
            if shared.get("quantization") == "w4a8":
                auxiliary_cpu += sum(entry.shape[0] * entry.shape[1] * 4
                                     for name, entry in entries.items()
                                     if ".experts." in name and name.endswith(".weight_s_rel"))
            # GPU staging uses the packed auxiliary rows, not the JSON format
            # descriptors or a separate copy of the original codebook tensors.
            expert = auxiliary_cpu + sum(
                entry.nbytes for name, entry in entries.items()
                if ".experts." in name and name.endswith(".weight"))
        fixed = sum(
            (entry.nbytes if quantization == "gptq" and name.endswith((".qweight", ".scales"))
             else 0 if quantization == "gptq" and name.endswith((".qzeros", ".g_idx"))
             else 2 * entry.nbytes // torch.empty((), dtype=entry.dtype).element_size())
            for name, entry in entries.items()
            if ".experts." not in name
            and name not in (embedding_key, head_key)
        )
        rows = config["semantic_end_token_id"] - config["semantic_start_token_id"] + 2
        if rows <= 1 or len(head_entry.shape) != 2:
            raise ValueError("Invalid semantic head metadata")
        compact = rows * head_entry.shape[1] * 2 + rows * 8
        if shared and shared.get("quantization") is None:
            # The shared FP8 loader retains exact BF16 checkpoint views.
            # These clean file pages are reclaimable, not private RAM copies.
            if embedding_entry.dtype == torch.bfloat16 and embedding_device == "cpu":
                mapped_embedding = embedding
            mapped_fixed = sum(
                entry.nbytes for name, entry in entries.items()
                if ".experts." not in name and name not in (embedding_key, head_key)
                and entry.dtype == torch.bfloat16
            )
            if compact_head and head_entry.dtype == torch.bfloat16:
                mapped_head = head
            largest = max(
                (entry.nbytes for name, entry in entries.items()
                 if ".experts." not in name
                 and not (name == embedding_key and mapped_embedding)),
                default=0,
            )
        experts_per_layer = config["text_config"]["num_experts"]
        expert_count = config["text_config"]["num_hidden_layers"] * experts_per_layer
        if expert_count < 1 or experts_per_layer < 1 or expert < 1:
            raise ValueError("Invalid routed expert metadata")
        if not (shared and shared.get("quantization") is None):
            largest = max(entry.nbytes for entry in entries.values())
        if include_codec:
            from torch._subclasses.fake_tensor import FakeTensorMode

            codec_path = checkpoint / "codec.pth"
            # mmap=True creates a whole-file COW mapping before map_location is
            # considered, reserving its full size as commit on Windows. Fake
            # tensors skip the storage records without opening that mapping.
            with FakeTensorMode():
                state = torch.load(str(codec_path), mmap=False, weights_only=True)
            if "state_dict" in state and isinstance(state["state_dict"], dict):
                state = state["state_dict"]
            if any("generator." in key for key in state):
                state = {key.replace("generator.", ""): value for key, value in state.items() if "generator." in key}
            tensors = {key: value for key, value in state.items() if isinstance(value, torch.Tensor)}
            masks = [tensor for key, tensor in tensors.items() if key.endswith(".causal_mask")]
            ropes = [tensor for key, tensor in tensors.items() if key.endswith(".freqs_cis")]
            codec_transformers = len(masks)
            codec_rope_head_dim = max((int(tensor.shape[-2]) * 2 for tensor in ropes), default=64)
            learned = {key: value for key, value in tensors.items() if not key.endswith((".causal_mask", ".freqs_cis"))}
            if not learned:
                raise ValueError("Codec checkpoint contains no learned tensors")
            codec_build = sum(tensor.numel() * 4 for tensor in learned.values())
            codec_resident = sum(
                tensor.numel() * 2 for key, tensor in learned.items()
                if not key.endswith(("parametrizations.weight.original0", "weight_g"))
            )
            codec_window = max(tensor.numel() * tensor.element_size() for tensor in learned.values())
            codec_commit = codec_path.stat().st_size
        source = str(checkpoint)
    cpu_weights = (embedding - mapped_embedding if embedding_device == "cpu" else 0) + (
        head if not compact_head and lm_head_device == "cpu" else 0
    ) + auxiliary_cpu
    parked = fixed - mapped_fixed + (embedding if embedding_device == "cuda" else 0) + (
        compact if compact_head else (head if lm_head_device == "cuda" else 0)
    ) - mapped_head
    # Each codec Transformer uses a bounded bool mask and BF16 rotary table.
    # Constructor peak includes the temporary ones/tril mask and one folded
    # FP32 weight; these temporaries are absent from the BF16 resident codec.
    context = codec_max_frames * 4
    codec_buffers = codec_transformers * (context * context + context * codec_rope_head_dim * 2)
    codec_scratch = codec_window + context * context
    compact_scratch = compact if compact_head else 0
    return HostMemoryRequirements(
        expert_bytes=expert,
        cpu_weight_bytes=cpu_weights,
        parking_bytes=parked if park_model and include_codec else 0,
        staging_bytes=(expert + expert_count - 1) // expert_count * staged_experts,
        ram_load_temporary_bytes=(head if compact_head and not direct_compact_head else 0) + compact_scratch,
        mmap_load_temporary_bytes=compact_scratch,
        load_file_window_bytes=largest,
        codec_resident_bytes=codec_resident + codec_buffers if include_codec else 0,
        codec_build_bytes=codec_build + codec_buffers if include_codec else 0,
        codec_load_temporary_bytes=codec_scratch if include_codec else 0,
        codec_checkpoint_commit_bytes=codec_commit if include_codec else 0,
        codec_file_window_bytes=codec_window if include_codec else 0,
        file_backed_cpu_weight_bytes=mapped_embedding,
        file_backed_parking_bytes=mapped_fixed + mapped_head if park_model and include_codec else 0,
        fixed_gpu_bytes=(
            fixed + (embedding if embedding_device == "cuda" else 0)
            + (compact if compact_head else head if lm_head_device == "cuda" else 0)
        ),
        num_experts_per_layer=experts_per_layer,
        source=source,
    )


def plan_host_memory(
    *,
    ram_available_bytes: int,
    commit_available_bytes: int | None = None,
    swap_available_bytes: int = 0,
    expert_storage: str = "auto",
    requirements: HostMemoryRequirements | None = None,
    allocated: HostAllocationState | None = None,
    buffer_bytes: int | None = None,
) -> HostMemoryPlan:
    """Prefer complete RAM residency when both future phase peaks fit.

    Private allocations can page when physical RAM is short and commit fits;
    auto chooses the reclaimable mmap bank in that situation. The small buffer
    protects a changing memory reading, not RAM already used by the desktop.
    ``expert_ram_budget_bytes`` is the remaining *private* RAM budget for a future
    partial resident bank; callers must avoid copying mmap pages into that bank
    without managing duplicate file residency.
    """
    _integer("ram_available_bytes", ram_available_bytes)
    _integer("swap_available_bytes", swap_available_bytes)
    if commit_available_bytes is not None:
        _integer("commit_available_bytes", commit_available_bytes)
    if expert_storage not in ("auto", "ram", "mmap"):
        raise ValueError("expert_storage must be auto, ram or mmap")
    requirements = requirements or estimate_host_requirements()
    allocated = allocated or HostAllocationState()
    if buffer_bytes is None:
        buffer_bytes = max(256 * MIB, min(512 * MIB, ram_available_bytes // 100))
    _integer("buffer_bytes", buffer_bytes)
    available_commit = (
        ram_available_bytes + swap_available_bytes
        if commit_available_bytes is None else commit_available_bytes
    )
    already = allocated.total_bytes

    def phases(storage: str) -> tuple[dict[str, int], dict[str, int]]:
        req = requirements
        bank = req.expert_bytes if storage == "ram" else 0
        base = max(bank, allocated.experts_bytes) + max(req.cpu_weight_bytes, allocated.cpu_weights_bytes)
        loaded_codec = allocated.codec_bytes
        runtime_codec = max(loaded_codec, req.codec_resident_bytes)
        parking = max(allocated.parking_bytes, req.parking_bytes)
        staging = max(allocated.staging_bytes, req.staging_bytes)
        needs_codec_load = allocated.codec_bytes == 0 and req.codec_build_bytes > 0
        needs_codec_growth = 0 < allocated.codec_bytes < req.codec_resident_bytes
        load_temp = req.ram_load_temporary_bytes if storage == "ram" else req.mmap_load_temporary_bytes
        # A cold codec is constructed and converted before the first model
        # parking operation. Already-created mirrors remain live, however.
        codec_load = (
            base + allocated.parking_bytes + staging
            + req.codec_build_bytes + req.codec_load_temporary_bytes
            + req.codec_checkpoint_commit_bytes
            if needs_codec_load else
            base + parking + staging + runtime_codec
            + (req.codec_load_temporary_bytes if needs_codec_growth else 0)
        )
        commit = {
            "model_load": base + loaded_codec + allocated.parking_bytes + allocated.staging_bytes + load_temp,
            # Parking mirrors survive model resume, and a CPU codec survives
            # reference encoding. Later generations therefore retain both.
            "generation": base + parking + runtime_codec + staging,
            "codec_load": codec_load,
            "codec_run": base + parking + runtime_codec + staging,
        }
        physical = dict(commit)
        physical["model_load"] += req.load_file_window_bytes
        if needs_codec_load:
            physical["codec_load"] += req.codec_file_window_bytes - req.codec_checkpoint_commit_bytes
        return (
            {name: max(0, value - already) for name, value in physical.items()},
            {name: max(0, value - already) for name, value in commit.items()},
        )

    ram_physical, ram_commit = phases("ram")
    ram_fits = (
        max(ram_physical.values()) + buffer_bytes <= ram_available_bytes
        and max(ram_commit.values()) + buffer_bytes <= available_commit
    )
    storage = ("ram" if ram_fits else "mmap") if expert_storage == "auto" else expert_storage
    physical, commit = phases(storage)
    physical_required = max(physical.values()) + buffer_bytes
    commit_required = max(commit.values()) + buffer_bytes
    physical_fits = physical_required <= ram_available_bytes
    commit_fits = commit_required <= available_commit
    nonexpert_physical, nonexpert_commit = phases("mmap")
    resident_budget = max(0, min(
        ram_available_bytes - max(nonexpert_physical.values()),
        available_commit - max(nonexpert_commit.values()),
    ) - buffer_bytes)
    resident_budget = min(requirements.expert_bytes, resident_budget + allocated.experts_bytes)
    warnings = []
    explanations = [
        "Use current available RAM; memory already occupied by Windows and other programs is not subtracted again.",
        "Compare model-load, generation, and codec phase peaks; their temporary allocations are not added together.",
        f"Keep {buffer_bytes / MIB:.0f} MiB for memory-reading changes and small unmodelled allocations.",
    ]
    if commit_available_bytes is None:
        explanations.append("Commit availability is estimated from current physical RAM plus available swap.")
    else:
        explanations.append("Use the OS commit capacity separately from physical RAM; read-only mappings do not charge private commit.")
    if expert_storage == "auto":
        explanations.append("All expert bytes fit the future RAM and commit peaks." if storage == "ram" else "Keep the expert bank file-backed because the complete RAM bank does not fit both future peaks.")
    if not physical_fits:
        warnings.append("Available physical RAM is below the future host peak; private weights may require very slow OS paging.")
    if not commit_fits:
        warnings.append("Available host commit is below the future allocation peak.")
    status = "ready" if commit_fits else "resource_busy"
    return HostMemoryPlan(
        requested_storage=expert_storage, expert_storage=storage,
        ram_available_bytes=ram_available_bytes, available_commit_bytes=available_commit,
        commit_budget_estimated=commit_available_bytes is None,
        buffer_bytes=buffer_bytes, already_allocated_bytes=already,
        required_physical_bytes=physical_required, required_commit_bytes=commit_required,
        expert_ram_budget_bytes=resident_budget,
        physical_fits=physical_fits, commit_fits=commit_fits,
        phase_physical_bytes=physical, phase_commit_bytes=commit,
        requirements=requirements, status=status,
        warnings=tuple(warnings), explanations=tuple(explanations),
    )


__all__ = [
    "HostAllocationState", "HostMemoryPlan", "HostMemoryRequirements",
    "estimate_host_requirements", "plan_host_memory",
]
