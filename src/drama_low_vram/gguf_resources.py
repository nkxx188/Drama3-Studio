"""Metadata-only resource planning for native Drama3 GGUF.

Estimates cover this runtime; free memory already excludes other applications.
They are admission estimates, not measured GPU peaks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

from .gguf_archive import Archive

MIB = 1024**2
GIB = 1024**3
BANK = re.compile(r"model_weights/model\.language_model\.layers\.(\d+)\.mlp\.experts\.(gate|up|down)_proj\.weight$")


def codec_capability(tensors):
    """Gate CUDA on the codec schema/types, independently of expert precision.

    These anchors describe the Fish DAC component the native Drama3 loader
    implements. The native loader still validates every required tensor.
    A metadata signature identifies layout/type compatibility, not audio quality
    or equality of weight payloads.
    """
    codec = {t.logical.removeprefix("codec_weights/"): t for t in tensors
             if t.logical.startswith("codec_weights/")}
    if not codec:
        return False, "", "No embedded Fish DAC codec tensors"
    signature = hashlib.sha256(json.dumps(
        sorted((name, t.kind, t.shape) for name, t in codec.items()),
        separators=(",", ":")).encode("utf-8")).hexdigest()
    if any(t.kind not in (0, 1, 30) for t in codec.values()):
        return False, signature, "Automatic CUDA codec requires F32/F16/BF16 codec weights"
    anchors = {
        "encoder.block.0.conv.weight": (64, 1, 7),
        "encoder.block.6.conv.weight": (1024, 1024, 3),
        "decoder.model.0.conv.weight": (1536, 1024, 7),
        "decoder.model.5.alpha": (96, 1),
        "decoder.model.6.conv.weight": (96, 7),
        "quantizer.semantic_quantizer.quantizers.0.codebook.weight": (4096, 8),
        **{f"quantizer.quantizer.quantizers.{i}.codebook.weight": (1024, 8) for i in range(9)},
    }
    for name, shape in anchors.items():
        if name not in codec or codec[name].shape != shape:
            return False, signature, "Unsupported Fish DAC tensor layout: " + name
    for module in ("pre_module", "post_module"):
        prefix = f"quantizer.{module}.layers.0.attention."
        packed = codec.get(prefix + "wqkv.weight")
        if packed is not None:
            if packed.shape != (3072, 1024):
                return False, signature, "Unsupported packed Fish DAC attention layout"
        elif any(prefix + projection + ".weight" not in codec or
                 codec[prefix + projection + ".weight"].shape != (1024, 1024)
                 for projection in ("q_proj", "k_proj", "v_proj")):
            return False, signature, "Incomplete Fish DAC attention projections"
    return True, signature, "Fish DAC 10-codebook floating codec; expert quantization is independent"


def resolve_gguf(value):
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        candidates = sorted(path.glob("*.gguf"))
        if len(candidates) != 1:
            raise ValueError("Choose one GGUF file; this folder has zero or multiple GGUF files")
        path = candidates[0]
    if path.suffix.lower() != ".gguf" or not path.is_file():
        raise ValueError("GGUF model file does not exist")
    return path


def is_gguf(value):
    path = Path(value)
    return path.suffix.lower() == ".gguf" or (path.is_dir() and any(path.glob("*.gguf")))


@lru_cache(maxsize=16)
def _inspect_cached(path, size, mtime_ns):
    return Layout.read(path)


def inspect_gguf(value):
    path = resolve_gguf(value)
    stat = path.stat()
    layout = _inspect_cached(str(path), stat.st_size, stat.st_mtime_ns)
    return {
        "directory": str(path), "ready": True, "runtime": "gguf",
        "storage_format": "GGUF " + ("Q8_0 / BF16" if layout.validated_q8 else f"expert types {layout.expert_types}"),
        "inspection_scope": "GGUF tensor directory/types; full weight SHA256 not verified",
        "total_bytes": stat.st_size, "present_bytes": stat.st_size, "partial_logical_bytes": 0,
        "missing": [], "invalid": [], "estimated_additional_bytes": 0,
        "disk_free_bytes": None, "layout": asdict(layout),
    }


@dataclass(frozen=True)
class Layout:
    model: str
    file_bytes: int
    layers: int
    experts: int
    expert_bytes: int
    slot_bytes: int
    dense_bytes: int
    codec_bytes: int
    decoder_bytes: int
    compact_head_saved_bytes: int
    host_embedding_saved_bytes: int
    validated_q8: bool
    expert_types: tuple[int, ...]
    codec_cuda_supported: bool = False
    codec_layout_signature: str = ""
    codec_note: str = ""
    host_embedding_supported: bool = False

    @classmethod
    def read(cls, path):
        archive = Archive(resolve_gguf(path))
        banks = [(BANK.fullmatch(t.logical), t) for t in archive.tensors]
        banks = [(m, t) for m, t in banks if m]
        layers = {int(m[1]) for m, _ in banks}
        experts = {t.shape[0] for _, t in banks if len(t.shape) == 3}
        if (not layers or layers != set(range(len(layers))) or len(banks) != 3 * len(layers)
                or len(experts) != 1 or any(len(t.shape) != 3 for _, t in banks)):
            raise ValueError("Not a supported Drama3 expert layout")
        count = experts.pop()
        if count != 256 or len(layers) != 40:
            raise ValueError("This runtime supports the Drama3 40-layer, 256-expert layout")
        head = archive.by_name.get("model_weights/lm_head.weight")
        embedding = archive.by_name.get("model_weights/model.language_model.embed_tokens.weight")
        if not head or not embedding:
            raise ValueError("Drama3 head or embedding tensor missing")
        expert_bytes = sum(t.nbytes for _, t in banks)
        dense = sum(t.nbytes for t in archive.tensors if t.logical.startswith("model_weights/")) - expert_bytes
        codec = sum(t.nbytes for t in archive.tensors if t.logical.startswith("codec_weights/"))
        decoder = sum(t.nbytes for t in archive.tensors
                      if t.logical.startswith(("codec_weights/decoder.", "codec_weights/quantizer."))) + 128 * MIB
        head_ok = head.kind == 30 and head.shape == (252288, 2048)
        embed_ok = embedding.kind == 30 and embedding.shape == (252288, 2048)
        # Reserve at most 8193 compact speech rows, conservatively.
        head_saved = head.nbytes - 8193 * 2048 * 2 if head_ok else 0
        types = tuple(sorted({t.kind for _, t in banks}))
        q8 = types == (8,) and head_ok and embed_ok
        codec_ok, codec_signature, codec_note = codec_capability(archive.tensors)
        return cls(str(archive.path), archive.path.stat().st_size, len(layers), count,
                   expert_bytes, expert_bytes // count, dense, codec, decoder,
                   head_saved, embedding.nbytes if embed_ok else 0, q8, types,
                   codec_ok, codec_signature, codec_note, embed_ok)


@dataclass(frozen=True)
class Plan:
    layout: Layout
    free_vram_bytes: int
    available_ram_bytes: int
    cache_experts: int
    host_storage: str
    host_embedding: bool
    codec_backend: str
    decoder_backend: str
    codec_strategy: str
    context_length: int
    threads: int
    resident_estimate_bytes: int
    workspace_estimate_bytes: int
    load_peak_host_estimate_bytes: int
    notes: tuple[str, ...]
    prefill_cache_mib: int = 256

    def report(self):
        return asdict(self) | {"cpu_expert_compute": False, "resource_estimates_are_measured_peaks": False}

    def session_options(self):
        return {
            "fish_drama3.disk_experts": "true",
            "fish_drama3.pin_experts": str(self.host_storage == "pin").lower(),
            "fish_drama3.expert_cache_size": str(self.cache_experts),
            "fish_drama3.cpu_expert_layers": "0",
            "fish_drama3.context_length": str(self.context_length),
            "fish_drama3.codec_backend": self.codec_backend,
            "fish_drama3.codec_decoder_backend": self.decoder_backend,
            "fish_drama3.codec_decode_strategy": self.codec_strategy,
            "fish_drama3.backbone_prefill_cache_mb": str(self.prefill_cache_mib),
            "fish_drama3.tower_prefill_cache_mb": str(self.prefill_cache_mib),
        }

    def fallbacks(self):
        """Each attempt starts after the previous process has exited."""
        current = self
        if current.host_storage == "pin":
            current = replace(current, host_storage="mapped",
                              notes=current.notes + ("Pinned source allocation failed: use mapped source.",))
            yield current
        if current.cache_experts > 8:
            current = replace(current, cache_experts=8,
                              resident_estimate_bytes=current.resident_estimate_bytes -
                              (current.cache_experts - 8) * current.layout.slot_bytes,
                              notes=current.notes + ("Allocation fallback: minimum top-8 cache.",))
            yield current
        if current.decoder_backend == "cuda":
            current = replace(current, decoder_backend="cpu", codec_strategy="full",
                              resident_estimate_bytes=current.resident_estimate_bytes - current.layout.decoder_bytes,
                              notes=current.notes + ("Allocation fallback: CPU waveform decoder.",))
            yield current


def plan_resources(layout, *, free_vram, available_ram, cores, context=1024, storage="auto", cache=None,
                   decoder="auto"):
    if free_vram <= 0 or available_ram <= 0:
        raise ValueError("Cannot plan without an available GPU/RAM memory snapshot")
    if context < 256 or context > 8192:
        raise ValueError("Service context must be between 256 and 8192")
    if storage not in ("auto", "mapped", "pin") or (cache is not None and not 8 <= cache <= layout.experts):
        raise ValueError("Invalid storage or cache setting")
    if decoder not in ("auto", "cpu", "cuda"):
        raise ValueError("Invalid waveform decoder setting")
    if decoder == "cuda" and not layout.codec_cuda_supported:
        raise ValueError("CUDA codec is not compatible: " + layout.codec_note)
    host_embedding = layout.host_embedding_supported
    base = layout.dense_bytes - layout.compact_head_saved_bytes
    if host_embedding:
        base -= layout.host_embedding_saved_bytes
    # Prefill/decode workspaces, KV/recurrent state, bounded allocation allowance.
    # This does not subtract system RAM again.
    workspace = 512 * MIB + context * 128 * 1024
    minimum = base + 8 * layout.slot_bytes + workspace
    if minimum > free_vram:
        raise MemoryError(
            f"GGUF minimum GPU estimate {minimum / GIB:.2f} GiB exceeds current free "
            f"{free_vram / GIB:.2f} GiB. Close GPU workloads or select a smaller model.")
    codec_minimum = base + (cache if cache is not None else 8) * layout.slot_bytes + workspace + layout.decoder_bytes
    codec_fits = layout.codec_cuda_supported and codec_minimum <= free_vram
    if decoder == "cuda" and not codec_fits:
        raise MemoryError("Requested CUDA codec plus expert cache exceeds current available VRAM estimate")
    chosen_decoder = "cuda" if decoder != "cpu" and codec_fits else "cpu"
    resident = base + (layout.decoder_bytes if chosen_decoder == "cuda" else 0)
    slots = min(layout.experts, max(8, (free_vram - resident - workspace) // layout.slot_bytes))
    # Prefill still scales with cache width; use a conservative temporary cap.
    slots = min(slots, 64) if cache is None else cache
    if resident + slots * layout.slot_bytes + workspace > free_vram:
        raise MemoryError("Requested expert cache exceeds current available VRAM estimate")
    # Pinning copies expert banks. The load also briefly materializes dense
    # sources and a full head. Page-cache source pages remain reclaimable.
    host_peak = layout.expert_bytes + layout.dense_bytes + layout.codec_bytes + 512 * MIB
    pinned = storage == "pin" or (storage == "auto" and available_ram >= host_peak)
    notes = ["Full top-8 GPU expert compute; mapped source may page to SSD under RAM pressure.",
             "Memory totals are estimates; failed allocations use bounded fallbacks."]
    prefill_cache = 256 if available_ram >= 16 * GIB else 64 if available_ram >= 8 * GIB else 0
    if prefill_cache < 256:
        notes.append("Host memory pressure limits optional prefill graph cache retention.")
    if not host_embedding:
        notes.append("Host embedding lookup requires an untied native BF16 embedding layout.")
    if not layout.codec_cuda_supported:
        notes.append("Automatic CUDA codec disabled: " + layout.codec_note)
    elif chosen_decoder == "cpu":
        notes.append("CPU waveform decoding selected by request or current available VRAM budget.")
    if slots == 64 and cache is None:
        notes.append("Automatic cache capped at 64 until prefill width is independently bounded.")
    return Plan(layout, int(free_vram), int(available_ram), int(slots),
                "pin" if pinned else "mapped", host_embedding, "cpu", chosen_decoder,
                "stream" if chosen_decoder == "cuda" else "full",
                context, max(1, min(8, cores or 1)), resident + slots * layout.slot_bytes,
                workspace, host_peak, tuple(notes), prefill_cache)
