"""Read-only disk-backed storage of the original Drama 3 FP8 expert bytes.

Windows copy-on-write mappings charge commit for the whole view. This backend
uses ``mmap.ACCESS_READ`` and zero-copy tensor views, so opening the bank does
not require enough RAM or pagefile to hold all experts. Consumers must only
read these buffers; OS-resident file pages remain subject to normal paging.
"""

from __future__ import annotations

import hashlib
import json
import math
import mmap
import os
import re
import shutil
import sys
import tempfile
import warnings
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from .offload import EXPERT_BUFFER_NAMES, FP8_BLOCK_SIZE, FP8_DTYPE

CACHE_FORMAT = "drama3-official-fp8-experts"
CACHE_FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
DATA_NAME = "experts.bin"
_COPY_CHUNK_BYTES = 1024 * 1024
_HASH_PATTERN = re.compile(r"[a-f0-9]{64}")
_EXPERT_KEY = re.compile(
    r"(?:model\.language_model|model)\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<projection>gate_proj|up_proj|down_proj)\."
    r"(?P<kind>weight|weight_scale_inv)"
)
_DTYPES = {
    "w13_weight": (FP8_DTYPE, "float8_e4m3fn", 1),
    "w13_scale_inv": (torch.float32, "float32", 4),
    "w2_weight": (FP8_DTYPE, "float8_e4m3fn", 1),
    "w2_scale_inv": (torch.float32, "float32", 4),
}
_PARTS = {
    "w13_weight": (("gate_proj", "weight"), ("up_proj", "weight")),
    "w13_scale_inv": (
        ("gate_proj", "weight_scale_inv"),
        ("up_proj", "weight_scale_inv"),
    ),
    "w2_weight": (("down_proj", "weight"),),
    "w2_scale_inv": (("down_proj", "weight_scale_inv"),),
}


def _json_file(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_shard(checkpoint: Path, name: str) -> Path:
    shard = (checkpoint / name).resolve()
    if not shard.is_relative_to(checkpoint):
        raise ValueError(f"Checkpoint shard escapes its directory: {name!r}")
    return shard


def _remove_temporary_directory(path: Path, parent: Path, prefix: str) -> None:
    resolved = path.resolve()
    if resolved.parent != parent.resolve() or not resolved.name.startswith(prefix):
        raise ValueError(f"Refusing cleanup outside the expert cache directory: {resolved}")
    shutil.rmtree(resolved)


def _read_shard_header(path: Path) -> tuple[dict[str, Any], int, str]:
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"Truncated safetensors file: {path}")
        length = int.from_bytes(length_bytes, "little")
        if not 1 <= length <= min(128 * 1024**2, path.stat().st_size - 8):
            raise ValueError(f"Invalid safetensors header size: {path}")
        header_bytes = handle.read(length)
    try:
        header = json.loads(header_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid safetensors header: {path}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors header object: {path}")
    signature = hashlib.sha256(length_bytes + header_bytes).hexdigest()
    return header, 8 + length, signature


def _layout(
    num_layers: int, num_experts: int, hidden: int, intermediate: int
) -> tuple[list[dict[str, Any]], int]:
    for label, value in (
        ("num_layers", num_layers),
        ("num_experts", num_experts),
        ("hidden", hidden),
        ("intermediate", intermediate),
    ):
        _integer(value, label, minimum=1)
    if hidden % FP8_BLOCK_SIZE or intermediate % FP8_BLOCK_SIZE:
        raise ValueError("Expert dimensions must be divisible by the FP8 block size")
    shapes = {
        "w13_weight": (num_experts, 2 * intermediate, hidden),
        "w13_scale_inv": (
            num_experts,
            2 * intermediate // FP8_BLOCK_SIZE,
            hidden // FP8_BLOCK_SIZE,
        ),
        "w2_weight": (num_experts, hidden, intermediate),
        "w2_scale_inv": (
            num_experts,
            hidden // FP8_BLOCK_SIZE,
            intermediate // FP8_BLOCK_SIZE,
        ),
    }
    offset = 0
    layers = []
    for layer in range(num_layers):
        buffers = {}
        for name in EXPERT_BUFFER_NAMES:
            _, dtype_name, element_size = _DTYPES[name]
            size = math.prod(shapes[name]) * element_size
            buffers[name] = {
                "shape": list(shapes[name]),
                "dtype": dtype_name,
                "offset": offset,
                "nbytes": size,
            }
            offset += size
        layers.append({"index": layer, "buffers": buffers})
    return layers, offset


@dataclass(frozen=True)
class _SourceTensor:
    shard: str
    offset: int
    nbytes: int


@dataclass(frozen=True)
class ExpertCacheSource:
    checkpoint_dir: Path
    model_fingerprint: str
    num_layers: int
    num_experts: int
    hidden: int
    intermediate: int
    nbytes: int
    source: dict[str, Any]
    tensors: dict[tuple[int, int, str, str], _SourceTensor]

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_dir": str(self.checkpoint_dir),
            "model_fingerprint": self.model_fingerprint,
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "hidden": self.hidden,
            "intermediate": self.intermediate,
            "nbytes": self.nbytes,
            "tensor_count": len(self.tensors),
            "source_verification": "config/index/header SHA256 and source size/mtime",
        }


def inspect_expert_cache_source(checkpoint_dir: str | Path) -> ExpertCacheSource:
    """Validate all expert names, shapes, dtypes and byte ranges from headers."""
    checkpoint = Path(checkpoint_dir).resolve()
    config, config_hash = _json_file(checkpoint / "config.json")
    index, index_hash = _json_file(checkpoint / "model.safetensors.index.json")
    if config.get("model_type") != "fish_omni":
        raise ValueError("Expert cache requires an official Fish Omni checkpoint")
    text = config.get("text_config")
    if not isinstance(text, dict):
        raise ValueError("Checkpoint config has no text_config")
    dimensions = {
        "num_layers": _integer(
            text.get("num_hidden_layers"), "num_hidden_layers", minimum=1
        ),
        "num_experts": _integer(text.get("num_experts"), "num_experts", minimum=1),
        "hidden": _integer(text.get("hidden_size"), "hidden_size", minimum=1),
        "intermediate": _integer(
            text.get("moe_intermediate_size"), "moe_intermediate_size", minimum=1
        ),
    }
    _, total_bytes = _layout(**dimensions)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Checkpoint index has no weight_map")
    if not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in weight_map.items()
    ):
        raise ValueError("Checkpoint weight_map must contain string names and shards")
    headers, source_files = {}, []
    for shard_name in sorted(set(weight_map.values())):
        shard = _safe_shard(checkpoint, shard_name)
        header, data_start, header_hash = _read_shard_header(shard)
        stat = shard.stat()
        headers[shard_name] = (header, data_start, stat.st_size)
        source_files.append(
            {
                "name": shard_name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "header_sha256": header_hash,
            }
        )

    tensors = {}
    hidden, intermediate = dimensions["hidden"], dimensions["intermediate"]
    for name, shard_name in weight_map.items():
        match = _EXPERT_KEY.fullmatch(name)
        if match is None:
            if ".mlp.experts." in name:
                raise ValueError(f"Unsupported routed expert checkpoint key: {name}")
            continue
        layer, expert = int(match["layer"]), int(match["expert"])
        projection, kind = match["projection"], match["kind"]
        if layer >= dimensions["num_layers"] or expert >= dimensions["num_experts"]:
            raise ValueError(f"Expert index outside model config: {name}")
        key = (layer, expert, projection, kind)
        if key in tensors:
            raise ValueError(f"Duplicate expert projection: {name}")
        header, data_start, file_size = headers[shard_name]
        entry = header.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"Checkpoint index/header mismatch: {name}")
        rows, columns = (
            (hidden, intermediate)
            if projection == "down_proj"
            else (intermediate, hidden)
        )
        if kind == "weight_scale_inv":
            rows, columns = rows // FP8_BLOCK_SIZE, columns // FP8_BLOCK_SIZE
            expected_dtype, element_size = "F32", 4
        else:
            expected_dtype, element_size = "F8_E4M3", 1
        if entry.get("shape") != [rows, columns] or entry.get("dtype") != expected_dtype:
            raise ValueError(f"Wrong expert shape/dtype: {name}")
        offsets = entry.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"Invalid source byte offsets: {name}")
        start = _integer(offsets[0], f"{name} byte offset")
        end = _integer(offsets[1], f"{name} byte end")
        size = rows * columns * element_size
        if end - start != size or data_start + end > file_size:
            raise ValueError(f"Source tensor size/range mismatch: {name}")
        tensors[key] = _SourceTensor(shard_name, data_start + start, size)
    expected_count = dimensions["num_layers"] * dimensions["num_experts"] * 6
    if len(tensors) != expected_count:
        raise ValueError(f"Missing expert tensors: {len(tensors)}/{expected_count}")
    source = {
        "config_sha256": config_hash,
        "index_sha256": index_hash,
        "files": source_files,
    }
    identity = {
        **source,
        "files": [
            {k: v for k, v in item.items() if k != "mtime_ns"} for item in source_files
        ],
    }
    return ExpertCacheSource(
        checkpoint_dir=checkpoint,
        model_fingerprint=_fingerprint(identity),
        nbytes=total_bytes,
        source=source,
        tensors=tensors,
        **dimensions,
    )


def build_expert_cache(
    checkpoint_dir: str | Path,
    cache_dir: str | Path,
    *,
    force: bool = False,
    disk_reserve_bytes: int = 512 * 1024**2,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Repack raw bytes with bounded copies, then commit a complete directory."""
    _integer(disk_reserve_bytes, "disk_reserve_bytes")
    source = inspect_expert_cache_source(checkpoint_dir)
    cache_dir = Path(cache_dir).resolve()
    if cache_dir == source.checkpoint_dir or cache_dir in source.checkpoint_dir.parents:
        raise ValueError("Expert cache cannot replace the checkpoint directory")
    if cache_dir.exists() and not force:
        raise FileExistsError(f"Expert cache already exists: {cache_dir}")
    if cache_dir.exists() and not cache_dir.is_dir():
        raise ValueError("Expert cache path must be a directory")
    if cache_dir.exists():
        existing, _ = _json_file(cache_dir / MANIFEST_NAME)
        if existing.get("format") != CACHE_FORMAT:
            raise ValueError("Refusing to replace a directory that is not an expert cache")
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(cache_dir.parent).free
    if free_bytes < source.nbytes + disk_reserve_bytes:
        raise OSError(
            f"Expert cache needs {source.nbytes:,} bytes plus {disk_reserve_bytes:,} "
            f"reserve; disk has {free_bytes:,} bytes free"
        )
    layers, size = _layout(
        source.num_layers, source.num_experts, source.hidden, source.intermediate
    )
    partial = Path(
        tempfile.mkdtemp(prefix=f".{cache_dir.name}.partial-", dir=cache_dir.parent)
    )
    committed = False
    try:
        digest = hashlib.sha256()
        copied = 0
        with ExitStack() as stack:
            handles = {
                item["name"]: stack.enter_context(
                    _safe_shard(source.checkpoint_dir, item["name"]).open(
                        "rb", buffering=0
                    )
                )
                for item in source.source["files"]
            }
            destination = stack.enter_context(
                (partial / DATA_NAME).open("wb", buffering=0)
            )
            for layer in range(source.num_layers):
                for name in EXPERT_BUFFER_NAMES:
                    for expert in range(source.num_experts):
                        for projection, kind in _PARTS[name]:
                            tensor = source.tensors[(layer, expert, projection, kind)]
                            handle = handles[tensor.shard]
                            handle.seek(tensor.offset)
                            remaining = tensor.nbytes
                            while remaining:
                                chunk = handle.read(min(remaining, _COPY_CHUNK_BYTES))
                                if not chunk:
                                    raise OSError(
                                        f"Truncated source shard: {tensor.shard}"
                                    )
                                if destination.write(chunk) != len(chunk):
                                    raise OSError(
                                        "Short write while building expert cache"
                                    )
                                digest.update(chunk)
                                remaining -= len(chunk)
                                copied += len(chunk)
                                if progress is not None:
                                    progress(copied, size)
            destination.flush()
            os.fsync(destination.fileno())
        if copied != size:
            raise ValueError(f"Expert cache byte count mismatch: {copied} != {size}")
        if inspect_expert_cache_source(source.checkpoint_dir).source != source.source:
            raise ValueError("Source checkpoint changed while building expert cache")
        manifest = {
            "format": CACHE_FORMAT,
            "format_version": CACHE_FORMAT_VERSION,
            "byte_order": "little",
            "model_fingerprint": source.model_fingerprint,
            "model": {
                "model_type": "fish_omni",
                "num_layers": source.num_layers,
                "num_experts": source.num_experts,
                "hidden": source.hidden,
                "intermediate": source.intermediate,
                "block_size": FP8_BLOCK_SIZE,
            },
            "source": source.source,
            "data": {"name": DATA_NAME, "size": size, "sha256": digest.hexdigest()},
            "layers": layers,
        }
        with (partial / MANIFEST_NAME).open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        backup = None
        if cache_dir.exists():
            backup = Path(
                tempfile.mkdtemp(
                    prefix=f".{cache_dir.name}.old-", dir=cache_dir.parent
                )
            )
            backup.rmdir()
            os.replace(cache_dir, backup)
        try:
            os.replace(partial, cache_dir)
            committed = True
        except BaseException:
            if backup is not None:
                os.replace(backup, cache_dir)
            raise
        if backup is not None:
            _remove_temporary_directory(
                backup, cache_dir.parent, f".{cache_dir.name}.old-"
            )
        return cache_dir
    finally:
        if not committed and partial.exists():
            _remove_temporary_directory(
                partial, cache_dir.parent, f".{cache_dir.name}.partial-"
            )


class ExpertCacheFile:
    """Validated mmap expert bank; mapped views retain their own file lifetime."""

    def __init__(self, cache_dir: Path, manifest: dict[str, Any]) -> None:
        self.cache_dir = cache_dir
        self.manifest = manifest
        self._mapping: torch.Tensor | None = None
        self._mmap: mmap.mmap | None = None
        self._closed = False
        if manifest.get("format") != CACHE_FORMAT:
            raise ValueError("Unsupported expert cache format")
        if manifest.get("format_version") != CACHE_FORMAT_VERSION:
            raise ValueError("Unsupported expert cache version")
        if manifest.get("byte_order") != "little" or sys.byteorder != "little":
            raise ValueError("Expert cache requires little-endian byte order")
        model = manifest.get("model")
        if not isinstance(model, dict) or model.get("model_type") != "fish_omni":
            raise ValueError("Expert cache is not a Fish Omni model")
        if model.get("block_size") != FP8_BLOCK_SIZE:
            raise ValueError("Unsupported expert block size")
        self.num_layers = _integer(model.get("num_layers"), "num_layers", minimum=1)
        self.num_experts = _integer(model.get("num_experts"), "num_experts", minimum=1)
        self.hidden = _integer(model.get("hidden"), "hidden", minimum=1)
        self.intermediate = _integer(model.get("intermediate"), "intermediate", minimum=1)
        expected_layers, self.nbytes = _layout(
            self.num_layers, self.num_experts, self.hidden, self.intermediate
        )
        if manifest.get("layers") != expected_layers:
            raise ValueError("Expert cache buffer layout mismatch")
        data = manifest.get("data")
        if not isinstance(data, dict) or data.get("name") != DATA_NAME:
            raise ValueError("Invalid expert cache data filename")
        if _integer(data.get("size"), "data size", minimum=1) != self.nbytes:
            raise ValueError("Expert cache data size mismatch")
        if not isinstance(data.get("sha256"), str) or not _HASH_PATTERN.fullmatch(
            data["sha256"]
        ):
            raise ValueError("Invalid expert cache data SHA256")
        fingerprint = manifest.get("model_fingerprint")
        if not isinstance(fingerprint, str) or not _HASH_PATTERN.fullmatch(fingerprint):
            raise ValueError("Invalid expert cache model fingerprint")
        self.model_fingerprint = fingerprint
        self._layers = expected_layers
        self._data_path = cache_dir / DATA_NAME
        if not self._data_path.is_file() or self._data_path.stat().st_size != self.nbytes:
            raise ValueError("Expert cache file is truncated or has extra data")

    @classmethod
    def open(
        cls,
        cache_dir: str | Path,
        *,
        checkpoint_dir: str | Path | None = None,
        verify_data: bool = False,
    ) -> "ExpertCacheFile":
        cache_dir = Path(cache_dir).resolve()
        manifest, _ = _json_file(cache_dir / MANIFEST_NAME)
        cache = cls(cache_dir, manifest)
        if checkpoint_dir is not None:
            cache.verify_source(checkpoint_dir)
        if verify_data:
            cache.verify_data()
        return cache

    @property
    def is_mapped(self) -> bool:
        return self._mapping is not None

    @property
    def expert_module_names(self) -> tuple[str, ...]:
        return tuple(
            f"model.layers.{layer}.mlp.experts" for layer in range(self.num_layers)
        )

    @property
    def expert_checkpoint_prefixes(self) -> tuple[str, ...]:
        return tuple(
            f"{prefix}.layers.{layer}.mlp.experts."
            for layer in range(self.num_layers)
            for prefix in ("model.language_model", "model")
        )

    def should_skip_checkpoint_key(self, name: str) -> bool:
        match = _EXPERT_KEY.fullmatch(name)
        return (
            match is not None
            and int(match["layer"]) < self.num_layers
            and int(match["expert"]) < self.num_experts
        )

    def verify_source(self, checkpoint_dir: str | Path) -> None:
        source = inspect_expert_cache_source(checkpoint_dir)
        if source.model_fingerprint != self.model_fingerprint:
            raise ValueError("Expert cache does not match this model checkpoint")
        if source.source != self.manifest.get("source"):
            raise ValueError("Expert cache source metadata changed")

    def verify_data(self) -> None:
        digest = hashlib.sha256()
        with self._data_path.open("rb") as handle:
            while chunk := handle.read(_COPY_CHUNK_BYTES):
                digest.update(chunk)
        if digest.hexdigest() != self.manifest["data"]["sha256"]:
            raise ValueError("Expert cache data checksum mismatch")

    def layer_buffers(self, layer: int) -> dict[str, torch.Tensor]:
        if self._closed:
            raise RuntimeError("Expert cache is closed")
        if (
            not isinstance(layer, int)
            or isinstance(layer, bool)
            or not 0 <= layer < self.num_layers
        ):
            raise IndexError(f"Invalid expert cache layer index: {layer}")
        if self._mapping is None:
            handle = self._data_path.open("rb")
            self._mmap = mmap.mmap(handle.fileno(), self.nbytes, access=mmap.ACCESS_READ)
            handle.close()
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The given buffer is not writable",
                    category=UserWarning,
                )
                self._mapping = torch.frombuffer(
                    self._mmap, dtype=torch.uint8, count=self.nbytes
                )
        result = {}
        for name in EXPERT_BUFFER_NAMES:
            spec = self._layers[layer]["buffers"][name]
            dtype = _DTYPES[name][0]
            result[name] = (
                self._mapping.narrow(0, spec["offset"], spec["nbytes"])
                .view(dtype)
                .view(spec["shape"])
            )
        return result

    def assign_to_backbone(self, backbone: torch.nn.Module) -> int:
        """Validate every target first, then bind mapped views with no CPU copy."""
        targets = []
        for layer, module_name in enumerate(self.expert_module_names):
            try:
                module = backbone.get_submodule(module_name)
            except AttributeError as exc:
                raise ValueError(f"Missing expert module: {module_name}") from exc
            if not getattr(module, "fp8", False):
                raise ValueError(f"{module_name} is not an FP8 expert module")
            for name in EXPERT_BUFFER_NAMES:
                tensor = module._buffers.get(name)
                if tensor is None or tensor.device.type not in {"meta", "cpu"}:
                    raise ValueError(f"{module_name}.{name} must be a meta/CPU buffer")
                spec = self._layers[layer]["buffers"][name]
                if list(tensor.shape) != spec["shape"] or tensor.dtype != _DTYPES[name][0]:
                    raise ValueError(f"{module_name}.{name} shape/dtype mismatch")
            targets.append(module)
        for layer, module in enumerate(targets):
            for name, tensor in self.layer_buffers(layer).items():
                module._buffers[name] = tensor
            module._drama_expert_storage = "mmap"
        return len(targets)

    def close(self) -> None:
        """Drop our view while allowing already-bound tensors to own the mapping."""
        self._mapping = None
        self._closed = True
        # torch.frombuffer holds a reference to mmap in its storage deleter.
        # Explicit mmap.close() would invalidate live backbone tensor views.
        self._mmap = None

    def __enter__(self) -> "ExpertCacheFile":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


inspect_expert_source = inspect_expert_cache_source

__all__ = [
    "CACHE_FORMAT",
    "CACHE_FORMAT_VERSION",
    "DATA_NAME",
    "MANIFEST_NAME",
    "ExpertCacheFile",
    "ExpertCacheSource",
    "build_expert_cache",
    "inspect_expert_cache_source",
    "inspect_expert_source",
]
