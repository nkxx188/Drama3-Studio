"""Strict safetensors reading through immutable file mappings.

The usual PyTorch safetensors backend uses copy-on-write storage, which can
charge an entire shard to Windows commit. This reader validates headers before
yielding any tensor, then exposes exact original bytes through ``ACCESS_READ``
mappings. Returned tensors must remain immutable; ``copy_`` into model weights
is safe. Tensor storage retains its mapping even after the iterator is closed.
"""

from __future__ import annotations

import json
import math
import mmap
import sys
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

_HEADER_LIMIT = 128 * 1024**2
_DTYPES: dict[str, tuple[torch.dtype, int]] = {
    "BF16": (torch.bfloat16, 2),
    "F32": (torch.float32, 4),
    "F16": (torch.float16, 2),
    "F8_E4M3": (torch.float8_e4m3fn, 1),
    "F8_E5M2": (torch.float8_e5m2, 1),
    "I64": (torch.int64, 8),
    "I32": (torch.int32, 4),
    "I8": (torch.int8, 1),
    "U8": (torch.uint8, 1),
    "BOOL": (torch.bool, 1),
}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _parse_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON in {label}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {label}")
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _prefixes(
    values: tuple[str, ...] | list[str] | None, label: str
) -> tuple[str, ...] | None:
    if values is None:
        return None
    if not isinstance(values, (tuple, list)) or not all(
        isinstance(value, str) for value in values
    ):
        raise TypeError(f"{label} must be a tuple/list of string prefixes")
    return tuple(values)


@dataclass(frozen=True)
class _TensorEntry:
    dtype: torch.dtype
    shape: tuple[int, ...]
    offset: int
    nbytes: int


@dataclass(frozen=True)
class _ShardHeader:
    path: Path
    file_size: int
    mtime_ns: int
    entries: dict[str, _TensorEntry]


def _read_header(path: Path) -> _ShardHeader:
    with path.open("rb") as handle:
        stat = path.stat()
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"Truncated safetensors length: {path}")
        header_size = int.from_bytes(raw_length, "little")
        if not 2 <= header_size <= min(_HEADER_LIMIT, stat.st_size - 8):
            raise ValueError(f"Invalid safetensors header size: {path}")
        raw_header = handle.read(header_size)
        if len(raw_header) != header_size or not raw_header.startswith(b"{"):
            raise ValueError(f"Invalid safetensors header bytes: {path}")
    header = _parse_json(raw_header, str(path))
    data_start = header_size + 8
    data_size = stat.st_size - data_start
    entries = {}
    intervals = []
    for name, value in header.items():
        if name == "__metadata__":
            if not isinstance(value, dict) or not all(
                isinstance(key, str) and isinstance(item, str)
                for key, item in value.items()
            ):
                raise ValueError(f"Invalid safetensors metadata: {path}")
            continue
        if not isinstance(value, dict):
            raise ValueError(f"Invalid tensor header for {name!r}")
        if set(value) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"Unexpected tensor header fields for {name!r}")
        dtype_name = value["dtype"]
        if not isinstance(dtype_name, str) or dtype_name not in _DTYPES:
            raise ValueError(f"Unsupported checkpoint dtype {dtype_name!r} for {name!r}")
        dtype, element_size = _DTYPES[dtype_name]
        shape = value["shape"]
        if not isinstance(shape, list):
            raise ValueError(f"Invalid tensor shape for {name!r}")
        shape = tuple(
            _nonnegative_integer(dimension, f"{name!r} shape dimension")
            for dimension in shape
        )
        offsets = value["data_offsets"]
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"Invalid tensor offsets for {name!r}")
        start, end = (
            _nonnegative_integer(offset, f"{name!r} data offset")
            for offset in offsets
        )
        nbytes = math.prod(shape) * element_size
        if end - start != nbytes or end > data_size:
            raise ValueError(f"Tensor byte range/shape mismatch for {name!r}")
        entries[name] = _TensorEntry(dtype, shape, data_start + start, nbytes)
        intervals.append((start, end, name))
    if not entries:
        if data_size:
            raise ValueError(f"Unreferenced safetensors payload: {path}")
        return _ShardHeader(path, stat.st_size, stat.st_mtime_ns, entries)
    previous_end = 0
    for start, end, name in sorted(intervals):
        if start != previous_end:
            raise ValueError(f"Overlapping or missing tensor payload bytes at {name!r}")
        previous_end = end
    if previous_end != data_size:
        raise ValueError(f"Unreferenced safetensors payload: {path}")
    return _ShardHeader(path, stat.st_size, stat.st_mtime_ns, entries)


def _checkpoint_headers(checkpoint: Path) -> list[_ShardHeader]:
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.exists():
        index = _parse_json(index_path.read_bytes(), str(index_path))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Checkpoint index has no weight_map")
        if not all(
            isinstance(name, str) and isinstance(shard, str)
            for name, shard in weight_map.items()
        ):
            raise ValueError("Checkpoint weight_map must contain string names and shards")
        names = sorted(set(weight_map.values()))
        paths = [(checkpoint / name).resolve() for name in names]
        if not all(path.is_relative_to(checkpoint) for path in paths):
            raise ValueError("Checkpoint shard path escapes its directory")
        if len(set(paths)) != len(paths):
            raise ValueError("Checkpoint index aliases the same shard file")
        headers = [_read_header(path) for path in paths]
        found = set()
        for shard_name, header in zip(names, headers):
            for name in header.entries:
                if name in found:
                    raise ValueError(f"Duplicate tensor across checkpoint shards: {name!r}")
                if weight_map.get(name) != shard_name:
                    raise ValueError(f"Checkpoint index/header mismatch for {name!r}")
                found.add(name)
        if found != set(weight_map):
            missing = sorted(set(weight_map) - found)[:4]
            raise ValueError(f"Checkpoint index tensors missing from shards: {missing}")
        return headers
    return [_read_header(checkpoint / "model.safetensors")]


def iter_readonly_checkpoint_tensors(
    checkpoint_dir: str | Path,
    *,
    skip_prefixes: tuple[str, ...] = (),
    include_prefixes: tuple[str, ...] | None = None,
) -> Iterator[tuple[str, Tensor]]:
    """Yield immutable CPU tensors in the official iterator's sorted key order.

    Every shard and index entry is validated before the first yield, including
    skipped tensors. No payload checksum is performed. Mapping does not allocate
    the shard in private RAM, although accessed OS file pages can become resident.
    """
    if sys.byteorder != "little":
        raise ValueError("Checkpoint reader requires a little-endian host")
    skip = _prefixes(skip_prefixes, "skip_prefixes")
    if skip is None:
        raise TypeError("skip_prefixes must contain string prefixes")
    include = _prefixes(include_prefixes, "include_prefixes")
    checkpoint = Path(checkpoint_dir).resolve()
    headers = _checkpoint_headers(checkpoint)
    yield from _iter_headers(headers, skip, include)


def iter_readonly_file_tensors(
    path: str | Path, *, skip_prefixes: tuple[str, ...] = (),
    include_prefixes: tuple[str, ...] | None = None,
):
    """Read an explicitly chosen standalone safetensors file without copying."""
    if sys.byteorder != "little":
        raise ValueError("Checkpoint reader requires a little-endian host")
    skip = _prefixes(skip_prefixes, "skip_prefixes")
    if skip is None:
        raise TypeError("skip_prefixes must contain string prefixes")
    include = _prefixes(include_prefixes, "include_prefixes")
    yield from _iter_headers([_read_header(Path(path).resolve())], skip, include)


def _iter_headers(headers, skip, include):
    for header in headers:
        keys = [
            name
            for name in sorted(header.entries)
            if not name.startswith(skip)
            and (include is None or name.startswith(include))
        ]
        if not keys:
            continue
        with header.path.open("rb") as handle:
            stat = header.path.stat()
            if stat.st_size != header.file_size or stat.st_mtime_ns != header.mtime_ns:
                raise ValueError(f"Checkpoint shard changed during loading: {header.path}")
            mapped = mmap.mmap(handle.fileno(), header.file_size, access=mmap.ACCESS_READ)
        try:
            for name in keys:
                entry = header.entries[name]
                if entry.nbytes == 0:
                    tensor = torch.empty(entry.shape, dtype=entry.dtype, device="cpu")
                else:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="The given buffer is not writable",
                            category=UserWarning,
                        )
                        byte_tensor = torch.frombuffer(
                            mapped,
                            dtype=torch.uint8,
                            count=entry.nbytes,
                            offset=entry.offset,
                        )
                    tensor = byte_tensor.view(entry.dtype).view(entry.shape)
                yield name, tensor
        finally:
            # Existing tensors own references to the mmap through frombuffer.
            # Explicit close() would invalidate views retained by a consumer.
            mapped = None


__all__ = ["iter_readonly_checkpoint_tensors"]
