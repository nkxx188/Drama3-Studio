"""Validate readonly checkpoint views against the installed safetensors reader."""

import ctypes
import gc
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from drama_low_vram.checkpoint_reader import iter_readonly_checkpoint_tensors


def _write_raw(path: Path, header: dict | bytes, payload: bytes = b"") -> None:
    if not isinstance(header, bytes):
        header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(len(header).to_bytes(8, "little"))
        handle.write(header)
        handle.write(payload)


def _fixture(checkpoint: Path):
    tensors = {
        "audio_decoder.values": torch.arange(16, dtype=torch.float32).view(4, 4),
        "model.language_model.fp16": torch.tensor([0.5, -2.25], dtype=torch.float16),
        "model.language_model.bf16": torch.tensor([0.5, -2.25], dtype=torch.bfloat16),
        "model.language_model.fp8": torch.arange(256, dtype=torch.uint8).view(
            torch.float8_e4m3fn
        ),
        "model.language_model.fp8_e5": torch.arange(256, dtype=torch.uint8).view(
            torch.float8_e5m2
        ),
        "model.language_model.ints": torch.tensor([0, 123456789], dtype=torch.int64),
        "model.language_model.i32": torch.tensor([-44, 55], dtype=torch.int32),
        "lm_head.weight": torch.arange(128, dtype=torch.bfloat16).view(16, 8),
        "z.empty": torch.empty(0, 3, dtype=torch.float32),
        "z.scalar": torch.tensor(0.25),
        "z.bool": torch.tensor([True, False]),
        "z.bytes": torch.tensor([123, 234], dtype=torch.uint8),
    }
    weight_map = {}
    for shard in range(2):
        name = f"model-{shard + 1:05d}-of-00002.safetensors"
        contents = {
            key: tensor
            for index, (key, tensor) in enumerate(tensors.items())
            if index % 2 == shard
        }
        save_file(contents, str(checkpoint / name), metadata={"test": "reader"})
        weight_map.update({key: name for key in contents})
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 123}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    return tensors, weight_map


def _bytes(tensor):
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes()


def test_order_dtypes_shapes_and_exact_bytes_match_installed_safetensors(tmp_path):
    tensors, weight_map = _fixture(tmp_path)
    expected = []
    for shard in sorted(set(weight_map.values())):
        with safe_open(str(tmp_path / shard), framework="pt", device="cpu") as handle:
            expected.extend((key, handle.get_tensor(key)) for key in handle.keys())
    actual = list(iter_readonly_checkpoint_tensors(tmp_path))
    assert [key for key, _ in actual] == [key for key, _ in expected]
    for (key, tensor), (_, baseline) in zip(actual, expected):
        assert tensor.device.type == "cpu"
        assert tensor.dtype == baseline.dtype == tensors[key].dtype
        assert tensor.shape == baseline.shape == tensors[key].shape
        assert _bytes(tensor) == _bytes(baseline) == _bytes(tensors[key])


def test_include_skip_filters_and_lifetime_after_iterator_close(tmp_path):
    tensors, _ = _fixture(tmp_path)
    iterator = iter_readonly_checkpoint_tensors(
        tmp_path,
        include_prefixes=("model.language_model",),
        skip_prefixes=("model.language_model.fp8",),
    )
    key, tensor = next(iterator)
    expected = _bytes(tensors[key])
    iterator.close()
    gc.collect()
    assert _bytes(tensor) == expected
    assert not list(
        iter_readonly_checkpoint_tensors(tmp_path, include_prefixes=())
    )


def test_single_shard_without_index_is_supported(tmp_path):
    tensors = {"value": torch.arange(17, dtype=torch.float32)}
    save_file(tensors, str(tmp_path / "model.safetensors"))
    actual = dict(iter_readonly_checkpoint_tensors(tmp_path))
    assert _bytes(actual["value"]) == _bytes(tensors["value"])


@pytest.mark.parametrize(
    "header,payload,error",
    [
        (
            {"tensor": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}},
            bytes(4),
            "range/shape",
        ),
        (
            {"tensor": {"dtype": "F32", "shape": [1], "data_offsets": [0, 8]}},
            bytes(4),
            "range/shape",
        ),
        (
            {"tensor": {"dtype": "INT4", "shape": [1], "data_offsets": [0, 1]}},
            bytes(1),
            "Unsupported",
        ),
        (
            {"tensor": {"dtype": "U8", "shape": [-1], "data_offsets": [0, 1]}},
            bytes(1),
            "nonnegative",
        ),
        (
            {"tensor": {"dtype": "U8", "shape": [True], "data_offsets": [0, 1]}},
            bytes(1),
            "nonnegative",
        ),
        (
            {"tensor": {"dtype": "U8", "shape": [1], "data_offsets": [False, 1]}},
            bytes(1),
            "nonnegative",
        ),
        (
            {"tensor": {"dtype": "U8", "shape": [1], "data_offsets": [1, 2]}},
            bytes(2),
            "missing tensor payload",
        ),
        (
            {"tensor": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}},
            bytes(2),
            "Unreferenced",
        ),
        (
            {
                "a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]},
                "b": {"dtype": "U8", "shape": [1], "data_offsets": [1, 2]},
            },
            bytes(2),
            "Overlapping",
        ),
        (
            {"__metadata__": {"test": 3}},
            b"",
            "metadata",
        ),
        (
            {"tensor": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1], "unknown": 9}},
            bytes(1),
            "fields",
        ),
    ],
)
def test_malformed_headers_rejected_even_when_all_tensors_skipped(
    tmp_path, header, payload, error
):
    _write_raw(tmp_path / "model.safetensors", header, payload)
    with pytest.raises(ValueError, match=error):
        list(iter_readonly_checkpoint_tensors(tmp_path, skip_prefixes=("",)))


@pytest.mark.parametrize(
    "header",
    [
        b'{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}',
        b'{"x":{"dtype":"U8","dtype":"U8","shape":[1],"data_offsets":[0,1]}}',
        b'{"__metadata__":{"x":"a","x":"b"}}',
    ],
)
def test_duplicate_json_names_are_rejected(tmp_path, header):
    _write_raw(tmp_path / "model.safetensors", header, bytes(1))
    with pytest.raises(ValueError, match="Duplicate"):
        list(iter_readonly_checkpoint_tensors(tmp_path))


@pytest.mark.parametrize("payload", [b"", bytes(7), (2**63).to_bytes(8, "little"), b"\x02" + bytes(7) + b"xx"])
def test_bad_length_or_truncated_header_rejected(tmp_path, payload):
    (tmp_path / "model.safetensors").write_bytes(payload)
    with pytest.raises(ValueError, match="Truncated|Invalid"):
        list(iter_readonly_checkpoint_tensors(tmp_path))


@pytest.mark.parametrize("change", ["wrong_shard", "missing", "extra", "escape"])
def test_index_mismatch_or_path_escape_is_rejected(tmp_path, change):
    _, weight_map = _fixture(tmp_path)
    if change == "wrong_shard":
        key = next(iter(weight_map))
        weight_map[key] = (
            "model-00002-of-00002.safetensors"
            if weight_map[key] == "model-00001-of-00002.safetensors"
            else "model-00001-of-00002.safetensors"
        )
    elif change == "missing":
        weight_map["missing.tensor"] = next(iter(weight_map.values()))
    elif change == "extra":
        weight_map.pop(next(iter(weight_map)))
    else:
        weight_map["missing.tensor"] = "../out.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="mismatch|missing|escapes"):
        list(iter_readonly_checkpoint_tensors(tmp_path))


def test_unaligned_dtype_offset_preserves_source_bytes(tmp_path):
    _write_raw(
        tmp_path / "model.safetensors",
        {
            "a.byte": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
            "b.float": {"dtype": "F32", "shape": [1], "data_offsets": [1, 5]},
        },
        b"\x01" + _bytes(torch.tensor([3.25])),
    )
    actual = dict(iter_readonly_checkpoint_tensors(tmp_path))
    assert actual["b.float"].item() == 3.25


@pytest.mark.skipif(sys.platform != "win32", reason="Windows commit accounting")
def test_reading_large_shard_does_not_commit_copy_on_write_file_size(tmp_path):
    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(Counters),
        ctypes.c_ulong,
    ]

    def commit():
        result = Counters()
        result.cb = ctypes.sizeof(result)
        if not psapi.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(result), result.cb
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return result.PrivateUsage

    size = 64 * 1024**2
    header = json.dumps(
        {"large": {"dtype": "U8", "shape": [size], "data_offsets": [0, size]}}
    ).encode()
    path = tmp_path / "model.safetensors"
    with path.open("wb") as handle:
        handle.write(len(header).to_bytes(8, "little"))
        handle.write(header)
        handle.truncate(8 + len(header) + size)
    gc.collect()
    before = commit()
    tensors = dict(iter_readonly_checkpoint_tensors(tmp_path))
    delta = commit() - before
    assert delta < 8 * 1024**2, delta
    assert tensors["large"].numel() == size
    assert tensors["large"][0].item() == 0
