"""Small GGUF directory reader shared by inference and conversion tools.

The upstream Python reader expands every byte-array element into Python
objects; audio.cpp embeds tokenizer resources in those arrays. This module
keeps metadata serialized and only decodes the few directory fields we need.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import struct

FORMATS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
           6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
# Payload geometry from the bundled ggml-common.h. These are formats the
# native TensorSource understands; admission does not claim quality/speed
# validation for every external package using them.
BLOCKS = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20),
    6: (32, 22), 7: (32, 24), 8: (32, 34),
    10: (256, 84), 11: (256, 110), 12: (256, 144),
    13: (256, 176), 14: (256, 210), 30: (1, 2),
}
INTEREST = {
    "general.alignment", "general.name", "general.architecture", "general.file_type",
    "audiocpp.tensor_names", "audiocpp.tensor_ranks", "audiocpp.tensor_shapes",
}


def read_number(file, format):
    size = struct.calcsize("<" + format)
    raw = file.read(size)
    if len(raw) != size:
        raise ValueError("Truncated GGUF")
    return struct.unpack("<" + format, raw)[0]


def read_string(file):
    length = read_number(file, "Q")
    raw = file.read(length)
    if len(raw) != length:
        raise ValueError("Truncated GGUF string")
    return raw.decode("utf-8")


def read_value(file, kind, *, capture):
    if kind in FORMATS:
        if capture:
            return read_number(file, FORMATS[kind])
        file.seek(struct.calcsize("<" + FORMATS[kind]), 1)
    elif kind == 8:
        if capture:
            return read_string(file)
        file.seek(read_number(file, "Q"), 1)
    elif kind == 9:
        item_kind, count = read_number(file, "I"), read_number(file, "Q")
        if not capture and item_kind in FORMATS:
            file.seek(count * struct.calcsize("<" + FORMATS[item_kind]), 1)
        else:
            values = [read_value(file, item_kind, capture=capture) for _ in range(count)]
            return values if capture else None
    else:
        raise ValueError(f"Unknown GGUF value type {kind}")


def pack_string(value):
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def scalar_metadata(name, value, kind=8):
    data = pack_string(value) if kind == 8 else struct.pack("<" + FORMATS[kind], value)
    return pack_string(name) + struct.pack("<I", kind) + data


@dataclass
class Tensor:
    physical: str
    logical: str
    shape: tuple[int, ...]
    kind: int
    offset: int
    nbytes: int


class Archive:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.metadata = []
        self.values = {}
        size = self.path.stat().st_size
        with self.path.open("rb") as file:
            if file.read(4) != b"GGUF" or read_number(file, "I") != 3:
                raise ValueError("Expected GGUF v3")
            count, keys = read_number(file, "Q"), read_number(file, "Q")
            for _ in range(keys):
                start = file.tell()
                name, kind = read_string(file), read_number(file, "I")
                value = read_value(file, kind, capture=name in INTEREST)
                end = file.tell()
                if end > size:
                    raise ValueError("GGUF metadata exceeds file")
                file.seek(start)
                self.metadata.append((name, file.read(end - start)))
                if name in INTEREST:
                    self.values[name] = value
            names = self.values.get("audiocpp.tensor_names")
            if names is None or len(names) != count:
                raise ValueError("Expected complete audio.cpp logical tensor names")
            self.tensors = []
            for index in range(count):
                physical, rank = read_string(file), read_number(file, "I")
                dims = tuple(read_number(file, "Q") for _ in range(rank))
                kind, offset = read_number(file, "I"), read_number(file, "Q")
                if kind not in BLOCKS:
                    raise ValueError(f"Unsupported input GGUF tensor type {kind}")
                block, byte_count = BLOCKS[kind]
                elements = math.prod(dims)
                if elements % block:
                    raise ValueError("GGUF block geometry mismatch")
                self.tensors.append(Tensor(
                    physical, names[index], tuple(reversed(dims)), kind,
                    offset, elements // block * byte_count))
            self.alignment = self.values.get("general.alignment", 32)
            self.data_offset = (file.tell() + self.alignment - 1) // self.alignment * self.alignment
            for tensor in self.tensors:
                tensor.offset += self.data_offset
                if tensor.offset + tensor.nbytes > size:
                    raise ValueError("Truncated GGUF tensor data")
        self.by_name = {tensor.logical: tensor for tensor in self.tensors}
        if len(self.by_name) != len(self.tensors):
            raise ValueError("Duplicate logical tensor name")


def write_header(file, metadata, tensors, *, alignment=32):
    file.write(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata)))
    for _, raw in metadata:
        file.write(raw)
    offset = 0
    for tensor in tensors:
        file.write(pack_string(tensor.physical))
        file.write(struct.pack("<I", len(tensor.shape)))
        for dimension in reversed(tensor.shape):
            file.write(struct.pack("<Q", dimension))
        file.write(struct.pack("<IQ", tensor.kind, offset))
        tensor.offset = offset
        offset = (offset + tensor.nbytes + alignment - 1) // alignment * alignment
    data_offset = (file.tell() + alignment - 1) // alignment * alignment
    file.write(bytes(data_offset - file.tell()))
    for tensor in tensors:
        tensor.offset += data_offset
    return data_offset + offset
