"""GGUF offsets and sidecar arrays survive bounded streaming conversion."""

from pathlib import Path
import struct
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from gguf_archive import Archive, Tensor, pack_string, scalar_metadata, write_header


def array_metadata(name, kind, values):
    raw = pack_string(name) + struct.pack("<IIQ", 9, kind, len(values))
    if kind == 8:
        return name, raw + b"".join(pack_string(value) for value in values)
    return name, raw + bytes(values)


def test_sidecar_byte_arrays_and_tensor_offsets_are_preserved_without_expanding_payloads(tmp_path):
    metadata = [
        ("general.architecture", scalar_metadata("general.architecture", "audiocpp")),
        array_metadata("audiocpp.embedded_files.data", 0, b"\x01" * 1024**2),
        array_metadata("audiocpp.tensor_names", 8, ["weights/a", "weights/b"]),
    ]
    tensors = [
        Tensor("_audiocpp.0", "weights/a", (2, 32), 8, 0, 68),
        Tensor("_audiocpp.1", "weights/b", (4,), 30, 0, 8),
    ]
    source = tmp_path / "model.gguf"
    with source.open("wb") as file:
        final = write_header(file, metadata, tensors)
        file.seek(tensors[0].offset)
        file.write(b"\x12" * 68)
        file.seek(tensors[1].offset)
        file.write(b"\x13" * 8)
        file.write(bytes(final - file.tell()))
    archive = Archive(source)
    assert [t.shape for t in archive.tensors] == [(2, 32), (4,)]
    assert [t.kind for t in archive.tensors] == [8, 30]
    assert [t.offset for t in archive.tensors] == [t.offset for t in tensors]
    assert [t.nbytes for t in archive.tensors] == [68, 8]
    assert archive.metadata == metadata
    assert "audiocpp.embedded_files.data" not in archive.values
    assert all(t.offset % 32 == 0 for t in archive.tensors)
    with source.open("rb") as file:
        file.seek(tensors[1].offset)
        assert file.read(8) == b"\x13" * 8
    # Real truncation must not be admitted as a successfully converted model.
    with source.open("r+b") as file:
        file.truncate(tensors[1].offset + 7)
    with pytest.raises(ValueError, match="Truncated"):
        Archive(source)


@pytest.mark.parametrize("kind,block,bytes_per_block", [
    (2, 32, 18), (3, 32, 20), (6, 32, 22), (7, 32, 24),
    (10, 256, 84), (11, 256, 110), (13, 256, 176), (14, 256, 210),
])
def test_common_native_quantizations_have_correct_payload_sizes(tmp_path, kind, block, bytes_per_block):
    """Compatibility admission only; not a GPU speed/quality test."""
    name = "model_weights/expert"
    tensor = Tensor("physical", name, (3, block), kind, 0, bytes_per_block * 3)
    path = tmp_path / "other-quantization.gguf"
    with path.open("wb") as file:
        end = write_header(file, [array_metadata("audiocpp.tensor_names", 8, [name])], [tensor])
        file.write(bytes(end - file.tell()))
    archive = Archive(path)
    assert archive.tensors[0].nbytes == bytes_per_block * 3
    assert archive.tensors[0].kind == kind
    assert archive.tensors[0].shape == (3, block)
