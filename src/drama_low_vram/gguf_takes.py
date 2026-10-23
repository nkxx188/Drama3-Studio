"""Data-only Drama 3 codec takes shared by native workflows and the WebUI."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from .protocols import PreviousTake, validate_codes, validate_content


# All supported Drama 3 exports retain the official DAC code vocabulary.
CODEC_CONFIG = SimpleNamespace(audio_decoder=SimpleNamespace(num_codebooks=10, vocab_size=4096))
MAX_TAKE_FRAMES = 65536


def take_descriptor(take):
    validate_content(take.text, "previous take")
    codes = validate_codes(take.codes, CODEC_CONFIG, "previous take")
    if codes.shape[1] > MAX_TAKE_FRAMES:
        raise ValueError("Saved take exceeds the supported codec frame count")
    result = {"text": take.text, "vq_codes": codes.tolist()}
    if take.lookahead is not None:
        validate_content(take.lookahead, "previous lookahead", empty=True)
        result["lookahead"] = take.lookahead
    return result


def _parse_take(value):
    if not isinstance(value, dict):
        raise ValueError("Saved take must be a JSON object")
    text, rows, lookahead = value.get("text"), value.get("vq_codes"), value.get("lookahead")
    validate_content(text, "saved take transcript")
    if (not isinstance(rows, list) or len(rows) != 10 or not isinstance(rows[0], list)
            or not 1 <= len(rows[0]) <= MAX_TAKE_FRAMES):
        raise ValueError("Saved take requires a nonempty [10, frames] codec grid")
    length = len(rows[0])
    for row in rows:
        if not isinstance(row, list) or len(row) != length or any(
            isinstance(code, bool) or not isinstance(code, int) or not 0 <= code < 4096
            for code in row
        ):
            raise ValueError("Saved codec codes must be equal-length integer rows in 0..4095")
    if lookahead is not None:
        validate_content(lookahead, "saved lookahead", empty=True)
    return PreviousTake(text, torch.tensor(rows, dtype=torch.int64), lookahead)


def load_native_take(path):
    """Validate legacy native JSON and new records without loading model weights."""
    path = Path(path)
    if path.stat().st_size > 128 * 1024**2:
        raise ValueError("Take sidecar is too large")
    value = json.loads(path.read_text(encoding="utf-8"))
    take = _parse_take(value)
    metadata = value.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Saved take metadata must be an object")
    if "spoken_text" in metadata:
        validate_content(metadata["spoken_text"], "saved speech text")
    chunks = value.get("chunks", [])
    if not isinstance(chunks, list) or len(chunks) > 4096:
        raise ValueError("Invalid saved take chunks")
    parsed = [_parse_take(chunk) for chunk in chunks]
    if parsed:
        joined = torch.cat([chunk.codes for chunk in parsed], dim=1)
        if not torch.equal(joined, take.codes):
            raise ValueError("Saved chunk codes do not match the complete take")
    return take, metadata, parsed


def load_conditioning_take(path):
    """Both project formats carry the same DAC codes; no re-encoding is needed."""
    if str(path).lower().endswith(".take.npz"):
        from .workflows import load_take
        take, metadata, _ = load_take(path, CODEC_CONFIG)
        records = metadata.get("chunks")
        if records is None and metadata.get("mode") not in ("infill", "scene"):
            report = Path(path).with_name(Path(path).name.removesuffix(".take.npz") + ".json")
            if report.is_file() and report.stat().st_size <= 8 * 1024**2:
                records = json.loads(report.read_text(encoding="utf-8")).get("chunks")
        if records is None:
            return take, metadata, []
        if not isinstance(records, list) or not 1 <= len(records) <= 4096:
            raise ValueError("Invalid saved take chunks")
        chunks, position = [], 0
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("Invalid saved take chunk")
            count, text, lookahead = record.get("frames"), record.get("text"), record.get("lookahead")
            if (isinstance(count, bool) or not isinstance(count, int) or count < 1
                    or position + count > take.codes.shape[1]):
                raise ValueError("Saved chunk codes do not match the complete take")
            validate_content(text, "saved chunk transcript")
            if lookahead is not None:
                validate_content(lookahead, "saved lookahead", empty=True)
            chunks.append(PreviousTake(text, take.codes[:, position:position + count], lookahead))
            position += count
        if position != take.codes.shape[1]:
            raise ValueError("Saved chunk codes do not match the complete take")
        return take, metadata, chunks
    if not str(path).lower().endswith(".take.json"):
        raise ValueError("Choose a Drama 3 .take.json or .take.npz record")
    return load_native_take(path)


def save_native_take(output, take, *, metadata=None, chunks=()):
    value = take_descriptor(take)
    value["schema"] = "drama3_take_v1"
    value["metadata"] = dict(metadata or {})
    if chunks:
        value["chunks"] = [take_descriptor(chunk) for chunk in chunks]
    target = Path(output).with_suffix(".take.json")
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return target
