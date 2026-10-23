"""Run the author's FP8/codec conversion with a bounded streaming destination.

The published safetensors header determines every destination offset. Only the
converter's output dictionary is replaced; tensor transformations are unchanged.
An exact published SHA256 match is required before the output is made visible.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import time
import urllib.request
from pathlib import Path


REVISION = "09f14f03cf33e5b6d1bf468af1020c099b8cce1a"
ARTIFACTS = {
    "model": (
        "drama3_fp8experts_bf16.safetensors",
        "5d053a200cf0f77faa13c087c78ef200c3b53f288451a6b9e7a8fa421c03a67b",
        38578417928,
    ),
    "codec": (
        "drama3_codec.safetensors",
        "4fecf102907a48414230da9a74d72d07c3a92368eab53fab655797837bddf5ae",
        1565647484,
    ),
}


class StreamingDestination:
    def __init__(self, header: bytes, path: Path, expected_size: int):
        import torch

        self.torch = torch
        self.entries = json.loads(header[8:])
        self.entries.pop("__metadata__", None)
        self.header_size = len(header)
        self.path = path
        self.handle = path.open("w+b", buffering=0)
        self.handle.write(header)
        self.handle.truncate(expected_size)
        self.written: set[str] = set()
        self.total = 0

    def __setitem__(self, name, tensor):
        if name in self.written:
            raise ValueError(f"Duplicate tensor: {name}")
        entry = self.entries[name]
        dtype_names = {
            self.torch.bfloat16: "BF16",
            self.torch.float32: "F32",
            self.torch.float8_e4m3fn: "F8_E4M3",
            self.torch.int64: "I64",
            self.torch.int32: "I32",
        }
        if list(tensor.shape) != entry["shape"] or dtype_names[tensor.dtype] != entry["dtype"]:
            raise ValueError(f"Published tensor metadata differs: {name}")
        tensor = tensor.detach().cpu().contiguous()
        data = memoryview(tensor.reshape(-1).view(self.torch.uint8).numpy()).cast("B")
        start, end = entry["data_offsets"]
        if len(data) != end - start:
            raise ValueError(f"Tensor byte length differs: {name}")
        self.handle.seek(self.header_size + start)
        self.handle.write(data)
        self.written.add(name)
        self.total += len(data)

    def finish(self):
        missing = set(self.entries) - self.written
        if missing:
            raise ValueError(f"Unwritten published tensors: {sorted(missing)[:8]}")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()


def published_header(name: str, cache: Path) -> bytes:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (name + ".header")
    if path.is_file():
        return path.read_bytes()
    url = f"https://huggingface.co/PedroMarinhoDev/Drama-3-ComfyUI/resolve/{REVISION}/{name}"
    request = urllib.request.Request(
        url,
        headers={"Range": "bytes=0-1048575", "User-Agent": "Drama3-local-reproduction/1.0"},
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        prefix = response.read(8)
        size = int.from_bytes(prefix, "little")
        if not 0 < size < 1024 * 1024 - 8:
            raise ValueError(f"Unexpected safetensors header size: {size}")
        header = prefix + response.read(size)
    if len(header) != 8 + size:
        raise ValueError("Incomplete safetensors header")
    json.loads(header[8:])
    path.write_bytes(header)
    return header


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=ARTIFACTS)
    parser.add_argument("--node", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    name, digest, expected_size = ARTIFACTS[args.kind]
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    header = published_header(name, args.receipt.parent / "headers")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    destination = StreamingDestination(header, partial, expected_size)
    sys.path.insert(0, str(args.node.resolve()))
    converter_path = args.node / "tools/convert.py"
    spec = importlib.util.spec_from_file_location("drama3_author_converter", converter_path)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)

    def finish(tensors, output, metadata):
        if tensors is destination:
            destination.finish()
        else:
            for key, value in tensors.items():
                destination[key] = value
            destination.finish()

    converter.write = finish
    options = argparse.Namespace(
        src=str(args.source.resolve()), out=str(args.output.resolve()),
        format="fp8", dense="bf16", device="cpu",
    )
    if args.kind == "model":
        tree = ast.parse(inspect.getsource(converter.convert_model))
        replacements = 0
        for statement in tree.body[0].body:
            if (isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                    and statement.targets[0].id == "out"
                    and isinstance(statement.value, ast.Dict)
                    and not statement.value.keys):
                statement.value = ast.Name(id="_streamed_destination", ctx=ast.Load())
                replacements += 1
        if replacements != 1:
            raise ValueError("Author converter's output initialization changed")
        converter._streamed_destination = destination
        exec(compile(ast.fix_missing_locations(tree), str(converter_path), "exec"), converter.__dict__)
        converter.convert_model(options)
    else:
        converter.convert_codec(options)
    conversion_seconds = time.perf_counter() - started
    print("Conversion complete; checking the published SHA256...", flush=True)
    with partial.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if partial.stat().st_size != expected_size or actual != digest:
        raise ValueError(f"Published SHA256 mismatch; partial retained: {actual}")
    os.replace(partial, args.output)
    receipt = {
        "kind": args.kind,
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "publisher_revision": REVISION,
        "sha256": actual,
        "published_sha256_match": True,
        "bytes": expected_size,
        "tensor_count": len(destination.written),
        "converter_sha256": hashlib.sha256(converter_path.read_bytes()).hexdigest(),
        "conversion_seconds": conversion_seconds,
        "total_seconds": time.perf_counter() - started,
        "math": "Author's converter, FP8 experts and BF16 dense; output dictionary streamed.",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
