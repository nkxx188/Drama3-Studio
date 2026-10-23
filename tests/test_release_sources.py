"""Required download support assets must survive a clean source checkout."""

import hashlib
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_catalog():
    source = ROOT / "src/drama_low_vram/model_catalog.py"
    spec = importlib.util.spec_from_file_location("release_model_catalog", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_catalog_loads_without_importing_the_inference_stack():
    catalog = load_catalog()
    models = catalog.download_catalog()
    assert {"fp8", "int4", "int8", "w4a8", "w6a8"} <= {row["id"] for row in models}


def test_all_bundled_model_components_exist_and_match_the_catalog():
    catalog = load_catalog()
    bundled = {}
    for choice in catalog.download_catalog():
        for row in catalog.get_model(choice["id"])["files"]:
            if row.get("bundled"):
                identity = (row["size"], row["sha256"])
                assert bundled.setdefault(row["name"], identity) == identity
    assert bundled, "The supported quantized models need their task-control row"
    for name, (size, digest) in bundled.items():
        source = ROOT / "src/drama_low_vram/data" / name
        assert source.is_file(), f"Required release asset is missing: {source}"
        assert source.stat().st_size == size
        with source.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == digest
