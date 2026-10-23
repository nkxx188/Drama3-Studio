"""Pinned, application-supported Drama 3 downloads."""

from __future__ import annotations

import json
import re
from pathlib import Path


CATALOG_PATH = Path(__file__).with_name("data") / "model-download-catalog.json"
_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,95}")
_REPOSITORY = re.compile(r"[\w.-]+/[\w.-]+")
_KINDS = {"official", "int4", "shared", "gguf"}
_SOURCES = {"huggingface", "modelscope"}


def _load():
    document = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or not isinstance(document.get("models"), list):
        raise RuntimeError("Invalid bundled model catalog")
    ids, targets = set(), set()
    for model in document["models"]:
        if (not isinstance(model, dict) or not _ID.fullmatch(model.get("id", ""))
                or model["id"] in ids or model.get("kind") not in _KINDS
                or not isinstance(model.get("target"), str)
                or not re.fullmatch(r"[A-Za-z0-9._-]{1,96}", model["target"])
                or model["target"] in targets
                or not _REPOSITORY.fullmatch(model.get("repo", ""))
                or not isinstance(model.get("revision"), str)
                or not isinstance(model.get("hf_revision"), str)
                or not isinstance(model.get("files"), list)
                or not isinstance(model.get("sources"), list)
                or not set(model["sources"]) <= _SOURCES
                or not model["sources"]):
            raise RuntimeError("Invalid bundled model catalog entry")
        ids.add(model["id"])
        targets.add(model["target"])
        names = set()
        for row in model["files"]:
            if (not isinstance(row, dict) or not isinstance(row.get("name"), str)
                    or not row["name"] or "\\" in row["name"] or ":" in row["name"]
                    or row["name"].startswith("/")
                    or ".." in Path(row["name"]).parts
                    or row["name"] in names
                    or isinstance(row.get("size"), bool)
                    or not isinstance(row.get("size"), int) or row["size"] < 1
                    or not re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", ""))):
                raise RuntimeError(f"Invalid pinned file in model catalog: {model['id']}")
            if row.get("repository") and not _REPOSITORY.fullmatch(row["repository"]):
                raise RuntimeError(f"Invalid file repository in model catalog: {model['id']}")
            names.add(row["name"])
    return document["models"]


_MODELS = _load()
_BY_ID = {model["id"]: model for model in _MODELS}


def get_model(model_id):
    if not isinstance(model_id, str) or model_id not in _BY_ID:
        raise ValueError("Choose a model from the supported download list")
    return _BY_ID[model_id]


def download_catalog():
    return [{
        "id": model["id"],
        "label": model["label"],
        "format": model["format"],
        "total_bytes": sum(row["size"] for row in model["files"]),
        "sources": list(model["sources"]),
    } for model in _MODELS]
