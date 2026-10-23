"""Pinned model discovery and cancellable install/verification jobs.

This module uses no inference framework. Directory checks never load model
weights; full SHA256 verification is an explicit background operation.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
import uuid

from .model_download import Downloader, DownloadCancelled, manifest, sha256_file
from .model_catalog import get_model


def model_directory(value):
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("Choose a model directory")
    return Path(value).expanduser().resolve()


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _quantization_method(directory):
    try:
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        return config.get("quantization_config", {}).get("quant_method")
    except (OSError, ValueError, AttributeError, TypeError):
        # The regular inspector reports missing/corrupt pinned metadata.
        return None


def inspect_checkpoint(directory, *, files=None):
    directory = model_directory(directory)
    shared_path = directory / "shared-model.json"
    if files is None and shared_path.is_file():
        return inspect_shared_checkpoint(directory)
    if files is None and _quantization_method(directory) == "drama3-int4":
        return inspect_quantized_checkpoint(directory)
    files = manifest() if files is None else files
    missing, invalid, present, partial = [], [], 0, 0
    for entry in files:
        target = directory / entry["name"]
        if not target.is_file():
            missing.append(entry["name"])
            partial_file = directory / (entry["name"] + ".partial")
            if partial_file.is_file():
                partial += min(entry["size"], partial_file.stat().st_size)
        elif target.stat().st_size != entry["size"]:
            invalid.append({"name": entry["name"], "reason": "File size differs from the pinned model"})
        elif entry["size"] <= 32 * 1024**2 and sha256_file(target) != entry["sha256"]:
            invalid.append({"name": entry["name"], "reason": "Metadata SHA256 differs from the pinned model"})
        else:
            present += entry["size"]
    ancestor = directory
    while not ancestor.exists() and ancestor.parent != ancestor:
        ancestor = ancestor.parent
    free = shutil.disk_usage(ancestor).free if ancestor.exists() else None
    total = sum(entry["size"] for entry in files)
    return {
        "directory": str(directory), "ready": not missing and not invalid,
        "inspection_scope": "Pinned file sizes and small-file SHA256; full weight hashes require Verify",
        "total_bytes": total, "present_bytes": present,
        "partial_logical_bytes": partial,
        "missing": missing, "invalid": invalid,
        "disk_free_bytes": free,
        "estimated_additional_bytes": max(0, total - present - partial),
    }


def inspect_shared_checkpoint(directory):
    """Metadata-only discovery; no torch import or second model payload."""
    directory = model_directory(directory)
    missing, invalid, present = [], [], 0
    try:
        entry = json.loads((directory / "shared-model.json").read_text(encoding="utf-8"))
        if (entry.get("format") != "drama3-packed-shared-v1"
                or not isinstance(entry.get("model"), str)
                or isinstance(entry.get("size"), bool) or not isinstance(entry.get("size"), int)
                or entry["size"] < 1
                or not isinstance(entry.get("sha256"), str)
                or not re.fullmatch(r"[a-f0-9]{64}", entry["sha256"])
                or entry.get("quantization") not in (None, "int8_convrot", "w6a8", "w4a8")):
            raise ValueError("Invalid shared model descriptor")
        model = (directory / entry["model"]).resolve()
        if not model.is_file():
            missing.append(str(model))
        elif model.stat().st_size != entry["size"]:
            invalid.append({"name": str(model), "reason": "Shared model size changed"})
        else:
            present += entry["size"]
        for name in ("config.json", "tokenizer.json", "codec.pth", "control-head.safetensors"):
            path = directory / name
            if not path.is_file():
                missing.append(name)
            else:
                present += path.stat().st_size
        total = entry["size"] + sum((directory / name).stat().st_size for name in (
            "config.json", "tokenizer.json", "codec.pth", "control-head.safetensors")
                                   if (directory / name).is_file())
    except (OSError, ValueError, TypeError, KeyError) as exc:
        invalid.append({"name": "shared-model.json", "reason": str(exc)})
        entry, total = {}, present
    ancestor = directory
    while not ancestor.exists() and ancestor.parent != ancestor:
        ancestor = ancestor.parent
    return {
        "directory": str(directory), "ready": not missing and not invalid,
        "inspection_scope": "Shared safetensors path/size and local metadata; Verify checks full SHA256",
        "storage_format": entry.get("precision", "packed"), "shared_model": entry.get("model"),
        "total_bytes": total, "present_bytes": present, "partial_logical_bytes": 0,
        "missing": missing, "invalid": invalid,
        "disk_free_bytes": shutil.disk_usage(ancestor).free,
        "estimated_additional_bytes": max(0, total - present),
    }


def inspect_quantized_checkpoint(directory):
    directory = model_directory(directory)
    invalid, missing, present = [], [], 0
    try:
        receipt = json.loads((directory / "download-verification.json").read_text(encoding="utf-8"))
        if receipt.get("model") != "fishaudio/drama-3-int4":
            raise ValueError("INT4 checkpoint has no matching verified download receipt")
        files = _quantized_files(receipt)
        for row in files:
            path = (directory / row["name"]).resolve()
            if not path.is_relative_to(directory):
                raise ValueError("INT4 file escapes selected directory")
            if not path.is_file():
                missing.append(row["name"])
            elif path.stat().st_size != row["size"]:
                invalid.append({"name": row["name"], "reason": "File size changed"})
            elif row["size"] <= 32 * 1024**2 and sha256_file(path) != row["sha256"]:
                invalid.append({"name": row["name"], "reason": "Metadata SHA256 changed"})
            else:
                present += path.stat().st_size
        for name in ("codec.pth", "tokenizer.json"):
            if not (directory / name).is_file():
                missing.append(name)
        total = sum(row["size"] for row in files)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        total = present
        invalid.append({"name": "download-verification.json", "reason": str(exc)})
    return {
        "directory": str(directory), "ready": not missing and not invalid,
        "storage_format": "W4A16-GPTQ",
        "inspection_scope": "Pinned INT4 download receipt and current sizes; full Verify checks SHA256",
        "total_bytes": total, "present_bytes": present,
        "partial_logical_bytes": 0, "missing": missing, "invalid": invalid,
        "disk_free_bytes": shutil.disk_usage(directory).free,
        "estimated_additional_bytes": max(0, total - present),
    }


def _quantized_files(receipt):
    files = list(receipt["files"])
    # These payloads are hardlinked from the verified original installation.
    # They still participate in discovery and full verification.
    names = {row["name"] for row in files}
    files.extend(row for row in manifest()
                 if row["name"] in ("codec.pth", "tokenizer.json") and row["name"] not in names)
    return files


class ModelInstallation:
    def __init__(self, data_root, *, files=None, checkpoint_root=None):
        self.data_root = Path(data_root)
        self.files = manifest() if files is None else files
        self.checkpoint_root = Path(checkpoint_root or Path(__file__).resolve().parents[2] / "checkpoints")
        self.state_file = self.data_root / "model-task.json"
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="drama-model")
        self._cancel = threading.Event()
        self._task = None
        self._future = None
        self._last_save = 0.0
        if self.state_file.is_file():
            try:
                value = json.loads(self.state_file.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    if value.get("status") in ("queued", "running", "cancelling"):
                        value["status"] = "interrupted"
                    self._task = value
            except (ValueError, OSError):
                pass

    def _save(self):
        self.data_root.mkdir(parents=True, exist_ok=True)
        temporary = self.state_file.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self._task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.state_file)
        self._last_save = time.monotonic()

    def snapshot(self):
        with self._lock:
            return json.loads(json.dumps(self._task)) if self._task else None

    def start(self, operation, directory=None, *, source="auto", model_id=None,
              workers=16, chunk_mib=64):
        if operation not in ("download", "verify"):
            raise ValueError("Unknown model operation")
        if source not in ("auto", "modelscope", "huggingface"):
            raise ValueError("Unknown model source")
        if (isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 32
                or isinstance(chunk_mib, bool) or not isinstance(chunk_mib, int)
                or not 1 <= chunk_mib <= 512):
            raise ValueError("Invalid model download parallelism")
        selected = get_model(model_id) if model_id is not None else None
        if selected:
            if source != "auto" and source not in selected["sources"]:
                raise ValueError("That download source is not available for this model")
            if source == "auto" and len(selected["sources"]) < 2:
                raise ValueError("Automatic source testing requires both download channels")
            root = self.checkpoint_root.expanduser().resolve()
            directory = model_directory(root / selected["target"])
            if directory.parent != root:
                raise ValueError("The model destination must stay inside the application checkpoint folder")
        else:
            directory = model_directory(directory)
        with self._lock:
            if self._task and self._task.get("status") in ("queued", "running", "cancelling"):
                raise RuntimeError("A model operation is already running")
            self._cancel = threading.Event()
            self._task = {
                "id": uuid.uuid4().hex, "operation": operation, "directory": str(directory),
                "model_id": selected["id"] if selected else None,
                "model_name": selected["label"] if selected else None,
                "model_format": selected["format"] if selected else None,
                "source_requested": source, "status": "queued", "created_at": time.time(),
                "progress": {
                    "stage": "preparing", "done_bytes": 0,
                    "total_bytes": sum(row["size"] for row in
                                       (selected["files"] if selected else self.files)),
                },
                "error": None,
            }
            self._save()
            self._future = self._executor.submit(
                self._run, operation, directory, source, workers, chunk_mib, self._cancel, selected)
            return self.snapshot()

    def cancel(self):
        with self._lock:
            if self._task and self._task.get("status") in ("queued", "running", "cancelling"):
                self._task["status"] = "cancelling"
                self._cancel.set()
                if self._future is not None and self._future.cancel():
                    self._task["status"] = "cancelled"
                    self._task["finished_at"] = time.time()
                self._save()
            return self.snapshot()

    def _progress(self, event):
        with self._lock:
            self._task["progress"] = event
            self._task["updated_at"] = time.time()
            if (time.monotonic() - self._last_save >= 0.5
                    or event["stage"] in ("source_selected", "complete")):
                self._save()

    def _run(self, operation, directory, source, workers, chunk_mib, cancel, selected=None):
        try:
            with self._lock:
                if cancel.is_set():
                    raise DownloadCancelled("Model operation cancelled before starting")
                self._task["status"] = "running"
                self._save()
            if selected:
                self._run_catalog_model(
                    operation, directory, source, workers, chunk_mib, cancel, selected)
                return
            shared = (directory / "shared-model.json").is_file()
            quantized = _quantization_method(directory) == "drama3-int4"
            inspected = inspect_checkpoint(directory) if shared or quantized else inspect_checkpoint(directory, files=self.files)
            if inspected["invalid"]:
                raise ValueError(
                    "Existing files differ from the pinned model. Choose a separate folder; existing files are preserved.")
            if operation == "verify" and not inspected["ready"]:
                raise ValueError("The model is incomplete; download missing files before full verification")
            if shared:
                if operation != "verify":
                    raise ValueError("This directory reuses an existing shared model; choose an official directory to download")
                entry = json.loads((directory / "shared-model.json").read_text(encoding="utf-8"))
                downloader = Downloader(
                    directory, [{"name": Path(entry["model"]).name,
                                 "size": entry["size"], "sha256": entry["sha256"]}],
                    workers, chunk_mib * 1024**2,
                    progress=self._progress, cancel_event=cancel)
                if downloader.verify_hash((directory / entry["model"]).resolve()) != entry["sha256"]:
                    raise ValueError("Shared model SHA256 mismatch")
                downloader.done_bytes = entry["size"]
                downloader.emit("complete")
                with self._lock:
                    self._task.update(status="complete", inspection=inspect_checkpoint(directory),
                                      finished_at=time.time())
                    self._save()
                return
            if quantized:
                if operation != "verify":
                    raise ValueError("Use the quantized model downloader for this INT4 repository")
                files = _quantized_files(json.loads((directory / "download-verification.json").read_text(
                    encoding="utf-8")))
                downloader = Downloader(directory, files, workers, chunk_mib * 1024**2,
                                        progress=self._progress, cancel_event=cancel)
                for row in files:
                    if downloader.verify_hash(directory / row["name"]) != row["sha256"]:
                        raise ValueError(f"INT4 model SHA256 mismatch: {row['name']}")
                    downloader.done_bytes += row["size"]
                downloader.emit("complete")
                with self._lock:
                    self._task.update(status="complete", inspection=inspect_checkpoint(directory),
                                      finished_at=time.time())
                    self._save()
                return
            if (operation == "download" and inspected["disk_free_bytes"] is not None
                    and inspected["estimated_additional_bytes"] > inspected["disk_free_bytes"]):
                raise OSError("The selected drive does not have enough free space for the remaining model files")
            downloader = Downloader(
                directory, self.files, workers, chunk_mib * 1024**2, source=source,
                progress=self._progress, cancel_event=cancel)
            if operation == "download":
                downloader.run()
            else:
                for entry in self.files:
                    if downloader.verify_hash(directory / entry["name"]) != entry["sha256"]:
                        raise ValueError(f"Model SHA256 mismatch: {entry['name']}")
                    downloader.done_bytes += entry["size"]
                downloader.emit("complete")
            with self._lock:
                self._task["status"] = "complete"
                self._task["inspection"] = inspect_checkpoint(directory, files=self.files)
                self._task["finished_at"] = time.time()
                self._save()
        except DownloadCancelled as exc:
            with self._lock:
                self._task.update(status="cancelled", error=str(exc), finished_at=time.time())
                self._save()
        except Exception as exc:
            with self._lock:
                self._task.update(
                    status="error", error=f"{type(exc).__name__}: {exc}", finished_at=time.time())
                self._save()

    def _run_catalog_model(self, operation, directory, source, workers, chunk_mib, cancel, model):
        files = model["files"]
        downloaded_files = [row for row in files if not row.get("bundled")]
        inspected = inspect_checkpoint(directory, files=files)
        if inspected["invalid"]:
            raise ValueError(
                "Files in the destination differ from this pinned model. They were preserved; "
                "choose another model location or resolve the mismatched files.")
        if operation == "verify" and not inspected["ready"]:
            raise ValueError("The model is incomplete; finish its download before verification")
        if (operation == "download" and inspected["disk_free_bytes"] is not None
                and inspected["estimated_additional_bytes"] > inspected["disk_free_bytes"]):
            raise OSError(
                f"Not enough free space. The remaining files need about "
                f"{inspected['estimated_additional_bytes'] / 1024**3:.2f} GiB.")
        downloader = Downloader(
            directory, downloaded_files, workers, chunk_mib * 1024**2, source=source,
            progress=self._progress, cancel_event=cancel, model=model["repo"],
            revision=model["revision"], hf_revision=model["hf_revision"])
        if operation == "download":
            downloader.run()
        else:
            for row in downloaded_files:
                if downloader.verify_hash(directory / row["name"]) != row["sha256"]:
                    raise ValueError(f"Model SHA256 mismatch: {row['name']}")
                downloader.done_bytes += row["size"]
                downloader.emit("verifying", file=row["name"])

        for row in files:
            if not row.get("bundled"):
                continue
            source_path = Path(__file__).with_name("data") / row["name"]
            if (not source_path.is_file() or source_path.stat().st_size != row["size"]
                    or sha256_file(source_path) != row["sha256"]):
                raise ValueError(f"Bundled model component failed its pinned SHA256: {row['name']}")
            target = directory / row["name"]
            if not target.exists():
                temporary = target.with_suffix(target.suffix + ".tmp")
                shutil.copyfile(source_path, temporary)
                if sha256_file(temporary) != row["sha256"]:
                    temporary.unlink(missing_ok=True)
                    raise ValueError(f"Copied model component failed SHA256: {row['name']}")
                os.replace(temporary, target)
            if sha256_file(target) != row["sha256"]:
                raise ValueError(f"Model SHA256 mismatch: {row['name']}")
            downloader.done_bytes += row["size"]

        if model["kind"] == "shared":
            weight = next(row for row in files if row["name"].endswith(".safetensors"))
            descriptor = {
                "format": "drama3-packed-shared-v1",
                "model": weight["name"],
                "size": weight["size"],
                "sha256": weight["sha256"],
                "quantization": model["quantization"],
                "precision": model["precision"],
                "source": model["repo"],
                "revision": model["revision"],
            }
            _write_json(directory / "shared-model.json", descriptor)

        receipt = {
            "model": model["repo"],
            "source": downloader.selected_source,
            "source_requested": source,
            "revision": model["revision"],
            "equivalent_huggingface_weight_revision": model["hf_revision"],
            "total_bytes": sum(row["size"] for row in files),
            "files": [{**row, "verified": True} for row in files],
        }
        _write_json(directory / "download-verification.json", receipt)
        downloader.emit("complete")
        inspection = (inspect_checkpoint(directory) if model["kind"] == "shared"
                      else inspect_checkpoint(directory, files=files))
        if not inspection.get("ready"):
            raise ValueError("Downloaded files passed their hashes but the model layout is incomplete")
        with self._lock:
            self._task.update(
                status="complete", inspection=inspection,
                source=downloader.selected_source, finished_at=time.time())
            self._save()

    def close(self):
        self.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)
