"""Local Windows interface for the validated Drama 3 engine.

The server is deliberately dependency-light: Python's standard library serves
the UI and serializes GPU jobs, while the existing engine remains the only
inference implementation. It binds to localhost by default.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
import hashlib
import ipaddress
from email import policy
from email.parser import BytesParser
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import shutil
import socket
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, unquote, urlparse
import uuid
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
UI_ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor/fish-speech")]

for key, folder in (
    ("TRITON_CACHE_DIR", "triton"),
    ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
    ("TORCH_EXTENSIONS_DIR", "torch_extensions"),
    ("HF_HOME", "huggingface"),
    ("TEMP", "tmp"),
    ("TMP", "tmp"),
):
    target = ROOT / ".cache" / folder
    target.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(key, str(target))

from drama_low_vram.autotune import inspect_hardware  # noqa: E402
from drama_low_vram.audio_stream import AudioStreamSession  # noqa: E402
from drama_low_vram.engine import DramaEngine, EngineOptions  # noqa: E402
from drama_low_vram.model_installation import (  # noqa: E402
    ModelInstallation, inspect_checkpoint, model_directory,
)
from drama_low_vram.model_catalog import download_catalog  # noqa: E402
from drama_low_vram.protocols import (  # noqa: E402
    Character, InfillSpan, PreviousTake, SceneLine, VoiceReference,
    design_character, validate_content, validate_scene_inputs,
)
from drama_low_vram.scenes import SceneSession  # noqa: E402
from drama_low_vram.text_chunks import _content_width, split_text  # noqa: E402
from drama_low_vram.text_stream import TextStreamSession  # noqa: E402
from drama_low_vram.workflows import DramaWorkflows, load_take  # noqa: E402
from drama_low_vram.gguf_resources import is_gguf, resolve_gguf, inspect_gguf  # noqa: E402
from drama_low_vram.gguf_runtime import GGUFEngine, GenerationCancelled, MAX_AUTOMATIC_CONTEXT  # noqa: E402
from drama_low_vram.gguf_workflows import GGUFWorkflows  # noqa: E402
from drama_low_vram.gguf_takes import load_conditioning_take, save_native_take  # noqa: E402
from drama_low_vram.runtime_files import load_native_runtime  # noqa: E402
from drama_low_vram.diagnostics import select_python_device  # noqa: E402
from drama_low_vram.reference_asr import ReferenceTranscriber  # noqa: E402
from drama_low_vram.studio_store import StudioStore, clone, record_id  # noqa: E402
from drama_low_vram.job_timing import JobClock  # noqa: E402
from drama_low_vram.studio_logging import console_session, detail, status  # noqa: E402


def _safe_name(value: str, fallback: str = "item") -> str:
    value = "".join(character for character in str(value) if character.isalnum() or character in "-_. ")
    value = value.strip(" .")
    return value[:80] or fallback


def _automatic_generation_limit(payload):
    """Bound one generated chunk, never the duration of an entire long job."""
    target = payload.get("max_chunk_chars", 160)
    lines = payload.get("lines", [])
    if payload.get("mode") == "scene" and payload.get("scene_strategy") == "joint":
        widths = [_content_width(line["text"]) for line in lines]
        width = max(max(widths, default=0), min(sum(widths), math.ceil(target * 1.5)))
    else:
        texts = [line["text"] for line in lines] if payload.get("mode") == "scene" else [
            payload.get("text", "")]
        whole = (payload.get("single_chunk") or payload.get("duration_frames")
                 or payload.get("mode") in ("infill", "design", "audio_stream"))
        pieces = texts if whole else [
            chunk for text in texts for chunk in split_text(text, max_chars=target)]
        width = max((_content_width(piece) for piece in pieces), default=0)
    # Eight audio frames per spoken character plus a tail allowance is a
    # conservative budget, not a duration hint. EOS still ends speech naturally.
    return min(2048, max(768, width * 8 + 128, payload.get("duration_frames") or 0))


def _inside(path: Path, parent: Path) -> Path:
    path = path.resolve()
    parent = parent.resolve()
    if path != parent and parent not in path.parents:
        raise ValueError("Path is outside the local project directory")
    return path


def inspect_model(path):
    return inspect_gguf(path) if is_gguf(path) else inspect_checkpoint(path)


def _portable_checkpoint_path(path):
    """Rebind a moved package's stale absolute model paths to its local pool."""
    path = model_directory(path)
    if path.exists():
        return path
    parts = path.parts
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].casefold() != "checkpoints":
            continue
        previous_root = Path(*parts[:index])
        if not ((previous_root / "ui/server.py").is_file()
                and (previous_root / "src/drama_low_vram").is_dir()):
            continue
        relative = Path(*parts[index + 1:]) if index + 1 < len(parts) else Path()
        candidate = (ROOT / "checkpoints" / relative).resolve()
        if candidate.exists():
            return candidate
    return path


def timing_result(result):
    """Small legacy timing fields, without graph plans or other large diagnostics."""
    result = result or {}
    value = {key: result[key] for key in (
        "total_seconds", "client_request_seconds", "request_seconds", "load_seconds",
        "session_reused", "backend", "cold_model_load_seconds",
    ) if key in result}
    chunks = result.get("chunks")
    if isinstance(chunks, list) and chunks:
        value["chunks"] = [{"native_request_seconds": row.get("native_request_seconds")}
                           for row in chunks if isinstance(row, dict)]
    return value


class AppState:
    def __init__(self, *, data_root=None):
        self.data_root = Path(data_root) if data_root is not None else ROOT / "ui_data"
        self.voices_root = self.data_root / "voices"
        self.uploads_root = self.data_root / "uploads"
        self.jobs_root = self.data_root / "jobs"
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.voices_root.mkdir(exist_ok=True)
        self.uploads_root.mkdir(exist_ok=True)
        self.jobs_root.mkdir(exist_ok=True)
        self.settings_file = self.data_root / "settings.json"
        self.checkpoint = ROOT / "checkpoints/drama-3"
        self.model_paths = []
        self.model_roots = []
        if self.settings_file.is_file():
            try:
                settings = json.loads(self.settings_file.read_text(encoding="utf-8"))
                saved = Path(settings["checkpoint"])
                self.checkpoint = _portable_checkpoint_path(
                    saved if saved.is_absolute() else ROOT / saved)
                paths = settings.get("model_paths", [])
                for value in paths[:32] if isinstance(paths, list) else []:
                    if not isinstance(value, str) or not value.strip():
                        continue
                    path = Path(value)
                    path = _portable_checkpoint_path(path if path.is_absolute() else ROOT / path)
                    if path not in self.model_paths:
                        self.model_paths.append(path)
                roots = settings.get("model_roots", [])
                for value in roots[:32] if isinstance(roots, list) else []:
                    if not isinstance(value, str) or not value.strip():
                        continue
                    path = Path(value)
                    path = _portable_checkpoint_path(path if path.is_absolute() else ROOT / path)
                    if path.is_dir() and path not in self.model_roots:
                        self.model_roots.append(path)
            except (OSError, ValueError, TypeError, KeyError):
                pass
        self.model_installation = ModelInstallation(
            self.data_root, checkpoint_root=ROOT / "checkpoints")
        self._engine = None
        self._engine_lock = threading.RLock()
        self._releasing_resources = False
        self._jobs: dict[str, dict] = {}
        self._streams: dict[str, TextStreamSession] = {}
        self._jobs_lock = threading.RLock()
        self._job_persist_times: dict[str, float] = {}
        self._job_clocks: dict[str, JobClock] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="drama-ui")
        self._reference_prewarm_lock = threading.RLock()
        self._reference_prewarm_tasks: dict[str, dict] = {}
        self._active_reference_prewarm: str | None = None
        self.asr = ReferenceTranscriber(ROOT, self.data_root, self._executor)
        self.studio = StudioStore(self.data_root)
        self._reference_assets_lock = threading.RLock()
        self.started_at = time.time()
        self._load_persisted_jobs()

    def _job_file(self, job_id: str) -> Path:
        return _inside(self.jobs_root / job_id / "job.json", self.jobs_root)

    def _persist_job_locked(self, job_id: str):
        job = self._jobs.get(job_id)
        if job is None:
            return
        target = self._job_file(job_id)
        temporary = target.with_suffix(".tmp")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(job, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, target)
        self._job_persist_times[job_id] = time.time()

    def _load_persisted_jobs(self):
        for path in self.jobs_root.glob("*/job.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(value, dict) and value.get("id"):
                    was_active = value.get("status") in ("queued", "running")
                    value["status"] = (
                        "interrupted" if was_active
                        else value.get("status", "unknown"))
                    if was_active and isinstance(value.get("timing"), dict):
                        value["timing"]["phase"] = "finished"
                    if was_active and value.get("mode") in ("scene", "stream", "audio_stream"):
                        result = value.get("result") or {}
                        value["result"] = result
                        if not result.get("segments"):
                            result["segments"] = self._scene_rows(value)
                        result["status"] = "interrupted"
                        if (value["mode"] == "scene"
                                and value.get("recipe", {}).get("scene_strategy", "per_line") == "per_line"):
                            result.setdefault("strategy", "per_line_tts")
                    self._jobs[str(value["id"])] = value
            except (OSError, ValueError, TypeError):
                continue

    def hardware(self):
        try:
            active = self._engine
            device = (f"cuda:{active.selected_device}" if getattr(active, "is_gguf", False)
                      else str(active.device) if active is not None and hasattr(active, "device")
                      else select_python_device())
            info = inspect_hardware(device)
            return {
                "platform": info.platform,
                "device": device,
                "gpu": info.gpu_name,
                "vram_gib": round(info.vram_total_bytes / 1024**3, 2),
                "ram_gib": round(info.ram_total_bytes / 1024**3, 2),
                "ram_available_gib": round(info.ram_available_bytes / 1024**3, 2),
                "commit_available_gib": round(
                    (info.commit_available_bytes or 0) / 1024**3, 2),
                "cpu_logical_cores": info.cpu_logical_cores,
            }
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    def capabilities(self):
        if is_gguf(self.checkpoint):
            try:
                protocol = load_native_runtime(ROOT).get("protocol", 1)
            except (OSError, ValueError, RuntimeError):
                protocol = 0
            advanced = protocol >= 3
            try:
                quality = inspect_model(self.checkpoint)["storage_format"]
            except (OSError, ValueError, RuntimeError):
                # Runtime features remain inspectable while model setup is incomplete.
                quality = "GGUF"
            return {
                "ordinary_tts": True, "voice_cloning": True, "multi_reference": True,
                "cross_language": True, "director_revision": advanced, "point_and_scoped_controls": True,
                "infill": advanced, "audio_import": protocol >= 5,
                "streaming_text": advanced, "streaming_audio": advanced,
                "scene_scheduler": advanced, "native_scene_separator": False,
                "joint_dialogue": protocol >= 4, "voice_design": protocol >= 4,
                "direction": True, "language_hint": True, "duration_hint": protocol >= 4,
                "automatic_compute_policy": True, "persistent_session": True,
                "cpu_expert_compute": False, "quality": quality,
                "native_protocol": protocol,
                "native_scene_separator_note": (
                    "Joint dialogue produces one take; per-line scheduling saves individual takes. "
                    "Native audio separators remain unavailable."),
                "note": (
                    "Drama 3 tasks use real saved codec codes. Joint dialogue and per-line scenes are separate workflows."
                    if advanced else "Update the native runtime to enable advanced Drama 3 workflows."
                ),
            }
        return {
            "ordinary_tts": True,
            "voice_cloning": True,
            "multi_reference": True,
            "cross_language": True,
            "director_revision": True,
            "point_and_scoped_controls": True,
            "direction": True, "language_hint": True, "duration_hint": True,
            "infill": True,
            "audio_import": True,
            "streaming_text": True,
            "streaming_audio": True,
            "scene_scheduler": True,
            "joint_dialogue": True, "voice_design": True,
            "native_scene_separator": False,
            "native_scene_separator_note": (
                "Published weights did not emit reliable separators in local tests; "
                "the interface uses validated per-line scheduling."
            ),
            "automatic_compute_policy": True,
            "quality": inspect_checkpoint(self.checkpoint).get(
                "storage_format", "official FP8 routed experts / BF16 dense and audio tower"),
        }

    def expert_cache_directory(self):
        if self.checkpoint == (ROOT / "checkpoints/drama-3").resolve():
            return ROOT / ".cache/drama3-fp8-experts"
        identity = hashlib.sha256(str(self.checkpoint).encode("utf-8")).hexdigest()[:16]
        return ROOT / ".cache" / f"drama3-fp8-experts-{identity}"

    def model_status(self, directory=None):
        inspected = inspect_model(directory if directory is not None else self.checkpoint)
        selected = (inspected if directory is None or model_directory(directory) == self.checkpoint
                    else inspect_model(self.checkpoint))
        shared = inspected.get("shared_model")
        with self._jobs_lock:
            busy = any(job["status"] in ("queued", "running") for job in self._jobs.values())
        with self._reference_prewarm_lock:
            prewarm = self._reference_prewarm_tasks.get(self._active_reference_prewarm)
        return {
            "selected_directory": str(self.checkpoint),
            "selected_name": self.checkpoint.parent.name if is_gguf(self.checkpoint) else self.checkpoint.name,
            "selected_ready": selected["ready"],
            "busy": busy,
            "engine_loaded": self._engine is not None and (
                not getattr(self._engine, "is_gguf", False) or self._engine.loaded),
            "releasing_resources": self._releasing_resources,
            "reference_prewarm": self._public_reference_prewarm(prewarm) if prewarm else None,
            "expert_cache_directory": str(self.expert_cache_directory()),
            "inspection": inspected,
            "selected_format": selected.get("storage_format", "FP8 / BF16"),
            "uses_checkpoint_directly": inspected.get("runtime") == "gguf" or bool(shared) or inspected.get("storage_format") == "W4A16-GPTQ",
            "available_models": self.available_models(),
            "model_locations": self.model_locations(),
            "runtime": "gguf" if is_gguf(self.checkpoint) else "python",
            "task": self.model_installation.snapshot(),
        }

    def model_search_roots(self):
        """Return only model pools chosen by the user or provided by this app."""
        roots, seen = [], set()

        def add(folder):
            try:
                folder = folder.resolve()
            except (OSError, RuntimeError):
                return
            identity = os.path.normcase(str(folder))
            if folder.parent != folder and identity not in seen:
                roots.append(folder); seen.add(identity)

        add(ROOT / "checkpoints")
        for folder in self.model_roots:
            add(folder)
        return roots

    def model_locations(self):
        locations, seen = [], set()
        default = os.path.normcase(str((ROOT / "checkpoints").resolve()))
        custom = {os.path.normcase(str(path.resolve())) for path in self.model_roots}
        for folder in (ROOT / "checkpoints", *self.model_roots):
            try:
                folder = folder.resolve()
            except (OSError, RuntimeError):
                continue
            identity = os.path.normcase(str(folder))
            if identity in seen:
                continue
            seen.add(identity)
            locations.append({"path": str(folder), "removable": identity in custom and identity != default})
        return locations

    def _save_model_settings(self, *, checkpoint=None, model_paths=None, model_roots=None):
        checkpoint = checkpoint or self.checkpoint
        model_paths = self.model_paths if model_paths is None else model_paths
        model_roots = self.model_roots if model_roots is None else model_roots

        def saved_path(path):
            path = Path(path).resolve()
            return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)

        temporary = self.settings_file.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({
                "checkpoint": saved_path(checkpoint),
                "model_paths": [saved_path(path) for path in model_paths],
                "model_roots": [saved_path(path) for path in model_roots],
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        os.replace(temporary, self.settings_file)

    def add_model_root(self, directory):
        folder = model_directory(directory)
        if not folder.is_dir():
            raise ValueError("Choose an existing model folder")
        identity = os.path.normcase(str(folder))
        if all(os.path.normcase(str(path)) != identity for path in self.model_roots):
            if len(self.model_roots) >= 32:
                raise ValueError("The model location limit has been reached")
            roots = [*self.model_roots, folder]
            with self._engine_lock:
                self._save_model_settings(model_roots=roots)
                self.model_roots = roots
        return self.model_status()

    def remove_model_root(self, directory):
        folder = model_directory(directory)
        identity = os.path.normcase(str(folder))
        roots = [path for path in self.model_roots if os.path.normcase(str(path)) != identity]
        if len(roots) == len(self.model_roots):
            raise ValueError("That model location cannot be removed")
        model_paths = [path for path in self.model_paths
                       if path == self.checkpoint or not path.is_relative_to(folder)]
        with self._engine_lock:
            self._save_model_settings(model_roots=roots, model_paths=model_paths)
            self.model_roots = roots
            self.model_paths = model_paths
        return self.model_status()

    def available_models(self):
        """Discover loadable local models without reading weight payloads."""
        rows, seen = [], set()

        def children(folder, limit=None):
            try:
                paths = []
                for path in folder.iterdir():
                    paths.append(path)
                    if limit is not None and len(paths) >= limit:
                        break
                return sorted(paths, key=lambda path: path.name.casefold())
            except OSError:
                return []

        def contains_drama_name(path, root):
            try:
                parts = path.relative_to(root).parts
            except ValueError:
                parts = (path.name,)
            return any("drama" in part.casefold() for part in parts)

        def model_file_label(path, root):
            parent = path.parent
            while parent != root and parent.is_relative_to(root):
                if "drama" in parent.name.casefold():
                    return parent.name
                parent = parent.parent
            return path.stem

        def add(path, *, name=None, location=None):
            try:
                path = path.resolve()
            except (OSError, RuntimeError):
                return
            identity = os.path.normcase(str(path))
            if identity in seen:
                return
            seen.add(identity)
            native = path.suffix.lower() == ".gguf"
            display_name = name or path.name
            rows.append({
                "label": display_name, "name": display_name,
                "directory": str(path),
                "location": str((location or (path.parent if native else path)).resolve()),
                "runtime": "gguf" if native else "python",
            })

        def is_drama_directory(path, *, allow_legacy):
            try:
                config = json.loads((path / "config.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return False
            if not isinstance(config, dict):
                return False
            model_type = config.get("model_type")
            return model_type == "fish_omni" or (allow_legacy and model_type is None)

        custom_roots = {os.path.normcase(str(path.resolve())) for path in self.model_roots}
        for folder in self.model_search_roots():
            custom = os.path.normcase(str(folder.resolve())) in custom_roots
            for path in children(folder):
                if custom and path.is_symlink():
                    continue
                if path.is_file() and path.suffix.lower() == ".gguf":
                    if not custom or contains_drama_name(path, folder):
                        label = model_file_label(path, folder) if custom else (
                            path.parent.name if path.parent != folder else path.stem)
                        add(path, name=label, location=path.parent)
                elif custom and path.is_file() and path.suffix.lower() == ".safetensors":
                    if contains_drama_name(path, folder):
                        label = model_file_label(path, folder)
                        add(path.parent, name=label, location=path.parent)
                elif path.is_dir():
                    # Auxiliary ASR weights are not a selectable Drama 3 checkpoint.
                    if is_drama_directory(
                            path, allow_legacy=not custom or contains_drama_name(path, folder)):
                        add(path)
                    # Some GGUF distributions include config.json beside their files.
                    for model in children(path):
                        if model.is_file() and model.suffix.lower() == ".gguf":
                            if not custom or contains_drama_name(model, folder):
                                label = model_file_label(model, folder) if custom else path.name
                                add(model, name=label, location=path)

        # User-selected libraries such as ComfyUI/models may nest model folders
        # several levels deep. Traverse only those explicit roots, with a hard
        # depth and entry bound and without following symlinks.
        for root in self.model_roots:
            if is_drama_directory(root, allow_legacy=True):
                add(root)
                continue
            stack, visited = [(root, 0)], 0
            while stack and visited < 20_000:
                folder, depth = stack.pop()
                entries = children(folder, 20_000 - visited)
                visited += len(entries)
                for path in entries:
                    if path.is_symlink():
                        continue
                    if path.is_file():
                        if path.suffix.lower() == ".gguf" and (
                                contains_drama_name(path, root)):
                            label = model_file_label(path, root)
                            add(path, name=label, location=path.parent)
                        elif path.suffix.lower() == ".safetensors" and (
                                contains_drama_name(path, root)):
                            label = model_file_label(path, root)
                            add(path.parent, name=label, location=path.parent)
                        continue
                    if not path.is_dir():
                        continue
                    if is_drama_directory(
                            path, allow_legacy=contains_drama_name(path, root)):
                        add(path)
                        continue
                    if depth < 5 and path.name.casefold() not in {
                            ".cache", ".git", "__pycache__", "node_modules"}:
                        stack.append((path, depth + 1))
                if visited >= 20_000:
                    break
        for path in self.model_paths:
            if path.exists():
                if path.suffix.lower() == ".gguf":
                    add(path, name=path.parent.name if "drama" in path.parent.name.casefold() else path.stem,
                        location=path.parent)
                else:
                    add(path)
        # A selected external or incomplete installation must remain visible.
        add(self.checkpoint)
        return rows

    def select_checkpoint(self, directory):
        checkpoint = model_directory(directory)
        if is_gguf(checkpoint):
            checkpoint = resolve_gguf(checkpoint)
        inspected = inspect_model(checkpoint)
        if not inspected["ready"]:
            raise ValueError("Model files are incomplete or do not match; check the selected model")
        changed = False
        with self._engine_lock:
            if self._releasing_resources:
                raise RuntimeError("Model resources are being released. Please wait.")
            with self._jobs_lock:
                if any(job["status"] in ("queued", "running") for job in self._jobs.values()):
                    raise RuntimeError("Finish or cancel generation jobs before switching models")
            if checkpoint != self.checkpoint:
                self.cancel_reference_prewarm()
                model_paths = list(dict.fromkeys([checkpoint, self.checkpoint, *self.model_paths]))[:32]
                temporary = self.settings_file.with_suffix(".tmp")
                temporary.write_text(json.dumps({
                    "checkpoint": str(checkpoint.relative_to(ROOT)) if checkpoint.is_relative_to(ROOT) else str(checkpoint),
                    "model_paths": [str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
                                    for path in model_paths],
                    "model_roots": [str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
                                    for path in self.model_roots],
                }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                # Publish the selection only after any old runtime is closed.
                # A close failure must not leave a new path on a live engine.
                if self._engine is not None:
                    self._engine.close()
                    self._engine = None
                os.replace(temporary, self.settings_file)
                self.checkpoint = checkpoint
                self.model_paths = model_paths
                with self._reference_prewarm_lock:
                    self._active_reference_prewarm = None
                changed = True
        if changed:
            status("Model selected: %s", checkpoint.name)
        return self.model_status()

    def release_resources(self):
        with self._engine_lock:
            if self._releasing_resources:
                raise RuntimeError("Model resources are being released. Please wait.")
            with self._jobs_lock:
                if any(job["status"] in ("queued", "running") for job in self._jobs.values()):
                    raise RuntimeError("Finish or cancel generation jobs before releasing resources.")
            self._releasing_resources = True
            self.cancel_reference_prewarm()
            try:
                future = self._executor.submit(self._release_engine_resources)
            except BaseException:
                self._releasing_resources = False
                raise
        try:
            future.result()
        finally:
            with self._engine_lock:
                self._releasing_resources = False
        return self.model_status()

    def _release_engine_resources(self):
        # Cancelled reference preparation must finish before closing its engine.
        with self._engine_lock:
            if self._engine is not None:
                self._engine.close()
                self._engine = None
            with self._reference_prewarm_lock:
                self._active_reference_prewarm = None
                self._reference_prewarm_tasks.clear()
        status("Model resources released.")

    def _event(self, job_id: str, event: dict):
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            now = time.time()
            clock = self._job_clocks.get(job_id)
            if clock:
                phase, action = event.get("timing_phase"), event.get("timing_action")
                if phase and action:
                    (clock.begin if action == "begin" else clock.end)(phase)
                elif event.get("stage") == "prepare":
                    clock.switch("initialization")
                elif event.get("stage") == "waiting_for_text":
                    clock.switch("waiting")
                elif event.get("stage") in ("references", "prefill", "generate", "decode", "audio"):
                    clock.switch("generation")
                job["timing"] = clock.snapshot()
            if event.get("stage") == "timing":
                detail("Job %s: %s", job_id, event)
                job["updated_at"] = now
                self._persist_job_locked(job_id)
                return
            if event.get("stage") in ("segment", "audio") and (event.get("path") or event.get("output")):
                result = job.setdefault("result", {"status": "running"})
                rows = result.setdefault("segments", [])
                path = event.get("path") or event.get("output")
                if not any((row.get("path") or row.get("output")) == path for row in rows):
                    rows.append({key: clone(value) for key, value in event.items() if key not in ("stage", "time")})
            job["events"].append({"time": now, **event})
            job["events"] = job["events"][-250:]
            job["updated_at"] = now
            immediate = event.get("stage") in {
                "error", "complete", "cancel", "cancelled", "segment", "audio",
            }
            if immediate or now - self._job_persist_times.get(job_id, 0.0) >= 0.75:
                detail("Job %s: %s", job_id, event)
                self._persist_job_locked(job_id)

    def create_job(self, payload: dict) -> str:
        # A model switch cannot slip between recipe validation and queueing.
        with self._engine_lock:
            if self._releasing_resources:
                raise RuntimeError("Model resources are being released. Please wait.")
            return self._create_job(payload)

    def _create_job(self, payload: dict) -> str:
        if not isinstance(payload, dict):
            raise ValueError("Job payload must be an object")
        payload = clone(payload)
        payload = self._creative_request(payload)
        automatic_length = payload.pop("automatic_length", False)
        if not isinstance(automatic_length, bool):
            raise ValueError("automatic_length must be a boolean")
        if automatic_length:
            payload["max_new_tokens"] = 2048
        mode = payload.get("mode", "tts")
        if mode not in ("tts", "revision", "edit", "infill", "scene", "design", "stream", "audio_stream"):
            raise ValueError("Unsupported interface mode")
        payload.setdefault("mode", mode)
        self._validate_sampling(payload)
        for key, default in {
            "seed": 123, "max_new_tokens": 768, "max_chunk_chars": 160,
            "temperature": 1.0 if is_gguf(self.checkpoint) else 0.7,
            "top_p": 0.9 if is_gguf(self.checkpoint) else 0.7, "top_k": 0,
        }.items():
            payload.setdefault(key, default)
        if mode == "design":
            validate_content(payload.get("text"), "design audition text")
            design_character(payload.get("character_name", "Voice"), payload.get("voice_description", ""),
                             payload.get("scene", ""))
            if payload.get("voice_ids") or payload.get("voice_bindings") or payload.get("uploaded_references"):
                raise ValueError("Voice design does not use voice references")
        if mode == "scene":
            self._scene_script(payload)
            if payload.get("scene_strategy", "per_line") not in ("joint", "per_line"):
                raise ValueError("Scene strategy must be joint or per_line")
            order = payload.get("scene_order")
            if order is not None:
                if (not isinstance(order, list) or not 1 <= len(order) <= 4096
                        or any(not isinstance(item, str) for item in order)
                        or len(set(order)) != len(order)
                        or not set(payload.get("line_ids", [])) <= set(order)):
                    raise ValueError("Invalid dialogue line order")
                for identifier in order:
                    record_id(identifier)
        elif payload.get("scene_strategy") == "joint" and payload.get("characters") and payload.get("lines"):
            self._scene_script(payload)
        if automatic_length:
            payload["max_new_tokens"] = _automatic_generation_limit(payload)
            payload["length_policy"] = "automatic-v1"
            self._validate_sampling(payload)
        if is_gguf(self.checkpoint):
            self._validate_gguf_job(payload)
            capabilities = self.capabilities()
            if ((mode == "design" and not capabilities["voice_design"])
                    or (payload.get("scene_strategy") == "joint"
                        and not capabilities["joint_dialogue"])):
                raise ValueError("Update the native runtime to protocol 4 for voice design and joint dialogue")
            feature = {"revision": "director_revision", "edit": "director_revision", "infill": "infill",
                       "scene": "scene_scheduler", "stream": "streaming_text", "audio_stream": "streaming_audio"}.get(mode)
            if ((feature and capabilities.get(feature) is False)
                    or ((payload.get("previous_take") or payload.get("previous_sources"))
                        and capabilities.get("director_revision") is False)):
                raise ValueError("Update the native runtime to protocol 3 to use Drama 3 advanced workflows")
            whole_text = payload.get("single_chunk") or (mode != "infill" and payload.get("duration_frames"))
            if whole_text and capabilities.get("duration_hint") is False:
                raise ValueError("Update the native runtime to protocol 4 for whole-text generation.")
        self._uploaded_references(payload)
        # Resolve mutable voice-library metadata before queueing, not at execution time.
        snapshot = payload.pop("_trusted_references", None)
        references = ([VoiceReference(**item) for item in snapshot] if snapshot is not None
                      else self._references(payload))
        if len(references) > 32:
            raise ValueError("At most 32 reference clips are supported")
        for reference in references:
            reference.transcript()
        if is_gguf(self.checkpoint) and payload.get("scene_strategy") == "joint":
            self._preflight_joint_context(payload, references)
        recipe = clone(payload)
        recipe["reference_snapshot"] = [asdict(reference) for reference in references]
        recipe["model"] = str(self.checkpoint)
        recipe["schema"] = 1
        recipe["codec"] = {"family": "drama3-dac", "sample_rate": 44100, "samples_per_frame": 2048,
                           "codebooks": 10, "vocabulary": 4096}
        recipe["runtime"] = {"backend": "gguf" if is_gguf(self.checkpoint) else "python"}
        if is_gguf(self.checkpoint):
            try:
                runtime = load_native_runtime(ROOT)
                recipe["runtime"].update(protocol=runtime.get("protocol"), executable_sha256=runtime.get("sha256"))
            except (OSError, ValueError, RuntimeError):
                recipe["runtime"]["protocol"] = None
        context = payload.get("studio")
        if context:
            if not isinstance(context, dict):
                raise ValueError("Studio context must be an object")
            record_id(context.get("work_id"))
            record_id(context.get("segment_id"))
        job_id = uuid.uuid4().hex
        job_dir = self.jobs_root / job_id
        stream = None
        if mode == "stream":
            live = payload.get("live_stream", False)
            chunks = payload.get("chunks", [])
            if not isinstance(live, bool) or not isinstance(chunks, list):
                raise ValueError("Text stream input must be a list of chunks with a boolean live_stream option")
            if not live and not chunks:
                raise ValueError("Provide text chunks or start a live text stream")
            if (payload.get("direction") or payload.get("language")
                    or payload.get("duration_frames") not in (None, "")
                    or payload.get("previous_take")):
                raise ValueError(
                    "Text streams manage their own history; use inline controls in each text chunk")
            stream = TextStreamSession(
                None, job_dir / "stream",
                max_chunk_chars=payload.get("max_chunk_chars", 160))
            for chunk in chunks:
                stream.push(chunk)
            if not live:
                stream.finish()
        job_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "id": job_id,
            "mode": mode,
            "status": "queued",
            "created_at": time.time(),
            "updated_at": time.time(),
            "events": [],
            "output": None,
            "take_file": None,
            "error": None,
            "recipe": recipe,
            "studio": clone(context) if context else None,
            "payload_summary": {
                "mode": mode,
                "text_length": len(str(payload.get("text", ""))),
                "reference_count": len(references),
            },
        }
        with self._jobs_lock:
            self._jobs[job_id] = record
            if stream is not None:
                self._streams[job_id] = stream
                record["stream_input"] = {
                    "live": payload.get("live_stream", False),
                    "finished": not payload.get("live_stream", False),
                    "pending": stream.pending,
                    "chunks": clone(payload.get("chunks", [])),
                }
            self._persist_job_locked(job_id)
        self.studio.register(record)
        self._executor.submit(self._run_job, job_id, clone(payload))
        return job_id

    @staticmethod
    def _validate_sampling(payload):
        if "silence_retry" in payload and not isinstance(payload["silence_retry"], bool):
            raise ValueError("silence_retry must be a boolean")
        for key, low, high in (("seed", 0, 2**32 - 1), ("max_new_tokens", 1, 2048),
                               ("top_k", 0, 8193), ("candidates", 1, 3)):
            if key in payload:
                value = payload[key]
                if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                    raise ValueError(f"{key} must be an integer from {low} to {high}")
        for key, low, high in (("temperature", 0, 2), ("top_p", 0, 1)):
            if key in payload:
                value = payload[key]
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or not low < value <= high):
                    raise ValueError(f"Invalid {key}")
        if "max_chunk_chars" in payload:
            value = payload["max_chunk_chars"]
            if (isinstance(value, bool) or not isinstance(value, int)
                    or not 80 <= value <= 320):
                raise ValueError("max_chunk_chars must be an integer from 80 to 320")
        duration = payload.get("duration_frames")
        if duration not in (None, ""):
            if isinstance(duration, bool) or not isinstance(duration, int) or not 1 <= duration <= 2048:
                raise ValueError("Target duration must be between 1 and 2048 frames")
            if duration > payload.get("max_new_tokens", 768):
                raise ValueError("Target duration exceeds the maximum generation length")

    @staticmethod
    def _preflight_joint_context(payload, references):
        import soundfile as sf
        minimum = 0
        for reference in references:
            try:
                info = sf.info(reference.audio)
            except (OSError, RuntimeError):
                # The regular format converter validates formats libsndfile cannot inspect.
                continue
            if not 0 < info.duration <= 120:
                raise ValueError("Reference audio must have a duration between zero and 120 seconds")
            minimum += max(0, math.floor(info.duration * 44100 / 2048) - 2)
        if payload.get("previous_sources"):
            for source in payload["previous_sources"]:
                if isinstance(source.get("frames"), int):
                    minimum += max(0, source["frames"] - 2)
        elif payload.get("previous_take"):
            take, _, _ = load_conditioning_take(_inside(Path(payload["previous_take"]), ROOT))
            minimum += max(0, take.codes.shape[1] - 2)
        if minimum > MAX_AUTOMATIC_CONTEXT:
            raise ValueError(
                "Joint dialogue exceeds the native context budget. Shorten references or script, or choose per-line generation.")

    @staticmethod
    def _scene_script(payload):
        characters = [Character(item["name"], item["description"], item["speaker"])
                      for item in payload.get("characters", [])]
        lines = [SceneLine(item["character"], item["text"]) for item in payload.get("lines", [])]
        validate_scene_inputs(payload.get("scene", ""), characters, lines)
        roles = {role.name: role for role in characters}
        texts = []
        for line in lines:
            speaker = roles[line.character].speaker
            if any(int(tag) != speaker for tag in re.findall(r"<\|speaker:(\d+)\|>", line.text)):
                raise ValueError("A line's speaker tag conflicts with its character")
            texts.append(line.text if re.match(r"\s*<\|speaker:\d+\|>", line.text)
                         else f"<|speaker:{speaker}|> {line.text}")
        payload["text"] = "\n".join(texts)
        for field, count in (("line_ids", len(lines)), ("character_ids", len(characters))):
            identifiers = payload.get(field)
            if identifiers is not None:
                if (not isinstance(identifiers, list) or len(identifiers) != count
                        or any(not isinstance(item, str) for item in identifiers)
                        or len(set(identifiers)) != count):
                    raise ValueError(f"Each scene entry needs a unique ID: {field}")
                for identifier in identifiers:
                    record_id(identifier)

    @staticmethod
    def _joint_script_lines(payload):
        roles = {role["speaker"]: role["name"] for role in payload["characters"]}
        text = payload["text"]
        tags = list(re.finditer(r"<\|speaker:(\d+)\|>", text))
        if not tags or text[:tags[0].start()].strip():
            raise ValueError("Keep the speaker labels in the complete dialogue transcript")
        lines = []
        for index, tag in enumerate(tags):
            speaker = int(tag[1])
            end = tags[index + 1].start() if index + 1 < len(tags) else len(text)
            if speaker not in roles:
                raise ValueError("Dialogue transcript contains an unassigned speaker")
            lines.append({"character": roles[speaker], "text": text[tag.end():end].strip()})
        payload["lines"] = lines
        old_ids = payload.get("line_ids") or []
        payload["line_ids"] = [old_ids[index] if index < len(old_ids) else uuid.uuid4().hex
                               for index in range(len(lines))]
        payload["scene_order"] = clone(payload["line_ids"])
        AppState._scene_script(payload)

    def version(self, payload):
        job = self.job(payload.get("job_id"))
        if not job:
            raise ValueError("Source version does not exist")
        recipe = clone(job.get("recipe") or {})
        recipe.setdefault("mode", job["mode"])
        complete = recipe.get("schema") == 1 and {
            "model", "seed", "max_new_tokens", "temperature", "top_p", "top_k",
            "reference_snapshot", "runtime", "codec",
        } <= recipe.keys()
        value = {"job_id": job["id"], "output": job.get("output"), "take_file": job.get("take_file"),
                 "recipe": recipe, "complete_recipe": complete, "studio": job.get("studio"),
                 "status": job["status"]}
        value["timing"] = clone(job.get("timing"))
        value["timing_result"] = timing_result(job.get("result"))
        if job.get("stream_input"):
            value["stream_input"] = clone(job["stream_input"])
            if "chunks" not in value["stream_input"]:
                rows = job.get("result", {}).get("segments", [])
                value["stream_input"]["chunks"] = (
                    [row["text"] for row in rows] if rows and all(isinstance(row.get("text"), str) for row in rows)
                    else clone(recipe.get("chunks", [])))
        index = payload.get("part_index")
        if (index is None and (job.get("studio") or {}).get("line_id")
                and recipe.get("mode") == "scene"
                and len(job.get("result", {}).get("segments", [])) == 1):
            index = 0
        if index is not None:
            row = self._scene_row(job, index)
            value.update(output=row.get("path"), take_file=row.get("take_file"),
                         part_index=index, line_id=self._scene_line_id(job, row, index))
            if value["take_file"]:
                value["status"] = "complete"
            if value.get("studio"):
                value["studio"]["line_id"] = value["line_id"]
            character = next((item for item in recipe.get("characters", [])
                              if item["name"] == row.get("character")), None)
            if character:
                speaker = character["speaker"]
                recipe.setdefault("scene_origin", {
                    "job_id": job["id"], "part_index": index, "speaker": speaker,
                    "line_id": value["line_id"],
                })
                recipe["reference_snapshot"] = [
                    item for item in recipe.get("reference_snapshot", []) if item["speaker"] == speaker]
                recipe["text"] = (row["text"] if re.match(r"\s*<\|speaker:\d+\|>", row["text"])
                                  else f"<|speaker:{speaker}|> {row['text']}")
                recipe["direction"] = (
                    f"Scene: {recipe.get('scene', '')}\n"
                    f"Character: {character['name']}. {character['description']}\n"
                    f"{recipe.get('direction', '')}").strip()
                recipe["mode"] = "tts"
                recipe["seed"] = (recipe.get("seed", 123) + index) % 2**32
                recipe["previous_sources"] = row.get("history_sources", [])
                for field in ("characters", "lines", "scene", "scene_strategy"):
                    recipe.pop(field, None)
            elif job["mode"] == "stream":
                recipe["text"] = row.get("text", "")
                recipe["mode"] = "tts"
                for field in ("chunks", "live_stream"):
                    recipe.pop(field, None)
                value["complete_recipe"] = False
        if value["take_file"]:
            try:
                take = self.inspect_take(value["take_file"])
                value["text"] = recipe.get("text", take["text"])
                value["take"] = take
            except FileNotFoundError:
                value.update(take_file=None, take_missing=True, text=recipe.get("text", ""))
        else:
            value["text"] = ("\n".join(value["stream_input"]["chunks"])
                             if job.get("stream_input") and index is None else recipe.get("text", ""))
        value["audio_missing"] = bool(value["output"] and not Path(value["output"]).is_file())
        return value

    def reference_upload(self, upload_id):
        if not isinstance(upload_id, str) or not re.fullmatch(r"[a-f0-9]{32}", upload_id):
            raise ValueError("Invalid reference upload ID")
        folder = _inside(self.uploads_root / upload_id, self.uploads_root)
        metadata = json.loads((folder / "reference.json").read_text(encoding="utf-8"))
        audio = _inside(folder / metadata["audio"], folder)
        if not audio.is_file():
            raise ValueError("Reference audio is missing")
        return {"id": upload_id, "filename": metadata["filename"], "audio_path": str(audio),
                "audio_url": "/files/" + "/".join(quote(part) for part in audio.relative_to(ROOT).parts)}

    def reference_from_version(self, payload):
        import soundfile as sf

        version = self.version(payload["source_version"])
        if not version["output"]:
            raise ValueError("This version has no audio")
        path = _inside(Path(version["output"]), self.jobs_root)
        selected = "start" in payload or "end" in payload
        if selected:
            uploaded = self._crop_reference_audio(path, path.name, payload.get("start"), payload.get("end"))
        else:
            if not 0 < sf.info(str(path)).duration <= 120:
                raise ValueError("Select up to 120 seconds from this audio to use as a reference.")
            if path.stat().st_size > 128 * 1024**2:
                raise ValueError("Selected reference audio exceeds the 128 MiB limit.")
            uploaded = self.add_reference_upload(path.name, path.read_bytes())
        return {**self.reference_upload(uploaded["id"]), **uploaded,
                "suggested_text": "" if selected else version["text"]}

    def save_uploaded_voice(self, payload):
        upload = self.reference_upload(payload["upload_id"])
        path = Path(upload["audio_path"])
        return self.add_voice(payload["name"], payload["transcript"], 0, path.name, path.read_bytes())

    @staticmethod
    def _audio_url(path):
        return "/files/" + "/".join(quote(part) for part in _inside(Path(path), ROOT).relative_to(ROOT).parts)

    @staticmethod
    def _link_or_copy(source, destination):
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)

    def save_reference_group(self, payload):
        name = payload.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Voice name must not be empty")
        bindings = payload.get("references")
        if not isinstance(bindings, list) or not 1 <= len(bindings) <= 32:
            raise ValueError("Choose between 1 and 32 reference clips")
        references = self._references({"voice_bindings": [
            {**binding, "speaker": 0} if isinstance(binding, dict) else binding for binding in bindings]})
        prepared, seen = [], set()
        for reference in references:
            reference.transcript()
            audio = _inside(Path(reference.audio), ROOT)
            with audio.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            text = re.sub(r"^\s*<\|speaker:\d+\|>\s*", "", reference.text, count=1)
            identity = (digest, text)
            if identity in seen:
                continue
            seen.add(identity)
            prepared.append((audio, text, digest))
        if not 1 <= len(prepared) <= 32:
            raise ValueError("Choose between 1 and 32 reference clips")
        voice_id = f"{uuid.uuid4().hex[:10]}-{_safe_name(name, 'voice')}"
        folder = _inside(self.voices_root / voice_id, self.voices_root)
        folder.mkdir()
        clips = []
        for index, (audio, transcript, digest) in enumerate(prepared):
            filename = f"reference-{index + 1:04d}{audio.suffix.lower()}"
            self._link_or_copy(audio, folder / filename)
            clips.append({"audio": filename, "transcript": transcript, "sha256": digest})
        metadata = {
            "schema": 2, "name": name.strip()[:120], "speaker": 0, "created_at": time.time(),
            "audio": clips[0]["audio"], "transcript": clips[0]["transcript"], "references": clips,
        }
        (folder / "voice.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return self._voice(voice_id)

    def _prepare_reference_assets(self, snapshot):
        references = []
        for reference in snapshot:
            audio = _inside(Path(reference["audio"]), ROOT)
            with audio.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            identifier = digest[:32]
            folder = _inside(self.uploads_root / identifier, self.uploads_root)
            with self._reference_assets_lock:
                folder.mkdir(exist_ok=True)
                metadata_file = folder / "reference.json"
                if not metadata_file.exists():
                    filename = "reference" + audio.suffix.lower()
                    target = folder / filename
                    if not target.exists():
                        self._link_or_copy(audio, target)
                    metadata_file.write_text(json.dumps({
                        "filename": audio.name, "audio": filename, "sha256": digest,
                    }, ensure_ascii=False) + "\n", encoding="utf-8")
                else:
                    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
                    if metadata.get("sha256") != digest:
                        raise ValueError("Reference asset ID collision")
            references.append({
                **self.reference_upload(identifier), "text": reference["text"], "speaker": reference["speaker"],
            })
        return references

    def _supplement_source_references(self, version, bindings, checked):
        if checked is not True:
            raise ValueError("Check the reference transcript first.")
        source_speakers = {int(speaker) for speaker in re.findall(
            r"<\|speaker:(\d+)\|>", version["text"])} or {0}
        frozen = version["recipe"].get("reference_snapshot", [])
        missing = source_speakers - {reference["speaker"] for reference in frozen}
        prepared = self._references({"voice_bindings": bindings})
        if any(reference.speaker not in missing for reference in prepared):
            raise ValueError("Prepared references may only fill missing source voices")
        if len(frozen) + len(prepared) > 32:
            raise ValueError("At most 32 reference clips are supported")
        return [asdict(reference) for reference in prepared]

    def prepare_version_restore(self, payload):
        version = self.version(payload)
        snapshot = clone(version["recipe"].get("reference_snapshot", []))
        if payload.get("prepared_references"):
            snapshot.extend(self._supplement_source_references(
                version, payload["prepared_references"], payload.get("reference_transcripts_checked")))
        return {"version": version, "references": self._prepare_reference_assets(snapshot)}

    @staticmethod
    def _scene_rows(job):
        rows = job.get("result", {}).get("segments")
        if not rows:
            rows = [event for event in job.get("events", [])
                    if event.get("stage") == "segment" and event.get("take_file")]
        return rows or []

    @staticmethod
    def _scene_line_id(job, row, offset):
        identifiers = (job.get("recipe") or {}).get("line_ids") or []
        index = row.get("index", offset)
        return row.get("line_id") or (identifiers[index] if 0 <= index < len(identifiers) else f"line-{index}")

    def _scene_row(self, job, index):
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("Invalid source line")
        row = next((row for offset, row in enumerate(self._scene_rows(job))
                    if row.get("index", offset) == index), None)
        if row is None:
            raise ValueError("Source dialogue line is unavailable")
        return row

    def _source_metadata(self, source):
        job = self.job(source.get("job_id"))
        if not job:
            raise ValueError("Source version does not exist")
        context, recipe = job.get("studio") or {}, job.get("recipe") or {}
        index = source.get("part_index")
        if index is None and context.get("line_id") and recipe.get("mode") == "scene":
            index = 0
        if index is not None:
            row = self._scene_row(job, index)
            return job, row.get("take_file"), row.get("history_sources", [])
        return job, job.get("take_file"), recipe.get("previous_sources", [])

    def _context_dependencies(self, source, seen=None):
        seen = set() if seen is None else seen
        identity = (source.get("job_id"), source.get("part_index"))
        if identity in seen:
            raise ValueError("Cyclic source history")
        seen.add(identity)
        job, _take, sources = self._source_metadata(source)
        if (sources or source.get("part_index") is not None
                or (job.get("studio") or {}).get("line_id") and job["mode"] == "scene"):
            return clone(sources)
        recipe = job.get("recipe") or {}
        parent = recipe.get("source_version")
        conditioning = recipe.get("previous_take") or recipe.get("source_take")
        # Retrying a derived version preserves its original conditioning path,
        # which can point past the immediate parent. Read metadata, not codes.
        while conditioning and parent:
            identity = (parent.get("job_id"), parent.get("part_index"))
            if identity in seen:
                raise ValueError("Cyclic source history")
            _job, parent_take, _history = self._source_metadata(parent)
            if parent_take == conditioning:
                return self._context_dependencies(parent, seen)
            seen.add(identity)
            parent = (_job.get("recipe") or {}).get("source_version")
        if conditioning and recipe.get("scene_origin"):
            raise ValueError("Recorded dialogue history is unavailable")
        return []

    def scene_context_status(self, payload):
        job = self.job(payload.get("job_id"))
        if not job:
            raise ValueError("Source version does not exist")
        if job["mode"] != "scene" or job.get("recipe", {}).get("scene_strategy", "per_line") != "per_line":
            raise ValueError("This result has no per-line context")
        context = job.get("studio") or {}
        work = self.studio.snapshot()["works"].get(context.get("work_id"), {})
        part = next((part for part in work.get("segments", []) if part["id"] == context.get("segment_id")), {})
        choices = part.get("line_versions", {})
        base_source = part.get("adopted") or part.get("scene_version") or {"job_id": job["id"]}
        base = self.job(base_source["job_id"])
        base_recipe = (base or {}).get("recipe") or {}
        per_line = bool(base and base["mode"] == "scene"
                        and base_recipe.get("scene_strategy", "per_line") == "per_line")
        base_sources, speakers, order = {}, {}, []
        if per_line:
            order = base_recipe.get("line_ids") or [
                f"line-{index}" for index in range(len(base_recipe.get("lines", [])))]
            roles = {role["name"]: role["speaker"] for role in base_recipe.get("characters", [])}
            speakers = {identifier: roles[line["character"]]
                        for identifier, line in zip(order, base_recipe.get("lines", []))}
            base_sources = {self._scene_line_id(base, row, offset): {
                                "job_id": base["id"], "part_index": row.get("index", offset)}
                            for offset, row in enumerate(self._scene_rows(base))}
        path_to_line = {}
        for identifier in part.get("versions", [job["id"]]):
            candidate = self.job(identifier)
            if not candidate:
                continue
            line_id = (candidate.get("studio") or {}).get("line_id")
            if line_id and candidate.get("take_file"):
                path_to_line[candidate["take_file"]] = line_id
            for offset, row in enumerate(self._scene_rows(candidate)):
                if row.get("take_file"):
                    path_to_line[row["take_file"]] = self._scene_line_id(candidate, row, offset)
        rows, statuses = self._scene_rows(job), []
        for offset, row in enumerate(rows):
            line_id = self._scene_line_id(job, row, offset)
            chosen = choices.get(line_id) or base_sources.get(line_id)
            prior = {identifier for identifier in order[:order.index(line_id)]
                     if speakers[identifier] == speakers[line_id]} if line_id in order else set()
            changed, available = False, per_line and chosen is not None
            try:
                if available:
                    _selected, selected_take, _history = self._source_metadata(chosen)
                    if not selected_take or not Path(selected_take).is_file():
                        raise ValueError("The selected dialogue take is unavailable")
                dependencies = self._context_dependencies(chosen) if available else []
                for dependency in dependencies:
                    previous_id = path_to_line.get(dependency["take_file"])
                    if previous_id is None:
                        raise ValueError("Recorded dialogue history is unavailable")
                    if previous_id not in prior:
                        continue
                    current = choices.get(previous_id) or base_sources.get(previous_id)
                    if current is None:
                        raise ValueError("A preceding dialogue line is unavailable")
                    _parent, current_take, _history = self._source_metadata(current)
                    if not current_take or not Path(current_take).is_file():
                        raise ValueError("A preceding codec take is unavailable")
                    changed |= current_take != dependency["take_file"]
            except (OSError, ValueError):
                available = False
            statuses.append({"line_id": line_id, "changed": changed, "available": available,
                             "has_previous": bool(prior)})
        return {"lines": statuses}

    def _adopted_line_history(self, version):
        context = version.get("studio") or {}
        work = self.studio.snapshot()["works"].get(context.get("work_id"), {})
        part = next((part for part in work.get("segments", []) if part["id"] == context.get("segment_id")), None)
        origin = version["recipe"].get("scene_origin")
        if not origin and version["recipe"].get("source_version"):
            parent = self.version(version["recipe"]["source_version"])
            origin = parent["recipe"].get("scene_origin")
        if part is None or not origin:
            raise ValueError("This result has no recorded dialogue context")
        base_source = part.get("adopted") or part.get("scene_version") or {"job_id": origin["job_id"]}
        base = self.job(base_source["job_id"])
        if (not base or base["mode"] != "scene"
                or base.get("recipe", {}).get("scene_strategy", "per_line") != "per_line"):
            raise ValueError("The adopted dialogue has no per-line context")
        recipe = base["recipe"]
        ids = recipe.get("line_ids") or [f"line-{index}" for index in range(len(recipe.get("lines", [])))]
        line_id = context.get("line_id", origin["line_id"])
        if line_id not in ids:
            raise ValueError("This line is not in the adopted dialogue")
        index, speaker = ids.index(line_id), origin["speaker"]
        roles = {role["name"]: role["speaker"] for role in recipe["characters"]}
        if roles[recipe["lines"][index]["character"]] != speaker:
            raise ValueError("The line's character changed. Restore the current script first.")
        previous = [position for position in range(index)
                    if roles[recipe["lines"][position]["character"]] == speaker]
        rows = {row.get("index", offset) for offset, row in enumerate(self._scene_rows(base))}
        selected = []
        for position in previous:
            source = part.get("line_versions", {}).get(ids[position])
            if source is None:
                if position not in rows:
                    continue
                source = {"job_id": base["id"], "part_index": position}
            _job, take_path, _history = self._source_metadata(source)
            if not take_path:
                raise ValueError("A preceding line has no completed codec take")
            selected.append((source, take_path))
        if not selected:
            raise ValueError("There is no completed preceding line for this character")
        history, chunks_seen = [], set()
        for (source, take_path), edge in ((selected[0], "first"), (selected[-1], "last")):
            take, _metadata, chunks = load_conditioning_take(_inside(Path(take_path), ROOT))
            anchor = (chunks[0] if edge == "first" else chunks[-1]) if chunks else take
            descriptor = {"take_file": take_path, "text": anchor.text, "frames": anchor.codes.shape[1],
                          "edge": edge, "source_version": clone(source)}
            if anchor.lookahead:
                descriptor["lookahead"] = anchor.lookahead
            offset = 0 if edge == "first" else take.codes.shape[1] - anchor.codes.shape[1]
            identity = (take_path, offset, anchor.codes.shape[1])
            if identity in chunks_seen:
                continue
            chunks_seen.add(identity)
            history.append(descriptor)
        return history

    def _continuation_history(self, version):
        """Keep exact first/recent chunks instead of an ever-growing full take."""
        path = _inside(Path(version["take_file"]), ROOT)
        take, _metadata, chunks = load_conditioning_take(path)
        if not chunks:
            if take.codes.shape[1] > 2048:
                raise ValueError("This older audio has no recorded chunk boundaries. Choose a completed part to continue.")
            chunks = [take]
        selected = [("first", chunks[0])]
        if len(chunks) > 1:
            selected.append(("last", chunks[-1]))
        if any(chunk.codes.shape[1] > 2048 for _edge, chunk in selected):
            raise ValueError("This older audio has no recorded chunk boundaries. Choose a completed part to continue.")
        sources = []
        for edge, chunk in selected:
            source = {"take_file": str(path), "edge": edge, "text": chunk.text, "frames": chunk.codes.shape[1]}
            if chunk.lookahead is not None:
                source["lookahead"] = chunk.lookahead
            sources.append(source)
        return sources

    def work_jobs(self, work_id, *, offset=0, limit=100):
        work = self.studio.snapshot()["works"].get(record_id(work_id))
        if work is None:
            raise ValueError("Studio work does not exist")
        if not 0 <= offset <= 100000 or not 1 <= limit <= 100:
            raise ValueError("Invalid work history page")
        identifiers = list(dict.fromkeys(identifier for part in work["segments"] for identifier in part["versions"]))
        rows, missing = [], []
        for identifier in identifiers[offset:offset + limit]:
            job = self.job(identifier)
            if job is None:
                missing.append(identifier)
                continue
            value = {key: job.get(key) for key in (
                "id", "mode", "status", "created_at", "updated_at", "studio",
                "output", "take_file", "error", "payload_summary", "timing")}
            recipe = job.get("recipe") or {}
            value["recipe"] = {key: recipe[key] for key in (
                "mode", "text", "model", "operation", "scene_strategy") if key in recipe} or None
            result = job.get("result") or {}
            value["result"] = {**timing_result(result), **{key: result[key] for key in (
                "audio", "strategy", "segments") if key in result}}
            rows.append(value)
        return {"jobs": rows, "missing": missing, "total": len(identifiers),
                "next_offset": offset + limit if offset + limit < len(identifiers) else None}

    def export_work(self, payload):
        from drama_low_vram.codec import concatenate_wavs
        versions = payload.get("versions")
        if payload.get("work_id"):
            work = self.studio.snapshot()["works"].get(record_id(payload["work_id"]))
            if not work:
                raise ValueError("Studio work does not exist")
            versions = []
            for segment in work["segments"]:
                base = segment.get("adopted") or segment.get("scene_version")
                if base:
                    versions.append(base)
                elif segment.get("line_order"):
                    choices = segment.get("line_versions", {})
                    if not all(identifier in choices for identifier in segment["line_order"]):
                        raise ValueError("Every dialogue line needs a completed version before export")
                    versions.extend(choices[identifier] for identifier in segment["line_order"])
        if not isinstance(versions, list) or not 1 <= len(versions) <= 4096:
            raise ValueError("Choose the adopted versions to export")
        resolved = [self.version(version) for version in versions]
        files, parts = [], []
        for version, value in zip(versions, resolved):
            job = self.job(version["job_id"])
            context = job.get("studio") or {}
            if (version.get("part_index") is None and job["mode"] == "scene"
                    and not context.get("line_id")
                    and job.get("result", {}).get("strategy") == "per_line_tts"):
                work = self.studio.snapshot()["works"].get(context.get("work_id"), {})
                segment = next((item for item in work.get("segments", [])
                                if item["id"] == context.get("segment_id")), {})
                choices = segment.get("line_versions", {})
                rows = {row["index"]: row for row in job["result"]["segments"]}
                line_ids = job.get("recipe", {}).get("line_ids", [])
                count = len(job.get("recipe", {}).get("lines", [])) or len(rows)
                for index in range(count):
                    line_id = line_ids[index] if index < len(line_ids) else f"line-{index}"
                    chosen = choices.get(line_id) or {"job_id": job["id"], "part_index": index}
                    if index not in rows and line_id not in choices:
                        raise ValueError("Every dialogue line needs a completed version before export")
                    selected = self.version(chosen)
                    if selected["status"] != "complete" or not selected["output"]:
                        raise ValueError("Only completed audio can be exported")
                    files.append(_inside(Path(selected["output"]), self.jobs_root))
                    parts.append({"source_version": clone(chosen), "line_id": line_id,
                                  "output": selected["output"], "take_file": selected["take_file"],
                                  "recipe": selected["recipe"]})
            else:
                if value["status"] != "complete" or not value["output"]:
                    raise ValueError("Only completed audio can be exported")
                files.append(_inside(Path(value["output"]), self.jobs_root))
                parts.append({"source_version": clone(version), "output": value["output"],
                              "take_file": value["take_file"], "recipe": value["recipe"],
                              **({"line_id": value.get("line_id", context.get("line_id"))}
                                 if value.get("line_id", context.get("line_id")) else {})})
        folder = self.data_root / "exports" / uuid.uuid4().hex
        folder.mkdir(parents=True)
        output = folder / "result.wav"
        concatenate_wavs(files, output)
        (folder / "composition.json").write_text(
            json.dumps({"schema": 2, "versions": versions, "parts": parts},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        return {"output": str(output), "audio_url": "/files/" + "/".join(
            quote(part) for part in output.relative_to(ROOT).parts)}

    def adopt_version(self, payload):
        version = self.version(payload["version"])
        if version["status"] != "complete" or not version["output"]:
            raise ValueError("Only completed audio can be adopted")
        context = version.get("studio") or {}
        if (context.get("work_id") != payload["work_id"]
                or context.get("segment_id") != payload["segment_id"]):
            raise ValueError("Version does not belong to this segment")
        line_id = payload.get("line_id")
        if line_id:
            if version.get("line_id", context.get("line_id")) != line_id:
                raise ValueError("Version does not belong to this dialogue line")
            return self.studio.adopt_line(
                payload["work_id"], payload["segment_id"], line_id, payload["version"])
        if context.get("line_id"):
            raise ValueError("A dialogue line cannot replace the entire scene")
        return self.studio.adopt(payload["work_id"], payload["segment_id"], payload["version"])

    def _creative_request(self, payload):
        payload.pop("_trusted_references", None)
        payload.pop("reference_snapshot", None)
        requested_model = payload.get("model")
        if requested_model and Path(requested_model).resolve() != self.checkpoint.resolve():
            raise ValueError("This version uses a different model. Switch models explicitly before generating.")
        source = payload.get("source_version")
        if not source:
            if payload.get("condition_source"):
                self.version(payload["condition_source"])
            return payload
        version = self.version(source)
        operation = payload.get("operation", "retry")
        recoverable = operation == "retry" and version["status"] in {
            "interrupted", "error", "failed", "cancelled", "incomplete"}
        if version["status"] != "complete" and not recoverable:
            raise ValueError("A source version must be complete")
        if not version["complete_recipe"]:
            raise ValueError("This older version has no complete recipe. Open it as a draft and choose its voice.")
        original = clone(version["recipe"])
        if Path(original["model"]).resolve() != self.checkpoint.resolve():
            raise ValueError("This version uses a different model. Switch models explicitly before generating.")
        expected_codec = {"family": "drama3-dac", "sample_rate": 44100, "samples_per_frame": 2048,
                          "codebooks": 10, "vocabulary": 4096}
        if (not isinstance(original["codec"], dict)
                or any(original["codec"].get(key) != value for key, value in expected_codec.items())):
            raise ValueError("This source uses an incompatible audio codec.")
        original.pop("reference_snapshot", None)
        original.pop("source_version", None)
        original.update(
            seed=payload.get("seed", int.from_bytes(os.urandom(4), "little")),
            studio=payload.get("studio") or version["studio"],
            source_version=source, operation=operation,
            _trusted_references=version["recipe"].get("reference_snapshot", []))
        if operation == "retry":
            if original.get("mode") == "stream":
                chunks = version.get("stream_input", {}).get("chunks", original.get("chunks", []))
                if not chunks:
                    raise ValueError("This stream has no received text to replay.")
                original.update(chunks=clone(chunks), live_stream=False, replay_stream=True)
                original["text"] = "\n".join(original["chunks"])
            return original
        if operation == "update_context":
            original.update(mode="tts", previous_sources=self._adopted_line_history(version),
                            conditioning="adopted_scene_history")
            original.pop("previous_take", None)
            original.pop("replay_stream", None)
            return original
        original.pop("conditioning", None)
        if not version["take_file"]:
            raise ValueError("This source has no editable codec take")
        if operation in ("revision", "reword", "continue"):
            original["mode"] = "revision" if operation == "revision" else "edit"
            original["text"] = version["text"] if operation == "revision" else payload.get("text", "")
            original["direction"] = payload.get("direction", original.get("direction", ""))
            original["previous_take"] = version["take_file"]
            original.pop("previous_sources", None)
            if operation == "continue":
                original.update(previous_sources=self._continuation_history(version),
                                conditioning="bounded_continuation_history")
            if original.get("scene_strategy") == "joint":
                original["single_chunk"] = True
                if operation != "revision":
                    if "lines" in payload:
                        original["lines"] = clone(payload["lines"])
                        original["line_ids"] = clone(payload.get("line_ids"))
                        original["scene_order"] = clone(payload.get("scene_order", payload.get("line_ids")))
                        self._scene_script(original)
                    else:
                        self._joint_script_lines(original)
            if operation == "reword":
                if payload.get("reword_references"):
                    original["_trusted_references"].extend(self._supplement_source_references(
                        version, payload["reword_references"], payload.get("reference_transcripts_checked")))
                    original["reference_transcripts_checked"] = True
                available = {reference["speaker"] for reference in original["_trusted_references"]}
                required = {int(speaker) for speaker in re.findall(r"<\|speaker:(\d+)\|>", original["text"])} or {0}
                original["conditioning"] = "source_history"
                if required <= available:
                    # Regenerate new speech from the same voices. Old performance
                    # history can otherwise pull the output back to old wording.
                    original["mode"] = "tts"
                    original.pop("previous_take", None)
                    original["conditioning"] = "frozen_reference_regeneration"
        elif operation == "infill":
            original.update(
                mode="infill", text=payload.get("text", ""), source_take=version["take_file"],
                start_frame=payload.get("start_frame"), end_frame=payload.get("end_frame"),
                _trusted_references=[], voice_ids=[], uploaded_references=[])
            for key in ("previous_take", "previous_sources", "direction", "language", "duration_frames"):
                original.pop(key, None)
            if original.get("scene_strategy") == "joint":
                self._joint_script_lines(original)
        else:
            raise ValueError("Unknown creative operation")
        for key in (
            "temperature", "top_k", "top_p", "max_new_tokens",
            "max_chunk_chars", "duration_frames", "automatic_length", "silence_retry",
        ):
            if key in payload:
                original[key] = payload[key]
        if original.get("duration_frames") and operation != "infill":
            original["single_chunk"] = True
        return original

    def job(self, job_id):
        with self._jobs_lock:
            value = self._jobs.get(job_id)
            if value is not None and job_id in self._job_clocks:
                value["timing"] = self._job_clocks[job_id].snapshot()
            return None if value is None else json.loads(json.dumps(value))

    def jobs(self):
        with self._jobs_lock:
            for job_id, clock in self._job_clocks.items():
                if job_id in self._jobs:
                    self._jobs[job_id]["timing"] = clock.snapshot()
            values = [job for job in self._jobs.values()
                      if job.get("mode") != "audio_import" or job.get("status") in ("queued", "running")]
        values.sort(key=lambda item: item["created_at"], reverse=True)
        return [json.loads(json.dumps(value)) for value in values[:100]]

    def cancel(self, job_id):
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if not job:
                raise KeyError(job_id)
            job["cancel_requested"] = True
            stream = self._streams.get(job_id)
            self._persist_job_locked(job_id)
        if stream is not None:
            stream.cancel()
        self._event(job_id, {"stage": "cancel", "message": "Cancellation requested"})

    def stream_input(self, job_id, *, text=None, finish=False):
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            stream = self._streams.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if (stream is None or job["status"] not in ("queued", "running")
                    or job.get("cancel_requested") or job["stream_input"]["finished"]):
                raise ValueError("This text stream no longer accepts input")
            if text is not None:
                if len(job["stream_input"].get("chunks", [])) >= 4096:
                    raise ValueError("Too many text chunks in one request")
                stream.push(text)
                job["stream_input"].setdefault("chunks", []).append(text)
            if finish:
                stream.finish()
            job["stream_input"].update(finished=finish, pending=stream.pending)
            self._persist_job_locked(job_id)
        self._event(job_id, {
            "stage": "text_input",
            "message": "Text input finished" if finish else "Text chunk queued",
            "pending": stream.pending,
        })
        return dict(job["stream_input"])

    def cancelled(self, job_id):
        with self._jobs_lock:
            return bool(self._jobs.get(job_id, {}).get("cancel_requested"))

    def _get_engine(self, job_id):
        with self._engine_lock:
            if self._engine is None:
                inspected = inspect_model(self.checkpoint)
                if not inspected["ready"]:
                    raise ValueError("Choose a complete local model in Settings")
                self._event(job_id, {"stage": "runtime", "message": "Loading the model"})
                status("Loading model: %s", self.checkpoint.name)
                if is_gguf(self.checkpoint):
                    self._engine = GGUFEngine(resolve_gguf(self.checkpoint), root=ROOT)
                    self._engine.status_callback = lambda event: self._event(
                        job_id, {"stage": "runtime", **event})
                    return self._engine
                import torch

                # Match the validated inference configuration. Native expert
                # workers retain their separately calibrated thread count.
                torch.set_num_threads(1)
                self._engine = DramaEngine(EngineOptions(
                    checkpoint=self.checkpoint,
                    device=select_python_device(),
                    expert_cache_dir=self.expert_cache_directory(),
                    compute_profile_dir=ROOT / ".cache/compute-profiles",
                    expert_storage="auto",
                    cache_experts=None,
                    backbone_graph=True,
                    combined_decode_bank=True,
                    host_gather="auto",
                    prefix_cache_mib=256,
                    codec_max_frames=2048,
                    context_bucket_size=512,
                    compute_policy="auto",
                ))
            self._engine.status_callback = lambda event: self._event(
                job_id, {"stage": "runtime", **event})
            return self._engine

    def _voice(self, voice_id: str) -> dict:
        if not isinstance(voice_id, str) or not voice_id:
            raise ValueError("Invalid voice id")
        path = _inside(self.voices_root / voice_id, self.voices_root)
        metadata_path = path / "voice.json"
        if not metadata_path.is_file():
            raise ValueError("Voice does not exist")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        raw = metadata.get("references") or [{"audio": metadata["audio"], "transcript": metadata["transcript"]}]
        if not isinstance(raw, list) or not 1 <= len(raw) <= 32:
            raise ValueError("Invalid saved reference group")
        references = []
        for item in raw:
            audio = _inside(path / item["audio"], path)
            if not audio.is_file():
                raise ValueError("Reference audio is missing")
            references.append({**item, "audio_path": str(audio), "audio_url": self._audio_url(audio)})
        return {**metadata, "id": voice_id, "audio_path": references[0]["audio_path"],
                "audio_url": references[0]["audio_url"], "references": references}

    def list_voices(self):
        result = []
        for path in sorted(self.voices_root.iterdir()):
            if not path.is_dir() or not (path / "voice.json").is_file():
                continue
            try:
                result.append(self._voice(path.name))
            except (OSError, ValueError, KeyError):
                continue
        return result

    def add_voice(self, name: str, transcript: str, speaker: int, filename: str, data: bytes):
        if len(data) < 32:
            raise ValueError("Audio file is empty")
        VoiceReference("uploaded", str(transcript), int(speaker)).transcript()
        if not str(name).strip():
            raise ValueError("Voice name must not be empty")
        voice_id = f"{uuid.uuid4().hex[:10]}-{_safe_name(name, 'voice')}"
        folder = self.voices_root / voice_id
        folder.mkdir(parents=True)
        suffix = Path(filename).suffix.lower()
        if suffix not in (".wav", ".flac", ".mp3", ".ogg", ".m4a"):
            suffix = ".wav"
        audio_name = f"reference{suffix}"
        (folder / audio_name).write_bytes(data)
        metadata = {
            "name": str(name).strip()[:120] or "Untitled voice",
            "transcript": str(transcript),
            "speaker": int(speaker),
            "audio": audio_name,
            "created_at": time.time(),
        }
        (folder / "voice.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {**metadata, "id": voice_id}

    def add_reference_upload(self, filename: str, data: bytes):
        """Store an uploaded reference without creating a voice-library entry."""
        suffix = Path(filename).suffix.lower()
        if suffix not in (".wav", ".flac", ".mp3", ".ogg", ".m4a"):
            raise ValueError("Choose a WAV, MP3, FLAC, OGG or M4A reference")
        if not 32 <= len(data) <= 128 * 1024**2:
            raise ValueError("Reference audio must be nonempty and no larger than 128 MiB")
        upload_id = uuid.uuid4().hex
        folder = self.uploads_root / upload_id
        folder.mkdir()
        audio_name = f"reference{suffix}"
        (folder / audio_name).write_bytes(data)
        metadata = {"filename": Path(filename).name, "audio": audio_name}
        (folder / "reference.json").write_text(
            json.dumps(metadata, ensure_ascii=False) + "\n", encoding="utf-8")
        return {"id": upload_id, "filename": metadata["filename"]}

    def start_uploaded_audio_edit(self, payload):
        """Create one edit job; uploaded audio is encoded only when the job runs."""
        import soundfile as sf
        from drama_low_vram.gguf_runtime import reference_wav

        if not isinstance(payload, dict):
            raise ValueError("Audio edit request must be an object")
        payload = clone(payload)
        automatic_length = payload.pop("automatic_length", False)
        if not isinstance(automatic_length, bool):
            raise ValueError("automatic_length must be a boolean")
        if automatic_length:
            payload["max_new_tokens"] = 2048
        upload = self.reference_upload(payload.get("upload_id"))
        reference = VoiceReference(upload["audio_path"], payload.get("original_transcript"), 0)
        reference.transcript()
        text = VoiceReference("", payload.get("text"), 0).transcript()
        operation = payload.get("operation")
        if operation not in ("revision", "reword", "infill"):
            raise ValueError("Unknown uploaded audio edit operation")
        if operation == "revision" and text != reference.transcript():
            raise ValueError("Revision must keep the original transcript")
        direction = payload.get("direction", "")
        validate_content(direction, "direction", empty=True)
        if operation == "infill" and direction:
            raise ValueError("Use inline controls for selected-passage editing")
        self._validate_sampling(payload)
        audio = reference_wav(reference.audio, ROOT)
        info = sf.info(audio)
        if (not .5 <= info.duration <= 120 or not 1 <= info.channels <= 8
                or not 8000 <= info.samplerate <= 384000):
            raise ValueError("Audio editing supports recordings from 0.5 to 120 seconds")
        frames = math.ceil(info.frames * 44100 / (info.samplerate * 2048))
        if operation == "infill":
            start, end = payload.get("start_frame"), payload.get("end_frame")
            if (any(isinstance(value, bool) or not isinstance(value, int) for value in (start, end))
                    or not 0 <= start <= end <= frames):
                raise ValueError("Infill boundaries must lie within the source codec frames")
        with self._engine_lock:
            if self._releasing_resources:
                raise RuntimeError("Model resources are being released. Please wait.")
            if not self.capabilities().get("audio_import"):
                raise ValueError("The selected model or runtime does not support audio editing")
            if payload.get("model") and Path(payload["model"]).resolve() != self.checkpoint.resolve():
                raise ValueError("The selected model changed. Please try again.")
            job_id = uuid.uuid4().hex
            folder = self.jobs_root / job_id
            folder.mkdir(parents=True)
            source_audio = folder / "source.wav"
            native = is_gguf(self.checkpoint)
            studio = clone(payload.get("studio") or {})
            if not isinstance(studio, dict):
                raise ValueError("Studio context must be an object")
            for key in ("work_id", "segment_id"):
                studio.setdefault(key, uuid.uuid4().hex)
                record_id(studio[key])
            studio.update(layout="single", result_scope="audio_editor")
            studio.setdefault("edit_source", "upload:" + upload["id"])
            recipe = {
                "schema": 1, "mode": {"revision": "revision", "reword": "tts", "infill": "infill"}[operation],
                "text": text, "direction": direction, "operation": operation,
                "model": str(self.checkpoint), "studio": studio,
                "reference_snapshot": ([] if operation == "infill" else [
                    asdict(VoiceReference(str(source_audio), reference.text, 0))]),
                "codec": {"family": "drama3-dac", "sample_rate": 44100,
                          "samples_per_frame": 2048, "codebooks": 10, "vocabulary": 4096},
                "runtime": {"backend": "gguf" if native else "python"},
                "audio_edit": {"upload_id": upload["id"], "original_transcript": reference.text},
                "seed": payload.get("seed", 123), "max_new_tokens": payload.get("max_new_tokens", 768),
                "max_chunk_chars": payload.get("max_chunk_chars", 160),
                "temperature": payload.get("temperature", 1.0 if native else .7),
                "top_k": payload.get("top_k", 0), "top_p": payload.get("top_p", .9 if native else .7),
                "duration_frames": payload.get("duration_frames"),
                "silence_retry": payload.get("silence_retry", False),
            }
            if automatic_length:
                recipe["max_new_tokens"] = _automatic_generation_limit(recipe)
                recipe["length_policy"] = "automatic-v1"
            if operation == "infill":
                recipe.update(start_frame=start, end_frame=end)
                recipe.pop("direction")
            if native:
                runtime = load_native_runtime(ROOT)
                recipe["runtime"].update(protocol=runtime.get("protocol"),
                                         executable_sha256=runtime.get("sha256"))
            job = {
                "id": job_id, "mode": recipe["mode"], "status": "queued",
                "created_at": time.time(), "updated_at": time.time(), "events": [],
                "output": None, "take_file": None, "error": None, "recipe": recipe, "studio": studio,
                "payload_summary": {"mode": recipe["mode"], "text_length": len(text),
                                    "reference_count": len(recipe["reference_snapshot"])},
            }
            with self._jobs_lock:
                self._jobs[job_id] = job
                self._persist_job_locked(job_id)
            self.studio.register(job)
            self._executor.submit(self._run_uploaded_audio_edit, job_id, reference, audio, source_audio)
        return {"id": job_id}

    def _prepare_uploaded_audio(self, job_id, reference, encoded_audio, source_audio):
        import soundfile as sf
        import torch
        from drama_low_vram.gguf_takes import CODEC_CONFIG
        from drama_low_vram.protocols import validate_codes

        if self.cancelled(job_id):
            raise GenerationCancelled("Audio preparation cancelled")
        clock = self._job_clocks[job_id]
        clock.begin("model_load")
        try:
            engine = self._get_engine(job_id)
            if getattr(engine, "is_gguf", False):
                engine.load(cancelled=lambda: self.cancelled(job_id))
            else:
                engine.load()
        finally:
            clock.end("model_load")
        if self.cancelled(job_id):
            raise GenerationCancelled("Audio preparation cancelled")
        self._event(job_id, {"stage": "prepare", "message": "Encoding uploaded audio"})
        shutil.copyfile(encoded_audio, source_audio)
        digest = hashlib.sha256(b"drama3-uploaded-edit-codec-v1")
        paths = {self.checkpoint.resolve(), Path(getattr(engine, "model_path", self.checkpoint)).resolve()}
        if self.checkpoint.is_dir():
            paths.update(self.checkpoint / name for name in ("codec.pth", "config.json"))
        for path in sorted(paths):
            digest.update(str(path).encode("utf-8"))
            if path.exists():
                stat = path.stat()
                digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
        with source_audio.open("rb") as handle:
            digest.update(hashlib.file_digest(handle, "sha256").digest())
        cache_audio = self.data_root / "audio-edit-cache" / (digest.hexdigest() + ".wav")
        cache = cache_audio.with_suffix(".take.json")
        codes = None
        if cache.is_file():
            try:
                codes = load_conditioning_take(cache)[0].codes
            except (OSError, ValueError, RuntimeError):
                pass
        hit = codes is not None
        if not hit:
            prepared = VoiceReference(str(source_audio), reference.text, 0)
            if getattr(engine, "is_gguf", False):
                engine.prewarm_references([prepared], cancelled=lambda: self.cancelled(job_id))
                artifact = engine._cached_reference_codes(source_audio)
                if not artifact:
                    raise RuntimeError("The audio codec did not return source codes")
                codes = torch.tensor(artifact["vq_codes"], dtype=torch.int64)
            else:
                codes, _ = engine.prepare_reference(source_audio, reference.transcript())
            codes = validate_codes(codes, CODEC_CONFIG, "imported audio")
            if self.cancelled(job_id):
                raise GenerationCancelled("Audio preparation cancelled")
            cache.parent.mkdir(parents=True, exist_ok=True)
            save_native_take(cache_audio, PreviousTake(reference.transcript(), codes))
        if self.cancelled(job_id):
            raise GenerationCancelled("Audio preparation cancelled")
        if getattr(engine, "is_gguf", False):
            remember = getattr(engine, "_remember_reference_codes", None)
            if callable(remember):
                remember(source_audio, {"text": reference.transcript(), "vq_codes": codes.tolist()})
        elif callable(getattr(engine, "cache_reference_codes", None)):
            engine.cache_reference_codes(source_audio, reference.transcript(), codes)
        take_file = save_native_take(
            source_audio, PreviousTake(reference.transcript(), codes),
            metadata={"mode": "audio_import", "spoken_text": reference.text,
                      "model": str(self.checkpoint)},
        )
        with self._jobs_lock:
            self._jobs[job_id]["source_preparation"] = {
                "codec_cache_hit": hit, "take_file": str(take_file),
            }
        return take_file, sf.info(source_audio)

    def _run_uploaded_audio_edit(self, job_id, reference, audio, source_audio):
        with self._jobs_lock:
            job = self._jobs[job_id]
            job["status"] = "running"
            self._job_clocks[job_id] = JobClock(queued_seconds=time.time() - job["created_at"])
            self._persist_job_locked(job_id)
        try:
            take_file, _info = self._prepare_uploaded_audio(job_id, reference, audio, source_audio)
            with self._jobs_lock:
                recipe = self._jobs[job_id]["recipe"]
                if recipe["mode"] == "infill":
                    recipe["source_take"] = str(take_file)
                    count = load_conditioning_take(take_file)[0].codes.shape[1]
                    recipe["start_frame"] = min(count, recipe["start_frame"])
                    recipe["end_frame"] = min(count, recipe["end_frame"])
                elif recipe["mode"] == "revision":
                    recipe["previous_take"] = str(take_file)
                self._persist_job_locked(job_id)
                request = clone(recipe)
            self._run_job(job_id, request)
        except GenerationCancelled:
            self._record_result(job_id, {"status": "cancelled", "output": None, "take_file": None})
        except Exception as exc:
            with self._jobs_lock:
                self._jobs[job_id].update(status="error", error=f"{type(exc).__name__}: {exc}")
                self._persist_job_locked(job_id)
            self._event(job_id, {"stage": "error", "message": str(exc)})
            detail("Uploaded audio edit %s failed", job_id, exc_info=True)
        finally:
            with self._jobs_lock:
                clock = self._job_clocks.pop(job_id, None)
                if clock:
                    clock.finish()
                    self._jobs[job_id]["timing"] = clock.snapshot()
                    self._persist_job_locked(job_id)

    def crop_reference_upload(self, payload):
        """Create a separate reference; never modify the uploaded source."""
        from drama_low_vram.gguf_runtime import reference_wav

        if not isinstance(payload, dict) or not re.fullmatch(
            r"[a-f0-9]{32}", str(payload.get("id", ""))
        ):
            raise ValueError("Invalid reference upload ID")
        folder = _inside(self.uploads_root / payload["id"], self.uploads_root)
        metadata = json.loads((folder / "reference.json").read_text(encoding="utf-8"))
        path = reference_wav(_inside(folder / metadata["audio"], folder), ROOT)
        return self._crop_reference_audio(path, metadata["filename"], payload.get("start"), payload.get("end"))

    def _crop_reference_audio(self, path, filename, start, end):
        import io
        import numpy as np
        import soundfile as sf

        if (isinstance(start, bool) or isinstance(end, bool)
                or not isinstance(start, (int, float)) or not isinstance(end, (int, float))
                or not math.isfinite(start) or not math.isfinite(end)
                or start < 0 or end <= start):
            raise ValueError("Select an ordered audio range")
        if end - start > 120:
            raise ValueError("A reference selection must be no longer than 120 seconds.")
        with sf.SoundFile(str(path)) as source:
            duration = len(source) / source.samplerate
            if (duration <= 0 or not 1 <= source.channels <= 8
                    or not 8000 <= source.samplerate <= 384000):
                raise ValueError("Unsupported reference audio format")
            rate = source.samplerate
            if end > duration + 1 / rate:
                raise ValueError("Selection exceeds the reference duration")
            first, last = math.floor(start * rate), min(len(source), math.ceil(end * rate))
            if last - first < rate * 0.5:
                raise ValueError("Choose at least 0.5 seconds of reference audio")
            if last - first > rate * 120:
                raise ValueError("A reference selection must be no longer than 120 seconds.")
            if (last - first) * source.channels * 4 + 128 > 128 * 1024**2:
                raise ValueError("Selected reference audio exceeds the 128 MiB limit.")
            source.seek(first)
            samples = source.read(last - first, dtype="float32", always_2d=True)
            if not np.isfinite(samples).all():
                raise ValueError("Reference audio contains invalid samples")
        buffer = io.BytesIO()
        # FLOAT preserves the decoded source without another quantization step.
        sf.write(buffer, samples, rate, format="WAV", subtype="FLOAT")
        result = self.add_reference_upload(Path(filename).stem + "-clip.wav", buffer.getvalue())
        result.update(start=first / rate, end=last / rate, duration_seconds=(last - first) / rate,
                      sample_rate=rate, samples=last - first)
        audio = self.uploads_root / result["id"] / "reference.wav"
        result["audio_url"] = "/files/" + "/".join(quote(part) for part in audio.relative_to(ROOT).parts)
        return result

    def start_transcription(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("Transcription payload must be an object")
        if bool(payload.get("upload_id")) == bool(payload.get("voice_id")):
            raise ValueError("Choose one uploaded reference or saved voice to transcribe")
        if payload.get("voice_id"):
            voice = self._voice(payload["voice_id"])
            index = payload.get("reference_index", 0)
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(voice["references"]):
                raise ValueError("Invalid saved reference index")
            audio = Path(voice["references"][index]["audio_path"])
        else:
            upload_id = payload["upload_id"]
            if not isinstance(upload_id, str) or not re.fullmatch(r"[a-f0-9]{32}", upload_id):
                raise ValueError("Invalid reference upload ID")
            folder = _inside(self.uploads_root / upload_id, self.uploads_root)
            metadata = json.loads((folder / "reference.json").read_text(encoding="utf-8"))
            audio = _inside(folder / metadata["audio"], folder)
        self.asr.executor = self._executor
        return self.asr.start(
            audio, language=payload.get("language", "auto"), force=payload.get("force", False),
            local_only=payload.get("local_only", True))

    def _uploaded_references(self, payload: dict):
        uploads = payload.get("uploaded_references", [])
        if not isinstance(uploads, list) or len(uploads) > 32:
            raise ValueError("Uploaded references must be a list of at most 32 clips")
        references = []
        for item in uploads:
            if not isinstance(item, dict) or not re.fullmatch(
                r"[a-f0-9]{32}", str(item.get("id", ""))
            ):
                raise ValueError("Invalid reference upload ID")
            transcript, speaker = item.get("text"), item.get("speaker", 0)
            if not isinstance(transcript, str):
                raise ValueError("Reference transcript is required")
            reference = VoiceReference("", transcript, speaker)
            reference.transcript()
            folder = _inside(self.uploads_root / item["id"], self.uploads_root)
            metadata = json.loads((folder / "reference.json").read_text(encoding="utf-8"))
            audio = _inside(folder / metadata["audio"], folder)
            if not audio.is_file():
                raise ValueError("Reference audio is missing")
            references.append(VoiceReference(str(audio), transcript, speaker))
        return references

    def _references(self, payload: dict):
        references = []
        bindings = payload.get("voice_bindings")
        if bindings is not None:
            if not isinstance(bindings, list) or len(bindings) > 32:
                raise ValueError("Voice bindings must be a list of at most 32 references")
            for item in bindings:
                if not isinstance(item, dict):
                    raise ValueError("Each voice binding must be an object")
                if "voice_id" in item:
                    voice = self._voice(item["voice_id"])
                    index = item.get("reference_index")
                    if index is not None and (
                            isinstance(index, bool) or not isinstance(index, int)
                            or not 0 <= index < len(voice["references"])):
                        raise ValueError("Invalid saved reference index")
                    for position, clip in enumerate(voice["references"]):
                        if index is not None and position != index:
                            continue
                        transcript = (item["text"] if "text" in item and (index is not None or position == 0)
                                      else re.sub(r"^\s*<\|speaker:\d+\|>\s*", "", clip["transcript"], count=1))
                        references.append(VoiceReference(
                            clip["audio_path"], transcript, item.get("speaker", 0)))
                else:
                    references.extend(self._uploaded_references({"uploaded_references": [item]}))
            if len(references) > 32:
                raise ValueError("At most 32 reference clips are supported")
            return references
        identifiers = payload.get("voice_ids", []) or []
        if not isinstance(identifiers, list) or len(identifiers) > 32:
            raise ValueError("Voice IDs must be a list of at most 32 voices")
        for voice_id in identifiers:
            voice = self._voice(voice_id)
            references.extend(VoiceReference(clip["audio_path"], clip["transcript"], int(voice.get("speaker", 0)))
                              for clip in voice["references"])
        references.extend(self._uploaded_references(payload))
        if len(references) > 32:
            raise ValueError("At most 32 reference clips are supported")
        return references

    @staticmethod
    def _public_reference_prewarm(task):
        if task is None:
            return None
        return {
            key: task[key] for key in (
                "id", "status", "stage", "error", "elapsed_seconds", "result")
            if key in task
        }

    @staticmethod
    def _reference_prewarm_fingerprint(checkpoint, references):
        digest = hashlib.sha256(str(checkpoint).casefold().encode("utf-8"))
        for reference in references:
            digest.update(reference.transcript().encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(reference.speaker).encode("ascii"))
            digest.update(b"\0")
            with Path(reference.audio).open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            digest.update(b"\0")
        return digest.hexdigest()

    def _set_reference_prewarm(self, task_id, **values):
        with self._reference_prewarm_lock:
            task = self._reference_prewarm_tasks.get(task_id)
            if task is None:
                return None
            task.update(values)
            return self._public_reference_prewarm(task)

    def cancel_reference_prewarm(self, task_id=None):
        with self._reference_prewarm_lock:
            task = self._reference_prewarm_tasks.get(
                task_id if task_id is not None else self._active_reference_prewarm)
            if task is None:
                return {"status": "idle"}
            if task["status"] not in ("queued", "running"):
                if self._active_reference_prewarm == task["id"]:
                    self._active_reference_prewarm = None
                return {"status": "idle"}
            task["cancel_event"].set()
            future = task.get("future")
            if future is not None and hasattr(future, "cancel") and future.cancel():
                task.update(status="cancelled", stage="cancelled",
                            elapsed_seconds=max(0.0, time.perf_counter() - task["started_at"]))
                if self._active_reference_prewarm == task["id"]:
                    self._active_reference_prewarm = None
            else:
                task["stage"] = "canceling"
            return self._public_reference_prewarm(task)

    def start_reference_prewarm(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("Reference warm-up payload must be an object")
        with self._engine_lock:
            return self._start_reference_prewarm(payload)

    def _start_reference_prewarm(self, payload):
        # Reuse a resident model; restoring a page must never cold-load one.
        if self._engine is None or self._releasing_resources:
            return {"status": "idle"}
        checkpoint = self.checkpoint
        references = self._references(payload)
        if not references:
            return self.cancel_reference_prewarm()
        if is_gguf(checkpoint):
            try:
                protocol = load_native_runtime(ROOT).get("protocol", 1)
            except (OSError, ValueError, RuntimeError):
                protocol = 0
            if protocol < 5:
                self.cancel_reference_prewarm()
                return {"status": "unsupported"}
        fingerprint = self._reference_prewarm_fingerprint(checkpoint, references)
        with self._reference_prewarm_lock:
            current = self._reference_prewarm_tasks.get(self._active_reference_prewarm)
            if (current and current.get("fingerprint") == fingerprint
                    and not current["cancel_event"].is_set()):
                if current["status"] in ("queued", "running"):
                    return self._public_reference_prewarm(current)
                if current["status"] == "ready" and self._engine is not None:
                    if is_gguf(checkpoint):
                        if self._engine.references_prepared(references):
                            current["fingerprint"] = fingerprint
                            return self._public_reference_prewarm(current)
                    elif self._engine.reference_prefix_cached(current.get("cache_key")):
                        return self._public_reference_prewarm(current)
        self.cancel_reference_prewarm()
        task_id = uuid.uuid4().hex
        task = {
            "id": task_id, "fingerprint": fingerprint, "checkpoint": str(checkpoint),
            "references": references, "status": "queued", "stage": "queued",
            "started_at": time.perf_counter(), "cancel_event": threading.Event(),
            "error": None, "result": None,
        }
        with self._reference_prewarm_lock:
            self._reference_prewarm_tasks[task_id] = task
            self._active_reference_prewarm = task_id
            for old_id in list(self._reference_prewarm_tasks)[:-16]:
                if old_id != task_id:
                    self._reference_prewarm_tasks.pop(old_id, None)
            task["future"] = self._executor.submit(self._run_reference_prewarm, task_id)
            return self._public_reference_prewarm(task)

    def reference_prewarm(self, task_id):
        with self._reference_prewarm_lock:
            return self._public_reference_prewarm(self._reference_prewarm_tasks.get(task_id))

    def _run_reference_prewarm(self, task_id):
        with self._reference_prewarm_lock:
            task = self._reference_prewarm_tasks.get(task_id)
            if task is None:
                return
            if task["cancel_event"].is_set():
                task.update(status="cancelled", stage="cancelled",
                            elapsed_seconds=max(0.0, time.perf_counter() - task["started_at"]))
                if self._active_reference_prewarm == task_id:
                    self._active_reference_prewarm = None
                return
            task["status"], task["stage"] = "running", "loading_model"
            checkpoint = Path(task["checkpoint"])
            references = task["references"]
            cancel_event = task["cancel_event"]
        engine = None
        previous_callback = None
        started = time.perf_counter()
        try:
            with self._engine_lock:
                if checkpoint != self.checkpoint or cancel_event.is_set():
                    self._set_reference_prewarm(task_id, status="cancelled", stage="cancelled")
                    return
                engine = self._get_engine(task_id)
            previous_callback = engine.status_callback
            engine.status_callback = lambda event: self._set_reference_prewarm(
                task_id, stage=event.get("stage", "warming_prefix"))
            if is_gguf(checkpoint):
                result = engine.prewarm_references(
                    references, cancelled=cancel_event.is_set)
            else:
                result = engine.prewarm_reference_prefix(
                    references,
                    cancelled=cancel_event.is_set,
                    on_stage=lambda stage: self._set_reference_prewarm(task_id, stage=stage),
                )
            if cancel_event.is_set() or result.get("status") == "cancelled":
                status_value, stage = "cancelled", "cancelled"
            elif result.get("status") == "ready":
                status_value, stage = "ready", "complete"
            else:
                status_value, stage = "skipped", "complete"
            cache_key = result.pop("_cache_key", None)
            self._set_reference_prewarm(
                task_id, status=status_value, stage=stage, result=result,
                cache_key=cache_key,
                elapsed_seconds=max(0.0, time.perf_counter() - started),
            )
        except GenerationCancelled:
            self._set_reference_prewarm(
                task_id, status="cancelled", stage="cancelled",
                elapsed_seconds=max(0.0, time.perf_counter() - started),
            )
        except Exception as exc:
            self._set_reference_prewarm(
                task_id, status="failed", stage="failed", error=f"{type(exc).__name__}: {exc}",
                elapsed_seconds=max(0.0, time.perf_counter() - started),
            )
            detail("Reference warm-up %s failed: %s", task_id, exc, exc_info=True)
        finally:
            if engine is not None:
                engine.status_callback = previous_callback

    def _previous(self, payload: dict, engine):
        sources = payload.get("previous_sources")
        if sources:
            if not isinstance(sources, list) or len(sources) > 32:
                raise ValueError("At most 32 conditioning history sources are supported")
            history = []
            for source in sources:
                path = _inside(Path(source["take_file"]), ROOT)
                take, _, _ = load_conditioning_take(path)
                frames = source["frames"]
                if (isinstance(frames, bool) or not isinstance(frames, int)
                        or not 1 <= frames <= take.codes.shape[1]
                        or source.get("edge") not in ("first", "last")):
                    raise ValueError("Invalid history chunk boundary")
                validate_content(source["text"], "history chunk transcript")
                codes = take.codes[:, :frames] if source["edge"] == "first" else take.codes[:, -frames:]
                history.append(PreviousTake(source["text"], codes, source.get("lookahead")))
            return history
        path_value = payload.get("previous_take")
        if not path_value:
            return []
        path = _inside(Path(path_value), ROOT)
        if getattr(engine, "is_gguf", False) or str(path).lower().endswith(".take.json"):
            take, _, _ = load_conditioning_take(path)
        else:
            take, _, _ = load_take(path, engine.model.config)
        return [take]

    def inspect_take(self, value):
        """Read a saved take for editing without loading GPU model weights."""
        path = _inside(Path(value), ROOT)
        if is_gguf(self.checkpoint) or str(path).lower().endswith(".take.json"):
            take, metadata, _ = load_conditioning_take(path)
        else:
            from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig
            architecture = json.loads((self.checkpoint / "config.json").read_text(encoding="utf-8"))
            if architecture.get("quantization_config", {}).get("quant_method") == "drama3-int4":
                architecture.pop("quantization_config")
            config = FishOmniConfig.from_dict(architecture)
            take, metadata, _ = load_take(path, config)
        frames = take.codes.shape[1]
        audio = path.with_name(
            path.name.removesuffix(".take.npz").removesuffix(".take.json") + ".wav")
        return {
            "path": str(path), "text": metadata.get("spoken_text", take.text), "frames": frames,
            "prompt_text": take.text,
            "sample_rate": 44100, "samples_per_frame": 2048,
            "duration_seconds": frames * 2048 / 44100,
            "audio": str(audio) if audio.is_file() else None,
            "mode": metadata.get("mode"),
        }

    def _progress(self, job_id):
        def callback(event):
            self._event(job_id, event)
        return callback

    @staticmethod
    def _validate_gguf_job(payload):
        mode = payload.get("mode", "tts")
        if mode in ("revision", "edit") and not payload.get("previous_take"):
            raise ValueError("Revision and continuation editing require a previous take")
        if mode == "infill":
            if not payload.get("source_take"):
                raise ValueError("Infilling requires a saved source take")
            if any(payload.get(key) for key in (
                "voice_ids", "uploaded_references", "previous_take", "direction", "language",
            )):
                raise ValueError("Infill uses its source audio; use inline controls in the revised transcript")
            if "start_frame" not in payload or "end_frame" not in payload:
                raise ValueError("Infilling requires start and end frames")

    def _run_job(self, job_id: str, payload: dict):
        if self.cancelled(job_id):
            self._record_result(job_id, {"status": "cancelled", "output": None, "take_file": None})
            with self._jobs_lock:
                self._streams.pop(job_id, None)
            return
        with self._jobs_lock:
            self._jobs[job_id]["status"] = "running"
            self._job_clocks[job_id] = clock = self._job_clocks.get(job_id) or JobClock(
                queued_seconds=time.time() - self._jobs[job_id]["created_at"])
            self._jobs[job_id]["timing"] = clock.snapshot()
            self._persist_job_locked(job_id)
        task = {
            "scene": "Dialogue", "design": "Voice design", "infill": "Audio edit",
            "revision": "Audio revision",
        }.get(payload.get("mode"), "Speech")
        status("%s #%s started.", task, job_id[:6])
        output = self.jobs_root / job_id / "result.wav"
        cancel = lambda: self.cancelled(job_id)
        try:
            with self._engine_lock:
                first_use = self._engine is None
            engine = self._get_engine(job_id)
            # History/infill validation needs the checkpoint config before the
            # workflow builder runs; ordinary TTS loads lazily inside workflows.
            native = getattr(engine, "is_gguf", False)
            if not native:
                clock.begin("model_load")
                try:
                    engine.load()
                finally:
                    clock.end("model_load")
            clock.switch("generation")
            mode = payload.get("mode", "tts")
            snapshot = self.job(job_id).get("recipe", {}).get("reference_snapshot")
            references = ([VoiceReference(**item) for item in snapshot] if snapshot is not None
                          else self._references(payload))
            if first_use and references and mode in ("tts", "audio_stream", "scene"):
                prepare = (getattr(engine, "prewarm_references", None) if native
                           else getattr(engine, "prewarm_reference_prefix", None))
                supported = not native or getattr(engine, "build", {}).get("protocol", 1) >= 5
                if callable(prepare) and supported:
                    clock.begin("initialization")
                    try:
                        self._event(job_id, {"stage": "prepare", "message": "Warming voice reference…"})
                        preparation = {"cancelled": cancel}
                        if not native:
                            preparation["max_new_tokens"] = int(payload.get("max_new_tokens", 768))
                        prepare(references, **preparation)
                    except GenerationCancelled:
                        raise
                    except Exception:
                        detail("Initial reference preparation failed; generation will prepare it on demand",
                               exc_info=True)
                    finally:
                        clock.end("initialization")
            previous = (
                self._previous(payload, engine)
                if mode in ("tts", "revision", "edit", "audio_stream")
                else []
            )
            common = {
                "seed": int(payload.get("seed", 123)),
                "max_new_tokens": int(payload.get("max_new_tokens", 768)),
                "max_chunk_chars": int(payload.get("max_chunk_chars", 160)),
                "direction": str(payload.get("direction", "")),
                "language": payload.get("language") or None,
                "duration_frames": (
                    int(payload["duration_frames"])
                    if payload.get("duration_frames") not in (None, "") else None),
                "references": references,
                "previous_takes": previous,
                "silence_retry": payload.get("silence_retry", False),
                "progress": self._progress(job_id),
                "cancelled": cancel,
            }
            for key in ("temperature", "top_k", "top_p"):
                if key in payload:
                    common[key] = payload[key]
            if payload.get("scene_strategy") == "joint" and mode in ("tts", "revision", "edit"):
                common["direction"] = (
                    f"Scene: {payload.get('scene', '')}\nCharacters:\n"
                    + "\n".join(f"Speaker {role['speaker']}: {role['description']}"
                                for role in payload["characters"])
                    + ("\n" + common["direction"] if common["direction"] else ""))
            workflows = GGUFWorkflows(engine) if native else DramaWorkflows(engine)
            if mode == "scene":
                characters = [
                    Character(str(item["name"]), str(item["description"]), int(item["speaker"]))
                    for item in payload.get("characters", [])
                ]
                lines = [SceneLine(str(item["character"]), str(item["text"]))
                         for item in payload.get("lines", [])]
                session = SceneSession(
                    engine, output, scene=str(payload.get("scene", "")),
                    characters=characters, lines=lines, references=references,
                    direction=str(payload.get("direction", "")),
                    seed=common["seed"], max_new_tokens=common["max_new_tokens"],
                    max_chunk_chars=common["max_chunk_chars"],
                    strategy=payload.get("scene_strategy", "per_line"),
                    silence_retry=common["silence_retry"],
                    **{key: common[key] for key in ("temperature", "top_k", "top_p") if key in common})
                try:
                    for row in session.run(progress=self._progress(job_id), cancelled=cancel):
                        if payload.get("line_ids"):
                            row["line_id"] = payload["line_ids"][row["index"]]
                        with self._jobs_lock:
                            self._jobs[job_id]["result"] = {
                                "status": "running", "strategy": "per_line_tts",
                                "segments": clone(session.results),
                            }
                        self._event(job_id, {"stage": "segment", **row})
                finally:
                    # An exception on a later line must not orphan completed takes.
                    result = session.report
                    if result:
                        with self._jobs_lock:
                            self._jobs[job_id]["result"] = clone(result)
                            self._persist_job_locked(job_id)
            elif mode == "design":
                for key in ("language", "duration_frames"):
                    common.pop(key, None)
                result = workflows.run(
                    "design", str(payload.get("text", "")), output, **common,
                    voice_description=str(payload.get("voice_description", "")),
                    character_name=str(payload.get("character_name", "Voice")),
                    scene=str(payload.get("scene", "")))
            elif mode == "stream":
                with self._jobs_lock:
                    session = self._streams[job_id]
                session.workflows = workflows
                session.options = {
                    "seed": common["seed"],
                    "max_new_tokens": common["max_new_tokens"],
                    "references": references,
                    **{key: common[key] for key in ("temperature", "top_k", "top_p") if key in common},
                }
                if cancel():
                    session.cancel()
                self._event(job_id, {
                    "stage": "waiting_for_text",
                    "message": "Waiting for text",
                })
                delivered_segments = []
                try:
                    for row in session.run(progress=self._progress(job_id)):
                        delivered_segments.append(row)
                        with self._jobs_lock:
                            self._jobs[job_id]["stream_input"]["pending"] = session.pending
                            waiting = not self._jobs[job_id]["stream_input"]["finished"] and session.pending < 2
                        self._event(job_id, {"stage": "segment", **row})
                        if waiting:
                            self._event(job_id, {"stage": "waiting_for_text", "message": "Waiting for text"})
                finally:
                    with self._jobs_lock:
                        self._jobs[job_id]["result"] = {
                            "status": session.status, "segments": delivered_segments,
                            "take_files": [row.get("take_file") for row in delivered_segments],
                        }
                        self._persist_job_locked(job_id)
                complete = session.status == "complete" and bool(delivered_segments)
                exported = session.export(output) if complete else None
                result = {
                    "status": session.status if delivered_segments or session.status != "complete" else "incomplete",
                    "output": str(exported) if exported else None,
                    "segments": delivered_segments,
                    "take_files": [row.get("take_file") for row in delivered_segments],
                    "exported_pcm_matches_delivered_chunks": bool(exported),
                }
            elif mode == "audio_stream":
                if native:
                    result = workflows.run(
                        "tts", str(payload.get("text", "")), output, **common,
                        single_chunk=bool(payload.get("single_chunk", False)),
                        streaming_audio=True, chunk_frames=int(payload.get("chunk_frames", 16)),
                        segment_ready=lambda row: self._event(job_id, {"stage": "audio", **row}))
                    result["mode"] = "audio_stream"
                else:
                    common.pop("silence_retry", None)
                    session = AudioStreamSession(
                        engine, str(payload.get("text", "")), output,
                        references=references, previous_takes=previous,
                        direction=common["direction"], language=common["language"],
                        duration_frames=common["duration_frames"],
                        seed=common["seed"], max_new_tokens=common["max_new_tokens"],
                        chunk_frames=int(payload.get("chunk_frames", 16)),
                        **{key: common[key] for key in ("temperature", "top_k", "top_p") if key in common})
                    for row in session.run(progress=self._progress(job_id), cancelled=cancel):
                        self._event(job_id, {"stage": "audio", **row})
                    result = session.report
            else:
                kwargs = dict(common)
                if payload.get("single_chunk"):
                    kwargs["single_chunk"] = True
                if mode == "infill":
                    for key in (
                        "direction", "language", "duration_frames", "previous_takes",
                    ):
                        kwargs.pop(key, None)
                    source = _inside(Path(payload["source_take"]), ROOT)
                    if native or str(source).lower().endswith(".take.json"):
                        source_take, _, _ = load_conditioning_take(source)
                    else:
                        source_take, _, _ = load_take(source, engine.model.config)
                    kwargs["infill"] = InfillSpan(
                        source_take.codes,
                        int(payload["start_frame"]),
                        int(payload["end_frame"]),
                        int(payload["duration_frames"])
                        if payload.get("duration_frames") not in (None, "") else None)
                result = workflows.run(
                    mode, str(payload.get("text", "")), output, **kwargs)
            self._record_result(job_id, result)
        except GenerationCancelled:
            self._record_result(job_id, {"status": "cancelled", "output": None, "take_file": None})
        except Exception as exc:
            details = f"{type(exc).__name__}: {exc}"
            message = ("Joint dialogue exceeds the native context budget. Shorten references or script, or choose per-line generation."
                       if payload.get("scene_strategy") == "joint" and "required_context=" in details else details)
            with self._jobs_lock:
                self._jobs[job_id]["status"] = "error"
                self._jobs[job_id]["error"] = message
                self._persist_job_locked(job_id)
            self._event(job_id, {
                "stage": "error", "message": message, "technical_error": details,
                "traceback": traceback.format_exc(limit=8),
            })
            detail("Job %s failed", job_id, exc_info=True)
            status("[Error] Job #%s failed: %s", job_id[:6], message)
        finally:
            with self._jobs_lock:
                self._streams.pop(job_id, None)
                clock = self._job_clocks.pop(job_id, None)
                if clock:
                    clock.finish()
                    self._jobs[job_id]["timing"] = clock.snapshot()
                    self._persist_job_locked(job_id)

    def _record_result(self, job_id, result):
        status = result.get("status", "complete")
        with self._jobs_lock:
            job = self._jobs[job_id]
            clock = self._job_clocks.get(job_id)
            if clock:
                clock.finish()
                job["timing"] = clock.snapshot()
            if "segments" not in result and job.get("result", {}).get("segments"):
                result = {**result, "segments": job["result"]["segments"],
                          "take_files": job["result"].get("take_files", [])}
            job["status"] = status
            job["result"] = result
            job["output"] = result.get("output")
            job["take_file"] = result.get("take_file")
            stream = self._streams.get(job_id)
            if stream is not None:
                job["stream_input"]["pending"] = stream.pending
            self._persist_job_locked(job_id)
        self.studio.first_completed(self.job(job_id))
        self._event(job_id, {"stage": status, "output": result.get("output")})
        self._report_result(job_id)

    def _report_result(self, job_id):
        job = self.job(job_id)
        outcome = job["status"]
        if outcome == "complete":
            seconds = job.get("timing", {}).get("generation_seconds")
            if isinstance(seconds, (int, float)):
                status("Job #%s completed. Generation: %.3f s.", job_id[:6], seconds)
            else:
                status("Job #%s completed.", job_id[:6])
        elif outcome == "cancelled":
            status("Job #%s cancelled.", job_id[:6])
        else:
            status("[Warning] Job #%s ended without a complete result (%s).", job_id[:6], outcome)

    def shutdown(self):
        # Wake live text consumers before waiting for the serialized worker.
        # Otherwise a consumer waiting for lookahead can hold the engine forever.
        self.cancel_reference_prewarm()
        with self._jobs_lock:
            active = [job_id for job_id, job in self._jobs.items()
                      if job["status"] in ("queued", "running")]
        for job_id in active:
            self.cancel(job_id)
        self._executor.shutdown(wait=True, cancel_futures=True)
        self.model_installation.close()
        with self._engine_lock:
            if self._engine is not None:
                self._engine.close()
                self._engine = None


STATE = AppState()


def _json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False).encode("utf-8")


def _local_model_picker_request(client_address, host_header, origin, port):
    try:
        remote = ipaddress.ip_address(client_address)
        if isinstance(remote, ipaddress.IPv6Address) and remote.ipv4_mapped:
            remote = remote.ipv4_mapped
        host = urlparse(f"//{host_header}").hostname
        parsed_origin = urlparse(origin or "")
        return (remote.is_loopback and host in {"127.0.0.1", "localhost", "::1"}
                and parsed_origin.scheme == "http" and parsed_origin.hostname == host
                and parsed_origin.port == port)
    except ValueError:
        return False


def _choose_model_directory(initial_directory=None):
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError("The native folder picker is unavailable in this Python runtime") from exc

    initial = Path(str(initial_directory or "")).expanduser()
    if initial.is_file():
        initial = initial.parent
    if not initial.is_dir():
        initial = ROOT / "checkpoints"
    if not initial.is_dir():
        initial = Path.home()

    window = None
    try:
        window = tk.Tk()
        window.withdraw()
        window.attributes("-topmost", True)
        window.update_idletasks()
        selected = filedialog.askdirectory(
            parent=window, title="Select Drama 3 model directory",
            initialdir=str(initial), mustexist=True,
        )
        return str(Path(selected).resolve()) if selected else ""
    except Exception as exc:
        raise RuntimeError(f"Could not open the native folder picker: {exc}") from exc
    finally:
        if window is not None:
            window.destroy()


class Handler(BaseHTTPRequestHandler):
    server_version = "Drama3LocalUI/0.1"

    def log_message(self, fmt, *args):
        detail("HTTP %s %s", self.address_string(), fmt % args)

    def _send(self, status, data: bytes, content_type="application/json; charset=utf-8"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _json(self, status, value):
        self._send(status, _json_bytes(value))

    def _body(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length > 128 * 1024**2:
            raise ValueError("Request body is too large")
        return self.rfile.read(length)

    def _payload(self):
        raw = self._body()
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _multipart_payload(self):
        content_type = self.headers.get("Content-Type", "")
        raw = self._body()
        envelope = (
            f"Content-Type: {content_type}\r\n"
            "MIME-Version: 1.0\r\n\r\n"
        ).encode("utf-8") + raw
        message = BytesParser(policy=policy.default).parsebytes(envelope)
        values = {}
        for part in message.iter_parts():
            disposition = part.get_content_disposition()
            name = part.get_param("name", header="content-disposition")
            if disposition != "form-data" or not name:
                continue
            filename = part.get_filename()
            content = part.get_payload(decode=True) or b""
            values[name] = {
                "filename": filename,
                "data": content,
            } if filename else content.decode(part.get_content_charset() or "utf-8")
        return values

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/":
                return self._file(UI_ROOT / "index.html", "text/html; charset=utf-8")
            if path.startswith("/assets/"):
                return self._file(_inside(UI_ROOT / path.removeprefix("/assets/"), UI_ROOT))
            if path.startswith("/files/"):
                target = _inside(ROOT / path.removeprefix("/files/"), ROOT)
                return self._file(target)
            if path == "/api/instance":
                return self._json(200, {
                    "app": "drama3-studio", "pid": os.getpid(),
                    "instance_key": hashlib.sha256(str(ROOT.resolve()).casefold().encode("utf-8")).hexdigest(),
                })
            if path == "/api/health":
                return self._json(200, {
                    "ok": True, "hardware": STATE.hardware(),
                    "engine_loaded": STATE._engine is not None and (
                        not getattr(STATE._engine, "is_gguf", False) or STATE._engine.loaded),
                    "uptime_seconds": time.time() - STATE.started_at,
                })
            if path == "/api/diagnostics":
                from drama_low_vram.diagnostics import snapshot
                return self._json(200, snapshot(ROOT))
            if path == "/api/diagnostics/export":
                from drama_low_vram.diagnostics import export
                target = export(ROOT, ROOT / "diagnostics/support.zip")
                return self._file(target, "application/zip")
            if path == "/api/capabilities":
                return self._json(200, STATE.capabilities())
            if path == "/api/model":
                return self._json(200, STATE.model_status())
            if path == "/api/model/catalog":
                return self._json(200, {"models": download_catalog()})
            if path == "/api/model/browse":
                return self._json(200, {"available": True})
            if path == "/api/model/task":
                return self._json(200, {"task": STATE.model_installation.snapshot()})
            if path == "/api/voices":
                return self._json(200, {"voices": STATE.list_voices()})
            if path.startswith("/api/reference-prewarm/"):
                task_id = path.removeprefix("/api/reference-prewarm/")
                task = STATE.reference_prewarm(task_id)
                if task is None:
                    return self._json(404, {"error": "Reference warm-up not found"})
                return self._json(200, task)
            if path == "/api/studio":
                return self._json(200, STATE.studio.snapshot())
            if path.startswith("/api/studio/work/") and path.endswith("/jobs"):
                work_id = path.removeprefix("/api/studio/work/").removesuffix("/jobs")
                query = parse_qs(parsed.query)
                return self._json(200, STATE.work_jobs(
                    work_id, offset=int(query.get("offset", [0])[0]), limit=int(query.get("limit", [100])[0])))
            if path.startswith("/api/reference-audio/"):
                return self._json(200, STATE.reference_upload(path.removeprefix("/api/reference-audio/")))
            if path.startswith("/api/transcriptions/"):
                return self._json(200, STATE.asr.get(path.removeprefix("/api/transcriptions/")))
            if path == "/api/jobs":
                return self._json(200, {"jobs": STATE.jobs()})
            if path.startswith("/api/jobs/"):
                job_id = path.removeprefix("/api/jobs/")
                job = STATE.job(job_id)
                if job is None:
                    return self._json(404, {"error": "Job not found"})
                return self._json(200, job)
            return self._json(404, {"error": "Not found"})
        except (ValueError, OSError) as exc:
            return self._json(400, {"error": f"{type(exc).__name__}: {exc}"})

    def _file(self, path: Path, content_type=None):
        if not path.is_file():
            return self._json(404, {"error": "File not found"})
        content_type = content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        size, start, end, status = path.stat().st_size, 0, path.stat().st_size - 1, 200
        requested = self.headers.get("Range")
        if requested is not None:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested.strip())
            valid = match is not None and bool(match.group(1) or match.group(2)) and size > 0
            if valid:
                first, last = match.groups()
                if first:
                    start, end = int(first), min(int(last), size - 1) if last else size - 1
                    valid = 0 <= start <= end < size
                else:
                    count = int(last)
                    valid = count > 0
                    start, end = max(0, size - count), size - 1
            if not valid:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            status = 206
        length = max(0, end - start + 1)
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        try:
            with path.open("rb") as handle:
                handle.seek(start)
                while length:
                    block = handle.read(min(length, 256 * 1024))
                    if not block:
                        break
                    self.wfile.write(block)
                    length -= len(block)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # Browsers close an old range when seeking to another location.
            return

    def do_HEAD(self):
        self.do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/api/jobs":
                job_id = STATE.create_job(self._payload())
                return self._json(202, {"id": job_id})
            if path == "/api/reference-prewarm":
                return self._json(202, STATE.start_reference_prewarm(self._payload()))
            if path == "/api/reference-prewarm/cancel":
                return self._json(202, STATE.cancel_reference_prewarm(self._payload().get("id")))
            if path == "/api/reference-audio":
                form = self._multipart_payload()
                audio = form["audio"]
                if not isinstance(audio, dict):
                    raise ValueError("An audio file is required")
                return self._json(201, STATE.add_reference_upload(
                    audio["filename"], audio["data"]))
            if path == "/api/audio-edit/jobs":
                return self._json(202, STATE.start_uploaded_audio_edit(self._payload()))
            if path == "/api/reference-audio/crop":
                return self._json(201, STATE.crop_reference_upload(self._payload()))
            if path == "/api/reference-audio/from-version":
                return self._json(201, STATE.reference_from_version(self._payload()))
            if path == "/api/voices/from-upload":
                return self._json(201, STATE.save_uploaded_voice(self._payload()))
            if path == "/api/voices/from-references":
                return self._json(201, STATE.save_reference_group(self._payload()))
            if path == "/api/studio/prepare-restore":
                return self._json(200, STATE.prepare_version_restore(self._payload()))
            if path == "/api/studio/context-status":
                return self._json(200, STATE.scene_context_status(self._payload()))
            if path == "/api/studio/draft":
                payload = self._payload()
                return self._json(200, STATE.studio.save_draft(payload["key"], payload["value"]))
            if path == "/api/studio/version":
                return self._json(200, STATE.version(self._payload()))
            if path == "/api/studio/adopt":
                return self._json(200, STATE.adopt_version(self._payload()))
            if path == "/api/studio/export":
                return self._json(201, STATE.export_work(self._payload()))
            if path == "/api/transcriptions":
                return self._json(202, STATE.start_transcription(self._payload()))
            if path == "/api/model/inspect":
                return self._json(200, STATE.model_status(self._payload().get("directory")))
            if path == "/api/model/browse":
                if not _local_model_picker_request(
                    self.client_address[0], self.headers.get("Host", ""),
                    self.headers.get("Origin"), self.server.server_port,
                ):
                    return self._json(403, {"error": "The model folder picker is available only to this local Studio page"})
                return self._json(200, {"directory": _choose_model_directory(
                    self._payload().get("initial_directory"))})
            if path in ("/api/model/roots/add", "/api/model/roots/remove"):
                if not _local_model_picker_request(
                    self.client_address[0], self.headers.get("Host", ""),
                    self.headers.get("Origin"), self.server.server_port,
                ):
                    return self._json(403, {"error": "Model locations can only be changed from this local Studio page"})
                directory = self._payload().get("directory")
                if path.endswith("/add"):
                    return self._json(200, STATE.add_model_root(directory))
                return self._json(200, STATE.remove_model_root(directory))
            if path == "/api/model/select":
                return self._json(200, STATE.select_checkpoint(self._payload()["directory"]))
            if path == "/api/model/release":
                self._payload()
                return self._json(200, STATE.release_resources())
            if path == "/api/model/tasks":
                payload = self._payload()
                model_id = payload.get("model_id")
                if model_id is not None:
                    task = STATE.model_installation.start(
                        payload.get("operation", "download"), source=payload.get("source", "auto"),
                        model_id=model_id)
                    return self._json(202, {"task": task})
                if is_gguf(payload.get("directory", STATE.checkpoint)):
                    raise ValueError("Model downloads are unavailable for this local model; select its path directly")
                task = STATE.model_installation.start(
                    payload["operation"], payload.get("directory", str(STATE.checkpoint)),
                    source=payload.get("source", "auto"))
                return self._json(202, {"task": task})
            if path == "/api/model/cancel":
                return self._json(202, {"task": STATE.model_installation.cancel()})
            if path == "/api/takes/inspect":
                return self._json(200, STATE.inspect_take(self._payload()["path"]))
            if path.startswith("/api/jobs/") and path.endswith("/cancel"):
                job_id = path.removesuffix("/cancel").removeprefix("/api/jobs/")
                STATE.cancel(job_id)
                return self._json(202, {"id": job_id, "cancel_requested": True})
            if path.startswith("/api/jobs/") and path.endswith(("/input", "/finish")):
                job_id, action = path.removeprefix("/api/jobs/").rsplit("/", 1)
                payload = self._payload()
                if action == "input" and "text" not in payload:
                    raise ValueError("A text chunk is required")
                result = STATE.stream_input(
                    job_id, text=payload.get("text") if action == "input" else None,
                    finish=action == "finish")
                return self._json(202, {"id": job_id, **result})
            if path == "/api/voices":
                content_type = self.headers.get("Content-Type", "")
                if content_type.startswith("multipart/form-data"):
                    form = self._multipart_payload()
                    name = form.get("name", "Untitled voice")
                    transcript = form.get("transcript", "")
                    speaker = int(form.get("speaker", "0"))
                    audio_field = form["audio"]
                    data = audio_field["data"]
                    filename = audio_field["filename"] or "reference.wav"
                else:
                    payload = self._payload()
                    name, transcript = payload["name"], payload["transcript"]
                    speaker, filename = int(payload.get("speaker", 0)), payload.get("filename", "reference.wav")
                    data = base64.b64decode(payload["audio_base64"], validate=True)
                result = STATE.add_voice(name, transcript, speaker, filename, data)
                return self._json(201, result)
            return self._json(404, {"error": "Not found"})
        except KeyError as exc:
            return self._json(400, {"error": f"Missing field: {exc.args[0]}"})
        except (ValueError, BufferError, RuntimeError, OSError, json.JSONDecodeError) as exc:
            return self._json(400, {"error": f"{type(exc).__name__}: {exc}"})


class StudioHTTPServer(ThreadingHTTPServer):
    def server_bind(self):
        # Winsock SO_REUSEADDR can otherwise let two copies claim the same port.
        if os.name == "nt":
            self.allow_reuse_address = False
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def handle_error(self, request, client_address):
        error = sys.exc_info()[1]
        detail("HTTP request failed for %s", client_address, exc_info=True)
        if not isinstance(error, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
            status("[Error] A request failed: %s", error)


def serve(httpd, *, open_browser=False):
    with console_session(ROOT):
        try:
            host = httpd.server_address[0]
            host = "127.0.0.1" if host == "0.0.0.0" else host
            url = f"http://{host}:{httpd.server_port}/"
            status("Ready: %s", url)
            status("Model: %s (loads on first generation)", STATE.checkpoint.name)
            status("Detailed log: diagnostics/studio.log")
            status("Keep this window open. Press Ctrl+C to stop.")
            if open_browser:
                threading.Timer(
                    0.35, lambda: webbrowser.open(url)
                ).start()
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            status("Stopping Studio...")
            httpd.server_close()
            STATE.shutdown()
            status("Studio stopped.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8879)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    serve(StudioHTTPServer((args.host, args.port), Handler), open_browser=args.open_browser)


if __name__ == "__main__":
    main()
