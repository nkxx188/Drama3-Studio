"""Persistent, serialized native GGUF backend for CLI and local service.

Cancellation terminates and reaps the worker; a later request recreates it.
An interrupted CUDA operation is never reused. Native request validation errors
leave the worker alive; execution failures close it before any retry.
"""

from __future__ import annotations

from dataclasses import replace
from collections import OrderedDict
import json
import hashlib
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import threading
import time
from types import SimpleNamespace
import uuid

import psutil
from .sampling import DEFAULT_TOP_K

from .gguf_resources import Layout, plan_resources
from .runtime_files import load_native_runtime
from .nvidia_policy import policy_for

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "@drama3 "
MAX_AUTOMATIC_CONTEXT = 8192
DIAGNOSTICS = (
    "DRAMA_Q8_ROUTE_TRACE", "DRAMA_Q8_ROUTE_TRACE_REQUEST", "DRAMA_Q8_STEP_RECORD",
    "DRAMA_Q8_STEP_REPLAY", "DRAMA_Q8_RAW_CPU", "DRAMA_Q8_COMPATIBLE_QUANT",
    "DRAMA_GGUF_PIPELINE_WIDTH", "DRAMA_GGUF_PIPELINE_SERIAL", "DRAMA_GGUF_PREFILL_WIDTH",
)


class GenerationCancelled(RuntimeError):
    pass


class NativeFailure(RuntimeError):
    def __init__(self, message, *, recoverable=False):
        super().__init__(message)
        self.recoverable = recoverable


def allocation_failure(message):
    return any(s in str(message).lower() for s in (
        "out of memory", "out-of-memory", "bad_alloc", "cudamemoryallocation",
        "failed to allocate", "allocation failed", "cudahostalloc failed",
        "cannot allocate", "cannot reserve", "cuda error 2",
    ))


def build_environment(build, root=ROOT):
    env = {k.upper(): v for k, v in os.environ.items()}
    temp = root / ".cache/tmp"
    temp.mkdir(parents=True, exist_ok=True)
    env.update(TEMP=str(temp), TMP=str(temp))
    env["PATH"] = build.get("dll_directory", str(Path(build["cuda_toolkit"]) / "bin")) + os.pathsep + env.get("PATH", "")
    for key in (*DIAGNOSTICS, "DRAMA_HOST_GGUF_EMBEDDING", "DRAMA_COMPACT_GGUF_HEAD"):
        env.pop(key, None)
    return env


def native_memory(build, root=ROOT):
    result = subprocess.run([build["executable"], "--probe"], cwd=build["source"],
                            env=build_environment(build, root), capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=30,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    rows = [json.loads(line[len(PREFIX):]) for line in result.stdout.splitlines() if line.startswith(PREFIX)]
    if result.returncode or not rows or not rows[-1].get("available"):
        detail = rows[-1].get("error", "") if rows else result.stderr[-1000:]
        if rows and rows[-1].get("devices") and not rows[-1].get("available"):
            names = ", ".join(device["name"] for device in rows[-1]["devices"])
            detail = f"No compiled RTX target matches the detected GPU(s): {names}. " + detail
        raise RuntimeError(
            f"Native CUDA startup failed (exit {result.returncode}). Check the NVIDIA driver, "
            "runtime DLLs and supported GPU. " + detail)
    return rows[-1]


def reference_wav(path, root=ROOT):
    """Cache only format conversion; native session caches the actual encoding."""
    path = Path(path).resolve()
    if path.suffix.lower() == ".wav":
        return path
    hasher = hashlib.sha256(b"gguf-reference-wav-v1")
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024**2), b""):
            hasher.update(chunk)
    target = root / ".cache/gguf-references" / (hasher.hexdigest() + ".wav")
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.stem + "-" + uuid.uuid4().hex + ".wav")
    try:
        import soundfile as sf
        try:
            info = sf.info(str(path))
            if not 0 < info.duration <= 120:
                raise ValueError("Reference audio must have a duration between zero and 120 seconds")
            samples, rate = sf.read(str(path), dtype="float32", always_2d=True)
            sf.write(str(temporary), samples, rate, subtype="FLOAT")
        except (RuntimeError, sf.LibsndfileError) as error:
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg is None:
                raise ValueError("Convert this reference format to WAV; FFmpeg is unavailable") from error
            subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-y", "-i", str(path),
                            "-c:a", "pcm_f32le", str(temporary)], check=True, capture_output=True,
                           timeout=120, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def reference_content_key(path):
    digest = hashlib.sha256(b"gguf-reference-codes-v1")
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GGUFEngine:
    """One model/session per engine, one request at a time, no CPU expert math."""

    is_gguf = True

    def __init__(self, model, *, root=ROOT, context=1024, storage="auto", cache=None, plan=None,
                 pipeline_width=0, pipeline_serial=False, prefill_width=None, decoder="auto", device=None):
        self.root = Path(root).resolve()
        self.model_path = Path(model).resolve()
        self.context, self.storage, self.cache = context, storage, cache
        if decoder not in ("auto", "cpu", "cuda"):
            raise ValueError("Decoder must be auto, cpu or cuda")
        self.decoder = decoder
        self.device = device
        self.selected_device = 0 if device is None else int(device)
        self.hardware_policy = None
        self.plan = plan
        self._provided_plan = plan is not None
        self._reuse_fallback = False
        if pipeline_width not in (0, 2, 4):
            raise ValueError("Experimental pipeline width must be zero, two or four")
        self.pipeline_width, self.pipeline_serial = pipeline_width, pipeline_serial
        self.prefill_width = prefill_width
        self.status_callback = None
        self._lock = threading.RLock()
        self._process = None
        self._messages = None
        self._reader = None
        self._stderr = None
        self._directory = None
        self._identity = None
        self._count = 0
        self._busy = False
        self._context_resize = False
        self._reference_codes = OrderedDict()
        self._reference_codes_bytes = 0
        self._reference_codes_budget = 64 * 1024**2
        self._reference_codes_lock = threading.RLock()
        self.load_seconds = 0.0
        self.attempts = []
        self.build = load_native_runtime(self.root)
        self.options = SimpleNamespace(codec_max_frames=2048)

    @property
    def loaded(self):
        return self._process is not None and self._process.poll() is None

    def _event(self, **event):
        if self.status_callback:
            self.status_callback(event)

    def _signature(self):
        stat = self.model_path.stat()
        return (str(self.model_path), stat.st_size, stat.st_mtime_ns, self.context,
                self.pipeline_width, self.pipeline_serial, self.prefill_width, self.decoder, self.device)

    @staticmethod
    def _pump(process, messages):
        try:
            for line in process.stdout:
                if line.startswith(PREFIX):
                    try:
                        messages.put(json.loads(line[len(PREFIX):]))
                    except ValueError:
                        messages.put({"event": "error", "message": "Malformed native protocol response"})
        finally:
            messages.put({"event": "exit"})

    def _receive(self, *, cancelled=None, timeout=1800, request_id=None):
        until = time.monotonic() + timeout
        while time.monotonic() < until:
            if cancelled and cancelled():
                self.close()
                raise GenerationCancelled("GGUF generation cancelled")
            try:
                message = self._messages.get(timeout=0.1)
            except queue.Empty:
                continue
            if message["event"] == "exit":
                raise NativeFailure("Native worker exited: " + self._log_tail())
            if request_id is not None and message.get("id") != request_id:
                raise NativeFailure("Native request identity mismatch")
            if message["event"] == "error":
                raise NativeFailure(message.get("message", "Native execution failed"),
                                    recoverable=message.get("recoverable", False))
            return message
        self.close()
        raise NativeFailure("Native worker timed out")

    def _log_tail(self):
        if self._directory is None:
            return ""
        result = []
        for name in ("native.log", "stderr.log"):
            path = self._directory / name
            if path.exists():
                with path.open("rb") as f:
                    f.seek(max(0, path.stat().st_size - 8192))
                    result.append(f.read().decode("utf-8", errors="replace"))
        return "\n".join(result)

    def load(self, cancelled=None):
        with self._lock:
            if self.loaded and self._signature() == self._identity:
                return
            self._event(stage="timing", timing_phase="model_load", timing_action="begin")
            try:
                return self._load(cancelled)
            finally:
                self._event(stage="timing", timing_phase="model_load", timing_action="end")

    def _load(self, cancelled=None):
        with self._lock:
            signature = self._signature()
            if self.loaded and signature == self._identity:
                return
            self.close()
            if (self.plan is None or (not self._provided_plan and not self._reuse_fallback)
                    or (self._identity is not None and signature != self._identity)):
                memory = native_memory(self.build, self.root)
                self.selected_device = memory.get("selected_device", 0) if self.device is None else int(self.device)
                if memory.get("devices"):
                    chosen = next((d for d in memory["devices"] if d["index"] == self.selected_device), None)
                    if not chosen or not chosen["supported"]:
                        raise ValueError("Selected NVIDIA device is unavailable or below SM75")
                    memory = memory | {"free_bytes": chosen["free_bytes"]}
                    self.hardware_policy = policy_for((chosen["major"], chosen["minor"]))
                    if self.pipeline_width and not self.hardware_policy.experimental_pipeline:
                        raise ValueError("Experimental GPU pipeline is currently validated only for SM89; use default decoding")
                self.plan = plan_resources(
                    Layout.read(self.model_path), free_vram=memory["free_bytes"],
                    available_ram=psutil.virtual_memory().available,
                    cores=psutil.cpu_count(logical=False), context=self.context,
                    storage=self.storage, cache=self.cache, decoder=self.decoder)
            self._reuse_fallback = False
            self._identity = signature
            candidates = [self.plan, *self.plan.fallbacks()]
            self.attempts = []
            for index, plan in enumerate(candidates):
                if cancelled and cancelled():
                    raise GenerationCancelled("GGUF load cancelled")
                self.plan = plan
                self._directory = self.root / ".cache/gguf-sessions" / uuid.uuid4().hex
                self._directory.mkdir(parents=True)
                config = self._directory / "config.json"
                config.write_text(json.dumps({
                    "model": str(self.model_path), "threads": plan.threads, "device": self.selected_device,
                    "options": plan.session_options(), "log": str(self._directory / "native.log"),
                }, ensure_ascii=False), encoding="utf-8")
                environment = build_environment(self.build, self.root)
                if plan.layout.compact_head_saved_bytes:
                    environment["DRAMA_COMPACT_GGUF_HEAD"] = "1"
                if plan.host_embedding:
                    environment["DRAMA_HOST_GGUF_EMBEDDING"] = "1"
                if self.pipeline_width:
                    if not plan.layout.validated_q8:
                        raise ValueError("Experimental native pipeline currently requires Q8")
                    environment["DRAMA_GGUF_PIPELINE_WIDTH"] = str(self.pipeline_width)
                    if self.pipeline_serial:
                        environment["DRAMA_GGUF_PIPELINE_SERIAL"] = "1"
                work_width = self.prefill_width if self.prefill_width is not None else plan.prefill_work_width
                if not 1 <= work_width <= plan.cache_experts:
                    raise ValueError("Prefill work width exceeds cache capacity")
                environment["DRAMA_GGUF_PREFILL_WIDTH"] = str(work_width)
                self._event(message="Loading native GGUF model", plan=plan.report())
                self._stderr = (self._directory / "stderr.log").open("wb")
                self._process = subprocess.Popen(
                    [self.build["executable"], str(config)], cwd=self.build["source"],
                    env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
                    text=True, encoding="utf-8", errors="replace", bufsize=1,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                self._messages = queue.Queue()
                self._reader = threading.Thread(target=self._pump, args=(self._process, self._messages), daemon=True)
                self._reader.start()
                try:
                    response = self._receive(cancelled=cancelled)
                    if response.get("event") != "ready" or response.get("protocol") != self.build.get("protocol", 1):
                        raise NativeFailure("Unsupported worker handshake")
                    self.load_seconds = response["load_seconds"]
                    self._context_resize = response.get("context_resize") is True
                    self._count = 0
                    self.attempts.append({"plan": plan.report(), "loaded": True})
                    return
                except NativeFailure as exc:
                    details = str(exc) + "\n" + self._log_tail()
                    self.attempts.append({"plan": plan.report(), "loaded": False, "error": details[-3000:]})
                    self.close()
                    if index + 1 == len(candidates) or not allocation_failure(details):
                        raise
                    self._event(message="Allocation failed; retrying with a smaller resource plan")

    def _grow_context(self, capacity, cancelled=None):
        """Rebuild request caches while retaining the native process and weights."""
        identifier = uuid.uuid4().hex
        old_capacity = self.context
        self._event(stage="timing", timing_phase="initialization", timing_action="begin")
        try:
            self._process.stdin.write(json.dumps({
                "id": identifier, "op": "reserve_context", "capacity": capacity,
            }) + "\n")
            self._process.stdin.flush()
            response = self._receive(cancelled=cancelled, request_id=identifier)
            if response.get("event") != "context_ready" or response.get("capacity") != capacity:
                raise NativeFailure("Invalid native context resize response")
            self.context = capacity
            self.plan = replace(
                self.plan, context_length=capacity,
                workspace_estimate_bytes=self.plan.workspace_estimate_bytes
                + (capacity - old_capacity) * 128 * 1024)
            self._identity = self._signature()
            return {"from": old_capacity, "to": capacity, "seconds": response.get("seconds"),
                    "weights_reloaded": False}
        finally:
            self._event(stage="timing", timing_phase="initialization", timing_action="end")

    def _remember_reference_codes(self, audio_path, artifact):
        model = self.model_path.stat()
        key = (str(self.model_path), model.st_size, model.st_mtime_ns,
               reference_content_key(audio_path))
        value = json.dumps(artifact, ensure_ascii=False, separators=(",", ":"))
        size = len(value.encode("utf-8"))
        if size > self._reference_codes_budget:
            return
        with self._reference_codes_lock:
            previous = self._reference_codes.pop(key, None)
            if previous is not None:
                self._reference_codes_bytes -= previous[1]
            while self._reference_codes and self._reference_codes_bytes + size > self._reference_codes_budget:
                _, (_, removed_size) = self._reference_codes.popitem(last=False)
                self._reference_codes_bytes -= removed_size
            self._reference_codes[key] = (value, size)
            self._reference_codes_bytes += size

    def _cached_reference_codes(self, path):
        model = self.model_path.stat()
        key = (str(self.model_path), model.st_size, model.st_mtime_ns,
               reference_content_key(path))
        with self._reference_codes_lock:
            value = self._reference_codes.get(key)
            if value is None:
                return None
            self._reference_codes.move_to_end(key)
            return json.loads(value[0])

    def references_prepared(self, references):
        paths = [Path(reference.audio).resolve() for reference in references]
        model = self.model_path.stat()
        prefix = (str(self.model_path), model.st_size, model.st_mtime_ns)
        keys = [(*prefix, reference_content_key(path)) for path in paths]
        with self._reference_codes_lock:
            return bool(keys) and all(key in self._reference_codes for key in keys)

    def prewarm_references(self, references, *, cancelled=None):
        """Encode reference audio in the resident worker without generating speech."""
        if self.build.get("protocol", 1) < 5:
            raise RuntimeError("Update the native runtime to protocol 5 for reference prewarming")
        if not references or len(references) > 32:
            raise ValueError("Reference prewarming requires 1..32 audio clips")
        descriptors, audio_paths = [], []
        import soundfile as sf
        for reference in references:
            path = Path(reference.audio).resolve()
            if not path.is_file() or path.stat().st_size > 128 * 1024**2:
                raise ValueError("Reference audio is missing or exceeds 128 MiB")
            audio_path = reference_wav(path, self.root)
            info = sf.info(str(audio_path))
            if info.duration <= 0 or info.duration > 120:
                raise ValueError("Reference audio must have a duration between zero and 120 seconds")
            descriptors.append({"audio_path": str(audio_path), "text": reference.transcript()})
            audio_paths.append(path)

        request = {
            "text": "Reference preparation only.",
            "max_tokens": 1,
            "options": {
                "encode_reference_only": "true",
                "references": json.dumps(descriptors, ensure_ascii=False),
            },
        }
        with self._lock:
            if cancelled is not None and cancelled():
                raise GenerationCancelled("GGUF reference preparation cancelled")
            self.load(cancelled=cancelled)
            reused = self._count > 0
            identifier = uuid.uuid4().hex
            started = time.perf_counter()
            self._event(stage="encoding_reference", reference_count=len(references))
            try:
                self._busy = True
                self._process.stdin.write(json.dumps({
                    "id": identifier, "op": "prewarm_reference", "request": request,
                }, ensure_ascii=False) + "\n")
                self._process.stdin.flush()
                response = self._receive(cancelled=cancelled, request_id=identifier)
                prepared = response.get("references")
                if (response.get("event") != "prewarmed" or not isinstance(prepared, list)
                        or len(prepared) != len(references)):
                    raise NativeFailure("Invalid native reference preparation response")
                for path, reference, artifact_json in zip(audio_paths, references, prepared):
                    try:
                        artifact = json.loads(artifact_json)
                    except (TypeError, ValueError) as error:
                        raise NativeFailure("Native reference preparation returned invalid codes") from error
                    if (not isinstance(artifact, dict)
                            or artifact.get("text") != reference.transcript()
                            or not isinstance(artifact.get("vq_codes"), list)
                            or len(artifact["vq_codes"]) != 10):
                        raise NativeFailure("Native reference preparation returned invalid codes")
                    self._remember_reference_codes(path, artifact)
                self._count += 1
                return {
                    "status": "ready", "references": len(references),
                    "request_seconds": time.perf_counter() - started,
                    "session_reused": reused,
                }
            except (NativeFailure, BrokenPipeError, OSError) as error:
                if not isinstance(error, NativeFailure) or not error.recoverable:
                    self.close()
                raise
            finally:
                self._busy = False

    def generate(self, text, output, *, references=(), seed=123, max_frames=768,
                 temperature=1.0, top_k=DEFAULT_TOP_K, top_p=0.9, chunk_size=160, cancelled=None,
                 previous_takes=(), direction="", language=None, duration_frames=None,
                 task_mode="", lookahead="", infill=None, streaming_audio=False,
                 chunk_frames=16, segment_ready=None, design=None):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Provide nonempty speech text")
        if (any(isinstance(value, bool) or not isinstance(value, int)
                for value in (seed, max_frames, chunk_size, top_k, chunk_frames))
                or not 0 <= seed < 2**32 or not 1 <= max_frames <= 2048
                or not 1 <= chunk_size <= 2000 or not 1 <= chunk_frames <= 2048):
            raise ValueError("Invalid seed, frame limit or text chunk size")
        if not 0 <= temperature <= 2 or not 0 < top_p <= 1 or not 0 <= top_k <= 8193:
            raise ValueError("Invalid sampling parameters")
        from .protocols import PreviousTake, directed_text, validate_content
        validate_content(text, "text")
        if task_mode not in ("", "stream", "infill", "design", "joint"):
            raise ValueError("Unknown native task mode")
        if len(references) > 32 or len(previous_takes) > 32:
            raise ValueError("At most 32 references or previous takes are supported")
        if task_mode or direction or language is not None or duration_frames is not None or previous_takes:
            from .gguf_takes import CODEC_CONFIG, take_descriptor
            validate_content(lookahead, "lookahead", empty=True)
            if task_mode != "stream" and lookahead:
                raise ValueError("Lookahead requires the streaming text task")
            if task_mode == "design":
                from .protocols import design_character
                if references or previous_takes or language or duration_frames or direction or not isinstance(design, dict):
                    raise ValueError("Voice design requires a character without references/history")
                design = design_character(**design)
            elif task_mode == "infill":
                if infill is None or references or previous_takes or streaming_audio:
                    raise ValueError("Infill requires source codes and complete reconstruction without references/history")
                infill.parts(CODEC_CONFIG)
                text = directed_text(text, duration_frames=infill.duration_frames)
            elif infill is not None:
                raise ValueError("An infill span requires the infill task")
            elif task_mode == "stream":
                if direction or language is not None or duration_frames is not None:
                    raise ValueError("Use inline controls in each text stream chunk")
            elif direction or language is not None or duration_frames is not None:
                text = directed_text(text, direction, language=language, duration_frames=duration_frames)
        elif lookahead or infill is not None:
            raise ValueError("Task context requires its matching task mode")
        if task_mode not in ("stream", "design") and not re.match(r"\s*<\|speaker:\d+\|>", text):
            text = "<|speaker:0|> " + text
        if task_mode in ("design", "joint") and self.build.get("protocol", 1) < 4:
            raise RuntimeError("Update the native runtime to protocol 4 for voice design and joint dialogue")
        advanced = bool(task_mode or previous_takes or streaming_audio)
        if advanced and self.build.get("protocol", 1) < 3:
            raise RuntimeError("Update the native runtime to protocol 3 to use Drama 3 advanced workflows")
        output = Path(output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        request = {"text": text, "seed": seed, "max_tokens": max_frames,
                   "temperature": temperature, "top_k": top_k, "top_p": top_p,
                   "options": {"text_chunk_size": str(chunk_size)}}
        if previous_takes:
            request["options"]["previous_takes"] = json.dumps(
                [take_descriptor(take) for take in previous_takes], ensure_ascii=False)
        if task_mode:
            request["options"]["task_mode"] = task_mode
        if design:
            request["options"]["design_character"] = json.dumps(design, ensure_ascii=False)
        if task_mode == "stream":
            request["options"]["lookahead"] = lookahead
        if task_mode == "infill":
            request["options"]["infill"] = json.dumps({
                "source": take_descriptor(PreviousTake(text, infill.source)),
                "start_frame": infill.start, "end_frame": infill.end,
            }, ensure_ascii=False)
        descriptors = []
        for reference in references:
            path = Path(reference.audio).resolve()
            if not path.is_file():
                raise ValueError("Reference audio does not exist")
            if path.stat().st_size > 128 * 1024**2:
                raise ValueError("Reference audio exceeds 128 MiB; provide a shorter reference")
            audio_path = reference_wav(path, self.root)
            import soundfile as sf
            info = sf.info(str(audio_path))
            if info.duration <= 0 or info.duration > 120:
                raise ValueError("Reference audio must have a duration between zero and 120 seconds")
            cached = self._cached_reference_codes(path)
            descriptors.append(
                {"vq_codes": cached["vq_codes"], "text": reference.transcript()}
                if cached else {"audio_path": str(audio_path), "text": reference.transcript()})
        if descriptors:
            request["options"]["references"] = json.dumps(descriptors, ensure_ascii=False)
        with self._lock:
            self.load(cancelled=cancelled)
            if (len(references) > 1 and self.build.get("protocol", 1) >= 5
                    and any("audio_path" in descriptor for descriptor in descriptors)):
                missing = [
                    reference for reference, descriptor in zip(references, descriptors)
                    if "audio_path" in descriptor
                ]
                self.prewarm_references(missing, cancelled=cancelled)
                for reference, descriptor in zip(references, descriptors):
                    if "audio_path" not in descriptor:
                        continue
                    cached = self._cached_reference_codes(reference.audio)
                    if cached:
                        descriptor.pop("audio_path")
                        descriptor["vq_codes"] = cached["vq_codes"]
                request["options"]["references"] = json.dumps(descriptors, ensure_ascii=False)
            reused = self._count > 0
            context_resizes = []
            # The plan has a finite descending chain (mapped, minimum cache,
            # CPU codec). Allow an in-flight OOM to reach its final fallback,
            # rather than spending the only retry changing host storage.
            attempt_limit = 3 + len(list(self.plan.fallbacks()))
            for retry in range(attempt_limit):
                identifier = uuid.uuid4().hex
                envelope = {"id": identifier, "output": str(output), "request": request,
                            "streaming": streaming_audio, "chunk_frames": chunk_frames}
                segments, delivered_samples = [], 0
                started = time.perf_counter()
                log_path = self._directory / "native.log"
                log_offset = log_path.stat().st_size if log_path.exists() else 0
                try:
                    self._busy = True
                    self._process.stdin.write(json.dumps(envelope, ensure_ascii=False) + "\n")
                    self._process.stdin.flush()
                    while True:
                        response = self._receive(cancelled=cancelled, request_id=identifier)
                        if response.get("event") == "result":
                            break
                        if response.get("event") != "audio" or not streaming_audio:
                            raise NativeFailure("Unexpected native response")
                        path = Path(response["path"]).resolve()
                        if (path.parent != output.parent or not path.is_file()
                                or response["index"] != len(segments)
                                or response["start_sample"] != delivered_samples
                                or response["sample_rate"] != 44100 or response["samples"] <= 0):
                            raise NativeFailure("Invalid native audio segment")
                        row = {key: response[key] for key in (
                            "index", "path", "start_sample", "samples", "sample_rate")}
                        row["seconds"] = row["samples"] / row["sample_rate"]
                        delivered_samples += row["samples"]
                        row["end_sample"] = delivered_samples
                        segments.append(row)
                        if segment_ready:
                            segment_ready(dict(row))
                    generation = response["generation"] or {}
                    timings = {}
                    if log_path.is_file():
                        with log_path.open("rb") as log:
                            log.seek(log_offset)
                            for line in log.read().decode("utf-8", errors="replace").splitlines():
                                match = re.match(r"\[TIMING [^]]+\] (drama3\.\S+|session\.wall_ms) (\S+)$", line)
                                if match:
                                    key, value = match.groups()
                                    try:
                                        number = float(value)
                                        timings[key] = (timings.get(key, 0) + number if
                                                        key.endswith("_ms") or key.endswith("_rows") or
                                                        key == "drama3.generated_frames" else number)
                                    except ValueError:
                                        timings[key] = value
                    result = response | {
                        "status": "incomplete" if generation.get("reached_limit", True) else "complete",
                        "complete": not generation.get("reached_limit", True),
                        "take_file": response.get("native_take_file"),
                        "segments": segments,
                        "backend": "native GGUF CUDA",
                        "executable_sha256": self.build.get("sha256"),
                        "device": self.selected_device,
                        "hardware_policy": self.hardware_policy.report() if self.hardware_policy else None,
                        "session_reused": reused, "load_seconds": self.load_seconds,
                        "context_resizes": context_resizes,
                        "client_request_seconds": time.perf_counter() - started,
                        "resource_plan": self.plan.report(), "load_attempts": self.attempts,
                        "experimental_pipeline_width": self.pipeline_width,
                        "requested_waveform_decoder": self.decoder,
                        "pipeline_extra_host_staging_estimate_bytes": (
                            self.pipeline_width * self.plan.layout.slot_bytes // self.plan.layout.layers * 2),
                        "prefill_work_width": self.prefill_width or self.plan.prefill_work_width,
                        "native_log": str(self._directory / "native.log"),
                        "timings": timings,
                    }
                    self._count += 1
                    if not output.is_file() or response.get("output") != str(output):
                        raise NativeFailure("Native result did not produce the requested WAV")
                    if self.build.get("protocol", 1) >= 3:
                        import soundfile as sf
                        info = sf.info(str(output))
                        if info.samplerate != 44100 or info.channels != 1:
                            raise NativeFailure("Native WAV has an invalid audio format")
                        result["audio"] = {"samples": info.frames, "sample_rate": info.samplerate,
                                           "duration_seconds": info.duration}
                        take_path = output.with_suffix(".take.json")
                        if result["take_file"] != str(take_path) or not take_path.is_file():
                            raise NativeFailure("Native result did not produce the requested codec take")
                        if streaming_audio and delivered_samples != info.frames:
                            raise NativeFailure("Streamed samples do not match the final WAV length")
                    output.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                                           encoding="utf-8")
                    return result
                except (NativeFailure, BrokenPipeError, OSError) as exc:
                    # Once a listener has received audio, replaying the request
                    # would deliver duplicates. Leave the partial parts usable.
                    if segments:
                        self.close()
                        raise
                    if isinstance(exc, NativeFailure) and exc.recoverable:
                        required = re.search(r"required_context=(\d+)", str(exc))
                        if required and retry + 1 < attempt_limit:
                            capacity = (int(required[1]) + 511) // 512 * 512
                            if self.context < capacity <= MAX_AUTOMATIC_CONTEXT and not self._provided_plan:
                                if self._context_resize:
                                    self._event(message=f"Growing context to {capacity} while keeping model weights")
                                    try:
                                        context_resizes.append(self._grow_context(capacity, cancelled))
                                        continue
                                    except (NativeFailure, BrokenPipeError, OSError) as resize_error:
                                        details = str(resize_error) + "\n" + self._log_tail()
                                        self.close()
                                        if not allocation_failure(details):
                                            raise
                                        self._event(message="Context allocation failed; retrying a fresh resource plan")
                                self.close()
                                self.context = capacity
                                self.plan = None
                                self._event(message=f"Growing context to {capacity} for this reference/request")
                                self.load(cancelled=cancelled)
                                reused = False
                                continue
                        raise
                    details = str(exc) + "\n" + self._log_tail()
                    candidates = list(self.plan.fallbacks()) if allocation_failure(details) else []
                    self.close()
                    if not candidates or retry + 1 >= attempt_limit:
                        raise
                    # A failed in-flight request is retried from a fresh seed
                    # and empty decoder state, after the failed worker is reaped.
                    self.plan = candidates[0]
                    self._reuse_fallback = True
                    self.load(cancelled=cancelled)
                    reused = False
                except BaseException:
                    self.close()
                    raise
                finally:
                    self._busy = False

    def close(self):
        with self._lock:
            process = self._process
            self._process = None
            if process is not None:
                if process.poll() is None:
                    if not self._busy:
                        try:
                            process.stdin.write('{"id":"close","op":"close"}\n')
                            process.stdin.flush()
                            process.wait(timeout=5)
                        except (OSError, subprocess.TimeoutExpired):
                            process.terminate()
                    else:
                        process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                if process.stdin:
                    process.stdin.close()
                if self._reader:
                    self._reader.join(timeout=2)
                if process.stdout:
                    process.stdout.close()
            if self._stderr:
                self._stderr.close()
                self._stderr = None
