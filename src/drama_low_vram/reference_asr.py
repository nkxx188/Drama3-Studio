"""Optional local Qwen3-ASR reference transcription, independent of ComfyUI.

Uses the Qwen-maintained native Transformers checkpoint and processor. Only
reference preparation uses this model; TTS generation never retranscribes.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid

from .studio_logging import detail, status

MODEL_ID = "Qwen/Qwen3-ASR-0.6B-hf"
REVISION = "7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c"
MODEL_FILES = (
    "config.json", "generation_config.json", "processor_config.json",
    "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "model.safetensors",
)
LANGUAGES = {
    "auto": None, "zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean",
    "yue": "Cantonese", "ar": "Arabic", "de": "German", "fr": "French", "es": "Spanish",
    "pt": "Portuguese", "id": "Indonesian", "it": "Italian", "ru": "Russian", "th": "Thai",
    "vi": "Vietnamese", "tr": "Turkish", "hi": "Hindi", "ms": "Malay", "nl": "Dutch",
    "sv": "Swedish", "da": "Danish", "fi": "Finnish", "pl": "Polish", "cs": "Czech",
    "fil": "Filipino", "fa": "Persian", "el": "Greek", "hu": "Hungarian",
    "mk": "Macedonian", "ro": "Romanian",
}


class LocalASRUnavailable(Exception):
    """Automatic reference preparation must not download missing weights."""


def model_available(directory: Path):
    return all((directory / name).is_file() for name in MODEL_FILES)


def ensure_model(directory: Path, root: Path, progress, *, local_only=False):
    missing = [name for name in MODEL_FILES if not (directory / name).is_file()]
    if missing:
        if local_only:
            raise LocalASRUnavailable()
        progress("downloading")
        from huggingface_hub import snapshot_download

        snapshot_download(
            MODEL_ID, revision=REVISION, allow_patterns=list(MODEL_FILES),
            local_dir=directory, cache_dir=root / ".cache/huggingface/hub",
        )
    if any(not (directory / name).is_file() for name in MODEL_FILES):
        raise RuntimeError("Qwen3-ASR model files are incomplete")


def read_reference(audio: Path, root: Path):
    import numpy as np
    import soundfile as sf
    from .gguf_runtime import reference_wav

    converted = reference_wav(audio, root)
    info = sf.info(str(converted))
    if not 0.5 <= info.duration <= 120:
        raise ValueError("Transcription needs reference audio between 0.5 and 120 seconds")
    samples, rate = sf.read(str(converted), dtype="float32", always_2d=True)
    samples = samples.mean(axis=1)
    if not np.isfinite(samples).all():
        raise ValueError("Reference audio contains invalid samples")
    if rate != 16000:
        import librosa
        samples = librosa.resample(samples, orig_sr=rate, target_sr=16000)
    peak = float(np.max(np.abs(samples)))
    if peak > 1:
        samples /= peak
    return samples, info.duration


def _recognize(samples, model_dir, device, language, progress):
    import numpy as np
    import torch
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

    dtype = (torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16) if (
        device.startswith("cuda")
    ) else torch.float32
    model = processor = inputs = output = None
    peak_before = None
    started = time.perf_counter()
    try:
        progress("loading")
        processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        model = Qwen3ASRForConditionalGeneration.from_pretrained(
            model_dir, dtype=dtype, attn_implementation="sdpa", local_files_only=True,
        ).to(device).eval()
        if device.startswith("cuda"):
            peak_before = torch.cuda.memory_allocated(device)
        loaded_at = time.perf_counter()
        texts, languages = [], []
        # Bound attention workspace and output length for long references.
        for start in range(0, len(samples), 30 * 16000):
            part = samples[start:start + 30 * 16000]
            if len(part) < 8000:
                part = np.pad(part, (0, 8000 - len(part)))
            progress("transcribing")
            inputs = processor.apply_transcription_request(
                audio=part, language=language,
            ).to(device, dtype)
            limit = min(1024, max(256, math.ceil(len(part) / 16000 * 20)))
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=limit, do_sample=False)
            generated = output[:, inputs["input_ids"].shape[1]:]
            if generated.shape[1] >= limit and int(generated[0, -1]) not in (
                model.config.eos_token_id if isinstance(model.config.eos_token_id, (tuple, list))
                else [model.config.eos_token_id]
            ):
                raise RuntimeError("Transcription reached its length limit; shorten the reference")
            parsed = processor.decode(generated, return_format="parsed")
            if isinstance(parsed, list):
                parsed = parsed[0]
            text = parsed["transcription"].strip()
            found = parsed.get("language") or language or ""
            if text:
                texts.append(text)
                if found and found not in languages:
                    languages.append(found)
            inputs = output = generated = None
        result = " ".join(texts).strip()
        if not result:
            raise ValueError("No speech was recognized. Enter the transcript manually or try another clip.")
        return {
            "text": result, "language": ", ".join(languages), "device": device,
            "model": MODEL_ID, "model_weight_bytes": sum(
                p.numel() * p.element_size() for p in model.parameters()),
            "gpu_model_allocated_bytes": peak_before,
            "load_seconds": loaded_at - started,
            "inference_seconds": time.perf_counter() - loaded_at,
        }
    finally:
        model = processor = inputs = output = None
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def transcribe_audio(audio, model_dir, root, language, progress, *, local_only=False):
    import torch

    ensure_model(model_dir, root, progress, local_only=local_only)
    samples, duration = read_reference(audio, root)
    device = "cpu"
    if torch.cuda.is_available():
        free = [(torch.cuda.mem_get_info(i)[0], i) for i in range(torch.cuda.device_count())]
        device = f"cuda:{max(free)[1]}"
    gpu_failed = False
    try:
        result = _recognize(samples, model_dir, device, language, progress)
    except torch.cuda.OutOfMemoryError:
        gpu_failed = True
    if gpu_failed:
        # A resident TTS engine is not unloaded just to transcribe a new clip.
        # This fallback is only used after an actual GPU allocation failure.
        gc.collect()
        torch.cuda.empty_cache()
        progress("cpu_fallback")
        result = _recognize(samples, model_dir, "cpu", language, progress)
        result["fallback"] = "gpu_out_of_memory"
    result["duration_seconds"] = duration
    return result


class ReferenceTranscriber:
    """Persistent content cache and tasks on the host's serialized work queue."""

    def __init__(self, root, data_root, executor, runner=None):
        self.root, self.data_root = Path(root), Path(data_root)
        self.cache = self.data_root / "transcripts"
        self.cache.mkdir(exist_ok=True)
        self.model_dir = Path(os.environ.get(
            "DRAMA_ASR_MODEL_DIR", str(self.root / "checkpoints/Qwen3-ASR-0.6B-hf")))
        self.executor, self.runner = executor, runner or transcribe_audio
        self._lock = threading.RLock()
        self._tasks, self._pending = {}, {}

    def start(self, audio, *, language="auto", force=False, local_only=False):
        if (not isinstance(force, bool) or not isinstance(local_only, bool)
                or not isinstance(language, str) or language not in LANGUAGES):
            raise ValueError("Invalid transcription language or retry flag")
        audio = Path(audio)
        hasher = hashlib.sha256()
        with audio.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024**2), b""):
                hasher.update(chunk)
        digest = hashlib.sha256(
            f"qwen-asr-v1:{REVISION}:{language}:{hasher.hexdigest()}".encode()).hexdigest()
        cached = self.cache / f"{digest}.json"
        with self._lock:
            task = {
                "id": uuid.uuid4().hex, "status": "queued", "stage": "queued",
                "model": MODEL_ID, "cache_hit": False,
            }
            self._tasks[task["id"]] = task
            if cached.is_file() and not force:
                try:
                    result = json.loads(cached.read_text(encoding="utf-8"))
                    if (not isinstance(result, dict) or not isinstance(result.get("text"), str)
                            or not result["text"].strip()):
                        raise ValueError("Empty cached transcript")
                    task.update(status="complete", stage="complete", result=result, cache_hit=True)
                    return self.get(task["id"])
                except (OSError, ValueError, TypeError):
                    pass
            if local_only and not model_available(self.model_dir):
                task.update(status="skipped", stage="skipped", reason="model_unavailable")
                return self.get(task["id"])
            pending = self._pending.get(digest)
            if pending is not None:
                self._tasks.pop(task["id"])
                return self.get(pending)
            self._pending[digest] = task["id"]
            self.executor.submit(self._run, task["id"], digest, audio, language, cached, local_only)
            return self.get(task["id"])

    def get(self, task_id):
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise ValueError("Transcription task not found")
            return json.loads(json.dumps(task))

    def _run(self, task_id, digest, audio, language, cached, local_only):
        started = time.perf_counter()
        status("Transcribing reference audio...")

        def progress(stage):
            with self._lock:
                self._tasks[task_id].update(status="running", stage=stage)

        try:
            progress("preparing")
            result = self.runner(
                audio, self.model_dir, self.root, LANGUAGES[language], progress,
                local_only=local_only)
            if not isinstance(result.get("text"), str) or not result["text"].strip():
                raise ValueError("No speech was recognized. Enter the transcript manually or try another clip.")
            result["seconds"] = time.perf_counter() - started
            temporary = cached.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            os.replace(temporary, cached)
            with self._lock:
                self._tasks[task_id].update(status="complete", stage="complete", result=result)
            status("Reference transcript ready.")
        except LocalASRUnavailable:
            with self._lock:
                self._tasks[task_id].update(
                    status="skipped", stage="skipped", reason="model_unavailable")
        except Exception as exc:
            detail("Reference transcription %s failed", task_id, exc_info=True)
            status("[Error] Reference transcription failed: %s", exc)
            with self._lock:
                self._tasks[task_id].update(
                    status="error", stage="error", error=f"{type(exc).__name__}: {exc}")
        finally:
            with self._lock:
                self._pending.pop(digest, None)
