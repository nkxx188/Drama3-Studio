# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp==1.30.0", "httpx>=0.27.1,<1"]
# ///
"""Optional local MCP adapter for an already-running Drama 3 Studio."""

import argparse
import asyncio
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import time
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from script_text import NATIVE_SPEAKER, contextual_span, dialogue_lines, excerpt_spans, replace_span, speaker_label

APP_ID = "drama3-studio"
DEFAULT_URL = "http://127.0.0.1:8879"
UNCONFIRMED_SUBMISSION = "Check Studio's generation history before submitting again."
STUDIO_JOB_ID = r"[0-9a-f]{32}"
JOB_ID = rf"^(?:prep-)?{STUDIO_JOB_ID}$"
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
MAX_AUDIO_SIZE = 128 * 1024**2 - 4096  # Leave room for Studio's multipart body limit.
ASR_TIMEOUT = 15 * 60
CODEC_FRAMES_PER_SECOND = 44100 / 2048
SCRIPT_GUIDE = Path(__file__).with_name("script-prompt.zh-CN.md")
LOGGER = logging.getLogger(__name__)


class NeedsInput(ToolError):
    """The workflow needs user action before it can continue."""


class ReferenceFile(BaseModel):
    audio_path: Annotated[str, Field(description="Absolute path of the local reference audio file.")]
    transcript: Annotated[str | None, Field(description="Reference words; omit to use local ASR.")] = None


VoiceSource = str | tuple[Path, str | None]


def local_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        port = parsed.port
        if (parsed.scheme == "http" and parsed.hostname in ("localhost", "127.0.0.1", "::1")
                and port and parsed.username is None and parsed.password is None
                and parsed.path in ("", "/") and not parsed.query and not parsed.fragment):
            host = "[::1]" if parsed.hostname == "::1" else "127.0.0.1"
            return f"http://{host}:{port}"
    except (TypeError, ValueError):
        pass
    raise ValueError("Studio URL must be a local HTTP address with a port, such as " + DEFAULT_URL)


async def request_json(client: httpx.AsyncClient, path: str, payload: dict | None = None, *, files=None) -> dict:
    submitting = path in ("/api/jobs", "/api/audio-edit/jobs") and payload is not None
    try:
        response = (await client.post(path, json=payload, files=files)
                    if payload is not None or files is not None else await client.get(path))
    except httpx.RequestError as error:
        if submitting:
            raise ToolError(f"Studio did not confirm the submission. {UNCONFIRMED_SUBMISSION}") from error
        raise ToolError(f"Cannot reach Studio at {client.base_url}. Start it and check its address.") from error

    try:
        data = response.json()
    except ValueError:
        data = None
    if not response.is_success:
        if submitting and response.status_code >= 500:
            raise ToolError(f"Studio returned HTTP {response.status_code}. {UNCONFIRMED_SUBMISSION}")
        if path == "/api/instance":
            raise ToolError(f"No compatible Drama 3 Studio found at {client.base_url}.")
        detail = data.get("error") if isinstance(data, dict) else None
        raise ToolError(f"Studio HTTP {response.status_code}: {detail or response.reason_phrase}")
    if not isinstance(data, dict):
        suffix = f" {UNCONFIRMED_SUBMISSION}" if submitting else " Check the Studio address and version."
        raise ToolError(f"Unexpected response from Studio.{suffix}")
    return data


class StudioClient:
    def __init__(self, url: str = DEFAULT_URL, directory: Path | None = None):
        self.url = local_url(url)
        self.directory = directory.resolve() if directory is not None else None
        self.instance_key = (hashlib.sha256(str(self.directory).casefold().encode("utf-8")).hexdigest()
                             if self.directory is not None else None)

    def current_url(self) -> str:
        if self.directory is None:
            return self.url
        try:
            state = json.loads((self.directory / "diagnostics" / "ui-instance.json").read_text(encoding="utf-8"))
            return local_url(state["url"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ToolError(f"Start Studio from {self.directory}. For a manually started server, use --studio-url.") from error

    @asynccontextmanager
    async def connect(self):
        async with httpx.AsyncClient(base_url=self.current_url(), trust_env=False, follow_redirects=False,
                                     timeout=httpx.Timeout(15.0, connect=3.0)) as client:
            identity = await request_json(client, "/api/instance")
            if identity.get("app") != APP_ID:
                raise ToolError(f"No compatible Drama 3 Studio found at {client.base_url}.")
            if self.instance_key is not None and identity.get("instance_key") != self.instance_key:
                raise ToolError(f"The server at {client.base_url} belongs to a different Studio installation. "
                                f"Launch Studio from {self.directory}.")
            yield client


@dataclass
class Preparation:
    progress: dict = field(default_factory=lambda: {"stage": "prepare"})
    result: dict | None = None
    task: asyncio.Task | None = None

    def stage(self, name: str, **detail) -> None:
        self.progress = {"stage": name, **detail}


class Preparations:
    def __init__(self):
        self.jobs: dict[str, Preparation] = {}

    def start(self, work: Callable[[Preparation], Awaitable[dict]]) -> dict:
        if sum(job.result is None for job in self.jobs.values()) >= 8:
            raise ToolError("Too many audio preparations are in progress. Wait for an existing job.")
        if len(self.jobs) >= 64:
            expired = next(key for key, job in self.jobs.items() if job.result is not None)
            del self.jobs[expired]
        identifier = f"prep-{uuid4().hex}"
        job = self.jobs[identifier] = Preparation()
        job.task = asyncio.create_task(self.run(job, work))
        return {"job_id": identifier, "status": "preparing", "progress": job.progress}

    def get(self, identifier: str) -> Preparation:
        preparation = self.jobs.get(identifier)
        if preparation is None:
            raise ToolError("Preparation ID not found in this MCP session. Check Studio history before submitting again.")
        return preparation

    @staticmethod
    async def run(job: Preparation, work: Callable[[Preparation], Awaitable[dict]]) -> None:
        try:
            job.result = await work(job)
        except NeedsInput as error:
            job.result = {"status": "needs_input", "error": str(error)}
        except (ToolError, ValueError) as error:
            job.result = {"status": "error", "error": str(error)}
        except Exception:
            LOGGER.exception("Audio preparation failed")
            job.result = {"status": "error", "error": "Unexpected audio preparation failure. Check the MCP log."}


def nonempty(text: str, label: str) -> str:
    if not text.strip():
        raise ToolError(f"Provide nonempty {label}.")
    return text


def saved_voice(voice_id: str) -> str:
    if not voice_id.strip() or re.search(r"[/\\:]", voice_id) or voice_id in (".", ".."):
        raise ToolError("Use a saved voice ID from list_voices.")
    return voice_id


def audio_file(value: str) -> Path:
    path = Path(value)
    try:
        if (not path.is_absolute() or str(path).startswith(("\\\\", "//"))
                or path.suffix.lower() not in AUDIO_SUFFIXES or ":" in path.name or not path.is_file()):
            raise ToolError("Use an absolute path to an existing local WAV, FLAC, MP3, OGG, or M4A file.")
        if not 32 <= path.stat().st_size <= MAX_AUDIO_SIZE:
            raise ToolError("The audio file is empty or too large for Studio's 128 MiB upload limit.")
    except (OSError, ValueError) as error:
        raise ToolError("Cannot access the specified local audio file.") from error
    return path


def reference_file(source: ReferenceFile) -> tuple[Path, str | None]:
    if source.transcript is not None:
        nonempty(source.transcript, "reference transcript")
    return audio_file(source.audio_path), source.transcript


def response_id(data: dict, label: str) -> str:
    identifier = data.get("id")
    if not isinstance(identifier, str) or not re.fullmatch(STUDIO_JOB_ID, identifier):
        suffix = f" {UNCONFIRMED_SUBMISSION}" if label == "job" else " Check the Studio version."
        raise ToolError(f"Studio did not return a valid {label} ID.{suffix}")
    return identifier


async def selected_model(client: httpx.AsyncClient) -> str:
    model = await request_json(client, "/api/model")
    if model.get("selected_ready") is not True or not model.get("selected_directory"):
        raise ToolError("No ready speech model is selected. Choose an installed model in Studio first.")
    return model["selected_directory"]


async def require_features(client: httpx.AsyncClient, features: tuple[str, ...], task: str) -> None:
    capabilities = await request_json(client, "/api/capabilities")
    if not all(capabilities.get(feature) is True for feature in features):
        raise ToolError(f"The selected Studio model or runtime does not support {task}.")


async def queue_job(client: httpx.AsyncClient, path: str, model: str, payload: dict) -> dict:
    data = await request_json(client, path, {**payload, "model": model, "automatic_length": True,
                                            "silence_retry": True})
    return {"job_id": response_id(data, "job"), "status": "queued"}


async def upload_audio(client: httpx.AsyncClient, path: Path) -> str:
    try:
        with path.open("rb") as stream:
            if not 32 <= os.fstat(stream.fileno()).st_size <= MAX_AUDIO_SIZE:
                raise ToolError("The audio file is empty or too large for Studio's 128 MiB upload limit.")
            data = await request_json(client, "/api/reference-audio", files={"audio": (path.name, stream)})
    except OSError as error:
        raise ToolError("Cannot read the specified local audio file.") from error
    return response_id(data, "upload")


async def crop_audio(client: httpx.AsyncClient, upload_id: str, start: float, end: float) -> str:
    data = await request_json(client, "/api/reference-audio/crop", {"id": upload_id, "start": start, "end": end})
    return response_id(data, "crop")


def codec_span(start: float, end: float) -> tuple[int, int]:
    def snap(seconds):
        frames = seconds * CODEC_FRAMES_PER_SECOND
        nearest = round(frames)
        return nearest if abs(frames - nearest) < 0.00002 else frames

    return math.floor(snap(start)), math.ceil(snap(end))


async def transcribe(client: httpx.AsyncClient, upload_id: str, preparation: Preparation,
                     stage: str, supply: str | None = None) -> dict:
    failure = NeedsInput if supply is not None else ToolError
    manual = f" Provide {supply} manually." if supply is not None else ""
    preparation.stage(stage)
    state = await request_json(client, "/api/transcriptions", {"upload_id": upload_id, "local_only": True})
    asr_id = response_id(state, "transcription")
    deadline = time.monotonic() + ASR_TIMEOUT
    while state.get("status") in ("queued", "running"):
        if time.monotonic() >= deadline:
            raise failure(f"Local transcription did not finish in time.{manual}")
        preparation.stage(stage, asr_stage=state.get("stage") or state["status"])
        await asyncio.sleep(1)
        state = await request_json(client, f"/api/transcriptions/{asr_id}")
    result = state.get("result") or {}
    if state.get("status") == "complete" and isinstance(result.get("text"), str) and result["text"].strip():
        return result
    if state.get("status") == "skipped":
        raise NeedsInput(f"Install the local ASR model in Studio.{manual}")
    reason = state.get("error") or "No text was recognized."
    raise failure(f"Local transcription failed: {reason}{manual}")


async def transcribe_file(studio: StudioClient, preparation: Preparation, path: Path,
                          window: tuple[float, float] | None) -> dict:
    async with studio.connect() as client:
        preparation.stage("upload_audio")
        upload_id = await upload_audio(client, path)
        if window is not None:
            preparation.stage("crop_selection")
            upload_id = await crop_audio(client, upload_id, *window)
        result = await transcribe(client, upload_id, preparation, "transcribe_audio")
        return {"status": "complete", **{key: result[key] for key in ("text", "language", "duration_seconds")
                                        if key in result}}


async def generate(studio: StudioClient, preparation: Preparation, payload: dict, sources: dict[int, VoiceSource]) -> dict:
    async with studio.connect() as client:
        model = await selected_model(client)
        if payload["mode"] == "scene":
            features = ("scene_scheduler",)
            if payload["scene_strategy"] == "joint":
                features += ("joint_dialogue",)
            await require_features(client, features, "the requested dialogue strategy")
        elif payload["mode"] == "design":
            await require_features(client, ("voice_design",), "voice design")
        bindings = []
        for speaker, source in sources.items():
            if isinstance(source, str):
                bindings.append({"voice_id": source, "speaker": speaker})
                continue
            path, transcript = source
            preparation.stage("upload_reference", speaker=f"S{speaker + 1}")
            upload_id = await upload_audio(client, path)
            if transcript is None:
                recognized = await transcribe(client, upload_id, preparation, "transcribe_reference",
                                              f"the reference transcript for S{speaker + 1}")
                transcript = recognized["text"]
            bindings.append({"id": upload_id, "text": transcript, "speaker": speaker})
        preparation.stage("submit")
        return await queue_job(client, "/api/jobs", model, {**payload, "voice_bindings": bindings})


async def range_edit_text(client: httpx.AsyncClient, preparation: Preparation, upload_id: str,
                          original: str, replacement: str, start: float, end: float, duration: float | None,
                          excerpt: str | None, occurrence: int | None) -> str:
    if excerpt is None:
        preparation.stage("crop_selection")
        selection = await crop_audio(client, upload_id, start, end)
        excerpt = (await transcribe(client, selection, preparation, "transcribe_selection", "original_excerpt"))["text"]
    spans = excerpt_spans(original, excerpt)
    if not spans:
        raise NeedsInput(f"Could not match the selected words ({excerpt!r}) to the full transcript. "
                         "Provide the confirmed original_excerpt and correct original_transcript if needed.")
    if occurrence is not None:
        if occurrence > len(spans):
            raise NeedsInput(f"There are only {len(spans)} matching occurrences of the original excerpt.")
        span = spans[occurrence - 1]
    elif len(spans) == 1:
        span = spans[0]
    else:
        # Use one nearby clip to disambiguate. An uncertain text match never submits an edit.
        left = max(0, start - 2)
        right = min(duration, end + 2) if duration is not None else min(120, end + 2)
        span = None
        if left < start or right > end:
            try:
                preparation.stage("crop_context")
                try:
                    clip = await crop_audio(client, upload_id, left, right)
                except ToolError:
                    if duration is not None or left == start or right == end:
                        raise
                    clip = await crop_audio(client, upload_id, left, end)
                context = await transcribe(client, clip, preparation, "transcribe_context", "original_excerpt and occurrence")
                span = contextual_span(original, excerpt, context["text"])
            except ToolError:
                pass
        if span not in spans:
            raise NeedsInput(f"The selected words ({excerpt!r}) occur {len(spans)} times. "
                             "Confirm original_excerpt and its 1-based occurrence in the original transcript.")
    try:
        return replace_span(original, span, replacement)
    except ValueError as error:
        raise NeedsInput(str(error)) from error


async def edit_audio(studio: StudioClient, preparation: Preparation, *, path: Path | None, source_job_id: str | None,
                     original: str | None, replacement: str | None, direction: str,
                     window: tuple[float, float] | None, excerpt: str | None, occurrence: int | None) -> dict:
    async with studio.connect() as client:
        model = await selected_model(client)
        features = ("audio_import", "infill") if window else ("audio_import", "director_revision")
        await require_features(client, features, "the requested audio edit")
        preparation.stage("load_source")
        duration = None
        if path is not None:
            upload_id = await upload_audio(client, path)
        else:
            job = await request_json(client, f"/api/jobs/{source_job_id}")
            if job.get("status") != "complete" or not job.get("output"):
                raise ToolError("Choose a completed Studio job with a WAV result.")
            if job.get("mode") == "scene" or (job.get("recipe") or {}).get("scene_strategy"):
                raise ToolError("For editing a multi-speaker take with its role bindings, use Studio's dialogue editor.")
            source = await request_json(client, "/api/reference-audio/from-version", {"source_version": {"job_id": source_job_id}})
            upload_id = response_id(source, "upload")
            if original is None:
                original = source.get("suggested_text") or None
            result = job.get("result") or {}
            duration = (result.get("audio") or {}).get("duration_seconds", result.get("duration_seconds"))
        if original is None:
            recognized = await transcribe(client, upload_id, preparation, "transcribe_source", "original_transcript")
            original, duration = recognized["text"], recognized.get("duration_seconds")
        speakers = set(NATIVE_SPEAKER.findall(original))
        if len(speakers) > 1:
            raise ToolError("This audio edit tool expects single-speaker audio. Use the dialogue editor for a multi-speaker take.")
        original = nonempty(NATIVE_SPEAKER.sub("", original).strip(), "original transcript")
        payload = {"upload_id": upload_id, "original_transcript": original}
        if window:
            start, end = window
            if duration is not None and end > duration:
                raise ToolError("The edit selection extends past the end of the source audio.")
            text = await range_edit_text(client, preparation, upload_id, original, replacement, start, end,
                                         duration, excerpt, occurrence)
            # Studio uses an outward-rounded, half-open codec frame range.
            first, last = codec_span(start, end)
            payload.update(operation="infill", text=text, start_frame=first, end_frame=max(first + 1, last))
        else:
            text = original if replacement is None else replacement
            if text == original and not direction.strip():
                raise NeedsInput("Provide new wording or a performance direction for the whole recording.")
            payload.update(operation="revision" if text == original else "reword", text=text, direction=direction)
        preparation.stage("submit")
        return await queue_job(client, "/api/audio-edit/jobs", model, payload)


def build_server(studio: StudioClient) -> FastMCP:
    server = FastMCP(
        "Drama 3 Studio",
        log_level="WARNING",
        instructions=(
            "When the user asks for expressive speech, a performed voice audition, or dialogue with a scene, emotion, "
            "or speaking intent, use get_script_guide before preparing the script. Apply its guidance to the inputs; "
            "do not send the guide or its headings as spoken text. Separate overall single-speaker direction from "
            "spoken text. For dialogue use S1/S2 turns with inline tags, not an overall direction. "
            "For verbatim reading or an already prepared script, preserve the wording and existing tags unless "
            "the user asks for changes.\n"
            "Generate speech or dialogue, design a voice, edit a single-speaker recording, or transcribe audio with "
            "local ASR. Use list_voices for saved voice IDs. Use only local audio files the user has authorized. "
            "Pass any returned job ID to get_job until finished. Allow time between checks and do not resubmit "
            "a preparing, queued, or running job. For needs_input, request the missing information or action "
            "instead of guessing."
        ),
    )
    preparations = Preparations()
    read_only = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
    submits = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)

    async def submit(payload: dict, sources: dict[int, VoiceSource]) -> dict:
        if all(isinstance(source, str) for source in sources.values()):
            return await generate(studio, Preparation(), payload, sources)
        return preparations.start(lambda prep: generate(studio, prep, payload, sources))

    @server.tool(title="Read Drama 3 Script Guide", annotations=read_only)
    async def get_script_guide() -> dict[str, str]:
        """Read the full Drama 3 script guide before adapting a performance.

        Includes single-speaker direction, natural-language and paired tags, the Fish Audio S2 tag reference,
        and Studio's S1/S2 dialogue format. Use this as guidance, not as text to send to the speech model.
        """
        try:
            return {"guide": SCRIPT_GUIDE.read_text(encoding="utf-8")}
        except (OSError, UnicodeError) as error:
            raise ToolError("Cannot read script-prompt.zh-CN.md next to server.py.") from error

    @server.tool(title="List Studio Voices", annotations=read_only)
    async def list_voices() -> dict[str, Any]:
        """List saved voice names and IDs for single-speaker generation or dialogue."""
        async with studio.connect() as client:
            data = await request_json(client, "/api/voices")
        voices = data.get("voices")
        if not isinstance(voices, list):
            raise ToolError("Studio did not return a voice list. Check the Studio version.")
        return {"voices": [{"id": voice["id"], "name": voice.get("name") or voice["id"]} for voice in voices]}

    @server.tool(title="Submit Studio Speech", annotations=submits)
    async def submit_speech(
        text: Annotated[str, Field(description="Final text to speak, including any intended inline tags.")],
        voice_id: Annotated[str | None, Field(description="Saved voice ID from list_voices, not its name.")] = None,
        direction: Annotated[str, Field(description="Optional overall performance direction, separate from text.")] = "",
        reference_audio: Annotated[ReferenceFile | None, Field(description="Local reference instead of voice_id.")] = None,
    ) -> dict[str, Any]:
        """Queue single-speaker speech. Choose a saved voice, a local reference, or neither.

        Use get_script_guide first when adapting text to a requested scene, emotion, or delivery.
        Keep overall delivery in direction and local tags in text. Preserve ready-to-read scripts as supplied.
        """
        nonempty(text, "text to speak")
        if voice_id is not None and reference_audio is not None:
            raise ToolError("Choose either voice_id or reference_audio, not both.")
        sources = {}
        if voice_id is not None:
            sources[0] = saved_voice(voice_id)
        elif reference_audio is not None:
            sources[0] = reference_file(reference_audio)
        return await submit({"mode": "tts", "text": text, "direction": direction}, sources)

    @server.tool(title="Submit Studio Voice Design", annotations=submits)
    async def submit_voice_design(
        voice_description: Annotated[str, Field(description="Describe the desired voice itself, such as timbre or pitch.")],
        text: Annotated[str, Field(description="Audition text to speak, including any intended inline tags.")],
        direction: Annotated[str, Field(description="Optional performance direction for these lines, separate from the voice description.")] = "",
    ) -> dict[str, Any]:
        """Design a voice from a description and generate one audition WAV. No reference audio is used or saved.

        Use get_script_guide first if the user requests expressive delivery for the audition text.
        The WAV is a Studio result and is not automatically added to the voice library.
        """
        nonempty(voice_description, "voice description")
        nonempty(text, "audition text")
        return await submit({"mode": "design", "voice_description": voice_description,
                             "text": text, "direction": direction}, {})

    @server.tool(title="Submit Studio Dialogue", annotations=submits)
    async def submit_dialogue(
        script: Annotated[str, Field(description="Final dialogue with S1: or S2: at the start of each turn. Inline tags are kept.")],
        voices: Annotated[dict[str, str | ReferenceFile], Field(description=(
            "Map each used speaker label (S1 through S16) to a saved voice ID or an object with audio_path "
            "and optional transcript. Every used speaker needs a voice."))],
        strategy: Annotated[Literal["joint", "per_line"], Field(description="Joint or turn-by-turn generation.")] = "joint",
    ) -> dict[str, Any]:
        """Queue an S1/S2 dialogue with one assigned voice per speaker. Returns one combined WAV when done.

        Use get_script_guide first when adapting dialogue for a requested scene or performance.
        Keep performance tags within the relevant speaker's lines; do not include an overall direction.
        """
        try:
            lines = dialogue_lines(script)
            bindings = {speaker_label(label): voice for label, voice in voices.items()}
        except ValueError as error:
            raise ToolError(str(error)) from error
        if len(bindings) != len(voices):
            raise ToolError("Assign each speaker exactly once.")
        roles = list(dict.fromkeys(line["character"] for line in lines))
        labels = set(roles)
        missing, unused = labels - bindings.keys(), bindings.keys() - labels
        if missing or unused:
            details = []
            if missing:
                details.append(f"Missing voices: {', '.join(sorted(missing))}.")
            if unused:
                details.append(f"No dialogue for: {', '.join(sorted(unused))}.")
            raise ToolError(" ".join(details))
        characters, sources = [], {}
        for label in roles:
            speaker = int(label[1:]) - 1
            voice = bindings[label]
            characters.append({"name": label, "speaker": speaker, "description": "Natural conversational voice."})
            sources[speaker] = saved_voice(voice) if isinstance(voice, str) else reference_file(voice)
        return await submit({
            "mode": "scene", "scene": "A conversation.", "scene_strategy": strategy,
            "characters": characters, "lines": lines,
            "character_ids": [uuid4().hex for _ in roles], "line_ids": [uuid4().hex for _ in lines],
            "studio": {"work_id": uuid4().hex, "segment_id": uuid4().hex, "layout": "scene"},
        }, sources)

    @server.tool(title="Submit Studio Audio Edit", annotations=submits)
    async def submit_audio_edit(
        text: Annotated[str | None, Field(description=(
            "New words for the selected span, or the full new transcript for a whole-recording edit. "
            "Omit to keep the whole transcript and change the delivery using direction."))] = None,
        audio_path: Annotated[str | None, Field(description="Absolute path of a local single-speaker recording.")] = None,
        source_job_id: Annotated[str | None, Field(pattern=JOB_ID, description=(
            "Completed single-speaker job ID, instead of audio_path."))] = None,
        start_seconds: Annotated[float | None, Field(strict=True, ge=0, lt=120, allow_inf_nan=False)] = None,
        end_seconds: Annotated[float | None, Field(strict=True, gt=0, le=120, allow_inf_nan=False)] = None,
        direction: Annotated[str, Field(description="Whole-recording delivery instructions. Use inline tags for a local edit.")] = "",
        original_transcript: Annotated[str | None, Field(description="Verified full source transcript, if available.")] = None,
        original_excerpt: Annotated[str | None, Field(description="Verified old words in the selected span, if available.")] = None,
        occurrence: Annotated[int | None, Field(strict=True, ge=1, description=(
            "User-confirmed 1-based occurrence of original_excerpt when repeated. Do not guess."))] = None,
    ) -> dict[str, Any]:
        """Edit a single-speaker recording. Provide both time bounds for a local replacement or neither for a whole edit."""
        if (audio_path is None) == (source_job_id is None):
            raise ToolError("Choose exactly one source: audio_path or source_job_id.")
        if (start_seconds is None) != (end_seconds is None):
            raise ToolError("Provide both start_seconds and end_seconds for a local edit.")
        window = None
        if start_seconds is not None:
            if end_seconds <= start_seconds:
                raise ToolError("end_seconds must be greater than start_seconds.")
            if text is None or direction.strip():
                raise ToolError("Provide replacement text for a local edit; use inline tags instead of direction.")
            if original_excerpt is None and end_seconds - start_seconds < 0.5 - 1e-9:
                raise ToolError("Automatic selection transcription needs at least 0.5 seconds. Provide original_excerpt for a shorter range.")
            window = (start_seconds, end_seconds)
        elif original_excerpt is not None or occurrence is not None:
            raise ToolError("original_excerpt and occurrence apply only to a timed replacement.")
        elif text is None and not direction.strip():
            raise ToolError("Provide new text or direction for a whole-recording edit.")
        if occurrence is not None and original_excerpt is None:
            raise ToolError("Provide the confirmed original_excerpt with its occurrence.")
        if original_transcript is not None:
            nonempty(original_transcript, "original transcript")
        if original_excerpt is not None:
            nonempty(original_excerpt, "original excerpt")
        if text is not None:
            nonempty(text, "new text")
            if NATIVE_SPEAKER.search(text):
                raise ToolError("The new text is for one speaker; do not include internal speaker tags.")
        if source_job_id is not None and source_job_id.startswith("prep-"):
            source = preparations.get(source_job_id)
            source_job_id = (source.result or {}).get("job_id")
            if source_job_id is None:
                raise ToolError("Check get_job and choose a completed audio source.")
        path = audio_file(audio_path) if audio_path is not None else None
        return preparations.start(lambda prep: edit_audio(
            studio, prep, path=path, source_job_id=source_job_id, original=original_transcript,
            replacement=text, direction=direction, window=window, excerpt=original_excerpt, occurrence=occurrence))

    @server.tool(title="Transcribe Local Audio", annotations=submits)
    async def transcribe_audio(
        audio_path: Annotated[str, Field(description="Absolute path of the local audio file to transcribe.")],
        start_seconds: Annotated[float | None, Field(strict=True, ge=0, allow_inf_nan=False,
                                                   description="Optional selection start; provide both time bounds.")] = None,
        end_seconds: Annotated[float | None, Field(strict=True, gt=0, allow_inf_nan=False,
                                                 description="Optional selection end; provide both time bounds.")] = None,
    ) -> dict[str, Any]:
        """Transcribe all or part of a local audio file. Use get_job to retrieve the recognized text.

        The input to ASR must be 0.5 to 120 seconds. A selection can be clipped out of a longer WAV recording.
        Requires Studio's installed local ASR, but no speech generation model. Does not edit the source file.
        """
        if (start_seconds is None) != (end_seconds is None):
            raise ToolError("Provide both start_seconds and end_seconds, or neither to transcribe the whole recording.")
        window = None
        if start_seconds is not None:
            if not 0.5 - 1e-9 <= end_seconds - start_seconds <= 120:
                raise ToolError("Choose an ordered selection between 0.5 and 120 seconds long.")
            window = (start_seconds, end_seconds)
        path = audio_file(audio_path)
        return preparations.start(lambda prep: transcribe_file(studio, prep, path, window))

    @server.tool(title="Get Studio Job Status", annotations=read_only)
    async def get_job(
        job_id: Annotated[str, Field(pattern=JOB_ID, description="Any job ID returned by a submission tool.")],
    ) -> dict[str, Any]:
        """Check preparation, generation, or transcription once. Returns a local WAV path or recognized text when complete."""
        studio_job_id = job_id
        if job_id.startswith("prep-"):
            preparation = preparations.get(job_id)
            if preparation.result is None:
                return {"job_id": job_id, "status": "preparing", "progress": preparation.progress}
            studio_job_id = preparation.result.get("job_id")
            if studio_job_id is None:
                return {"job_id": job_id, **preparation.result}
        async with studio.connect() as client:
            job = await request_json(client, f"/api/jobs/{studio_job_id}")
        status = job.get("status")
        if not isinstance(status, str):
            raise ToolError("Studio did not return a job status. Check the Studio version.")
        response = {"job_id": job_id, "status": status}
        if studio_job_id != job_id:
            response["studio_job_id"] = studio_job_id
        result = job.get("result") or {}
        if status in ("queued", "running"):
            event = (job.get("events") or [{}])[-1]
            keys = ("stage", "line", "lines", "chunk", "chunks", "retry", "max_retries")
            progress = {key: event[key] for key in keys if key in event}
            phase = (job.get("timing") or {}).get("phase")
            if phase:
                progress["phase"] = phase
            if progress:
                response["progress"] = progress
        elif status == "complete":
            output = job.get("output") or result.get("output")
            if not isinstance(output, str) or not Path(output).is_absolute() or Path(output).suffix.lower() != ".wav":
                raise ToolError("Studio marked the job complete but did not return a local WAV path.")
            response["wav_path"] = output
            duration = (result.get("audio") or {}).get("duration_seconds", result.get("duration_seconds"))
            if duration is not None:
                response["duration_seconds"] = duration
        elif job.get("error"):
            response["error"] = job["error"]
        elif status == "incomplete" and (result.get("silence_retry") or {}).get("exhausted"):
            response["error"] = "Long silence remained after the automatic retries."
        return response

    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--studio-dir", type=Path, help="Studio installation folder; follows its launcher port")
    target.add_argument("--studio-url", type=local_url, default=DEFAULT_URL, help="Explicit local Studio URL")
    args = parser.parse_args()
    build_server(StudioClient(args.studio_url, args.studio_dir)).run(transport="stdio")


if __name__ == "__main__":
    main()
