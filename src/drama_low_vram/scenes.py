"""Explicit per-line scene scheduling with stable voice anchors and real takes.

This uses complete TTS utterances and supplies exact generated/reference codes.
It does not claim the model emitted native audio separators.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from math import ceil
from pathlib import Path
import time

import torch

from .codec import concatenate_wavs
from .protocols import (
    PreviousTake, SPEAKER_TAG_PATTERN, UNICODE_WHITE_SPACE,
    validate_content, validate_scene_inputs,
)
from .workflows import DramaWorkflows, load_take, save_take
from .sampling import DEFAULT_TOP_K
from .text_chunks import _content_width
from .silence_retry import retry_summary


class SceneSession:
    def __init__(self, engine, output, *, scene, characters, lines, references=(),
                 direction="", seed=123, max_new_tokens=768, max_chunk_chars=160,
                 strategy="per_line", temperature=None, top_k=DEFAULT_TOP_K, top_p=None,
                 silence_retry=False):
        validate_scene_inputs(scene, characters, lines)
        validate_content(direction, "scene direction", empty=True)
        voices = {character.speaker for character in characters}
        if len(references) > 32:
            raise ValueError("At most 32 reference clips are supported")
        for reference in references:
            reference.transcript()
            if reference.speaker not in voices:
                raise ValueError("Reference voice is not assigned to a scene character")
        by_name = {character.name: character for character in characters}
        for line in lines:
            speaker = by_name[line.character].speaker
            if any(int(tag) != speaker for tag in SPEAKER_TAG_PATTERN.findall(line.text)):
                raise ValueError("A line's speaker tag conflicts with its character")
        if (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64
                or isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int)
                or not 1 <= max_new_tokens <= engine.options.codec_max_frames
                or isinstance(max_chunk_chars, bool) or not isinstance(max_chunk_chars, int)
                or max_chunk_chars < 1):
            raise ValueError("Invalid scene generation parameters")
        self.engine, self.output = engine, Path(output).resolve()
        self.scene, self.characters, self.lines = scene, tuple(characters), tuple(lines)
        self.references, self.direction = tuple(references), direction
        self.seed, self.maximum, self.max_chunk_chars = seed, max_new_tokens, max_chunk_chars
        if strategy not in ("per_line", "joint"):
            raise ValueError("Scene strategy must be joint or per_line")
        self.strategy = strategy
        self.sampling = {"top_k": top_k}
        if not isinstance(silence_retry, bool):
            raise ValueError("silence_retry must be a boolean")
        self.sampling["silence_retry"] = silence_retry
        if temperature is not None:
            self.sampling["temperature"] = temperature
        if top_p is not None:
            self.sampling["top_p"] = top_p
        self.status, self.results, self.report, self._started = "waiting", [], None, False

    def _joint_batches(self, by_name):
        """Group complete dialogue lines, keeping no more than three speakers."""
        batches, widths, current, speakers, width = [], [], [], set(), 0
        for line in self.lines:
            character = by_name[line.character]
            text = line.text
            if SPEAKER_TAG_PATTERN.match(text.lstrip(UNICODE_WHITE_SPACE)) is None:
                text = f"<|speaker:{character.speaker}|> {text}"
            line_width = _content_width(line.text)
            new_speaker = character.speaker not in speakers
            if current and (
                width + line_width > self.max_chunk_chars
                or new_speaker and len(speakers) >= 3
            ):
                batches.append(current)
                widths.append(width)
                current, speakers, width = [], set(), 0
            current.append((character.speaker, text))
            speakers.add(character.speaker)
            width += line_width
        if current:
            batches.append(current)
            widths.append(width)
        # A small final turn should not pay another full reference/history prefill.
        if (len(batches) > 1 and widths[-1] < max(1, self.max_chunk_chars // 3)
                and widths[-2] + widths[-1] <= ceil(self.max_chunk_chars * 1.5)
                and len({speaker for batch in batches[-2:] for speaker, _text in batch}) <= 3):
            batches[-2].extend(batches.pop())
        return batches

    def run(self, *, progress=None, cancelled=None):
        if self._started:
            raise RuntimeError("A scene session can only run once")
        self._started = True
        self.output.parent.mkdir(parents=True, exist_ok=True)
        directory = self.output.with_name(self.output.stem + ".lines")
        directory.mkdir(exist_ok=True)
        started, position, first, recent = time.perf_counter(), 0, {}, {}
        first_sources, recent_sources, retry_reports = {}, {}, []
        by_name = {character.name: character for character in self.characters}
        with self.engine._lock:
            try:
                if getattr(self.engine, "is_gguf", False):
                    from .gguf_workflows import GGUFWorkflows
                    workflows = GGUFWorkflows(self.engine)
                else:
                    workflows = DramaWorkflows(self.engine)
                self.status = "generating"
                if self.strategy == "joint":
                    batches = self._joint_batches(by_name)
                    note = f"Scene: {self.scene}\nCharacters:\n" + "\n".join(
                        f"Speaker {role.speaker}: {role.description}" for role in self.characters)
                    if self.direction:
                        note += "\n" + self.direction
                    directory = self.output.with_name(self.output.stem + ".joint")
                    directory.mkdir(exist_ok=True)
                    outputs, takes, frames, history, batch_reports = [], [], [], [], []
                    joint_samples = 0
                    full_text = "\n".join(text for batch in batches for _speaker, text in batch)
                    for batch_index, batch in enumerate(batches):
                        if cancelled and cancelled():
                            self.status = "cancelled"
                            break
                        batch_text = "\n".join(text for _speaker, text in batch)
                        batch_speakers = {speaker for speaker, _text in batch}
                        batch_references = [
                            reference for reference in self.references
                            if reference.speaker in batch_speakers
                        ]

                        def batch_progress(event):
                            if progress:
                                progress({
                                    **event, "chunk": batch_index + 1, "chunks": len(batches),
                                })

                        result = workflows.run(
                            "tts", batch_text,
                            directory / f"joint-{batch_index + 1:05d}.wav",
                            references=batch_references, previous_takes=history,
                            direction=note, seed=(self.seed + batch_index) % 2**64,
                            max_new_tokens=self.maximum, max_chunk_chars=self.max_chunk_chars,
                            single_chunk=True, progress=batch_progress, cancelled=cancelled,
                            **self.sampling)
                        batch_reports.extend(result.get("chunks", []))
                        if result["status"] != "complete":
                            self.status = result["status"]
                            break
                        if getattr(self.engine, "is_gguf", False):
                            take, _, _ = workflows.load_take(result["take_file"])
                        else:
                            take, _, native_frames = load_take(
                                result["take_file"], self.engine.model.config)
                            frames.append(native_frames)
                        outputs.append(result["output"])
                        takes.append(take)
                        joint_samples += result.get("audio", {}).get("samples", 0)
                        history = [takes[0], take] if takes[0] is not take else [take]
                    else:
                        if cancelled and cancelled():
                            self.status = "cancelled"
                        else:
                            concatenate_wavs(outputs, self.output)
                            codes = torch.cat([take.codes for take in takes], dim=1)
                            metadata = {
                                "mode": "scene", "scene_strategy": "joint",
                                "scene": self.scene, "spoken_text": full_text,
                                "direction": self.direction, "complete": True,
                                "chunks": batch_reports,
                            }
                            if getattr(self.engine, "is_gguf", False):
                                from .gguf_takes import save_native_take
                                take_file = save_native_take(
                                    self.output, PreviousTake(full_text, codes),
                                    metadata=metadata, chunks=takes)
                            else:
                                take_file = save_take(
                                    self.output, full_text, codes, torch.cat(frames),
                                    metadata=metadata)
                            self.status = "complete"
                            self.report = {
                                "status": self.status, "strategy": "joint",
                                "native_audio_separators_used": False,
                                "output": str(self.output), "take_file": str(take_file),
                                "scene": self.scene,
                                "characters": [asdict(role) for role in self.characters],
                                "script": [asdict(line) for line in self.lines],
                                "direction": self.direction, "segments": [],
                                "chunks": batch_reports,
                                "silence_retry": retry_summary(batch_reports),
                                "samples": joint_samples,
                            }
                            position = joint_samples
                            self.report["sample_rate"] = 44100
                            self.report["duration_seconds"] = position / 44100
                            self.report["total_seconds"] = time.perf_counter() - started
                    if self.status == "complete":
                        return
                    self.report = {
                        "status": self.status, "strategy": "joint",
                        "native_audio_separators_used": False,
                        "output": None, "take_file": None,
                        "scene": self.scene,
                        "characters": [asdict(role) for role in self.characters],
                        "script": [asdict(line) for line in self.lines],
                        "direction": self.direction, "segments": [], "chunks": batch_reports,
                        "silence_retry": retry_summary(batch_reports),
                        "total_seconds": time.perf_counter() - started,
                    }
                    return
                for index, line in enumerate(self.lines):
                    if cancelled and cancelled():
                        self.status = "cancelled"
                        break
                    character = by_name[line.character]
                    tagged = line.text
                    if SPEAKER_TAG_PATTERN.match(tagged.lstrip(UNICODE_WHITE_SPACE)) is None:
                        tagged = f"<|speaker:{character.speaker}|> {tagged}"
                    note = f"Scene: {self.scene}\nCharacter: {character.name}. {character.description}"
                    if self.direction:
                        note += "\n" + self.direction
                    history = [first[character.speaker]] if character.speaker in first else []
                    if character.speaker in recent and recent[character.speaker] is not first.get(character.speaker):
                        history.append(recent[character.speaker])
                    history_sources = ([dict(first_sources[character.speaker])]
                                       if character.speaker in first_sources else [])
                    if len(history) > 1:
                        history_sources.append(dict(recent_sources[character.speaker]))
                    references = [reference for reference in self.references
                                  if reference.speaker == character.speaker]

                    def events(event):
                        if progress:
                            progress({"line": index + 1, "lines": len(self.lines),
                                      "character": character.name, **event})

                    result = workflows.run(
                        "tts", tagged, directory / f"line-{index + 1:05d}.wav",
                        references=references, previous_takes=history, direction=note,
                        seed=(self.seed + index) % 2**64, max_new_tokens=self.maximum,
                        max_chunk_chars=self.max_chunk_chars, progress=events, cancelled=cancelled,
                        **self.sampling)
                    if result.get("silence_retry"):
                        retry_reports.append({"silence_retry": result["silence_retry"]})
                    if result["status"] != "complete":
                        self.status = result["status"]
                        break
                    if getattr(self.engine, "is_gguf", False):
                        take, _, _ = workflows.load_take(result["take_file"])
                    else:
                        take, _, _ = load_take(result["take_file"], self.engine.model.config)
                    chunks = result["chunks"]
                    first_count, last_count = chunks[0]["frames"], chunks[-1]["frames"]
                    # Anchor actual first/last chunks, with their own actual
                    # transcripts. Never truncate audio under a full transcript.
                    new_anchor = character.speaker not in first
                    if new_anchor:
                        first[character.speaker] = PreviousTake(
                            chunks[0]["text"], take.codes[:, :first_count].clone())
                        first_sources[character.speaker] = {
                            "take_file": result["take_file"], "text": chunks[0]["text"],
                            "frames": first_count, "edge": "first"}
                    recent[character.speaker] = (
                        first[character.speaker] if new_anchor and len(chunks) == 1
                        else PreviousTake(chunks[-1]["text"], take.codes[:, -last_count:].clone()))
                    recent_sources[character.speaker] = {
                        "take_file": result["take_file"], "text": chunks[-1]["text"],
                        "frames": last_count, "edge": "last"}
                    samples = result["audio"]["samples"]
                    row = {
                        "index": index, "character": character.name, "speaker": character.speaker,
                        "text": line.text, "path": result["output"], "take_file": result["take_file"],
                        "start_sample": position, "end_sample": position + samples,
                        "start_seconds": position / 44100, "end_seconds": (position + samples) / 44100,
                        "samples": samples, "sample_rate": 44100,
                        "reference_count": len(references), "history_chunks": len(history),
                        "history_sources": history_sources,
                        "delivered_seconds": time.perf_counter() - started,
                    }
                    position += samples
                    self.results.append(row)
                    yield row
                else:
                    if cancelled and cancelled():
                        self.status = "cancelled"
                    else:
                        concatenate_wavs([row["path"] for row in self.results], self.output)
                        self.status = "complete"
            except GeneratorExit:
                self.status = "cancelled"
                raise
            except BaseException:
                self.status = "error"
                raise
            finally:
                self.report = self.report or {
                    "status": self.status, "strategy": "per_line_tts",
                    "native_audio_separators_used": False,
                    "output": str(self.output) if self.status == "complete" else None,
                    "scene": self.scene, "characters": [asdict(character) for character in self.characters],
                    "script": [asdict(line) for line in self.lines],
                    "references": [asdict(reference) for reference in self.references],
                    "direction": self.direction, "seed": self.seed,
                    "max_new_tokens": self.maximum, "max_chunk_chars": self.max_chunk_chars,
                    "segments": self.results, "sample_rate": 44100, "samples": position,
                    "duration_seconds": position / 44100,
                    "total_seconds": time.perf_counter() - started,
                    "silence_retry": retry_summary(retry_reports),
                }
                self.output.with_suffix(".json").write_text(
                    json.dumps(self.report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
