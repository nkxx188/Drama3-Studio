"""Serialized model workflows, shared by CLI/research and the future interface."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict
import hashlib
import json
import os
import zipfile
from pathlib import Path
import time

import numpy as np
import torch

from .codec import write_wav
from .engine import seed_sampling
from .generation import generate
from .protocols import (
    Character, InfillSpan, PreviousTake, SceneLine, VoiceReference,
    LINE_SEPARATOR, build_infill_prompt, build_scene_prompt, build_stream_prompt,
    build_tts_prompt, build_design_prompt, design_character, directed_text, validate_codes, validate_content,
)
from .text_chunks import split_text
from .sampling import DEFAULT_TOP_K


def save_take(output, text, codes, frames, *, metadata=None):
    """Data-only NumPy sidecar; no Python pickle or additional model bank."""
    output = Path(output)
    target = output.with_suffix(".take.npz")
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle, text=np.asarray(text), codes=codes.cpu().numpy(),
            frames=frames.cpu().numpy(),
            metadata=np.asarray(json.dumps(metadata or {}, ensure_ascii=False)),
        )
    os.replace(temporary, target)
    return target


def load_take(path, config):
    path = Path(path)
    if path.stat().st_size > 128 * 1024**2:
        raise ValueError("Take sidecar is too large")
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if (len(entries) != 4 or len({entry.filename for entry in entries}) != 4
                or any(entry.file_size > 64 * 1024**2 for entry in entries)
                or sum(entry.file_size for entry in entries) > 128 * 1024**2):
            raise ValueError("Take archive exceeds the supported expanded size")
    with np.load(path, allow_pickle=False) as values:
        if set(values.files) != {"text", "codes", "frames", "metadata"}:
            raise ValueError("Unrecognized take sidecar fields")
        if values["codes"].nbytes > 64 * 1024**2 or values["frames"].nbytes > 64 * 1024**2:
            raise ValueError("Take arrays exceed the supported size")
        text = str(values["text"].item())
        codes = validate_codes(torch.from_numpy(values["codes"].copy()), config, "saved take")
        metadata = json.loads(str(values["metadata"].item()))
        frames = torch.from_numpy(values["frames"].copy())
    if not isinstance(metadata, dict):
        raise ValueError("Saved take metadata must be an object")
    if (frames.ndim != 2 or frames.dtype != torch.int64
            or frames.shape[1] != 1 + config.audio_decoder.num_codebooks):
        raise ValueError("Saved take frames have invalid shape or dtype")
    validate_content(text, "saved take transcript")
    lookahead = metadata.get("lookahead") if metadata.get("mode") == "stream" else None
    if lookahead is not None:
        validate_content(lookahead, "saved lookahead", empty=True)
    return PreviousTake(text, codes, lookahead), metadata, frames


def audio_segments(frames, config, separator_id=None):
    """Control rows consume model positions but never become codec frames."""
    result, start = [], 0
    positions = []
    for index, row in enumerate(frames):
        main = int(row[0])
        if separator_id is not None and main == separator_id:
            if index == start:
                raise ValueError("The model emitted an empty audio segment")
            result.append(frames[start:index, 1:].T.contiguous())
            positions.append(index)
            start = index + 1
        elif not config.semantic_start_token_id <= main <= config.semantic_end_token_id:
            raise ValueError("Unexpected non-audio output token")
        elif int(row[1]) != main - config.semantic_start_token_id:
            raise ValueError("Semantic token and code zero disagree")
    if len(frames) > start:
        result.append(frames[start:, 1:].T.contiguous())
    elif len(frames):
        raise ValueError("The model ended with an empty audio segment")
    return result, positions


class DramaWorkflows:
    def __init__(self, engine):
        self.engine = engine

    def run(
        self, mode: str, text: str, output: Path, *,
        references: Sequence[VoiceReference] = (),
        previous_takes: Sequence[PreviousTake] = (),
        direction: str = "", language: str | None = None, duration_frames: int | None = None,
        infill: InfillSpan | None = None,
        scene: str = "", characters: Sequence[Character] = (), lines: Sequence[SceneLine] = (),
        lookahead: str = "", seed: int = 123, max_new_tokens: int = 768,
        max_chunk_chars: int = 160, temperature: float = 0.7, top_k: int = DEFAULT_TOP_K, top_p: float = 0.7,
        progress: Callable[[dict], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
        segment_ready: Callable[[dict], None] | None = None,
        single_chunk: bool = False, voice_description: str = "", character_name: str = "Voice",
    ):
        if mode not in ("tts", "revision", "edit", "infill", "scene", "stream", "design"):
            raise ValueError("Unknown Drama3 workflow")
        if mode == "design":
            design_character(character_name, voice_description, scene)
            if references or previous_takes:
                raise ValueError("Voice design does not use reference audio or previous takes")
        elif voice_description:
            raise ValueError("Voice description requires the design workflow")
        if mode != "scene":
            validate_content(text, "text")
        if mode in ("revision", "edit") and not previous_takes:
            raise ValueError("Revision and continuation editing require a previous take")
        if mode == "infill" and infill is None:
            raise ValueError("Infilling requires source codes and an edit window")
        if mode in ("scene", "stream", "infill") and (
            direction or language is not None or duration_frames is not None
        ):
            raise ValueError(
                "Use inline controls in this task's text; infill duration belongs to InfillSpan")
        if mode in ("scene", "infill") and references:
            raise ValueError(
                "This native task protocol has no reference blocks; use TTS for reference-conditioned speech")
        if mode == "infill" and previous_takes:
            raise ValueError("Infill context comes from its source codes")
        if mode != "infill" and infill is not None:
            raise ValueError("An infill window requires the infill workflow")
        if mode not in ("scene", "design") and (scene or characters or lines):
            raise ValueError("Scene descriptions and lines require the scene workflow")
        if mode == "scene" and text:
            raise ValueError("Scene speech comes from its script lines")
        if mode != "stream" and lookahead:
            raise ValueError("Lookahead requires the streaming workflow")
        if (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64
                or isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int)
                or not 1 <= max_new_tokens <= self.engine.options.codec_max_frames):
            raise ValueError("Invalid seed or per-chunk generation limit")
        if (not 0 < temperature < 2 or not 0 < top_p <= 1
                or isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0):
            raise ValueError("Invalid sampling parameters")
        if (isinstance(max_chunk_chars, bool) or not isinstance(max_chunk_chars, int)
                or max_chunk_chars < 1):
            raise ValueError("Chunk length must be a positive integer")
        if len(references) > 32:
            raise ValueError("At most 32 reference clips are supported per request")
        transcripts = [reference.transcript() for reference in references]
        validate_content(direction, "direction", empty=True)
        output = Path(output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        engine = self.engine
        with engine._lock:
            if engine._closed or engine._transition_failure:
                raise RuntimeError(engine._transition_failure or "Engine is closed")
            started = time.perf_counter()
            lifecycle_before = dict(engine._lifecycle)
            if progress:
                progress({"stage": "references", "count": len(references)})
            reference_codes, hits = [], []
            for reference, transcript in zip(references, transcripts):
                if cancelled and cancelled():
                    return self._cancelled(output)
                codes, hit = engine.prepare_reference(Path(reference.audio), transcript)
                reference_codes.append(codes)
                hits.append(hit)
            engine.load()
            config, tokenizer = engine.model.config, engine.model.tokenizer
            history = [
                PreviousTake(take.text, validate_codes(take.codes, config, "previous take"),
                             take.lookahead)
                for take in previous_takes
            ]
            if single_chunk or mode in ("scene", "infill", "stream", "design"):
                chunks = [text]
            else:
                chunks = split_text(text, max_chars=max_chunk_chars)
            first_chunk = None
            records, generated, all_frames, previews = [], [], [], []
            status = "complete"
            separator_id = tokenizer.token_to_id(LINE_SEPARATOR) if mode == "scene" else None
            controls = (LINE_SEPARATOR,) if mode == "scene" else ()
            for index, chunk in enumerate(chunks):
                if cancelled and cancelled():
                    status = "cancelled"
                    break
                if mode == "design":
                    user_text = f"[{direction}] {chunk}" if direction else chunk
                    prompt = build_design_prompt(
                        user_text, tokenizer, config, name=character_name,
                        description=voice_description, scene=scene)
                elif mode == "infill":
                    prompt = build_infill_prompt(chunk, tokenizer, config, infill)
                    user_text = chunk
                elif mode == "scene":
                    prompt = build_scene_prompt(scene, characters, lines, tokenizer, config, history=history)
                    user_text = "\n".join(f"{line.character}: {line.text}" for line in lines)
                elif mode == "stream":
                    prompt = build_stream_prompt(
                        chunk, lookahead, tokenizer, config, history=history,
                        transcripts=transcripts, references=reference_codes)
                    user_text = chunk
                else:
                    active_history = history
                    if first_chunk is not None:
                        active_history = [*previous_takes, first_chunk]
                        if history[-1] is not first_chunk:
                            active_history.append(history[-1])
                    prompt = build_tts_prompt(
                        chunk, tokenizer, config, transcripts=transcripts, references=reference_codes,
                        history=active_history, direction=direction, language=language,
                        duration_frames=duration_frames,
                    )
                    user_text = (
                        directed_text(chunk, direction, language=language, duration_frames=duration_frames)
                        if direction or language is not None or duration_frames is not None else chunk
                    )
                if progress:
                    progress({"stage": "prepare", "chunk": index + 1, "chunks": len(chunks),
                              "prompt_rows": len(prompt)})
                prepare_started = time.perf_counter()
                plan, error = engine._prepare(len(prompt) + max_new_tokens, output_controls=controls)
                prepare_seconds = time.perf_counter() - prepare_started
                if progress:
                    progress({"stage": "prefill", "chunk": index + 1, "chunks": len(chunks)})
                seed_sampling((seed + index) % 2**64)
                control_head = getattr(engine.model.backbone, "protocol_head", None) if controls else None

                def frame_progress(count, limit):
                    if progress:
                        progress({"stage": "generate", "chunk": index + 1,
                                  "chunks": len(chunks), "frames": count, "limit": limit})

                result = generate(
                    engine.model, prompt, max_new_tokens=max_new_tokens,
                    prefill_chunk_size=plan.prefill_chunk, temperature=temperature,
                    top_k=top_k, top_p=top_p, tower_graph=engine.graph,
                    backbone_graph=engine.backbone_graph, progress=frame_progress,
                    cancelled=cancelled,
                    prefix_cache=engine.prefix_cache
                    if mode in ("tts", "revision", "edit", "stream") else None,
                    logit_head=(
                        (lambda hidden: control_head.logits(hidden, engine.model.backbone.lm_head))
                        if control_head is not None else None),
                    control_token_ids=(separator_id,) if separator_id is not None else (),
                    candidate_ids=engine._task_candidates,
                    allow_prefill_fallback=engine.options.prefill_chunk is None,
                )
                engine._remember_prefill_chunk(result.prefix_cache)
                plan = engine._ready_plan or plan
                record = {
                    **result.as_dict(), "text": user_text, "prompt_rows": len(prompt),
                    "runtime_prepare_seconds": prepare_seconds,
                    "runtime_prepare_breakdown": dict(engine._last_prepare_timing),
                    "runtime_plan": plan.as_dict(), "graph_warning": error,
                }
                records.append(record)
                all_frames.append(result.frames)
                if result.finish_reason != "eos" or (not len(result.frames) and mode != "infill"):
                    status = "cancelled" if result.finish_reason == "cancelled" else "incomplete"
                    break
                if mode == "infill" and not len(result.frames):
                    segments, separators = [
                        torch.empty(config.audio_decoder.num_codebooks, 0, dtype=torch.int64)
                    ], []
                else:
                    segments, separators = audio_segments(result.frames, config, separator_id)
                if mode == "scene" and len(segments) != len(lines):
                    status = "segment_mismatch"
                    record["expected_segments"], record["actual_segments"] = len(lines), len(segments)
                    break
                codes = torch.cat(segments, dim=1)
                if mode == "infill":
                    replacement = codes
                    codes = infill.join(replacement, config)
                    if not codes.shape[1]:
                        status = "incomplete"
                        break
                    prefix, suffix = infill.parts(config)
                    record["infill"] = {
                        "start_frame": infill.start, "end_frame": infill.end,
                        "replacement_frames": replacement.shape[1],
                        "output_frames": codes.shape[1],
                        "prefix_codes_equal": bool(torch.equal(codes[:, :prefix.shape[1]], prefix)),
                        "suffix_codes_equal": bool(torch.equal(
                            codes[:, codes.shape[1] - suffix.shape[1]:], suffix)),
                    }
                if codes.shape[1] > engine.options.codec_max_frames:
                    raise ValueError("Decoded chunk exceeds this engine's codec frame budget")
                record["code_sha256"] = hashlib.sha256(codes.numpy().tobytes()).hexdigest()
                record["separator_rows"] = separators
                generated.append(codes)
                take = PreviousTake(user_text, codes, lookahead if mode == "stream" else None)
                history.append(take)
                if first_chunk is None:
                    first_chunk = take
                if segment_ready is not None and mode != "infill":
                    if progress:
                        progress({"stage": "decode", "chunk": index + 1, "chunks": len(chunks)})
                    audio = engine._decode_audio_segments([codes])
                    preview_path = output.with_name(f"{output.stem}.part-{index + 1:04d}.wav")
                    write_wav(audio, preview_path)
                    preview_take = save_take(
                        preview_path, user_text, codes, result.frames,
                        metadata={
                            "mode": mode, "completed_chunk": True, "chunk_index": index,
                            "direction": direction,
                            **({"lookahead": lookahead} if mode == "stream" else {}),
                            "references": [asdict(reference) for reference in references],
                        })
                    previews.append(audio)
                    segment_ready({
                        "index": index, "path": str(preview_path),
                        "take_file": str(preview_take),
                        "samples": audio.numel(), "sample_rate": 44100,
                        "seconds": audio.numel() / 44100,
                    })
            audio_info, take_file = None, None
            if status == "complete" and generated:
                if progress:
                    progress({"stage": "decode", "chunks": len(chunks)})
                audio = torch.cat(previews) if previews else engine._decode_audio_segments(generated)
                if cancelled and cancelled():
                    status = "cancelled"
                else:
                    write_wav(audio, output)
                    codes = torch.cat(generated, dim=1)
                    take_text = text if mode != "scene" else user_text
                    if mode in ("tts", "revision", "edit") and (direction or language or duration_frames):
                        take_text = directed_text(
                            text, direction, language=language, duration_frames=duration_frames)
                    saved_frames = (
                        torch.cat((codes[:1] + config.semantic_start_token_id, codes)).T
                        if mode == "infill" else torch.cat(all_frames)
                    )
                    take_file = save_take(
                        output, take_text, codes, saved_frames,
                        metadata={"mode": mode, "direction": direction, "spoken_text": text,
                                  "voice_description": voice_description,
                                  "character_name": character_name, "scene": scene,
                                  "language": language, "duration_frames": duration_frames,
                                  **({"lookahead": lookahead} if mode == "stream" else {}),
                                  **({"chunks": [
                                      {"text": record["text"], "frames": chunk.shape[1],
                                       **({"lookahead": lookahead} if mode == "stream" else {})}
                                      for record, chunk in zip(records, generated)
                                  ]} if mode not in ("scene", "infill") else {}),
                                  "references": [asdict(reference) for reference in references]},
                    )
                    audio_info = {"samples": audio.numel(), "sample_rate": 44100,
                                  "duration_seconds": audio.numel() / 44100}
            report = {
                "status": status, "mode": mode, "text": text,
                "output": str(output) if audio_info else None,
                "take_file": str(take_file) if take_file else None,
                "audio": audio_info, "chunks": records,
                "reference_cache_hits": hits, "seed": seed,
                "total_seconds": time.perf_counter() - started,
                "lifecycle_delta": {key: engine._lifecycle[key] - lifecycle_before[key] for key in lifecycle_before},
                "all_top8_preserved": True,
                **({"stream_reference_conditioning": "system_reference_blocks"}
                   if mode == "stream" and references else {}),
                "compute_policy": dict(getattr(engine, "compute_report", {})),
            }
            if status != "complete" and all_frames:
                frame_file = output.with_suffix(".diagnostic-frames.npy")
                np.save(frame_file, torch.cat(all_frames).numpy(), allow_pickle=False)
                report["diagnostic_frames"] = str(frame_file)
            output.with_suffix(".json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if progress:
                progress({"stage": status, "output": report["output"]})
            return report

    @staticmethod
    def _cancelled(output):
        return {"status": "cancelled", "output": None, "take_file": None,
                "audio": None, "chunks": [], "total_seconds": 0.0}
