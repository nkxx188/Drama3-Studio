"""Full Drama 3 workflows on the persistent GGUF engine.

Task prompts and codec reconstruction execute in the native worker. Python
owns bounded text splitting, real-take history, delivery and job metadata.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from .codec import concatenate_wavs
from .gguf_runtime import GenerationCancelled
from .gguf_takes import CODEC_CONFIG, load_conditioning_take, load_native_take, save_native_take
from .protocols import PreviousTake, design_character, directed_text, validate_content
from .text_chunks import split_text
from .workflows import _history_variants, _is_prompt_overflow
from .sampling import DEFAULT_TOP_K


class GGUFWorkflows:
    def __init__(self, engine):
        self.engine = engine

    @staticmethod
    def load_take(path):
        return load_conditioning_take(path)

    def run(self, mode, text, output, *, references=(), previous_takes=(),
            direction="", language=None, duration_frames=None, infill=None,
            lookahead="", seed=123, max_new_tokens=768, max_chunk_chars=160,
            temperature=1.0, top_k=DEFAULT_TOP_K, top_p=0.9, progress=None, cancelled=None,
            segment_ready=None, streaming_audio=False, chunk_frames=16, single_chunk=False,
            voice_description="", character_name="Voice", scene=""):
        if mode not in ("tts", "revision", "edit", "infill", "stream", "design"):
            raise ValueError("Unknown GGUF workflow")
        if mode == "design":
            design = design_character(character_name, voice_description, scene)
            if references or previous_takes or language or duration_frames:
                raise ValueError("Voice design does not use references/history or language/duration hints")
        else:
            design = None
            if voice_description:
                raise ValueError("Voice description requires the design workflow")
        validate_content(text, "text")
        validate_content(direction, "direction", empty=True)
        validate_content(lookahead, "lookahead", empty=True)
        if mode in ("revision", "edit") and not previous_takes:
            raise ValueError("Revision and continuation editing require a previous take")
        if mode == "infill":
            if infill is None or references or previous_takes or streaming_audio:
                raise ValueError("Infill requires source codes without references/history or audio streaming")
            if direction or language is not None or duration_frames is not None:
                raise ValueError("Use inline controls; infill duration belongs to the replacement span")
            infill.parts(CODEC_CONFIG)
        elif infill is not None:
            raise ValueError("An infill window requires the infill workflow")
        if mode == "stream":
            if direction or language is not None or duration_frames is not None:
                raise ValueError("Use inline controls in each text stream chunk")
        elif lookahead:
            raise ValueError("Lookahead requires the streaming text workflow")
        if (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32
                or isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int)
                or not 1 <= max_new_tokens <= 2048
                or isinstance(max_chunk_chars, bool) or not isinstance(max_chunk_chars, int)
                or not 1 <= max_chunk_chars <= 2000):
            raise ValueError("Invalid GGUF seed, frame limit or chunk length")
        chunks = [text] if single_chunk or mode in ("infill", "stream", "design") else split_text(text, max_chars=max_chunk_chars)
        if len(chunks) > 4096:
            raise ValueError("Too many text chunks in one request")
        # Validate all controls before loading the model.
        if direction or language is not None or duration_frames is not None:
            directed_text(text, direction, language=language, duration_frames=duration_frames)
        output = Path(output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        started, records, takes, outputs, segments = time.perf_counter(), [], [], [], []
        first, last, position, status = None, None, 0, "complete"
        last_result = {}
        with self.engine._lock:
            for index, chunk in enumerate(chunks):
                if cancelled and cancelled():
                    raise GenerationCancelled("GGUF generation cancelled")
                if progress:
                    progress({"stage": "generate", "chunk": index + 1, "chunks": len(chunks)})
                target = output if len(chunks) == 1 else output.with_name(
                    f"{output.stem}.chunk-{index + 1:04d}.wav")
                base_position = position

                def deliver(row):
                    row = dict(row)
                    row["index"] = len(segments)
                    row["start_sample"] += base_position
                    row["end_sample"] += base_position
                    row["delivered_seconds"] = time.perf_counter() - started
                    segments.append(row)
                    if segment_ready:
                        segment_ready(dict(row))
                    if progress:
                        progress({"stage": "generate", "chunk": index + 1, "chunks": len(chunks),
                                  "frames": (row["end_sample"] - base_position) // 2048,
                                  "limit": max_new_tokens})

                extra = {"design": design} if design else {}
                result = None
                history_error = None
                for active_history in _history_variants(
                    previous_takes, first, last):
                    try:
                        result = self.engine.generate(
                            f"[{direction}] {chunk}" if design and direction else chunk,
                            target, references=references, previous_takes=active_history,
                            direction="" if design else direction, language=language, duration_frames=duration_frames,
                            task_mode=mode if mode in ("infill", "stream", "design") else "joint" if single_chunk else "",
                            lookahead=lookahead, infill=infill, seed=(seed + index) % 2**32,
                            max_frames=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p,
                            chunk_size=2000, cancelled=cancelled, streaming_audio=streaming_audio,
                            chunk_frames=chunk_frames, segment_ready=deliver if streaming_audio else None, **extra)
                    except Exception as error:
                        if not _is_prompt_overflow(error):
                            raise
                        history_error = error
                        continue
                    break
                if result is None:
                    raise history_error or RuntimeError("No usable generation history")
                last_result = result
                status = result["status"]
                if not result.get("take_file"):
                    # Legacy workers may do basic TTS, but cannot advertise
                    # editing/history without real codec artifacts.
                    raise RuntimeError("The native runtime did not save a Drama 3 take")
                take, _, native_chunks = load_native_take(result["take_file"])
                current_takes = native_chunks or [take]
                for saved in current_takes:
                    records.append({"text": saved.text, "frames": saved.codes.shape[1],
                                    "native_request_seconds": result.get("request_seconds"),
                                    "status": status})
                takes.extend(current_takes)
                first = first or current_takes[0]
                last = current_takes[-1]
                outputs.append(result["output"])
                position += result["audio"]["samples"]
                if segment_ready and not streaming_audio and mode != "infill" and status == "complete":
                    row = {"index": index, "path": result["output"], "take_file": result["take_file"],
                           "start_sample": base_position, "end_sample": position,
                           "samples": result["audio"]["samples"], "sample_rate": 44100}
                    segments.append(row)
                    segment_ready(dict(row))
                if status != "complete":
                    break
            if cancelled and cancelled():
                raise GenerationCancelled("GGUF generation cancelled")
            if len(outputs) > 1:
                concatenate_wavs(outputs, output)
            codes = torch.cat([take.codes for take in takes], dim=1)
            prompt_text = text
            if mode not in ("stream", "infill"):
                prompt_text = directed_text(
                    text, direction, language=language, duration_frames=duration_frames)
            elif mode == "infill":
                prompt_text = directed_text(text, duration_frames=infill.duration_frames)
            metadata = {
                "mode": mode, "spoken_text": text, "direction": direction,
                "language": language, "duration_frames": duration_frames,
                "references": [asdict(reference) for reference in references],
                "model": str(self.engine.model_path), "complete": status == "complete",
                "voice_description": voice_description, "character_name": character_name, "scene": scene,
            }
            saved = save_native_take(
                output, PreviousTake(prompt_text, codes, lookahead if mode == "stream" else None),
                metadata=metadata, chunks=takes)
            result = {
                **last_result, "mode": mode, "text": text, "status": status,
                "complete": status == "complete", "output": str(output),
                "take_file": str(saved) if status == "complete" else None,
                "audio": {"samples": position, "sample_rate": 44100,
                          "duration_seconds": position / 44100},
                "audio_seconds": position / 44100, "chunks": records, "segments": segments,
                "seed": seed, "total_seconds": time.perf_counter() - started,
                "all_top8_preserved": True,
            }
            if streaming_audio:
                result["streaming_audio"] = True
                result["streamed_samples"] = sum(row["samples"] for row in segments)
            if mode == "infill":
                prefix, suffix = infill.parts(CODEC_CONFIG)
                suffix_start = codes.shape[1] - suffix.shape[1]
                result["infill"] = {
                    "start_frame": infill.start, "end_frame": infill.end,
                    "replacement_frames": codes.shape[1] - prefix.shape[1] - suffix.shape[1],
                    "output_frames": codes.shape[1],
                    "prefix_codes_equal": torch.equal(codes[:, :prefix.shape[1]], prefix),
                    "suffix_codes_equal": torch.equal(codes[:, suffix_start:], suffix),
                }
                if not result["infill"]["prefix_codes_equal"] or not result["infill"]["suffix_codes_equal"]:
                    raise RuntimeError("Native infill changed codec codes outside the selected span")
            output.with_suffix(".json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if progress:
                progress({"stage": status, "output": result["output"]})
            return result
