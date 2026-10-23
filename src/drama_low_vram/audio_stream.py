"""Generated-frame audio delivery with explicit state preservation and EOS."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from .codec import load_codec, write_wav
from .engine import CodecCoexistenceUnavailable, seed_sampling
from .protocols import build_tts_prompt, directed_text, validate_content
from .sequence_state import SequenceSnapshot
from .stream_codec import CausalCodecStream
from .verification import DecodeCursor
from .workflows import save_take
from .sampling import DEFAULT_TOP_K
from fish_speech.models.text2semantic.inference import resolve_codec_path


class AudioStreamSession:
    """One full TTS utterance, including references and ordinary controls.

    Audio is delivered after completed frame groups. The codec retains actual
    per-layer context; model/RNG state survives codec parking and graph rebuilds.
    Completion requires EOS. Codec coexistence keeps the live model state in
    place. When parking is necessary, a CPU snapshot protects continuation;
    insufficient snapshot RAM defers delivery until EOS.
    """

    def __init__(self, engine, text, output, *, references=(), previous_takes=(),
                 direction="", language=None, duration_frames=None,
                 seed=123, max_new_tokens=768, chunk_frames=32,
                 temperature=0.7, top_k=DEFAULT_TOP_K, top_p=0.7):
        validate_content(text, "stream text")
        validate_content(direction, "direction", empty=True)
        if (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64
                or isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int)
                or not 1 <= max_new_tokens <= engine.options.codec_max_frames
                or isinstance(chunk_frames, bool) or not isinstance(chunk_frames, int)
                or not 1 <= chunk_frames <= max_new_tokens):
            raise ValueError("Invalid audio stream seed or frame limits")
        if (not 0 < temperature < 2 or not 0 < top_p <= 1
                or isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0):
            raise ValueError("Invalid audio stream sampling parameters")
        if len(references) > 32:
            raise ValueError("At most 32 references are supported")
        for reference in references:
            reference.transcript()
        self.engine, self.text, self.output = engine, text, Path(output).resolve()
        self.references, self.previous_takes = tuple(references), tuple(previous_takes)
        self.direction, self.language, self.duration_frames = direction, language, duration_frames
        self.seed, self.maximum, self.stride = seed, max_new_tokens, chunk_frames
        self.temperature, self.top_k, self.top_p = temperature, top_k, top_p
        self.status, self.results, self.report = "waiting", [], None
        self._started = False

    def _prepare_codec(self):
        engine = self.engine
        if engine.codec is None:
            engine.codec = load_codec(
                resolve_codec_path(engine.checkpoint), device="cpu",
                dtype=torch.bfloat16, max_frames=self.maximum)
        elif engine.codec._drama_low_vram_max_frames < self.maximum:
            # Grow positional buffers before prefill, while no live sequence
            # can be damaged by admission/parking.
            with engine._codec_on_gpu(self.maximum, operation="decode_latents"):
                pass

    def _decode(self, codec_state, codes, cursor, capacity, *, continued):
        engine = self.engine
        checkpoint, checkpoint_seconds, restore_seconds = None, 0.0, 0.0
        if continued:
            # Codec and expert workspaces do not own KV/GDN/tower state. Try
            # coexistence first, enforcing that no admission/OOM retry can park
            # the model. The outer transaction also rolls back a completed
            # latent stage if waveform staging later cannot coexist.
            started = time.perf_counter()
            try:
                with codec_state.transaction(), engine._codec_preserving_sequence():
                    audio = self._decode_stages(codec_state, codes)
                    codec_seconds = time.perf_counter() - started
                    prepare_started = time.perf_counter()
                    engine._prepare(capacity)
                    cursor.tower_graph, cursor.backbone_graph = engine.graph, engine.backbone_graph
                    restore_seconds = time.perf_counter() - prepare_started
            except CodecCoexistenceUnavailable:
                # Failed staging has restored codec CPU ownership and the
                # transaction has restored all causal codec state. Snapshot
                # admission happens after the failed allocation scope exits.
                pass
            else:
                return audio, {
                    "codec_seconds": codec_seconds, "checkpoint_seconds": 0.0,
                    "restore_seconds": restore_seconds,
                    "sequence_preservation": "resident",
                }
            size = SequenceSnapshot.estimate_bytes(
                engine.model, cursor.position + 1, cursor.tower_rows)
            if not engine._admit_prefix_snapshot(size):
                return None
            started = time.perf_counter()
            try:
                checkpoint = cursor.capture()
            except MemoryError:
                return None
            checkpoint_seconds = time.perf_counter() - started
        started = time.perf_counter()
        audio = self._decode_stages(codec_state, codes)
        codec_seconds = time.perf_counter() - started
        if checkpoint is not None:
            started = time.perf_counter()
            engine._prepare(capacity)
            cursor.tower_graph, cursor.backbone_graph = engine.graph, engine.backbone_graph
            cursor.restore(checkpoint)
            restore_seconds = time.perf_counter() - started
            del checkpoint
        return audio, {
            "codec_seconds": codec_seconds, "checkpoint_seconds": checkpoint_seconds,
            "restore_seconds": restore_seconds,
            "sequence_preservation": "snapshot" if continued else "eos",
        }

    def _decode_stages(self, codec_state, codes):
        engine = self.engine
        latent = engine._run_codec(
            codes.shape[1], lambda codec: codec_state.decode_latents(codes),
            direction="decode_latents")
        try:
            audio = engine._run_codec(
                codes.shape[1], lambda codec: codec_state.decode_waveform(latent),
                direction="decode_waveform")
        finally:
            del latent
        return audio

    @torch.inference_mode()
    def run(self, *, progress=None, cancelled=None):
        if self._started:
            raise RuntimeError("An audio stream can only run once")
        self._started = True
        engine, codec_state = self.engine, None
        self.output.parent.mkdir(parents=True, exist_ok=True)
        started, parts, frames, deferred = time.perf_counter(), [], None, False
        before = {}
        with engine._lock:
            try:
                self.status = "initializing"
                before = dict(engine._lifecycle)
                if cancelled and cancelled():
                    self.status = "cancelled"
                    return
                transcripts, references = [], []
                for reference in self.references:
                    if cancelled and cancelled():
                        self.status = "cancelled"
                        return
                    transcript = reference.transcript()
                    codes, _ = engine.prepare_reference(Path(reference.audio), transcript)
                    transcripts.append(transcript)
                    references.append(codes)
                if progress:
                    progress({"stage": "prepare"})
                self._prepare_codec()
                engine.load()
                prompt = build_tts_prompt(
                    self.text, engine.model.tokenizer, engine.model.config,
                    transcripts=transcripts, references=references,
                    history=self.previous_takes, direction=self.direction,
                    language=self.language, duration_frames=self.duration_frames)
                capacity = len(prompt) + self.maximum + 1
                plan, _ = engine._prepare(capacity)
                if progress:
                    progress({"stage": "prefill"})
                if engine.graph is None or engine.backbone_graph is None or not engine.backbone_graph.enabled:
                    raise ValueError("Audio frame streaming requires the validated tower/backbone graphs")
                seed_sampling(self.seed)
                cursor = DecodeCursor.prefill(
                    engine.model, prompt, chunk_size=plan.prefill_chunk,
                    tower_graph=engine.graph, backbone_graph=engine.backbone_graph,
                    temperature=self.temperature, top_k=self.top_k, top_p=self.top_p,
                    prefix_cache=engine.prefix_cache,
                    allow_prefill_fallback=engine.options.prefill_chunk is None)
                engine._remember_prefill_chunk(cursor.prefix_report)
                codec_state = CausalCodecStream(engine.codec, max_frames=self.maximum)
                delivered = 0
                self.status = "generating"
                for index in range(self.maximum):
                    if cancelled and cancelled():
                        self.status = "cancelled"
                        break
                    cursor.step()
                    count = len(cursor.frames) - len(prompt)
                    if progress:
                        progress({"stage": "generate", "frames": count, "limit": self.maximum})
                    should_decode = cursor.finished or (not deferred and count - delivered >= self.stride)
                    if should_decode and count > delivered:
                        codes = cursor.frames[len(prompt) + delivered:, 1:].T.cpu().contiguous()
                        decoded = self._decode(
                            codec_state, codes, cursor, capacity, continued=not cursor.finished)
                        if decoded is None:
                            deferred = True
                        else:
                            audio, timing = decoded
                            path = self.output.with_name(
                                f"{self.output.stem}.part-{len(parts) + 1:04d}.wav")
                            write_wav(audio, path)
                            parts.append(audio)
                            row = {
                                "index": len(parts) - 1, "path": str(path),
                                "start_frame": delivered, "end_frame": count,
                                "samples": audio.numel(), "sample_rate": 44100,
                                "delivered_seconds": time.perf_counter() - started,
                                "before_eos": not cursor.finished,
                                "codec_cpu_state_bytes": codec_state.state_bytes, **timing,
                            }
                            self.results.append(row)
                            delivered = count
                            yield row
                    if cursor.finished:
                        self.status = "complete" if count else "incomplete"
                        break
                else:
                    self.status = "incomplete"
                frames = cursor.frames[len(prompt):].cpu()
                if self.status == "complete":
                    if cancelled and cancelled():
                        self.status = "cancelled"
                    else:
                        audio = torch.cat(parts)
                        write_wav(audio, self.output)
                        take_text = directed_text(
                            self.text, self.direction, language=self.language,
                            duration_frames=self.duration_frames) if (
                                self.direction or self.language or self.duration_frames
                            ) else self.text
                        take = save_take(
                            self.output, take_text, frames[:, 1:].T.contiguous(), frames,
                            metadata={
                                "mode": "tts", "audio_streaming": True,
                                "direction": self.direction,
                                "references": [asdict(reference) for reference in self.references],
                            })
                        self.report = {
                            "output": str(self.output), "take_file": str(take),
                            "reference_prefix_cache": dict(cursor.prefix_report),
                            "audio": {"samples": audio.numel(), "sample_rate": 44100,
                                      "duration_seconds": audio.numel() / 44100},
                        }
            except GeneratorExit:
                self.status = "cancelled"
                raise
            except BaseException:
                self.status = "error"
                raise
            finally:
                if codec_state is not None:
                    codec_state.close()
                self.report = {
                    "status": self.status, "text": self.text,
                    "output": None, "take_file": None, "audio": None,
                    **(self.report or {}), "segments": self.results,
                    "deferred_audio_for_host_budget": deferred,
                    "total_seconds": time.perf_counter() - started,
                    "sampling": {
                        "seed": self.seed, "max_new_tokens": self.maximum,
                        "chunk_frames": self.stride, "temperature": self.temperature,
                        "top_k": self.top_k, "top_p": self.top_p,
                    },
                    "all_top8_preserved": True,
                    "compute_policy": dict(getattr(engine, "compute_report", {})),
                    "lifecycle": {
                        key: engine._lifecycle[key] - value for key, value in before.items()},
                }
                self.report["status"] = self.status
                if self.status != "complete" and frames is not None and len(frames):
                    diagnostic = self.output.with_suffix(".diagnostic-frames.npy")
                    import numpy as np
                    np.save(diagnostic, frames.numpy(), allow_pickle=False)
                    self.report["diagnostic_frames"] = str(diagnostic)
                self.output.with_suffix(".json").write_text(
                    json.dumps(self.report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
