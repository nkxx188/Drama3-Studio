"""Actual codec stateful chunks, complete context and waveform boundary errors."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from benchmark_engine_decode import DramaEngine, EngineOptions, torch
from drama_low_vram.codec import load_codec, write_wav
from drama_low_vram.stream_codec import CausalCodecStream
from fish_speech.models.text2semantic.inference import resolve_codec_path
from probe_codec_prefix import compare


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--take", type=Path,
                        default=ROOT / "outputs/engine-workflow-references.take.npz")
    parser.add_argument("--chunk-frames", type=int, nargs="+", default=[10, 32, 64])
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--repeat-to", type=int)
    parser.add_argument("--with-model", action="store_true")
    parser.add_argument("--prefix", default="codec-state")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "research/stream-codec-validation.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with np.load(args.take, allow_pickle=False) as values:
        codes = torch.from_numpy(values["codes"].copy()).long()
    if args.repeat_to:
        codes = codes.repeat(1, math.ceil(args.repeat_to / codes.shape[1]))[:, :args.repeat_to]
    if min(args.chunk_frames) < 1:
        parser.error("Positive codec chunks required")
    if args.dtype == "float32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    report = {
        "scope": "Stateful official codec with per-layer causal KV and convolution history, original weights and global RoPE. Whole decoding comparison. Repetition, when requested, is a codec diagnostic, not generated speech.",
        "dtype": args.dtype, "frames": codes.shape[1], "with_model": args.with_model,
        "repeated_diagnostic": bool(args.repeat_to), "cases": [],
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    with DramaEngine(EngineOptions(
        checkpoint=ROOT / "checkpoints/drama-3", expert_storage="mmap",
        cache_experts=1, staged_experts=8, pair_rows=64, prefill_chunk=32,
        backbone_graph=True, combined_decode_bank=True,
        cpu_experts=2, native_threads=4, host_gather="auto",
    )) as engine:
        engine.codec = load_codec(
            resolve_codec_path(engine.checkpoint), device="cpu",
            dtype=getattr(torch, args.dtype), max_frames=codes.shape[1])
        if args.with_model:
            engine.load()
            engine._prepare(1024)
        engine._decode_audio_segments([codes])
        reference = engine._decode_audio_segments([codes])
        write_wav(reference, ROOT / "outputs" / f"{args.prefix}-reference-{args.dtype}.wav")
        for count in args.chunk_frames:
            if args.with_model and engine._parked:
                engine._prepare(1024)
            graphs = engine.graph, engine.backbone_graph
            state = CausalCodecStream(engine.codec, max_frames=codes.shape[1])
            parts, maximum_state = [], 0
            before = dict(engine._lifecycle)
            started = time.perf_counter()
            for start in range(0, codes.shape[1], count):
                chunk = codes[:, start:start + count]
                latent = engine._run_codec(
                    chunk.shape[1], lambda codec: state.decode_latents(chunk),
                    direction="decode_latents")
                try:
                    audio = engine._run_codec(
                        chunk.shape[1], lambda codec: state.decode_waveform(latent),
                        direction="decode_waveform")
                finally:
                    del latent
                parts.append(audio)
                maximum_state = max(maximum_state, state.state_bytes)
            elapsed = time.perf_counter() - started
            audio = torch.cat(parts)
            error = compare(reference, audio)
            signal = float(reference.float().square().mean())
            noise = error["rms_error"] ** 2
            row = {
                "chunk_frames": count, "seconds": elapsed, "chunks": len(parts),
                "decoded_frames": state.decoded_frames, "maximum_cpu_state_bytes": maximum_state,
                "finite_waveform": bool(torch.isfinite(audio).all()), **error,
                "graphs_preserved": not args.with_model or (
                    engine.graph is graphs[0] and engine.backbone_graph is graphs[1]),
                "signal_to_error_db": 10 * math.log10(signal / noise) if noise else None,
                "lifecycle": {key: engine._lifecycle[key] - before[key] for key in before},
            }
            state.close()
            output = ROOT / "outputs" / f"{args.prefix}-stream-{args.dtype}-{count}.wav"
            write_wav(audio, output)
            row["output"] = str(output)
            report["cases"].append(row)
            save()
            print(json.dumps(row), flush=True)
            if args.dtype == "float32" and (
                error["maximum_absolute_error"] > 5e-5 or error["rms_error"] > 5e-6
            ):
                raise RuntimeError("FP32 causal state diverged from whole-sequence math")
        report["completed"] = True
        save()


if __name__ == "__main__":
    main()
