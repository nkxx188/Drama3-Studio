"""Measure official codec workspace with live model graphs in an isolated probe."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from benchmark_memory_pressure import SHORT, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release-expert-buffers", action="store_true")
    parser.add_argument("--cache-experts", type=int, choices=(0, 1), default=1)
    parser.add_argument("--bounded-snake", action="store_true")
    parser.add_argument("--flat-decoder", action="store_true")
    parser.add_argument("--frames", type=int, choices=(16, 375, 650), default=375)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve earlier probe output")
    os.environ.update(environment())
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor/fish-speech")]
    import torch
    from drama_low_vram.codec import codec_workspace_bytes
    from drama_low_vram.engine import DramaEngine, EngineOptions
    from drama_low_vram.sequence_state import _bindings

    torch.set_num_threads(1)
    report = {
        "scope": "Isolated admission probe only; no production safety override.",
        "status": "running", "stages": [],
        "cudnn_workspace_cap_env": os.environ.get("CUDNN_CONV_WSCAP_DBG"),
        "cudnn_version": torch.backends.cudnn.version(),
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    with DramaEngine(EngineOptions(
        checkpoint=ROOT / "checkpoints/drama-3", expert_storage="mmap",
        cache_experts=args.cache_experts, staged_experts=8, pair_rows=64, prefill_chunk=32,
        cpu_experts=2, native_threads=4, backbone_graph=True,
        combined_decode_bank=True, context_bucket_size=512,
    )) as engine:
        try:
            codes, _ = engine.prepare_reference(
                ROOT / "outputs/quality-official-fp8.wav", SHORT)
            frames = args.frames
            report["frames"] = frames
            inputs = codes.repeat(
                1, (frames + codes.shape[1] - 1) // codes.shape[1])[:, :frames]
            # Establish a warm codec before creating the resident main model.
            engine._decode_audio_segments([inputs])
            expected = engine._decode_audio_segments([inputs])
            report["baseline_wave_sha256"] = hashlib.sha256(
                expected.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            if args.bounded_snake:
                import importlib.util

                spec = importlib.util.spec_from_file_location(
                    "failed_codec_activation",
                    ROOT / "research/phase24-failed-codec-activation.py")
                candidate = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(candidate)

                report["bounded_activations"] = candidate.bound_decoder_activations(engine.codec)
            if args.flat_decoder:
                from drama_low_vram.codec_decoder import flatten_decoder_calls

                report["flat_decoder"] = flatten_decoder_calls(engine.codec)
            engine.load()
            engine._prepare(947)
            if args.release_expert_buffers:
                import gc

                engine.manager.release_runtime_buffers()
                gc.collect()
                torch.cuda.empty_cache()
            report["expert_buffers_released"] = args.release_expert_buffers
            report["cache_experts"] = args.cache_experts
            graphs = engine.graph, engine.backbone_graph
            addresses = [tensor.data_ptr() for _, tensor, _, _ in _bindings(
                engine.model, 1, 1)]
            admission = engine._codec_can_coexist
            original_codec = engine._codec_on_gpu
            engine._full_codec_decode_fits = lambda _frames: False

            def probe_admission(frames, **kwargs):
                normal = admission(frames, **kwargs)
                if kwargs.get("operation") == "decode_waveform":
                    report["normal_waveform_admission"] = normal
                    # The live-sequence guard below turns actual OOM into a
                    # recorded probe failure without parking or clearing graphs.
                    return True
                return normal

            engine._codec_can_coexist = probe_admission

            @contextmanager
            def trace(frames, **kwargs):
                row = {
                    "operation": kwargs.get("operation", "all"),
                    "frames": frames,
                    "free_before": int(torch.cuda.mem_get_info(engine.device)[0]),
                    "reserved_before": int(torch.cuda.memory_reserved(engine.device)),
                    "allocated_before": int(torch.cuda.memory_allocated(engine.device)),
                }
                started = time.perf_counter()
                try:
                    with original_codec(frames, **kwargs) as codec:
                        torch.cuda.synchronize(engine.device)
                        allocated = int(torch.cuda.memory_allocated(engine.device))
                        torch.cuda.reset_peak_memory_stats(engine.device)
                        row["allocated_after_staging"] = allocated
                        try:
                            yield codec
                        finally:
                            row["peak_operation_extra_bytes"] = (
                                int(torch.cuda.max_memory_allocated(engine.device)) - allocated)
                            row["reserved_peak"] = int(torch.cuda.max_memory_reserved(engine.device))
                finally:
                    row["seconds"] = time.perf_counter() - started
                    report["stages"].append(row)
                    save()

            engine._codec_on_gpu = trace
            with engine._codec_preserving_sequence():
                actual = engine._decode_audio_segments([inputs])
            report.update(
                status="complete",
                finite=bool(torch.isfinite(actual).all()),
                waveform_bytes_equal=torch.equal(
                    expected.view(torch.uint8), actual.view(torch.uint8)),
                wave_sha256=hashlib.sha256(
                    actual.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
                graphs_preserved=(engine.graph is graphs[0] and engine.backbone_graph is graphs[1]),
                cache_addresses_preserved=addresses == [
                    tensor.data_ptr() for _, tensor, _, _ in _bindings(engine.model, 1, 1)],
                static_waveform_workspace_bytes=codec_workspace_bytes(frames, "decode_waveform"),
            )
        except Exception as error:
            report.update(status="error", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            save()
            print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
