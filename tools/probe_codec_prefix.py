"""Causal codec prefix stability and bounded-window boundary diagnostics."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from benchmark_engine_decode import DramaEngine, EngineOptions, torch


def compare(expected, actual):
    import soundfile as sf

    delta = expected.float() - actual.float()
    def pcm(samples):
        handle = io.BytesIO()
        sf.write(handle, samples.clamp(-1, 1).float().numpy(), 44100,
                 format="WAV", subtype="PCM_16")
        handle.seek(0)
        return sf.read(handle, dtype="int16")[0]
    pcm_expected, pcm_actual = pcm(expected), pcm(actual)
    return dict(
        bytes_equal=torch.equal(expected.view(torch.uint8), actual.view(torch.uint8)),
        maximum_absolute_error=float(delta.abs().max()),
        rms_error=float(delta.square().mean().sqrt()),
        pcm16_different_samples=int(np.count_nonzero(pcm_expected != pcm_actual)),
        samples=expected.numel())


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--take", type=Path,
                        default=ROOT / "outputs/engine-workflow-references.take.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "research/codec-prefix-probe.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with np.load(args.take, allow_pickle=False) as data:
        codes = torch.from_numpy(data["codes"].copy()).long()
    report = {
        "scope": "Real generated codes. Whole-prefix redecoding versus whole-sequence decoding, plus an explicit finite-context window diagnostic. No new speech generation or perceptual-quality claim.",
        "take": str(args.take), "frames": codes.shape[1], "prefixes": [], "windows": [],
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    with DramaEngine(EngineOptions(checkpoint=ROOT / "checkpoints/drama-3")) as engine:
        # Construct codec buffers to the same final capacity before all decodes.
        with engine._codec_on_gpu(codes.shape[1], operation="decode"):
            pass
        whole = engine._decode_audio_segments([codes])
        for count in sorted({10, 20, 32, 64, codes.shape[1] // 2}):
            if not 1 <= count < codes.shape[1]:
                continue
            started = time.perf_counter()
            audio = engine._decode_audio_segments([codes[:, :count]])
            row = dict(frames=count, seconds=time.perf_counter() - started,
                       **compare(whole[:audio.numel()], audio))
            report["prefixes"].append(row)
            save()
            print(json.dumps(row), flush=True)
        # Compare dropping remote context, as used by finite overlap codecs.
        end = min(codes.shape[1], 100)
        start = max(1, end // 2)
        for overlap in (0, 16, 32, 64):
            begin = max(0, start - overlap)
            audio = engine._decode_audio_segments([codes[:, begin:end]])
            trimmed = audio[(start - begin) * 2048:]
            reference = whole[start * 2048:end * 2048]
            row = dict(start_frame=start, end_frame=end, overlap=overlap,
                       context_start=begin, **compare(reference, trimmed))
            report["windows"].append(row)
            save()
            print(json.dumps(row), flush=True)
        report["all_complete_prefix_bytes_equal"] = all(
            row["bytes_equal"] for row in report["prefixes"])
        report["production_streaming_enabled"] = False
        save()


if __name__ == "__main__":
    main()
