"""Real TTS/cloning frame streaming, EOS, VQ equality and delivered PCM export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from benchmark_engine_decode import DramaEngine, EngineOptions, torch
from drama_low_vram.audio_stream import AudioStreamSession
from drama_low_vram.codec import load_codec
from drama_low_vram.protocols import VoiceReference
from drama_low_vram.workflows import DramaWorkflows
from fish_speech.models.text2semantic.inference import resolve_codec_path
from probe_codec_prefix import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clone", action="store_true")
    parser.add_argument("--force-park", action="store_true")
    parser.add_argument("--warm-baseline", action="store_true")
    parser.add_argument("--chunk-frames", type=int, default=16)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--expert-storage", choices=("auto", "ram", "mmap"), default="mmap")
    parser.add_argument("--deny-snapshot", action="store_true",
                        help="Simulate host snapshot refusal after the baseline; resident streaming must still work.")
    parser.add_argument("--prefix", default="audio-frame-stream")
    parser.add_argument("--output", type=Path, default=ROOT / "research/audio-stream-validation.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    text = "早上好。" if args.force_park else "<|speaker:0|>你好，这是一次语音测试。"
    report = {
        "scope": "Real model generation with early audio frames. Target VQ must match complete baseline, including after codec coexistence/parking. Final export equals delivered PCM. Waveform rounding is measured separately.",
        "dtype": args.dtype, "cloned": args.clone,
        "force_park": args.force_park, "segments": [],
        "expert_storage_requested": args.expert_storage,
        "snapshot_admission_forced_false": args.deny_snapshot,
    }
    references = [VoiceReference(
        str(ROOT / "outputs/quality-official-fp8.wav"),
        "你好，这是官方 FP8 质量优先测试。")] if args.clone else []
    with DramaEngine(EngineOptions(
        checkpoint=ROOT / "checkpoints/drama-3", expert_storage=args.expert_storage,
        cache_experts=1, staged_experts=8, pair_rows=64, prefill_chunk=32,
        backbone_graph=True, combined_decode_bank=True,
        cpu_experts=2, native_threads=4, host_gather="auto", prefix_cache_mib=256,
        codec_policy="park" if args.force_park else "auto",
    )) as engine:
        if args.dtype == "float32":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            engine.codec = load_codec(
                resolve_codec_path(engine.checkpoint), device="cpu",
                dtype=torch.float32, max_frames=256)
        baseline_output = ROOT / "outputs" / f"{args.prefix}-baseline.wav"
        baseline = DramaWorkflows(engine).run(
            "tts", text, baseline_output, references=references, seed=123, max_new_tokens=256)
        if baseline["status"] != "complete":
            raise RuntimeError("Streaming baseline did not reach EOS")
        if args.warm_baseline:
            report["cold_baseline_seconds"] = baseline["total_seconds"]
            baseline = DramaWorkflows(engine).run(
                "tts", text, baseline_output, references=references, seed=123, max_new_tokens=256)
            if baseline["status"] != "complete":
                raise RuntimeError("Warm baseline did not reach EOS")
        output = ROOT / "outputs" / f"{args.prefix}.wav"
        report["expert_storage_selected"] = engine.storage
        if args.deny_snapshot:
            engine._admit_prefix_snapshot = lambda _size: False
        session = AudioStreamSession(
            engine, text, output, references=references, seed=123,
            max_new_tokens=256, chunk_frames=args.chunk_frames)
        last = time.perf_counter()

        def progress(event):
            nonlocal last
            now = time.perf_counter()
            if now - last >= 15:
                print(json.dumps(event), flush=True)
                last = now

        for row in session.run(progress=progress):
            report["segments"].append(row)
            print(json.dumps(row), flush=True)
        if session.status != "complete":
            raise RuntimeError("Stream did not reach EOS")
        report["stream"] = session.report
        with np.load(baseline["take_file"], allow_pickle=False) as a, np.load(
            session.report["take_file"], allow_pickle=False
        ) as b:
            report["vq_codes_equal"] = bool(np.array_equal(a["codes"], b["codes"]))
            report["all_frames_equal"] = bool(np.array_equal(a["frames"], b["frames"]))
        delivered = np.concatenate([sf.read(row["path"], dtype="int16")[0]
                                    for row in report["segments"]])
        final, rate = sf.read(output, dtype="int16")
        report["exported_pcm_matches_delivered"] = bool(np.array_equal(delivered, final))
        report["sample_rate"] = rate
        report["delivered_before_eos"] = any(row["before_eos"] for row in report["segments"])
        report["resident_segments"] = sum(
            row.get("sequence_preservation") == "resident" for row in report["segments"])
        a = torch.from_numpy(sf.read(baseline_output, dtype="float32")[0])
        b = torch.from_numpy(sf.read(output, dtype="float32")[0])
        report["waveform_difference"] = compare(a, b)
        report["baseline"] = {
            "audio": baseline["audio"], "total_seconds": baseline["total_seconds"],
            "report_file": str(baseline_output.with_suffix(".json"))}
        report["gates_passed"] = all(report[key] for key in (
            "vq_codes_equal", "all_frames_equal", "exported_pcm_matches_delivered",
            "delivered_before_eos"))
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        if not report["gates_passed"]:
            raise RuntimeError("Real frame-stream state or export gate failed")
        print(json.dumps({key: value for key, value in report.items()
                          if key not in ("stream", "segments")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
