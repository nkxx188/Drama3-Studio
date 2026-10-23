"""Complete utterance acceptance for measured CPU/GPU compute policies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from benchmark_engine_decode import DramaEngine, EngineOptions, torch
from drama_low_vram.calibration import apply_compute_selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--cases", nargs="+", choices=("short", "clone"),
                        default=["short", "clone"])
    parser.add_argument("--output", type=Path,
                        default=ROOT / "research/compute-profile-full-acceptance.json")
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("At least two counterbalanced repetitions required")
    torch.set_num_threads(1)
    short = "<|speaker:0|>你好，这是官方 FP8 质量优先测试。"
    long = ("<|speaker:0|>今天我们继续测试这套本地语音引擎。"
            "希望普通电脑也能保持清楚自然的声音，读完一整句话，而不需要购买昂贵的专业显卡。"
            "接下来还会仔细检查语音克隆、长句停顿和声音的稳定性。")
    variants = [
        ("gpu-torch", dict(cpu_experts=0, native_threads=4, host_gather="torch")),
        ("gpu-native", dict(cpu_experts=0, native_threads=4, host_gather="auto")),
        ("cpu2-threads4", dict(cpu_experts=2, native_threads=4, host_gather="auto")),
        ("cpu2-threads2", dict(cpu_experts=2, native_threads=2, host_gather="auto")),
    ]
    report = {
        "scope": "Completed EOS utterances, original official bytes and top-8. Identical inputs, sampling, cache and codec policy; two counterbalanced repetitions after a full warmup per case.",
        "configuration": dict(
            cache_experts=1, staged_experts=8, pair_rows=64, prefill_chunk=32,
            backbone_graph=True, combined_decode_bank=True, context_bucket_size=512,
            torch_threads=1, expert_storage="mmap"),
        "warmups": [], "requests": [],
    }

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")

    with DramaEngine(EngineOptions(
        checkpoint=ROOT / "checkpoints/drama-3", expert_storage="mmap",
        cache_experts=1, staged_experts=8, pair_rows=64, prefill_chunk=32,
        backbone_graph=True, combined_decode_bank=True,
        cpu_experts=2, native_threads=2, host_gather="auto",
    )) as engine:
        engine.load()
        engine.prepare_reference(ROOT / "outputs/quality-official-fp8.wav", short)
        engine._prepare(1024)
        for label, text, maximum, cloned in (
            ("short", short, 256, False), ("clone", long, 768, True),
        ):
            if label not in args.cases:
                continue
            kwargs = dict(
                seed=123, max_new_tokens=maximum,
                reference_audio=ROOT / "outputs/quality-official-fp8.wav" if cloned else None,
                reference_text=short if cloned else None)

            def run(name, repetition):
                output = ROOT / "outputs" / f"compute-policy-{label}-{name}-{repetition}.wav"
                last = time.perf_counter()

                def progress(count, _limit):
                    nonlocal last
                    now = time.perf_counter()
                    if now - last >= 20:
                        print(json.dumps(dict(
                            case=label, variant=name, repeat=repetition, frames=count)),
                            flush=True)
                        last = now

                result = engine.synthesize(text, output, progress=progress, **kwargs)
                row = dict(
                    case=label, variant=name, repeat=repetition, status=result["status"],
                    audio=result["audio"], total_seconds=result["total_seconds"],
                    generation_seconds=sum(c["generation_seconds"] for c in result["chunks"]),
                    decode_seconds=sum(c["decode_seconds"] for c in result["chunks"]),
                    prefill_seconds=sum(c["prefill_seconds"] for c in result["chunks"]),
                    frames=[c["frames"] for c in result["chunks"]],
                    codes=[c["semantic_codes_sha256"] for c in result["chunks"]],
                    lifecycle=result["lifecycle_delta"],
                    h2d_bytes=result["offload_delta"]["h2d_bytes"],
                    peak_allocated_bytes=max(c["peak_allocated_bytes"] for c in result["chunks"]),
                    wav_sha256=hashlib.sha256(output.read_bytes()).hexdigest()
                    if output.is_file() else None,
                    report_file=str(output.with_suffix(".json")))
                if result["status"] != "complete":
                    raise RuntimeError(f"Incomplete {label}/{name}; no acceptance claim")
                return row

            warmup = run("warmup", 0)
            report["warmups"].append(warmup)
            save()
            print(json.dumps(warmup, ensure_ascii=False), flush=True)
            for repetition in range(args.repeats):
                order = variants if repetition % 2 == 0 else list(reversed(variants))
                for name, candidate in order:
                    apply_compute_selection(engine, {"candidate": candidate})
                    row = run(name, repetition + 1)
                    row["candidate"] = candidate
                    report["requests"].append(row)
                    save()
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                    if row["codes"] != warmup["codes"]:
                        raise RuntimeError("Free-running code bytes differ from the warmup")
        report["summary"] = []
        for label in args.cases:
            rows = [row for row in report["requests"] if row["case"] == label]
            baseline = statistics.median(
                row["total_seconds"] for row in rows if row["variant"] == "gpu-torch")
            for name, _candidate in variants:
                values = [row for row in rows if row["variant"] == name]
                median = statistics.median(row["total_seconds"] for row in values)
                report["summary"].append(dict(
                    case=label, variant=name, median_total_seconds=median,
                    median_decode_seconds=statistics.median(
                        row["decode_seconds"] for row in values),
                    time_reduction_vs_gpu_torch=1 - median / baseline,
                    all_case_codes_equal=len({tuple(row["codes"]) for row in rows}) == 1,
                    all_case_wavs_equal=len({row["wav_sha256"] for row in rows}) == 1,
                    all_case_frames_equal=len({tuple(row["frames"]) for row in rows}) == 1))
        report["all_complete"] = True
        save()
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
