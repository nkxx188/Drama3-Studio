"""Matched completed speech measurements of GPU-only and shared CPU/GPU decode."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from benchmark_engine_decode import (
    DramaEngine, EngineOptions, build_drama3_prompt, generate, seed_sampling, torch,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--cpu-jobs", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--warmup-utterances", type=int, default=2,
                        help="Complete GPU utterances per case before timing, to warm expert file pages and codec lifecycle.")
    parser.add_argument("--cases", nargs="+", choices=("short", "clone"), default=["short", "clone"])
    parser.add_argument("--codec-policy", choices=("auto", "park"), default="auto")
    parser.add_argument("--output", type=Path, default=ROOT / "research/engine-hybrid-comparison.json")
    parser.add_argument("--prefix", default="engine-hybrid")
    args = parser.parse_args()
    if (args.repeats < 1 or not args.cpu_jobs or args.torch_threads < 1
            or args.warmup_utterances < 0
            or any(not 1 <= jobs <= 8 for jobs in args.cpu_jobs)):
        parser.error("Positive repeats/threads and 1..8 CPU experts required")
    # Host gathering and native work share CPU bandwidth. State the setting and
    # apply it equally to baseline and hybrid; do not change the engine global default.
    torch.set_num_threads(args.torch_threads)
    short = "<|speaker:0|>你好，这是官方 FP8 质量优先测试。"
    long = "<|speaker:0|>今天我们继续测试这套本地语音引擎。希望普通电脑也能保持清楚自然的声音，读完一整句话，而不需要购买昂贵的专业显卡。接下来还会仔细检查语音克隆、长句停顿和声音的稳定性。"
    report = {
        "scope": "Same engine, direct-slot GPU baseline, official FP8/BF16 weights, top-8, seed and completed utterances. GPU prefill in every variant.",
        "requests": [], "warmups": [], "repeats_per_variant": args.repeats,
        "warmup_utterances_per_case": args.warmup_utterances,
        "configuration": {
            "cache_experts": 2, "staged_experts": 8, "pair_rows": 64,
            "prefill_chunk": 32, "backbone_graph": True,
            "combined_decode_bank": True, "native_threads": args.threads,
            "torch_intraop_threads": torch.get_num_threads(),
            "expert_storage": "mmap", "codec_policy": args.codec_policy,
        },
        "runtime": {"torch": torch.__version__, "device": torch.cuda.get_device_name()},
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with DramaEngine(EngineOptions(
        checkpoint=ROOT / "checkpoints/drama-3", expert_storage="mmap",
        cache_experts=2, staged_experts=8, pair_rows=64, prefill_chunk=32,
        backbone_graph=True, combined_decode_bank=True,
        cpu_experts=args.cpu_jobs[0], native_threads=args.threads,
        codec_policy=args.codec_policy,
    )) as engine:
        engine.load()
        hybrid = engine.manager.hybrid_decode
        engine.prepare_reference(ROOT / "outputs/quality-official-fp8.wav", short)
        engine._prepare(1024)
        prompt = build_drama3_prompt(short, engine.model.tokenizer, engine.model.config)
        variants = [0, *dict.fromkeys(args.cpu_jobs)]
        for jobs in variants:
            hybrid.cpu_jobs = jobs or args.cpu_jobs[0]
            with hybrid.suspend() if not jobs else nullcontext():
                seed_sampling(123)
                generate(
                    engine.model, prompt, max_new_tokens=8, prefill_chunk_size=32,
                    tower_graph=engine.graph, backbone_graph=engine.backbone_graph)
        for label, text, maximum, cloned in (
            ("short", short, 256, False), ("clone", long, 768, True),
        ):
            if label not in args.cases:
                continue
            # Eight-frame compilation alone does not warm all experts used by
            # a completed utterance. Exclude full warmups from timed samples.
            for warmup in range(args.warmup_utterances):
                output = ROOT / "outputs" / f"{args.prefix}-{label}-warmup-{warmup + 1}.wav"
                with hybrid.suspend():
                    result = engine.synthesize(
                        text, output, seed=123, max_new_tokens=maximum,
                        reference_audio=ROOT / "outputs/quality-official-fp8.wav" if cloned else None,
                        reference_text=short if cloned else None)
                if result["status"] != "complete":
                    raise RuntimeError("Incomplete warmup; no performance claim")
                report["warmups"].append({
                    "case": label, "report_file": str(output.with_suffix(".json")),
                    "frames": [chunk["frames"] for chunk in result["chunks"]],
                    "generation_seconds": sum(chunk["generation_seconds"] for chunk in result["chunks"]),
                    "code_hashes": [chunk.get("semantic_codes_sha256") for chunk in result["chunks"]],
                })
                save()
                print(json.dumps({"phase": "complete_utterance_warmup", **report["warmups"][-1]}), flush=True)
            for repeat in range(args.repeats):
                # Forward then reverse ordering reduces a simple warming bias.
                order = variants if repeat % 2 == 0 else list(reversed(variants))
                for jobs in order:
                    hybrid.cpu_jobs = jobs or args.cpu_jobs[0]
                    name = "gpu" if jobs == 0 else f"cpu{jobs}"
                    output = ROOT / "outputs" / f"{args.prefix}-{label}-{name}-{repeat + 1}.wav"
                    before = hybrid.report()
                    last_progress = time.perf_counter()

                    def progress(count, _limit):
                        nonlocal last_progress
                        now = time.perf_counter()
                        if now - last_progress >= 20:
                            print(json.dumps({
                                "phase": label, "variant": name, "repeat": repeat + 1,
                                "frames": count,
                            }), flush=True)
                            last_progress = now

                    with hybrid.suspend() if not jobs else nullcontext():
                        result = engine.synthesize(
                            text, output, seed=123, max_new_tokens=maximum,
                            reference_audio=ROOT / "outputs/quality-official-fp8.wav" if cloned else None,
                            reference_text=short if cloned else None, progress=progress)
                    after = hybrid.report()
                    row = {
                        "case": label, "variant": name, "cpu_jobs": jobs,
                        "repeat": repeat + 1, "status": result["status"],
                        "audio": result["audio"], "report_file": str(output.with_suffix(".json")),
                        "frames": [chunk["frames"] for chunk in result["chunks"]],
                        "code_hashes": [chunk.get("semantic_codes_sha256") for chunk in result["chunks"]],
                        "generation_seconds": sum(chunk["generation_seconds"] for chunk in result["chunks"]),
                        "prefill_seconds": sum(chunk["prefill_seconds"] for chunk in result["chunks"]),
                        "decode_seconds": sum(chunk["decode_seconds"] for chunk in result["chunks"]),
                        "first_frame_seconds": [chunk["first_frame_seconds"] for chunk in result["chunks"]],
                        "total_seconds": result["total_seconds"],
                        "codec_seconds": result["codec_seconds"],
                        "lifecycle": result["lifecycle_delta"],
                        "h2d_bytes": result["offload_delta"]["h2d_bytes"],
                        "hybrid_delta": {
                            key: after[key] - before[key]
                            for key in ("calls", "cpu_pairs", "gpu_pairs", "control_d2h_bytes",
                                        "partial_h2d_bytes", "workspace_allocations", "fallbacks",
                                        "boundary_fallback_calls", "boundary_recomputed_pairs")},
                        "peak_tensor_bytes": max(chunk["peak_allocated_bytes"] for chunk in result["chunks"]),
                        "peak_reserved_bytes": max(chunk["peak_reserved_bytes"] for chunk in result["chunks"]),
                        "memory": result["memory"],
                        "batching": [{
                            key: chunk["runtime_plan"][key]
                            for key in ("max_staged_experts", "max_pair_rows", "prefill_chunk", "cache_experts")
                        } for chunk in result["chunks"]],
                        "wav_sha256": hashlib.sha256(output.read_bytes()).hexdigest() if output.is_file() else None,
                    }
                    report["requests"].append(row)
                    save()
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                    if result["status"] != "complete":
                        raise RuntimeError("Incomplete speech; no performance claim")
                    peers = [item for item in report["requests"] if item["case"] == label]
                    if any(item["code_hashes"] != row["code_hashes"] for item in peers):
                        raise RuntimeError("Free-running codes differ; inspect before reporting equal-quality performance")
    report["summary"] = []
    for label in dict.fromkeys(args.cases):
        rows = [row for row in report["requests"] if row["case"] == label]
        baseline = [row for row in rows if row["cpu_jobs"] == 0]
        for jobs in dict.fromkeys(args.cpu_jobs):
            mixed = [row for row in rows if row["cpu_jobs"] == jobs]
            report["summary"].append({
                "case": label, "cpu_jobs": jobs,
                "all_code_hashes_equal": len({tuple(row["code_hashes"]) for row in rows}) == 1,
                "all_wav_hashes_equal": len({row["wav_sha256"] for row in rows}) == 1,
                "all_frames_equal": len({tuple(row["frames"]) for row in rows}) == 1,
                "all_batching_equal": len({json.dumps(row["batching"], sort_keys=True) for row in rows}) == 1,
                "all_lifecycle_counts_equal": len({
                    tuple(row["lifecycle"][key] for key in (
                        "runtime_reuses", "runtime_rebuilds", "park_calls", "resume_calls",
                        "codec_coexist_calls", "codec_park_calls", "codec_cache_reclaims"))
                    for row in rows
                }) == 1,
                **{
                    f"{name}_{key}": statistics.median(row[key] for row in values)
                    for name, values in (("gpu", baseline), ("hybrid", mixed))
                    for key in ("generation_seconds", "decode_seconds", "total_seconds", "h2d_bytes")
                },
                "generation_time_reduction_fraction": 1 - (
                    statistics.median(row["generation_seconds"] for row in mixed)
                    / statistics.median(row["generation_seconds"] for row in baseline)),
            })
    save()
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
