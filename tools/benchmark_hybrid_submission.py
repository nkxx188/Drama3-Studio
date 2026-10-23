"""Matched hybrid completion-fence A/B, with optional completed voice clones."""

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
from benchmark_engine_decode import (
    DramaEngine, EngineOptions, build_drama3_prompt, generate, seed_sampling, torch,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cache-experts", type=int, default=2)
    parser.add_argument("--complete", action="store_true")
    parser.add_argument("--compare", choices=("completion", "gather", "combined", "all"), default="completion")
    parser.add_argument("--output", type=Path, default=ROOT / "research/hybrid-submission-comparison.json")
    args = parser.parse_args()
    if args.frames < 2 or args.repeats < 1 or args.cache_experts < 0:
        parser.error("At least two frames, positive repeats and nonnegative cache required")
    torch.set_num_threads(1)
    short = "<|speaker:0|>你好，这是官方 FP8 质量优先测试。"
    text = "<|speaker:0|>今天我们继续测试这套本地语音引擎。希望普通电脑也能保持清楚自然的声音，读完一整句话，而不需要购买昂贵的专业显卡。接下来还会仔细检查语音克隆、长句停顿和声音的稳定性。"
    report = {
        "scope": (
            "Matched completed clone, same original weights, prefix reuse and forced codec parking."
            if args.complete else
            "Matched truncated clone diagnostic; no codec or completed-speech speedup claim."
        ),
        "configuration": {
            "cpu_experts": 2, "native_threads": 4, "host_threads": 1,
            "cache_experts": args.cache_experts, "staged_experts": 8,
            "pair_rows": 64, "prefill_chunk": 32, "prefix_cache_mib": 256,
            "backbone_graph": True, "codec_policy": "park",
            "comparison": args.compare,
        },
        "warmups": [], "requests": [],
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with DramaEngine(EngineOptions(
        checkpoint=ROOT / "checkpoints/drama-3", expert_storage="mmap",
        cache_experts=args.cache_experts, staged_experts=8, pair_rows=64,
        prefill_chunk=32, backbone_graph=True, combined_decode_bank=True,
        prefix_cache_mib=256, cpu_experts=2, native_threads=4, codec_policy="park",
    )) as engine:
        engine.load()
        hybrid = engine.manager.hybrid_decode
        variants = {
            "completion": [("host_fence", True, "torch"), ("stream_fence", False, "torch")],
            "gather": [("torch_gather", True, "torch"), ("native_gather", True, "auto")],
            "combined": [("baseline", True, "torch"), ("optimized", False, "auto")],
            "all": [
                ("baseline", True, "torch"), ("native_only", True, "auto"),
                ("async_only", False, "torch"), ("combined", False, "auto"),
            ],
        }[args.compare]
        codes, _ = engine.prepare_reference(ROOT / "outputs/quality-official-fp8.wav", short)
        prompt = build_drama3_prompt(
            text if args.complete else short, engine.model.tokenizer, engine.model.config,
            prompt_texts=[short], prompt_tokens=[codes],
        )
        if not args.complete:
            plan, _ = engine._prepare(len(prompt) + args.frames)
        for repeat in range(-1, args.repeats):
            order = variants if repeat % 2 == 0 else list(reversed(variants))
            # A completed warmup also warms the cold expert file pages and codec.
            if args.complete and repeat < 0:
                order = variants[:1]
            for label, synchronize, gather in order:
                hybrid.synchronize_return = synchronize
                engine.manager.configure_host_gather(gather)
                suffix = f"{label}-{repeat + 1}"
                before = hybrid.report()
                before_offload = engine.manager.stats.as_dict()
                if args.complete:
                    output = ROOT / "outputs" / f"engine-phase6-submission-{suffix}.wav"
                    last = time.perf_counter()

                    def progress(frames, _limit):
                        nonlocal last
                        now = time.perf_counter()
                        if now - last >= 20:
                            print(json.dumps({"variant": suffix, "frames": frames}), flush=True)
                            last = now

                    result = engine.synthesize(
                        text, output, seed=123, max_new_tokens=768,
                        reference_audio=ROOT / "outputs/quality-official-fp8.wav",
                        reference_text=short, progress=progress,
                    )
                    if result["status"] != "complete":
                        raise RuntimeError("Incomplete clone; no performance claim")
                    row = {
                        "report_file": str(output.with_suffix(".json")),
                        "code_hashes": [chunk["semantic_codes_sha256"] for chunk in result["chunks"]],
                        "wav_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                        "frames": [chunk["frames"] for chunk in result["chunks"]],
                        "decode_seconds": sum(chunk["decode_seconds"] for chunk in result["chunks"]),
                        "generation_seconds": sum(chunk["generation_seconds"] for chunk in result["chunks"]),
                        "runtime_prepare_seconds": sum(chunk["runtime_prepare_seconds"] for chunk in result["chunks"]),
                        "total_seconds": result["total_seconds"],
                        "lifecycle": result["lifecycle_delta"],
                    }
                else:
                    seed_sampling(123)
                    result = generate(
                        engine.model, prompt, max_new_tokens=args.frames,
                        prefill_chunk_size=plan.prefill_chunk, tower_graph=engine.graph,
                        backbone_graph=engine.backbone_graph, prefix_cache=engine.prefix_cache,
                    )
                    row = {
                        **result.as_dict(),
                        "code_hashes": [hashlib.sha256(result.frames.numpy().tobytes()).hexdigest()],
                    }
                after = hybrid.report()
                after_offload = engine.manager.stats.as_dict()
                row.update({
                    "variant": label, "repeat": repeat + 1,
                    "host_gather": gather,
                    "offload_delta": {
                        key: after_offload[key] - before_offload[key]
                        for key in ("native_gather_calls", "torch_gather_calls",
                                    "host_gather_seconds", "h2d_bytes")
                    },
                    "hybrid_delta": {
                        key: after[key] - before[key]
                        for key in ("calls", "completion_host_waits", "completion_stream_waits",
                                    "boundary_fallback_calls", "boundary_recomputed_pairs")
                    },
                })
                report["warmups" if repeat < 0 else "requests"].append(row)
                save()
                print(json.dumps(row, ensure_ascii=False), flush=True)
                peers = report["warmups"] + report["requests"]
                if any(peer["code_hashes"] != row["code_hashes"] for peer in peers):
                    raise RuntimeError("Completion scheduling changed generated codes")
                if args.complete and any(peer["wav_sha256"] != row["wav_sha256"] for peer in peers):
                    raise RuntimeError("Completion scheduling changed WAV")
        rows = report["requests"]
        keys = ["decode_seconds", "generation_seconds"]
        if args.complete:
            keys += ["total_seconds", "runtime_prepare_seconds"]
        report["summary"] = {
            label: {
                key: statistics.median(row[key] for row in rows if row["variant"] == label)
                for key in keys
            } for label, _, _ in variants
        }
        report["all_codes_equal"] = len({tuple(row["code_hashes"]) for row in rows}) == 1
        save()
        print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
