"""Matched full-utterance official/direct FP8 decode comparison on one engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor/fish-speech")]
for key, name in (("TRITON_CACHE_DIR", "triton"), ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
                  ("HF_HOME", "huggingface"), ("TEMP", "tmp"), ("TMP", "tmp")):
    directory = ROOT / ".cache" / name
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(directory)

import torch

from drama_low_vram.engine import DramaEngine, EngineOptions, seed_sampling
from drama_low_vram.generation import generate
from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--cases", nargs="+", choices=("short", "clone"), default=["short", "clone"])
    parser.add_argument("--staged-experts", type=int, default=8)
    parser.add_argument("--pair-rows", type=int, default=64)
    parser.add_argument("--prefix", default="engine-decode")
    parser.add_argument("--output", type=Path, default=ROOT / "research/engine-decode-comparison.json")
    args = parser.parse_args()
    if min(args.repeats, args.staged_experts, args.pair_rows) < 1:
        parser.error("Positive repeats required")
    short = "<|speaker:0|>你好，这是官方 FP8 质量优先测试。"
    long = "<|speaker:0|>今天我们继续测试这套本地语音引擎。希望普通电脑也能保持清楚自然的声音，读完一整句话，而不需要购买昂贵的专业显卡。接下来还会仔细检查语音克隆、长句停顿和声音的稳定性。"
    report = {
        "scope": "Same engine, weights, sampling, text, references and cache; only FP8 batch-one kernel selection changes.",
        "repeats_per_variant": args.repeats, "requests": [],
        "runtime": {"torch": torch.__version__, "device": torch.cuda.get_device_name()},
        "configuration": {
            "expert_storage": "mmap", "cache_experts": 2, "staged_experts": args.staged_experts,
            "pair_rows": args.pair_rows, "prefill_chunk": 32, "backbone_graph": True,
            "combined_decode_bank": True, "codec_policy": "auto",
        },
    }

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with DramaEngine(EngineOptions(
        checkpoint=ROOT / "checkpoints/drama-3", expert_storage="mmap",
        cache_experts=2, staged_experts=args.staged_experts,
        pair_rows=args.pair_rows, prefill_chunk=32,
        backbone_graph=True, combined_decode_bank=True,
    )) as engine:
        engine.load()
        engine.prepare_reference(ROOT / "outputs/quality-official-fp8.wav", short)
        engine._prepare(1024)
        prompt = build_drama3_prompt(short, engine.model.tokenizer, engine.model.config)
        # Compile both paths outside the completed-utterance timings.
        for direct in (False, True):
            engine.manager.backend.decode_specialization = direct
            seed_sampling(123)
            generate(
                engine.model, prompt, max_new_tokens=8, prefill_chunk_size=32,
                tower_graph=engine.graph, backbone_graph=engine.backbone_graph)
        for label, text, maximum, cloned in (("short", short, 256, False),
                                               ("clone", long, 768, True)):
            if label not in args.cases:
                continue
            for repeat in range(args.repeats):
                order = (False, True) if repeat % 2 == 0 else (True, False)
                for direct in order:
                    engine.manager.backend.decode_specialization = direct
                    name = "direct" if direct else "official"
                    output = ROOT / "outputs" / f"{args.prefix}-{label}-{name}-{repeat + 1}.wav"
                    progress_started = time.perf_counter()
                    last_progress = progress_started

                    def progress(count, limit):
                        nonlocal last_progress
                        now = time.perf_counter()
                        if now - last_progress >= 20:
                            print(json.dumps({
                                "phase": label, "variant": name, "repeat": repeat + 1,
                                "frames": count, "elapsed_seconds": now - progress_started,
                            }), flush=True)
                            last_progress = now

                    result = engine.synthesize(
                        text, output, max_new_tokens=maximum, seed=123,
                        reference_audio=ROOT / "outputs/quality-official-fp8.wav" if cloned else None,
                        reference_text=short if cloned else None, progress=progress,
                    )
                    row = {
                        "case": label, "variant": name, "repeat": repeat + 1,
                        "report_file": str(output.with_suffix(".json")),
                        "status": result["status"], "audio": result["audio"],
                        "frames": [chunk["frames"] for chunk in result["chunks"]],
                        "code_hashes": [chunk.get("semantic_codes_sha256") for chunk in result["chunks"]],
                        "generation_seconds": sum(chunk["generation_seconds"] for chunk in result["chunks"]),
                        "prefill_seconds": sum(chunk["prefill_seconds"] for chunk in result["chunks"]),
                        "decode_seconds": sum(chunk["decode_seconds"] for chunk in result["chunks"]),
                        "total_seconds": result["total_seconds"],
                        "lifecycle": result["lifecycle_delta"],
                        "h2d_bytes": result["offload_delta"]["h2d_bytes"],
                        "peak_tensor_bytes": max(chunk["peak_allocated_bytes"] for chunk in result["chunks"]),
                        "peak_reserved_bytes": max(chunk["peak_reserved_bytes"] for chunk in result["chunks"]),
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
                        raise RuntimeError(f"Incomplete {label} {name}; no performance claim")
    summaries = []
    for label in dict.fromkeys(args.cases):
        rows = [row for row in report["requests"] if row["case"] == label]
        official = [row["generation_seconds"] for row in rows if row["variant"] == "official"]
        direct = [row["generation_seconds"] for row in rows if row["variant"] == "direct"]
        summaries.append({
            "case": label,
            "all_code_hashes_equal": len({tuple(row["code_hashes"]) for row in rows}) == 1,
            "all_frames_equal": len({tuple(row["frames"]) for row in rows}) == 1,
            "all_wav_hashes_equal": len({row["wav_sha256"] for row in rows}) == 1,
            "all_batching_equal": len({json.dumps(row["batching"], sort_keys=True) for row in rows}) == 1,
            "official_generation_median_seconds": statistics.median(official),
            "direct_generation_median_seconds": statistics.median(direct),
            "generation_time_reduction_fraction": 1 - statistics.median(direct) / statistics.median(official),
        })
    report["summary"] = summaries
    save()
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
