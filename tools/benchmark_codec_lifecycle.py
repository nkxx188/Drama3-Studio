"""Completed repeated voice clones: full codec parking vs serial branch staging."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from benchmark_engine_decode import DramaEngine, EngineOptions, torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--cache-experts", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "research/phase7-codec-lifecycle-comparison.json")
    args = parser.parse_args()
    if args.repeats < 2 or args.cache_experts < 0:
        parser.error("At least two timed repeats and nonnegative expert cache required")
    torch.set_num_threads(1)
    short = "<|speaker:0|>你好，这是官方 FP8 质量优先测试。"
    text = "<|speaker:0|>今天我们继续测试这套本地语音引擎。希望普通电脑也能保持清楚自然的声音，读完一整句话，而不需要购买昂贵的专业显卡。接下来还会仔细检查语音克隆、长句停顿和声音的稳定性。"
    report = {
        "scope": "Completed same-reference clone under continuous requests. Each block has its own full warmup to establish the previous-request lifecycle. Original FP8/BF16 and top-8; auto admission retained.",
        "configuration": {
            "cpu_experts": 2, "native_threads": 4, "host_threads": 1,
            "cache_experts": args.cache_experts, "staged_experts": 8,
            "pair_rows": 64, "prefill_chunk": 32, "prefix_cache_mib": 256,
            "host_gather": "auto", "codec_policy": "auto",
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
        prefix_cache_mib=256, cpu_experts=2, host_gather="auto", codec_policy="auto",
    )) as engine:
        engine.load()
        # ABBA blocks; the warmup pays for any placement change from the previous
        # block. Both variants then measure the state left by their own request.
        counts = {"full": 0, "auto": 0}
        blocks = [("full", (args.repeats + 1) // 2), ("auto", args.repeats),
                  ("full", args.repeats // 2)]
        for block, (placement, repeats) in enumerate(blocks):
            engine.options = replace(engine.options, codec_placement=placement)
            for sample in range(-1, repeats):
                if sample >= 0:
                    counts[placement] += 1
                label = f"{placement}-block{block + 1}-{'warmup' if sample < 0 else counts[placement]}"
                output = ROOT / "outputs" / f"engine-phase7-codec-{label}.wav"
                last = time.perf_counter()

                def progress(frames, _limit):
                    nonlocal last
                    now = time.perf_counter()
                    if now - last >= 20:
                        print(json.dumps({"phase": label, "frames": frames}), flush=True)
                        last = now

                graphs = engine.graph, engine.backbone_graph
                result = engine.synthesize(
                    text, output, seed=123, max_new_tokens=768,
                    reference_audio=ROOT / "outputs/quality-official-fp8.wav",
                    reference_text=short, progress=progress,
                )
                if result["status"] != "complete":
                    raise RuntimeError("Incomplete clone; no lifecycle speed claim")
                chunks = result["chunks"]
                row = {
                    "variant": placement, "block": block + 1,
                    "repeat": 0 if sample < 0 else counts[placement],
                    "report_file": str(output.with_suffix(".json")), "audio": result["audio"],
                    "frames": [chunk["frames"] for chunk in chunks],
                    "code_hashes": [chunk["semantic_codes_sha256"] for chunk in chunks],
                    "wav_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                    "prefill_seconds": sum(chunk["prefill_seconds"] for chunk in chunks),
                    "decode_seconds": sum(chunk["decode_seconds"] for chunk in chunks),
                    "generation_seconds": sum(chunk["generation_seconds"] for chunk in chunks),
                    "runtime_prepare_seconds": sum(chunk["runtime_prepare_seconds"] for chunk in chunks),
                    "runtime_prepare_breakdown": [chunk["runtime_prepare_breakdown"] for chunk in chunks],
                    "codec_seconds": result["codec_seconds"], "total_seconds": result["total_seconds"],
                    "prefix_cache": [chunk["prefix_cache"] for chunk in chunks],
                    "batching": [
                        {key: chunk["runtime_plan"][key] for key in (
                            "cache_experts", "max_staged_experts", "max_pair_rows", "prefill_chunk")}
                        for chunk in chunks
                    ],
                    "h2d_bytes": result["offload_delta"]["h2d_bytes"],
                    "lifecycle": result["lifecycle_delta"],
                    "graphs_preserved": (
                        all(graph is not None for graph in graphs)
                        and engine.graph is graphs[0] and engine.backbone_graph is graphs[1]),
                }
                report["warmups" if sample < 0 else "requests"].append(row)
                save()
                print(json.dumps(row, ensure_ascii=False), flush=True)
                peers = report["warmups"] + report["requests"]
                if any(peer["code_hashes"] != row["code_hashes"] or
                       peer["wav_sha256"] != row["wav_sha256"] for peer in peers):
                    raise RuntimeError("Codec lifecycle changed complete speech")
        metrics = (
            "prefill_seconds", "decode_seconds", "generation_seconds",
            "runtime_prepare_seconds", "codec_seconds", "total_seconds",
        )
        rows = report["requests"]
        report["summary"] = {
            label: {
                key: statistics.median(row[key] for row in rows if row["variant"] == label)
                for key in metrics
            } for label in ("full", "auto")
        }
        report["total_time_reduction_fraction"] = (
            1 - report["summary"]["auto"]["total_seconds"] / report["summary"]["full"]["total_seconds"]
        )
        report["all_codes_and_wavs_equal"] = True
        save()
        print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
