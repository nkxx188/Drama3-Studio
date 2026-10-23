"""Compare real cold-reference prefill with speculative reference-only warm-up."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor/fish-speech"), str(ROOT)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--text", default="你好，这是一次本地语音测试。")
    parser.add_argument("--seed", type=int, default=3040887657)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "research/2026-10-22-reference-prewarm-real")
    args = parser.parse_args()
    import torch
    from drama_low_vram.engine import DramaEngine, EngineOptions
    from drama_low_vram.protocols import VoiceReference
    from drama_low_vram.workflows import DramaWorkflows

    torch.set_num_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    references = [VoiceReference(str(args.reference.resolve()), args.transcript)]
    engine = DramaEngine(EngineOptions(
        checkpoint=args.checkpoint,
        expert_cache_dir=ROOT / ".cache/reference-prewarm-experts",
        compute_profile_dir=ROOT / ".cache/compute-profiles",
        expert_storage="auto", cache_experts=None, backbone_graph=True,
        combined_decode_bank=True, host_gather="auto", prefix_cache_mib=256,
        context_bucket_size=512, compute_policy="auto",
    ))
    report = {"checkpoint": str(args.checkpoint.resolve()), "seed": args.seed,
              "reference": str(args.reference.resolve()), "text": args.text,
              "scope": "One resident real GPU model; exact same reference, text and seed. "
                       "Cold control disables prefix reuse; warm-up performs no speech sampling."}
    try:
        started = time.perf_counter()
        engine.load()
        report["model_load_seconds"] = time.perf_counter() - started
        workflows = DramaWorkflows(engine)

        def run(name, *, cache=True):
            print(json.dumps({"phase": name, "time": time.time()}), flush=True)
            started = time.perf_counter()
            if cache:
                result = workflows.run(
                    "tts", args.text, output / (name + ".wav"),
                    references=references, seed=args.seed)
            else:
                with engine.prefix_cache.suspend():
                    result = workflows.run(
                        "tts", args.text, output / (name + ".wav"),
                        references=references, seed=args.seed)
            chunk = result["chunks"][0]
            measured = {
                "status": result["status"], "request_seconds": time.perf_counter() - started,
                "prefill_seconds": chunk["prefill_seconds"],
                "decode_seconds": chunk["decode_seconds"], "frames": chunk["frames"],
                "prefix_cache": chunk["prefix_cache"], "code_sha256": chunk["code_sha256"],
                "runtime_prepare_seconds": chunk["runtime_prepare_seconds"],
                "output": result["output"], "audio": result["audio"],
            }
            report[name] = measured
            print(json.dumps({"phase": name, **measured}), flush=True)
            return measured

        baseline = run("cold-control", cache=False)
        print(json.dumps({"phase": "prewarm"}), flush=True)
        started = time.perf_counter()
        warmup = engine.prewarm_reference_prefix(
            references, on_stage=lambda stage: print(
                json.dumps({"phase": "prewarm", "stage": stage}), flush=True))
        warmup.pop("_cache_key", None)
        report["prewarm"] = {**warmup, "seconds": time.perf_counter() - started}
        print(json.dumps({"phase": "prewarm", **report["prewarm"]}), flush=True)
        warmed = run("warmed")
        report["passed"] = bool(
            baseline["status"] == warmed["status"] == "complete"
            and warmup["status"] == "ready" and warmed["prefix_cache"]["hit"]
            and baseline["code_sha256"] == warmed["code_sha256"]
        )
        report["prefill_speedup"] = (
            baseline["prefill_seconds"] / max(warmed["prefill_seconds"], 1e-9))
    finally:
        engine.close()
        (output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["passed"]:
        raise RuntimeError("Reference prewarm correctness check failed; evidence preserved")


if __name__ == "__main__":
    main()
