"""Fixed-frame numerical gate for the shared multi-layer CPU/GPU runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
# Reuse cache placement, recording and comparison contracts from phase 3.
from validate_native_teacher_forcing import (
    DramaEngine, EngineOptions, build_drama3_prompt, compare, forced_backbone,
    generate, seed_sampling, torch,
)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--cpu-jobs", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "research/hybrid-decode-numerics.json")
    parser.add_argument("--tensors", type=Path, default=ROOT / "research/hybrid-decode-numerics.pt")
    args = parser.parse_args()
    if args.frames < 4 or any(not 1 <= jobs <= 8 for jobs in args.cpu_jobs):
        parser.error("At least four frames and 1..8 CPU jobs required")
    report = {
        "scope": "GPU prefill, identical fixed audio frames, shared native pool. Direct-slot GPU baseline; not free-running speech.",
        "frames_per_condition": args.frames,
        "native_threads": args.threads, "conditions": [],
        "all_measured_equal": True,
    }
    artifacts = []
    text = "<|speaker:0|>你好，这是官方 FP8 质量优先测试。"
    with DramaEngine(EngineOptions(
        checkpoint=ROOT / "checkpoints/drama-3", expert_storage="mmap",
        cache_experts=2, staged_experts=8, pair_rows=64, prefill_chunk=32,
        backbone_graph=True, combined_decode_bank=True,
    )) as engine:
        engine.load()
        codes, _ = engine.prepare_reference(ROOT / "outputs/quality-official-fp8.wav", text)
        for cloned in (False, True):
            label = "clone" if cloned else "plain"
            prompt = build_drama3_prompt(
                text, engine.model.tokenizer, engine.model.config,
                prompt_texts=[text] if cloned else [], prompt_tokens=[codes] if cloned else [])
            engine._prepare(1024)
            seed_sampling(123)
            with patch.object(engine.manager, "hybrid_decode", None):
                target = generate(
                    engine.model, prompt, max_new_tokens=args.frames,
                    prefill_chunk_size=32, tower_graph=engine.graph,
                    backbone_graph=engine.backbone_graph)
            if len(target.frames) != args.frames:
                raise RuntimeError("Target ended before requested diagnostic window")
            reference = forced_backbone(engine, prompt, target.frames)
            row = {
                "condition": label, "prompt_rows": len(prompt),
                "target_sha256": hashlib.sha256(target.frames.numpy().tobytes()).hexdigest(),
                "variants": [],
            }
            artifact = {"condition": label, "prompt": prompt, "frames": target.frames,
                        "reference": reference, "variants": {}}
            for jobs in args.cpu_jobs:
                engine.manager.configure_hybrid_decode(jobs, threads=args.threads)
                hybrid = engine.manager.hybrid_decode
                selections = (
                    ("selected_0_20_39", [0, 20, 39]),
                    ("alternating_half", lambda step: range(step % 2, 40, 2)),
                    ("all_40", range(40)),
                )
                for name, layers in selections:
                    actual = forced_backbone(
                        engine, prompt, target.frames, hybrid_layers=layers)
                    summary = {
                        "name": name, "cpu_jobs": jobs, **compare(actual, reference),
                        "shared_runtime": hybrid.report(),
                    }
                    row["variants"].append(summary)
                    artifact["variants"][f"{name}_cpu{jobs}"] = actual
                    equal = (
                        summary["hidden"]["equal"]
                        and summary["semantic_logits"]["equal"]
                        and summary["route_slot_equal_fraction"] == 1
                    )
                    report["all_measured_equal"] &= equal
                    print(json.dumps({"condition": label, **summary}), flush=True)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(
                        report | {"conditions": report["conditions"] + [row]},
                        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    if not equal:
                        torch.save(artifacts + [artifact], args.tensors)
                        raise RuntimeError("Numerical gate failed; inspect saved frames before speech benchmarks")
            report["conditions"].append(row)
            artifacts.append(artifact)
    torch.save(artifacts, args.tensors)
    report["tensor_artifact"] = str(args.tensors)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
