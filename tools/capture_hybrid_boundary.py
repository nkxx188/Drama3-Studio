"""Reproduce the unguarded FP8 boundary, or check guarded expert outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from validate_native_teacher_forcing import (
    DramaEngine, EngineOptions, ExpertStore, forced_backbone, torch,
)
from diagnose_native_stages import gpu_stages


class FoundBoundary(Exception):
    pass


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "research/hybrid-decode-numerics.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "research/hybrid-boundary-case.pt")
    parser.add_argument("--guarded", action="store_true",
                        help="Retain production recovery and require every expert output to match.")
    args = parser.parse_args()
    torch.set_num_threads(1)
    artifact = torch.load(args.source, weights_only=True)[0]
    # The original diagnostic artifact includes frames in the reference input.
    prompt = artifact.get("prompt")
    frames = artifact.get("frames")
    if prompt is None or frames is None:
        # Older phase-4 artifacts omitted them; the corresponding GPU target
        # is deterministic and can be regenerated before enabling native work.
        from validate_native_teacher_forcing import build_drama3_prompt, generate, seed_sampling
    with DramaEngine(EngineOptions(
        checkpoint=ROOT / "checkpoints/drama-3", expert_storage="mmap",
        cache_experts=2, staged_experts=8, pair_rows=64, prefill_chunk=32,
        backbone_graph=True, combined_decode_bank=True,
    )) as engine:
        engine.load()
        engine._prepare(1024)
        if frames is None:
            text = "<|speaker:0|>你好，这是官方 FP8 质量优先测试。"
            prompt = build_drama3_prompt(text, engine.model.tokenizer, engine.model.config)
            seed_sampling(123)
            frames = generate(
                engine.model, prompt, max_new_tokens=len(artifact["reference"]["hidden"]),
                prefill_chunk_size=32, tower_graph=engine.graph,
                backbone_graph=engine.backbone_graph).frames
        engine.manager.configure_hybrid_decode(2, threads=4)
        hybrid = engine.manager.hybrid_decode
        if not args.guarded:
            # Disable recovery only in this diagnostic executor, to preserve a
            # reproducible example of the original phase-4 failure.
            hybrid.executor.library.check(
                hybrid.executor.library.set_quantization_guard(hybrid.executor._handle, 0))
        original = hybrid.forward
        cpu_original = hybrid._cpu
        last = {}
        lookup = {id(module): index for index, module in enumerate(engine.manager.modules.values())}
        calls = 0

        def capture_cpu(store, ids, weights):
            last["ids"], last["weights"] = ids.clone(), weights.clone()
            return cpu_original(store, ids, weights)

        def check(module, store, x, ids, weights):
            nonlocal calls
            actual = original(module, store, x, ids, weights)
            with hybrid.suspend():
                expected = engine.manager.forward(module, x, ids, weights)
            if not torch.equal(
                actual.contiguous().view(torch.uint8),
                expected.contiguous().view(torch.uint8),
            ):
                row = {
                    "layer": lookup[id(module)], "decode_step": calls // 40,
                    "guard_enabled": args.guarded,
                    "x": x.cpu(), "ids": ids.cpu(), "weights": weights.cpu(),
                    "actual": actual.cpu(), "expected": expected.cpu(),
                    "cpu_ids": last["ids"], "cpu_weights": last["weights"],
                }
                with engine.manager.pool.session(store, engine.device):
                    staged = engine.manager.pool.stage(store, last["ids"])
                    gpu = gpu_stages(
                        engine.manager.backend, x, last["weights"].to(engine.device), staged)
                row["gpu"] = {
                    name: value.cpu() if isinstance(value, torch.Tensor) else value
                    for name, value in gpu.items()}
                torch.save(row, args.output)
                torch.save({
                    "layer": row["layer"],
                    "cases": [{
                        "x": row["x"], "ids": row["cpu_ids"],
                        "weights": row["cpu_weights"],
                        "output": row["gpu"]["partials"].sum(0).bfloat16(),
                    }],
                }, args.output.with_name(args.output.stem + "-diagnostic.pt"))
                print(json.dumps({
                    "layer": row["layer"], "decode_step": row["decode_step"],
                    "cpu_ids": row["cpu_ids"].tolist(), "output": str(args.output),
                    "different_elements": int((actual != expected).sum()),
                }), flush=True)
                raise FoundBoundary()
            calls += 1
            return actual

        try:
            with patch.object(hybrid, "_cpu", capture_cpu), patch.object(hybrid, "forward", check):
                forced_backbone(engine, prompt, frames, hybrid_layers=range(40))
        except FoundBoundary:
            if args.guarded:
                raise RuntimeError("Guarded expert output differs from GPU")
            return
        if args.guarded:
            print(json.dumps({"guard_enabled": True, "checked_expert_calls": calls,
                              "all_outputs_equal": True}), flush=True)
            return
        raise RuntimeError("No mismatch found in the supplied diagnostic window")


if __name__ == "__main__":
    main()
