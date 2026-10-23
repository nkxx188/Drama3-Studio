"""Real model equality across native gathering and stream completion policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from validate_state_prefix import (
    DramaEngine, EngineOptions, TowerDecodeGraph, SequenceSnapshot,
    build_drama3_prompt, byte_equal, state_equal, collect, compare, torch,
)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--cache-experts", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "research/phase6-runtime-numerics.json")
    parser.add_argument("--tensors", type=Path, default=ROOT / "research/phase6-runtime-numerics.pt")
    args = parser.parse_args()
    if args.frames < 4:
        parser.error("At least four diagnostic frames required")
    torch.set_num_threads(1)
    report = {
        "scope": "Original weights, all top-8, free-running diagnostic frames, hidden/logits/routes and valid KV/GDN/tower bytes. No performance claim.",
        "frames": args.frames, "cache_experts": args.cache_experts,
        "cases": [], "all_measured_equal": True,
    }
    artifacts = {}

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    with DramaEngine(EngineOptions(
        checkpoint=ROOT / "checkpoints/drama-3", expert_storage="mmap",
        cache_experts=args.cache_experts, staged_experts=8, pair_rows=64,
        prefill_chunk=32, backbone_graph=True, combined_decode_bank=True,
        cpu_experts=2, prefix_cache_mib=256,
    )) as engine:
        engine.load()
        text = "<|speaker:0|>你好，这是官方 FP8 质量优先测试。"
        codes, _ = engine.prepare_reference(ROOT / "outputs/quality-official-fp8.wav", text)
        prompts = [
            ("plain", build_drama3_prompt(text, engine.model.tokenizer, engine.model.config)),
            ("clone", build_drama3_prompt(
                text, engine.model.tokenizer, engine.model.config,
                prompt_texts=[text], prompt_tokens=[codes])),
        ]
        engine._prepare(max(len(prompt) for _, prompt in prompts) + args.frames)
        engine.graph.close()
        engine.graph = TowerDecodeGraph(
            engine.model.audio_decoder, engine.model.cache_capacity,
            compact_logits=True, record_logits=True,
        )
        original_step = engine.graph.step
        for label, prompt in prompts:
            expected = None
            for name, synchronize, gather in (
                ("baseline", True, "torch"), ("native_only", True, "auto"),
                ("async_only", False, "torch"), ("combined", False, "auto"),
            ):
                engine.manager.hybrid_decode.synchronize_return = synchronize
                engine.manager.configure_host_gather(gather)
                acoustic = []

                def record_step(*values, **kwargs):
                    result = original_step(*values, **kwargs)
                    acoustic.append(engine.graph.codebook_logits.cpu().clone())
                    return result

                # Suspend prefix reuse equally; the native prefill path is
                # exercised rather than hidden behind a previously cached state.
                with patch.object(engine.graph, "step", record_step):
                    actual = collect(engine, prompt, args.frames, cache=False)
                actual["acoustic_logits"] = torch.stack(acoustic)
                valid = len(prompt) + len(actual["frames"]) - 1
                actual["state"] = SequenceSnapshot.capture(
                    engine.model, backbone_rows=valid, tower_rows=valid,
                    tower_graph=engine.graph, include_rng=True,
                )
                if expected is None:
                    expected = actual
                checks = {
                    **compare(actual, expected),
                    "acoustic_logits_bytes_equal": byte_equal(
                        actual["acoustic_logits"], expected["acoustic_logits"]),
                    "state_bytes_equal": state_equal(actual["state"], expected["state"]),
                    "cuda_rng_bytes_equal": all(
                        byte_equal(actual["state"].rng.cuda[index], value)
                        for index, value in expected["state"].rng.cuda.items()),
                }
                row = {"condition": label, "variant": name, **checks}
                report["cases"].append(row)
                report["all_measured_equal"] &= all(checks.values())
                artifacts[f"{label}_{name}"] = {
                    key: value for key, value in actual.items() if key != "state"
                }
                save()
                print(json.dumps(row), flush=True)
                if not all(checks.values()):
                    torch.save(artifacts, args.tensors)
                    raise RuntimeError("Runtime submission numerical gate failed")
    torch.save(artifacts, args.tensors)
    save()


if __name__ == "__main__":
    main()
