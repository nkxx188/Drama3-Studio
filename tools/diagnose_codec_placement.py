"""Capture real short-speech codes and locate memory-dependent codec differences."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from benchmark_engine_decode import (
    DramaEngine, EngineOptions, build_drama3_prompt, generate, seed_sampling, torch,
)
from drama_low_vram.codec import load_codec
from validate_state_prefix import byte_equal


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--order", nargs="+", choices=("auto", "full"),
                        default=["auto", "full", "auto", "full"])
    parser.add_argument("--force-park", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "research/phase7-codec-stage-diagnosis.json")
    parser.add_argument("--codes", type=Path, default=ROOT / "research/phase7-short-codes.pt")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = args.deterministic
    report = {
        "scope": "Identical real short-speech codes, stage outputs, full vs split codec under resident backbone. Diagnostic only.",
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark, "runs": [],
        "force_park": args.force_park,
    }
    with DramaEngine(EngineOptions(
        checkpoint=ROOT / "checkpoints/drama-3", expert_storage="mmap",
        cache_experts=1, staged_experts=8, pair_rows=64, prefill_chunk=32,
        backbone_graph=True, combined_decode_bank=True, cpu_experts=2,
        host_gather="auto",
        codec_policy="park" if args.force_park else "auto",
    )) as engine:
        engine.load()
        engine._prepare(288)
        if args.codes.is_file():
            codes = torch.load(args.codes, weights_only=True)
        else:
            text = "<|speaker:0|>你好，这是官方 FP8 质量优先测试。"
            prompt = build_drama3_prompt(text, engine.model.tokenizer, engine.model.config)
            seed_sampling(123)
            generated = generate(
                engine.model, prompt, max_new_tokens=256, prefill_chunk_size=32,
                backbone_graph=engine.backbone_graph, tower_graph=engine.graph)
            if generated.finish_reason != "eos":
                raise RuntimeError("Short diagnostic speech did not complete")
            codes = generated.frames[:, 1:].T.contiguous()
            torch.save(codes, args.codes)
        frames = codes.shape[-1]
        engine.codec = load_codec(
            ROOT / "checkpoints/drama-3/codec.pth", device="cpu", max_frames=frames)
        report["codes_sha256"] = hashlib.sha256(codes.numpy().tobytes()).hexdigest()
        reference = None
        for placement in args.order:
            engine.options = replace(engine.options, codec_placement=placement)
            engine._prepare(288)
            stages = {}
            hooks = []
            before = dict(engine._lifecycle)
            for name, module in (
                ("quantizer.post", engine.codec.quantizer.post_module),
                ("quantizer.upsample", engine.codec.quantizer.upsample),
                *[(f"decoder.{index}", module) for index, module in
                  enumerate(engine.codec.decoder.model)],
            ):
                def record(_module, _inputs, output, key=name):
                    stages[key] = output.detach().cpu().clone()
                hooks.append(module.register_forward_hook(record))
            try:
                audio = engine._decode_audio_segments([codes])
                stages["waveform"] = audio
            finally:
                for hook in hooks:
                    hook.remove()
            if reference is None:
                reference = stages
            comparison = {}
            for name, value in stages.items():
                expected = reference[name]
                delta = value.float() - expected.float()
                comparison[name] = {
                    "bytes_equal": byte_equal(value, expected),
                    "sha256": hashlib.sha256(
                        value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
                    "relative_l2": float(delta.norm() / expected.float().norm().clamp_min(1e-30)),
                    "max_abs": float(delta.abs().max()),
                }
            row = {"placement": placement, "stages": comparison, "lifecycle_delta": {
                key: engine._lifecycle[key] - before[key] for key in before
            }}
            report["runs"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
