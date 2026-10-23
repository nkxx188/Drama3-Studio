"""Measure fixed-frame backbone/logit/route effects of native expert substitution."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

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
from drama_low_vram.hybrid_layer import HybridLayerExperiment
from drama_low_vram.fp8_decode import build_gpu_silu_table
from drama_low_vram.offload import ExpertStore
from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt


def metrics(actual, expected):
    delta = actual.float() - expected.float()
    return {
        "equal": bool(
            actual.dtype == expected.dtype and actual.shape == expected.shape
            and torch.equal(
                actual.contiguous().view(torch.uint8),
                expected.contiguous().view(torch.uint8),
            )
        ),
        "relative_l2": float(delta.norm() / expected.float().norm().clamp_min(1e-30)),
        "max_abs": float(delta.abs().max()),
    }


@torch.inference_mode()
def forced_backbone(
    engine, prompt, frames, *, native_layer=None, repeated=False, cpu_jobs=4,
    arithmetic="reference",
    silu_table=None,
    hybrid_layers=None,
):
    model, manager = engine.model, engine.manager
    backbone, tower = model.backbone, model.audio_decoder
    capacity = len(prompt) + len(frames)
    plan, _ = engine._prepare(max(capacity, 1024))
    modules = list(manager.modules.values())
    layer_lookup = {id(module): index for index, module in enumerate(modules)}
    records = [[None] * len(modules) for _ in range(len(frames))]
    current = -1
    original = manager.forward
    hybrid = getattr(manager, "hybrid_decode", None)
    target = modules[native_layer] if native_layer is not None else None
    native_context = (
        HybridLayerExperiment(
            ExpertStore.from_module(target), cpu_jobs=cpu_jobs,
            cache_experts=2, threads=4, parallel=True,
            native_arithmetic=arithmetic,
            silu_table=silu_table,
        ) if target is not None else nullcontext(None)
    )
    native_calls = []
    hidden_rows, logits_rows = [], []
    with native_context as native:
        def expert_forward(module, x, ids, weights):
            if current >= 0:
                records[current][layer_lookup[id(module)]] = ids.detach().cpu().clone()
            if (current >= 0 and module is target
                    and (repeated or current == 2)):
                output, record = native.forward(x, ids, weights)
                native_calls.append({"forced_step": current, **record})
                return output
            before = 0 if hybrid is None else hybrid.stats.calls
            output = original(module, x, ids, weights)
            if hybrid is not None and hybrid.stats.calls > before:
                native_calls.append({
                    "forced_step": current, "layer_index": layer_lookup[id(module)],
                })
            return output

        with patch.object(manager, "forward", expert_forward):
            positions = torch.arange(capacity, device=tower.device)
            rows = prompt.to(tower.device)
            for start in range(0, len(rows), plan.prefill_chunk):
                end = min(start + plan.prefill_chunk, len(rows))
                chunk = rows[start:end]
                embedded = tower.embed_backbone(chunk, backbone.embed(chunk[:, 0]))
                backbone(inputs_embeds=embedded[None], input_pos=positions[start:end])
            for step, frame in enumerate(frames):
                current = step
                position = len(rows) + step
                frame = frame[None].to(tower.device)
                embedded = tower.embed_backbone(frame, backbone.embed(frame[:, 0]))
                layers = hybrid_layers(step) if callable(hybrid_layers) else hybrid_layers
                with hybrid.decode(layers=layers) if hybrid_layers is not None else nullcontext():
                    if engine.backbone_graph is not None and engine.backbone_graph.enabled:
                        hidden = engine.backbone_graph.forward(
                            inputs_embeds=embedded[None], input_pos=position)[0]
                    else:
                        hidden = backbone(
                            inputs_embeds=embedded[None], input_pos=positions[position:position + 1])[0]
                logits = backbone.lm_head.compact_logits(hidden)
                hidden_rows.append(hidden.cpu().clone())
                logits_rows.append(logits.cpu().clone())
    return {
        "hidden": torch.cat(hidden_rows), "logits": torch.cat(logits_rows),
        "routes": torch.stack([torch.cat(row, dim=0) for row in records]),
        "native_calls": native_calls,
    }


def compare(actual, reference):
    p_log = reference["logits"].float().log_softmax(-1)
    q_log = actual["logits"].float().log_softmax(-1)
    kl = (p_log.exp() * (p_log - q_log)).sum(-1)
    route_overlap = (
        actual["routes"][..., :, None] == reference["routes"][..., None, :]
    ).any(-1).float().mean(-1)
    return {
        "hidden": metrics(actual["hidden"], reference["hidden"]),
        "semantic_logits": metrics(actual["logits"], reference["logits"]),
        "max_kl_nats": float(kl.max().clamp_min(0)),
        "per_frame_kl_nats": kl.tolist(),
        "semantic_top1_equal_fraction": float(
            (actual["logits"].argmax(-1) == reference["logits"].argmax(-1)).float().mean()),
        "route_slot_equal_fraction": float(
            (actual["routes"] == reference["routes"]).float().mean()),
        "route_set_mean_overlap": float(route_overlap.mean()),
        "per_frame_route_set_overlap": route_overlap.mean(-1).tolist(),
        "native_expert_calls": len(actual["native_calls"]),
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--arithmetic", choices=("reference", "sm89"), default="reference")
    parser.add_argument("--gpu-silu-table", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "research/native-teacher-forcing.json")
    parser.add_argument("--tensors", type=Path, default=ROOT / "research/native-teacher-forcing.pt")
    args = parser.parse_args()
    if args.frames < 4:
        parser.error("At least four forced frames are required")
    text = "<|speaker:0|>你好，这是官方 FP8 质量优先测试。"
    report = {
        "scope": "Fixed real audio frames, whole backbone/cache continuation; one selected MoE layer substituted. Not free-running speech or perceptual quality.",
        "frames_per_condition": args.frames,
        "native_jobs": 4, "conditions": [], "default_native_backend_enabled": False,
        "native_arithmetic": args.arithmetic,
        "gpu_silu_table": args.gpu_silu_table,
    }
    artifacts = []
    silu_table = build_gpu_silu_table() if args.gpu_silu_table else None
    with DramaEngine(EngineOptions(
        checkpoint=ROOT / "checkpoints/drama-3", expert_storage="mmap",
        staged_experts=8, pair_rows=64, prefill_chunk=32, cache_experts=2,
        backbone_graph=True, combined_decode_bank=True,
    )) as engine:
        engine.load()
        engine.manager.backend.decode_specialization = False
        reference_codes, _ = engine.prepare_reference(ROOT / "outputs/quality-official-fp8.wav", text)
        for cloned in (False, True):
            condition = "clone" if cloned else "plain"
            prompt = build_drama3_prompt(
                text, engine.model.tokenizer, engine.model.config,
                prompt_texts=[text] if cloned else [],
                prompt_tokens=[reference_codes] if cloned else [],
            )
            engine._prepare(1024)
            engine.manager.backend.decode_specialization = False
            seed_sampling(123)
            target = generate(
                engine.model, prompt, max_new_tokens=args.frames,
                prefill_chunk_size=32, tower_graph=engine.graph,
                backbone_graph=engine.backbone_graph,
            )
            if len(target.frames) < args.frames:
                raise RuntimeError("Target ended before the requested diagnostic window")
            reference = forced_backbone(engine, prompt, target.frames)
            condition_report = {
                "condition": condition, "prompt_rows": len(prompt),
                "target_frames_sha256": hashlib.sha256(target.frames.numpy().tobytes()).hexdigest(),
                "target_finish_reason": target.finish_reason, "variants": [],
            }
            variants = [
                ("gpu_repeat", False, None, False),
                ("gpu_direct_decode", True, None, False),
                ("native_layer0_once", False, 0, False),
                ("native_layer20_once", False, 20, False),
                ("native_layer39_once", False, 39, False),
                ("native_layer0_repeated", False, 0, True),
            ]
            condition_artifact = {"condition": condition, "prompt": prompt,
                                  "frames": target.frames, "reference": reference, "variants": {}}
            for name, direct, layer, repeated in variants:
                engine.manager.backend.decode_specialization = direct
                actual = forced_backbone(
                    engine, prompt, target.frames, native_layer=layer, repeated=repeated,
                    arithmetic=args.arithmetic, silu_table=silu_table)
                summary = {"name": name, "injected_layer": layer,
                           "repeated_each_forced_frame": repeated, **compare(actual, reference)}
                condition_report["variants"].append(summary)
                condition_artifact["variants"][name] = actual
                print(json.dumps({"condition": condition, **summary}), flush=True)
                args.output.write_text(json.dumps(report | {
                    "conditions": report["conditions"] + [condition_report]
                }, indent=2) + "\n", encoding="utf-8")
            report["conditions"].append(condition_report)
            artifacts.append(condition_artifact)
    args.tensors.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifacts, args.tensors)
    report["tensor_artifact"] = str(args.tensors)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
