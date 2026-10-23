"""Real-checkpoint prefix equality, stochastic rollback and oracle verification."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from benchmark_engine_decode import (
    DramaEngine, EngineOptions, build_drama3_prompt, generate, seed_sampling, torch,
)
from drama_low_vram.sequence_state import SequenceSnapshot
from drama_low_vram.tower_graph import TowerDecodeGraph
from drama_low_vram.verification import DecodeCursor


def byte_equal(a, b):
    return (
        a.shape == b.shape and a.dtype == b.dtype
        and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                        b.contiguous().reshape(-1).view(torch.uint8))
    )


def state_equal(a, b):
    return (
        a.backbone_rows == b.backbone_rows and a.tower_rows == b.tower_rows
        and a.buffers.keys() == b.buffers.keys()
        and all(byte_equal(a.buffers[key], b.buffers[key]) for key in a.buffers)
        and a.sampling.keys() == b.sampling.keys()
        and all(byte_equal(a.sampling[key], b.sampling[key]) for key in a.sampling)
    )


@torch.inference_mode()
def collect(engine, prompt, frames, *, cache=True):
    head, manager = engine.model.backbone.lm_head, engine.manager
    original_head, original_forward = head.compact_logits, manager.forward
    hidden, logits, routes = [], [], []

    def record_head(values):
        result = original_head(values)
        hidden.append(values.cpu().clone())
        logits.append(result.cpu().clone())
        return result

    def record_routes(module, values, ids, weights):
        if len(values) == 1:
            routes.append(ids.cpu().clone())
        return original_forward(module, values, ids, weights)

    seed_sampling(123)
    with patch.object(head, "compact_logits", record_head), patch.object(manager, "forward", record_routes):
        with engine.prefix_cache.suspend() if not cache else nullcontext():
            result = generate(
                engine.model, prompt, max_new_tokens=frames, prefill_chunk_size=32,
                tower_graph=engine.graph, backbone_graph=engine.backbone_graph,
                prefix_cache=engine.prefix_cache)
    return {
        "frames": result.frames, "hidden": torch.cat(hidden), "logits": torch.cat(logits),
        "routes": torch.cat(routes), "timing": result.as_dict(),
    }


def compare(actual, reference):
    return {f"{name}_bytes_equal": byte_equal(actual[name], reference[name])
            for name in ("frames", "hidden", "logits", "routes")}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sections", nargs="+", choices=("prefix", "oracle"),
                        default=["prefix", "oracle"])
    parser.add_argument("--output", type=Path, default=ROOT / "research/state-prefix-validation.json")
    parser.add_argument("--tensors", type=Path, default=ROOT / "research/state-prefix-validation.pt")
    args = parser.parse_args()
    if args.frames < 4 or args.repeats < 1:
        parser.error("At least four frames and positive repetitions required")
    torch.set_num_threads(1)
    report = {"scope": "Official weights; exact prefix and state gates; oracle logits verification, no draft/acceptance sampler.",
              "prefix": [], "rollback": [], "oracle": [], "all_measured_equal": True}
    artifacts = {}

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def prefix_case(engine, label, prompt, reference=None):
        expected = collect(engine, prompt, args.frames, cache=False) if reference is None else reference
        actual = collect(engine, prompt, args.frames)
        checks = compare(actual, expected)
        row = {"name": label, "prompt_rows": len(prompt), **checks,
               "prefill_seconds": actual["timing"]["prefill_seconds"],
               "prefix_cache": actual["timing"]["prefix_cache"],
               "cache": engine.prefix_cache.report()}
        report["prefix"].append(row)
        artifacts[label] = {"reference": expected, "actual": actual}
        save()
        print(json.dumps(row), flush=True)
        if not all(checks.values()):
            torch.save(artifacts, args.tensors)
            raise RuntimeError(f"Prefix gate failed: {label}")
        return expected

    with DramaEngine(EngineOptions(
        checkpoint=ROOT / "checkpoints/drama-3", expert_storage="mmap",
        cache_experts=2, staged_experts=8, pair_rows=64, prefill_chunk=32,
        backbone_graph=True, combined_decode_bank=True, prefix_cache_mib=256,
    )) as engine:
        engine.load()
        text = "<|speaker:0|>你好，这是官方 FP8 质量优先测试。"
        longer = "<|speaker:0|>今天我们继续测试这套本地语音引擎。希望普通电脑也能保持清楚自然的声音，读完一整句话，而不需要购买昂贵的专业显卡。"
        codes, _ = engine.prepare_reference(ROOT / "outputs/quality-official-fp8.wav", text)

        def prompt(target=text, reference=codes, transcript=text):
            return build_drama3_prompt(
                target, engine.model.tokenizer, engine.model.config,
                prompt_texts=[transcript], prompt_tokens=[reference])

        engine._prepare(1024)
        base_prompt = prompt()
        if "prefix" in args.sections:
            reference = prefix_case(engine, "cache_fill", base_prompt)
            prefix_case(engine, "same_reference_hit", base_prompt, reference)
            prefix_case(engine, "changed_target_hit", prompt(longer))
            changed = codes.clone()
            changed[2, 0] = (changed[2, 0] + 1) % 4096
            prefix_case(engine, "changed_reference_miss", prompt(reference=changed))
            engine._park_model()
            engine._prepare(1024)
            prefix_case(engine, "after_parking_hit", base_prompt, reference)
            engine._prepare(1536)
            prefix_case(engine, "after_capacity_growth_hit", base_prompt, reference)
            torch.save(artifacts, args.tensors)
        if "oracle" not in args.sections:
            report["cache"] = engine.prefix_cache.report()
            save()
            return

        # Record all acoustic heads from the production sampler for comparison.
        engine.graph.close()
        engine.graph = TowerDecodeGraph(
            engine.model.audio_decoder, engine.model.cache_capacity,
            compact_logits=True, record_logits=True)
        seed_sampling(123)
        cursor = DecodeCursor.prefill(
            engine.model, base_prompt, tower_graph=engine.graph,
            backbone_graph=engine.backbone_graph)
        base = cursor.capture(storage_device=engine.device)
        snapshots = [cursor.capture(storage_device=engine.device, include_rng=False)]
        records = []
        for _ in range(4):
            row = cursor.step()
            if cursor.finished:
                raise RuntimeError("Unexpected EOS in diagnostic window")
            records.append({name: value.clone() for name, value in row.items()})
            snapshots.append(cursor.capture(storage_device=engine.device, include_rng=False))
        proposals = torch.cat([row["frame"] for row in records])
        expected_logits = {
            "semantic_logits": torch.cat([row["semantic_logits"] for row in records]),
            "acoustic_logits": torch.stack([row["acoustic_logits"] for row in records]),
        }
        cursor.restore(base)
        repeated = [cursor.step() for _ in range(4)]
        replay_equal = all(
            byte_equal(actual[name], expected[name])
            for actual, expected in zip(repeated, records)
            for name in ("frame", "semantic_logits", "acoustic_logits"))
        report["rollback"].append({"name": "stochastic_replay", "equal": replay_equal,
                                   "snapshot_bytes": base.nbytes})
        if not replay_equal:
            raise RuntimeError("Stochastic replay failed")
        # Rebuild each possible accepted prefix after rejecting a 4-frame window.
        for accepted in range(5):
            cursor.restore(base)
            for row in proposals[:accepted]:
                cursor.force(row[None])
            actual = cursor.capture(storage_device=engine.device, include_rng=False)
            equal = (
                state_equal(actual.sequence, snapshots[accepted].sequence)
                and byte_equal(actual.hidden, snapshots[accepted].hidden)
                and byte_equal(actual.frames, snapshots[accepted].frames))
            report["rollback"].append({"name": "accepted_prefix_replay", "accepted": accepted, "equal": equal})
            save()
            if not equal:
                raise RuntimeError(f"Accepted-prefix rollback differs at {accepted}")

        for count in (2, 4):
            for batch in (False, True):
                cursor.restore(base)
                actual = cursor.force(proposals[:count], block=True, batch_experts=batch)
                state = cursor.capture(storage_device=engine.device, include_rng=False)
                checks = {
                    **{f"{key}_bytes_equal": byte_equal(actual[key], expected_logits[key][:count])
                       for key in expected_logits},
                    "hidden_bytes_equal": byte_equal(state.hidden, snapshots[count].hidden),
                    "state_bytes_equal": state_equal(state.sequence, snapshots[count].sequence),
                }
                row = {"frames": count, "batch_experts": batch, **checks}
                report["oracle"].append(row)
                artifacts[f"oracle_{count}_{batch}"] = {
                    "actual": {key: value.cpu() for key, value in actual.items()},
                    "reference": {key: value[:count].cpu() for key, value in expected_logits.items()},
                }
                if not all(checks.values()):
                    report["all_measured_equal"] = False
                    save()
                    print(json.dumps(row), flush=True)
                    continue
                samples = {"production": [], "oracle": []}
                transfers = {"production": [], "oracle": []}
                # Completed windows, identical starting sequence and empty GPU
                # expert cache; original FP8 file pages were already warmed.
                for repeat in range(args.repeats + 1):
                    for variant in (("production", "oracle") if repeat % 2 == 0 else ("oracle", "production")):
                        cursor.restore(base)
                        engine.manager.clear_hot_cache()
                        torch.cuda.synchronize(engine.device)
                        before = engine.manager.stats.h2d_bytes
                        started = time.perf_counter()
                        if variant == "production":
                            for _ in range(count):
                                cursor.step()
                        else:
                            cursor.force(proposals[:count], block=True, batch_experts=batch)
                        torch.cuda.synchronize(engine.device)
                        elapsed = time.perf_counter() - started
                        if repeat:
                            samples[variant].append(elapsed)
                            transfers[variant].append(engine.manager.stats.h2d_bytes - before)
                row.update({
                    "samples_seconds": samples, "expert_h2d_bytes": transfers,
                    "production_median_seconds": statistics.median(samples["production"]),
                    "oracle_median_seconds": statistics.median(samples["oracle"]),
                    "oracle_ideal_speedup": statistics.median(samples["production"]) / statistics.median(samples["oracle"]),
                    "scope": "Perfect proposals supplied for free, all accepted; no drafter/acceptance/checkpoint cost included.",
                })
                save()
                print(json.dumps(row), flush=True)
        report["cache"] = engine.prefix_cache.report()
        report["all_measured_equal"] &= all(row["equal"] for row in report["rollback"])
    torch.save(artifacts, args.tensors)
    save()


if __name__ == "__main__":
    main()
