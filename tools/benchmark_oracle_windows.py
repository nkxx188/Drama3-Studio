"""Multi-window perfect-proposal target cost; not an end-to-end speculative decoder."""

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
from validate_state_prefix import (
    DramaEngine, EngineOptions, build_drama3_prompt, seed_sampling, torch,
    DecodeCursor, byte_equal, state_equal, TowerDecodeGraph,
)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offsets", type=int, nargs="+", default=[0, 8, 16])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--direct-pairs", action="store_true",
                        help="Experimentally avoid route alignment for small verification pair groups.")
    parser.add_argument("--output", type=Path, default=ROOT / "research/oracle-window-benchmark.json")
    args = parser.parse_args()
    if args.repeats < 1 or not args.offsets or min(args.offsets) < 0:
        parser.error("Nonnegative offsets and positive repetitions required")
    torch.set_num_threads(1)
    report = {
        "scope": "Perfect full-frame proposals supplied for free, all accepted. Serial GDN/attention/dense with batched experts. No draft, acceptance/rejection or full-speech speedup claim.",
        "conditions": [], "repeats": args.repeats,
        "configuration": {"cpu_experts": 0, "cache_experts": 2, "staged_experts": 8,
                          "pair_rows": 64, "prefill_chunk": 32, "torch_threads": 1},
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    text = "<|speaker:0|>你好，这是官方 FP8 质量优先测试。"
    with DramaEngine(EngineOptions(
        checkpoint=ROOT / "checkpoints/drama-3", expert_storage="mmap",
        cache_experts=2, staged_experts=8, pair_rows=64, prefill_chunk=32,
        backbone_graph=True, combined_decode_bank=True,
    )) as engine:
        engine.load()
        codes, _ = engine.prepare_reference(ROOT / "outputs/quality-official-fp8.wav", text)
        engine._prepare(1024)
        engine.graph.close()
        engine.graph = TowerDecodeGraph(
            engine.model.audio_decoder, engine.model.cache_capacity,
            compact_logits=True, record_logits=True)
        report["configuration"]["direct_verification_pairs"] = args.direct_pairs
        backend = engine.manager.backend
        original_pairs = backend.forward_pairs

        def direct_pairs(values, ids, weights, bank):
            if (values.shape[0] > 64 or torch.cuda.get_device_capability(values.device) != (8, 9)):
                return original_pairs(values, ids, weights, bank)
            from drama_low_vram.fp8_decode import quantized_pairs

            q, scales = backend.kernel.quantize_groups(
                backend.kernel.quant_kernel, values.contiguous(), values.shape[-1])
            return quantized_pairs(backend.kernel, q, scales, ids, weights, bank, dtype=values.dtype)

        def oracle(cursor, proposals):
            with patch.object(backend, "forward_pairs", direct_pairs) if args.direct_pairs else nullcontext():
                return cursor.force(proposals, block=True)
        for cloned in (False, True):
            label = "clone" if cloned else "plain"
            prompt = build_drama3_prompt(
                text, engine.model.tokenizer, engine.model.config,
                prompt_texts=[text] if cloned else [], prompt_tokens=[codes] if cloned else [])
            condition = {"condition": label, "windows": []}
            report["conditions"].append(condition)
            for offset in args.offsets:
                seed_sampling(123)
                cursor = DecodeCursor.prefill(
                    engine.model, prompt, tower_graph=engine.graph,
                    backbone_graph=engine.backbone_graph)
                for _ in range(offset):
                    cursor.step()
                base = cursor.capture()
                records = [cursor.step() for _ in range(4)]
                if cursor.finished:
                    raise RuntimeError("Diagnostic window reached EOS")
                proposals = torch.cat([row["frame"] for row in records])
                for count in (2, 4):
                    cursor.restore(base)
                    for _ in range(count):
                        cursor.step()
                    expected = cursor.capture()
                    cursor.restore(base)
                    actual = oracle(cursor, proposals[:count])
                    final = cursor.capture()
                    checks = {
                        "semantic_logits_bytes_equal": byte_equal(
                            actual["semantic_logits"], torch.cat([row["semantic_logits"] for row in records[:count]])),
                        "acoustic_logits_bytes_equal": byte_equal(
                            actual["acoustic_logits"], torch.stack([row["acoustic_logits"] for row in records[:count]])),
                        "hidden_bytes_equal": byte_equal(final.hidden, expected.hidden),
                        "state_bytes_equal": state_equal(final.sequence, expected.sequence),
                    }
                    row = {"offset": offset, "frames": count, **checks}
                    condition["windows"].append(row)
                    if not all(checks.values()):
                        row["performance_gate"] = False
                        save()
                        print(json.dumps({"condition": label, **row}), flush=True)
                        continue
                    routes = []
                    original = engine.manager.forward

                    def record(module, values, ids, weights):
                        routes.append(ids.detach().cpu().clone())
                        return original(module, values, ids, weights)

                    cursor.restore(base)
                    with patch.object(engine.manager, "forward", record):
                        oracle(cursor, proposals[:count])
                    row["route_slots"] = sum(ids.numel() for ids in routes)
                    row["route_union_experts"] = sum(torch.unique(ids).numel() for ids in routes)
                    row["union_to_route_slots_fraction"] = row["route_union_experts"] / row["route_slots"]
                    samples = {"production": [], "oracle": []}
                    transfers = {"production": [], "oracle": []}
                    restore_samples = []
                    for repeat in range(args.repeats + 1):
                        for variant in (("production", "oracle") if repeat % 2 == 0 else ("oracle", "production")):
                            restore_started = time.perf_counter()
                            cursor.restore(base)
                            restore_samples.append(time.perf_counter() - restore_started)
                            engine.manager.clear_hot_cache()
                            torch.cuda.synchronize(engine.device)
                            before = engine.manager.stats.h2d_bytes
                            started = time.perf_counter()
                            if variant == "production":
                                for _ in range(count):
                                    cursor.step()
                            else:
                                oracle(cursor, proposals[:count])
                            torch.cuda.synchronize(engine.device)
                            if repeat:
                                samples[variant].append(time.perf_counter() - started)
                                transfers[variant].append(engine.manager.stats.h2d_bytes - before)
                    row.update({
                        "performance_gate": True, "samples_seconds": samples,
                        "expert_h2d_bytes": transfers,
                        "cpu_snapshot_bytes": base.nbytes,
                        "cpu_snapshot_restore_median_seconds": statistics.median(restore_samples),
                        "production_median_seconds": statistics.median(samples["production"]),
                        "oracle_median_seconds": statistics.median(samples["oracle"]),
                        "ideal_speedup": statistics.median(samples["production"]) / statistics.median(samples["oracle"]),
                    })
                    save()
                    print(json.dumps({"condition": label, **row}), flush=True)
            summary = [
                row for row in condition["windows"] if row.get("performance_gate")
            ]
            condition["median_ideal_speedup_by_frames"] = {
                str(count): statistics.median(row["ideal_speedup"] for row in summary if row["frames"] == count)
                for count in (2, 4) if any(row["frames"] == count for row in summary)
            }
            save()


if __name__ == "__main__":
    main()
