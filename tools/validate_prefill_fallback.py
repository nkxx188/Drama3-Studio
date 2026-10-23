"""Validate official FP8 state and next logits after a partial prefill OOM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from validate_state_prefix import (
    DramaEngine, EngineOptions, DecodeCursor, TowerDecodeGraph,
    build_drama3_prompt, byte_equal, seed_sampling, state_equal, torch,
)
from benchmark_memory_pressure import LONG, SHORT


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new diagnostic output")
    torch.set_num_threads(1)
    report = {
        "scope": (
            "Official FP8/top-8 with an injected CUDA allocation exception after "
            "real partial GDN/KV writes. Not an actual low-VRAM speed benchmark."
        ),
        "status": "running",
    }
    try:
        with DramaEngine(EngineOptions(
            checkpoint=ROOT / "checkpoints/drama-3", expert_storage="mmap",
            backbone_graph=True, combined_decode_bank=True, prefix_cache_mib=256,
            host_gather="auto", compute_policy="auto",
            compute_profile_dir=ROOT / ".cache/compute-profiles",
        )) as engine:
            codes, _ = engine.prepare_reference(
                ROOT / "outputs/quality-official-fp8.wav", SHORT)
            engine.load()
            prompt = build_drama3_prompt(
                LONG, engine.model.tokenizer, engine.model.config,
                prompt_texts=[SHORT], prompt_tokens=[codes])
            engine._prepare(len(prompt) + 768)
            report["compute_policy"] = engine.compute_report
            engine.graph.close()
            engine.graph = TowerDecodeGraph(
                engine.model.audio_decoder, engine.model.cache_capacity,
                compact_logits=True, record_logits=True)
            seed_sampling(123)
            expected_cursor = DecodeCursor.prefill(
                engine.model, prompt, chunk_size=16,
                tower_graph=engine.graph, backbone_graph=engine.backbone_graph)
            expected_state = expected_cursor.capture()
            expected = [expected_cursor.step() for _ in range(4)]

            original_forward = engine.model.backbone.forward
            injected = []

            def fail_after_partial_writes(*values, **kwargs):
                output = original_forward(*values, **kwargs)
                positions = kwargs["input_pos"]
                if len(positions) == 32 and int(positions[0]) == 32:
                    injected.append(True)
                    raise torch.cuda.OutOfMemoryError("Injected after official GDN/KV writes")
                return output

            seed_sampling(123)
            with patch.object(engine.model.backbone, "forward", fail_after_partial_writes):
                actual_cursor = DecodeCursor.prefill(
                    engine.model, prompt, chunk_size=32,
                    tower_graph=engine.graph, backbone_graph=engine.backbone_graph,
                    allow_prefill_fallback=True)
            actual_state = actual_cursor.capture()
            actual = [actual_cursor.step() for _ in range(4)]
            checks = {
                "one_injected_oom": injected == [True],
                "reduced_to_16_rows": actual_cursor.prefix_report["chunk_size"] == 16,
                "one_prefill_retry": actual_cursor.prefix_report["oom_retries"] == 1,
                "prefill_state_bytes_equal": state_equal(actual_state.sequence, expected_state.sequence),
                "prefill_hidden_bytes_equal": byte_equal(actual_state.hidden, expected_state.hidden),
                "prefill_frames_bytes_equal": byte_equal(actual_state.frames, expected_state.frames),
                "cpu_rng_bytes_equal": byte_equal(
                    actual_state.sequence.rng.torch_cpu, expected_state.sequence.rng.torch_cpu),
                "cuda_rng_bytes_equal": all(
                    byte_equal(actual_state.sequence.rng.cuda[index], value)
                    for index, value in expected_state.sequence.rng.cuda.items()),
                "next_four_frames_bytes_equal": all(
                    byte_equal(row["frame"], reference["frame"])
                    for row, reference in zip(actual, expected, strict=True)),
                "next_four_semantic_logits_bytes_equal": all(
                    byte_equal(row["semantic_logits"], reference["semantic_logits"])
                    for row, reference in zip(actual, expected, strict=True)),
                "next_four_acoustic_logits_bytes_equal": all(
                    byte_equal(row["acoustic_logits"], reference["acoustic_logits"])
                    for row, reference in zip(actual, expected, strict=True)),
            }
            report.update(
                checks=checks, all_measured_equal=all(checks.values()),
                prompt_rows=len(prompt), prefix=actual_cursor.prefix_report,
                status="complete")
            if not report["all_measured_equal"]:
                raise RuntimeError("Official prefill OOM recovery changed valid state or next logits")
    except BaseException as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
