"""Check graph/cache/RNG identity and decode continuation after codec coexistence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from validate_state_prefix import (
    DramaEngine, EngineOptions, DecodeCursor, TowerDecodeGraph,
    build_drama3_prompt, byte_equal, seed_sampling, state_equal, torch,
)
from drama_low_vram.sequence_state import _bindings


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "research/phase7-codec-continuation.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = {"scope": "Official model state, RNG and next sampled frames across codec coexistence; not a speed benchmark.",
              "cases": [], "all_measured_equal": True}
    with DramaEngine(EngineOptions(
        checkpoint=ROOT / "checkpoints/drama-3", expert_storage="mmap",
        cache_experts=1, staged_experts=8, pair_rows=64, prefill_chunk=32,
        backbone_graph=True, combined_decode_bank=True, prefix_cache_mib=256,
        cpu_experts=2, host_gather="auto", codec_placement="auto",
    )) as engine:
        text = "<|speaker:0|>你好，这是官方 FP8 质量优先测试。"
        codes, _ = engine.prepare_reference(ROOT / "outputs/quality-official-fp8.wav", text)
        # Avoid codec context growth as an extra variable during continuation.
        with engine._codec_on_gpu(375, operation="decode_latents"):
            pass
        engine.load()
        engine._prepare(947)
        engine.graph.close()
        engine.graph = TowerDecodeGraph(
            engine.model.audio_decoder, engine.model.cache_capacity,
            compact_logits=True, record_logits=True,
        )
        prompt = build_drama3_prompt(
            text, engine.model.tokenizer, engine.model.config,
            prompt_texts=[text], prompt_tokens=[codes],
        )
        seed_sampling(123)
        cursor = DecodeCursor.prefill(
            engine.model, prompt, tower_graph=engine.graph, backbone_graph=engine.backbone_graph)
        base = cursor.capture()
        expected = [cursor.step() for _ in range(4)]
        for frames in (16, 375):
            cursor.restore(base)
            graphs = engine.graph, engine.backbone_graph
            addresses = [target.data_ptr() for _, target, _, _ in _bindings(
                engine.model, base.sequence.backbone_rows, base.sequence.tower_rows)]
            before = dict(engine._lifecycle)
            inputs = codes.repeat(1, (frames + codes.shape[1] - 1) // codes.shape[1])[:, :frames]
            audio = engine._decode_audio_segments([inputs])
            if engine._parked:
                raise RuntimeError("Coexistence continuation test unexpectedly parked the model")
            after = cursor.capture()
            actual = [cursor.step() for _ in range(4)]
            checks = {
                "graphs_identical": engine.graph is graphs[0] and engine.backbone_graph is graphs[1],
                "cache_addresses_identical": addresses == [
                    target.data_ptr() for _, target, _, _ in _bindings(
                        engine.model, base.sequence.backbone_rows, base.sequence.tower_rows)],
                "valid_state_bytes_equal": state_equal(base.sequence, after.sequence),
                "hidden_bytes_equal": byte_equal(base.hidden, after.hidden),
                "cuda_rng_bytes_equal": all(
                    byte_equal(after.sequence.rng.cuda[index], value)
                    for index, value in base.sequence.rng.cuda.items()),
                "next_frames_and_all_logits_equal": all(
                    byte_equal(a[key], b[key])
                    for a, b in zip(actual, expected)
                    for key in ("frame", "semantic_logits", "acoustic_logits")),
                "finite_waveform": bool(torch.isfinite(audio).all()),
            }
            row = {"frames": frames, **checks, "lifecycle_delta": {
                key: engine._lifecycle[key] - before[key] for key in before
            }}
            report["cases"].append(row)
            report["all_measured_equal"] &= all(checks.values())
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(row), flush=True)
            if not all(checks.values()):
                raise RuntimeError("Codec coexistence changed model continuation")


if __name__ == "__main__":
    main()
