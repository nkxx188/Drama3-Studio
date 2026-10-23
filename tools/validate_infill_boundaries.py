"""Real generated chunk boundaries for insertion and prefix-deletion infill."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from benchmark_engine_decode import DramaEngine, EngineOptions, torch
from drama_low_vram.codec import write_wav
from drama_low_vram.protocols import InfillSpan
from drama_low_vram.workflows import DramaWorkflows, load_take, save_take


def main():
    torch.set_num_threads(1)
    report = {
        "scope": "Two actual generated text-stream chunks establish known speech/code boundaries. Original joined codes are decoded as the source. Insertion and deletion use the native FIM target; neither forces output or fabricates a middle.",
        "cases": [],
    }
    with DramaEngine(EngineOptions(
        checkpoint=ROOT / "checkpoints/drama-3", expert_storage="mmap",
        cache_experts=1, staged_experts=8, pair_rows=64, prefill_chunk=32,
        backbone_graph=True, combined_decode_bank=True,
        cpu_experts=2, native_threads=4, host_gather="auto",
    )) as engine:
        engine.load()
        first, _, _ = load_take(
            ROOT / "outputs/live-text-stream-protocol/chunk-00001.take.npz", engine.model.config)
        second, _, _ = load_take(
            ROOT / "outputs/live-text-stream-protocol/chunk-00002.take.npz", engine.model.config)
        source = torch.cat((first.codes, second.codes), dim=1)
        original = ROOT / "outputs/infill-boundary-source.wav"
        write_wav(engine._decode_audio_segments([source]), original)
        source_text = first.text + second.text
        source_rows = torch.cat((source[:1] + engine.model.config.semantic_start_token_id, source)).T
        save_take(original, source_text, source, source_rows)
        report.update(source=str(original), source_text=source_text,
                      known_boundary_frame=first.codes.shape[1])
        workflows = DramaWorkflows(engine)
        for label, text, span in (
            ("insert", source_text, InfillSpan(
                first.codes, first.codes.shape[1], first.codes.shape[1], second.codes.shape[1])),
            ("delete", second.text, InfillSpan(
                source, 0, first.codes.shape[1], 1)),
        ):
            output = ROOT / "outputs" / f"infill-boundary-{label}.wav"
            result = workflows.run(
                "infill", text, output, infill=span, seed=123, max_new_tokens=192)
            row = {
                "case": label, "status": result["status"], "text": text,
                "audio": result["audio"], "output": str(output),
                "total_seconds": result["total_seconds"], "chunks": result["chunks"],
            }
            report["cases"].append(row)
            (ROOT / "research/infill-boundary-validation.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if result["status"] != "complete":
                raise RuntimeError(f"Native infill {label} did not complete")


if __name__ == "__main__":
    main()
