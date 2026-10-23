"""Real lookahead session with ordered chunks, export and exact delivered PCM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from benchmark_engine_decode import DramaEngine, EngineOptions, torch
from drama_low_vram.workflows import DramaWorkflows
from drama_low_vram.text_stream import TextStreamSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-bucket-size", type=int, default=512)
    parser.add_argument("--prefix", default="live-text-stream-bucket")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "research/text-stream-bucket-validation.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    import soundfile as sf
    import numpy as np

    report = {"scope": "Real one-chunk-lookahead text stream, output per completed chunk. Not frame-level or real-time streaming.",
              "segments": [], "context_bucket_size": args.context_bucket_size}
    with DramaEngine(EngineOptions(
        checkpoint=ROOT / "checkpoints/drama-3", expert_storage="mmap",
        cache_experts=1, staged_experts=8, pair_rows=64, prefill_chunk=32,
        backbone_graph=True, combined_decode_bank=True, cpu_experts=2,
        native_threads=2, host_gather="auto",
        context_bucket_size=args.context_bucket_size,
    )) as engine:
        session = TextStreamSession(
            DramaWorkflows(engine), ROOT / "outputs" / args.prefix,
            options={"max_new_tokens": 192, "seed": 123})
        session.push("早上好。")
        session.push("我们开始今天的测试。")
        session.finish()
        started = time.perf_counter()
        for row in session.run():
            row["delivered_seconds"] = time.perf_counter() - started
            row["lifecycle"] = dict(engine._lifecycle)
            row["context_capacity"] = engine.model.cache_capacity
            report["segments"].append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        if session.status != "complete" or len(report["segments"]) != 2:
            raise RuntimeError("Text stream did not complete every input chunk")
        output = session.export(ROOT / "outputs" / f"{args.prefix}.wav")
        expected = np.concatenate([sf.read(row["path"], dtype="int16")[0] for row in report["segments"]])
        actual, rate = sf.read(output, dtype="int16")
        report.update(
            status=session.status, output=str(output),
            sample_rate=rate, samples=len(actual),
            exported_pcm_matches_delivered_chunks=bool(np.array_equal(actual, expected)),
            all_lookaheads_correct=[row["lookahead"] for row in report["segments"]] == [
                "我们开始今天的测试。", ""],
        )
        if not report["exported_pcm_matches_delivered_chunks"] or not report["all_lookaheads_correct"]:
            raise RuntimeError("Text stream export/order mismatch")
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
