"""Real single-voice cloning with one and two distinct reference recordings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from benchmark_engine_decode import DramaEngine, EngineOptions, torch
from drama_low_vram.protocols import VoiceReference, build_tts_prompt
from drama_low_vram.workflows import DramaWorkflows


def main():
    torch.set_num_threads(1)
    target = "<|speaker:0|>我们正在检查同一音色的多段参考。"
    references = [
        VoiceReference(str(ROOT / "outputs/quality-official-fp8.wav"),
                       "你好，这是官方 FP8 质量优先测试。", 0),
        VoiceReference(str(ROOT / "outputs/audio-frame-resident-auto.wav"),
                       "你好，这是一次语音测试。", 0),
    ]
    output = ROOT / "research/same-speaker-multireference-validation.json"
    report = {
        "scope": "Actual one/two-reference requests for global speaker 0, using two distinct synthetic clips anchored to the same reference voice. Verifies conditioning, EOS and finite PCM; does not establish speaker similarity or a quality gain from more references.",
        "target": target, "references": [], "requests": [],
    }
    with DramaEngine(EngineOptions(
        checkpoint=ROOT / "checkpoints/drama-3", expert_storage="mmap",
        cache_experts=1, staged_experts=8, pair_rows=64, prefill_chunk=32,
        backbone_graph=True, combined_decode_bank=True,
        cpu_experts=2, native_threads=4, host_gather="auto", prefix_cache_mib=256,
    )) as engine:
        encoded = []
        for reference in references:
            codes, _ = engine.prepare_reference(Path(reference.audio), reference.transcript())
            encoded.append(codes)
            report["references"].append({
                "audio": reference.audio, "speaker": reference.speaker,
                "transcript": reference.transcript(), "frames": codes.shape[1],
                "audio_sha256": hashlib.sha256(Path(reference.audio).read_bytes()).hexdigest(),
                "codes_sha256": hashlib.sha256(codes.numpy().tobytes()).hexdigest(),
            })
        engine.load()
        config = engine.model.config
        prompt = build_tts_prompt(
            target, engine.model.tokenizer, config,
            transcripts=[reference.transcript() for reference in references], references=encoded)
        audio_rows = prompt[(prompt[:, 0] >= config.semantic_start_token_id)
                            & (prompt[:, 0] <= config.semantic_end_token_id)]
        report["all_reference_codes_enter_actual_prompt"] = bool(torch.equal(
            audio_rows[:, 1:], torch.cat(encoded, dim=1).T))
        workflows = DramaWorkflows(engine)
        for count in (1, 2):
            path = ROOT / "outputs" / f"same-speaker-reference-{count}.wav"
            result = workflows.run(
                "tts", target, path, references=references[:count],
                seed=123, max_new_tokens=256)
            if result["status"] != "complete":
                raise RuntimeError(f"{count}-reference request failed to reach EOS")
            audio, rate = sf.read(path, dtype="float32")
            with np.load(result["take_file"], allow_pickle=False) as take:
                metadata = json.loads(str(take["metadata"].item()))
            row = {
                "reference_count": count, "status": result["status"],
                "all_eos": all(chunk["finish_reason"] == "eos" for chunk in result["chunks"]),
                "all_top8_preserved": result["all_top8_preserved"],
                "audio": result["audio"], "sample_rate": rate,
                "finite": bool(np.isfinite(audio).all()),
                "saved_reference_speakers": [reference["speaker"] for reference in metadata["references"]],
                "reference_cache_hits": result["reference_cache_hits"],
                "total_seconds": result["total_seconds"], "output": result["output"],
                "take_file": result["take_file"], "report_file": str(path.with_suffix(".json")),
                "code_hashes": [chunk["code_sha256"] for chunk in result["chunks"]],
                "wav_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            report["requests"].append(row)
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(row, ensure_ascii=False), flush=True)
    report["passed"] = (
        report["all_reference_codes_enter_actual_prompt"]
        and len({reference["audio_sha256"] for reference in report["references"]}) == 2
        and all(row["all_eos"] and row["finite"] and row["sample_rate"] == 44100
                and row["saved_reference_speakers"] == [0] * row["reference_count"]
                for row in report["requests"]))
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise RuntimeError("Same-speaker reference conditioning audit failed")


if __name__ == "__main__":
    main()
