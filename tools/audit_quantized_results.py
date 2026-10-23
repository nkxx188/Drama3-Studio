"""Summarize completed requests, resources, ASR checks and tuning parity."""

import hashlib
import json
from pathlib import Path
import re
import statistics
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def normalized(text):
    text = re.sub(r"<\|speaker:\d+\|>", "", text).lower()
    text = re.sub(r"\s+", "", text).replace("fp8", "fp八")
    return "".join(character for character in text if character.isalnum())


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def audio_check(path):
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise ValueError("Expected PCM16 output")
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float64) / 32768
        return {
            "sha256": digest(path), "sample_rate": wav.getframerate(),
            "channels": wav.getnchannels(), "samples": wav.getnframes(),
            "rms": float(np.sqrt(np.mean(samples**2))),
            "peak": float(np.abs(samples).max()),
            "near_clipped_fraction": float(np.mean(np.abs(samples) >= 0.999)),
            "near_silent_fraction": float(np.mean(np.abs(samples) < 1e-4)),
        }


def main():
    asr = {}
    for name in ("quantized-first-asr.json", "quantized-final-asr.json"):
        for row in json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))["rows"]:
            asr[str(Path(row["audio"]).resolve())] = row
    studies, all_rows = {}, []
    for label in ("int4-auto", "int4-tuned", "int8-auto", "int8-tts", "w6-auto"):
        source = ROOT / "research" / f"quant-{label}-benchmark.json"
        report = json.loads(source.read_text(encoding="utf-8"))
        rows = []
        for index, result in enumerate(report["results"]):
            chunks = result["chunks"]
            frames = sum(chunk["frames"] for chunk in chunks)
            decode = sum(chunk["decode_seconds"] for chunk in chunks)
            path = Path(result["output"]) if result["output"] else None
            row = {
                "label": label, "index": index,
                "case": result.get("benchmark_case", "clone-short" if index == 0 else "clone-long"),
                "status": result["status"], "frames": frames,
                "audio_seconds": result["audio"]["duration_seconds"] if result["audio"] else None,
                "total_seconds": result["total_seconds"], "decode_seconds": decode,
                "prefill_seconds": sum(chunk["prefill_seconds"] for chunk in chunks),
                "decode_ms_per_frame": decode * 1000 / max(1, frames),
                "peak_gpu_allocated_bytes": max(chunk["peak_allocated_bytes"] for chunk in chunks),
                "peak_gpu_reserved_bytes": max(chunk["peak_reserved_bytes"] for chunk in chunks),
                "process": result["memory"]["process"],
                "cache_experts_per_layer": chunks[0]["runtime_plan"]["cache_experts"],
                "codes": [chunk.get("semantic_codes_sha256") for chunk in chunks],
                "output": str(path) if path else None,
                "audio_check": audio_check(path) if path else None,
                "asr": asr.get(str(path.resolve())) if path else None,
            }
            row["asr_normalized_text_match"] = (
                normalized(row["asr"]["text"]) == normalized(result["text"]) if row["asr"] else None)
            rows.append(row)
        longs = [row for row in rows if row["case"] == "clone-long" and row["status"] == "complete"]
        studies[label] = {
            "report": str(source), "cold_load_seconds": report["cold_load_seconds"], "requests": rows,
            "long_clone_medians": {
                field: statistics.median(row[field] for row in longs)
                for field in ("total_seconds", "decode_seconds", "decode_ms_per_frame", "audio_seconds")
            } if longs else None,
        }
        all_rows.extend(rows)
    original, tuned = studies["int4-auto"]["requests"], studies["int4-tuned"]["requests"]
    parity = [
        {"index": index, "codes_equal": original[index]["codes"] == tuned[index]["codes"],
         "wav_equal": original[index]["audio_check"]["sha256"] == tuned[index]["audio_check"]["sha256"]}
        for index in range(3)
    ]
    result = {
        "scope": "Native Windows, RTX4070 Laptop 8GB/Ryzen7940H/48GB. Torch CPU threads=1. "
        "Cold model loading is separate; complete request times include prefill/graphs/codec. "
        "Quantized models emit different durations and are not equal-quality speed comparisons. "
        "ASR checks text, not perceptual quality or speaker similarity.",
        "studies": studies, "int4_tuning_parity": parity,
        "unconditioned_failures": [
            row for row in all_rows if row["case"].startswith("tts")
            and (row["status"] != "complete" or row["asr_normalized_text_match"] is False)
        ],
        "default_model": "shared official FP8/BF16",
        "all_top8_preserved": True, "training_performed": False,
    }
    (ROOT / "research/phase18-quantized-results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"medians": {label: study["long_clone_medians"] for label, study in studies.items()},
                      "int4_tuning_parity": parity,
                      "unconditioned_failures": len(result["unconditioned_failures"])}, indent=2))


if __name__ == "__main__":
    main()
