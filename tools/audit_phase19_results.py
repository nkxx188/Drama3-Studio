"""Audit native Q4 parity, real-request timings and explicit quality failures."""

import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(text):
    text = re.sub(r"<\|speaker:\d+\|>", "", text).lower()
    text = re.sub(r"\s+", "", text).replace("fp8", "fp八")
    return "".join(c for c in text if c.isalnum())


def main():
    asr = {}
    for name in ("phase19-first-asr.json", "phase19-final-asr.json"):
        for row in read(name)["rows"]:
            asr[str(Path(row["audio"]).resolve())] = row
    native = {}
    for label in ("q4-quality", "q4-quality-compact", "q4-quality-cpu-codec",
                  "q4-quality-compact-plain"):
        report = read(f"phase19-gguf-{label}.json")
        log = Path(report["log"]).read_text(encoding="utf-8", errors="replace")
        times = {key: float(value) / 1000 for key, value in re.findall(
            r"metrics\[([^\]]+)\]\.wall_ms=([0-9.]+)", log)}
        codecs = [float(value) / 1000 for value in re.findall(r"drama3\.codec_ms ([0-9.]+)", log)]
        rows = []
        for index, audio in enumerate(report["audio"]):
            identifier = Path(audio["path"]).stem
            row = {**audio, "request_seconds": times[identifier], "codec_seconds": codecs[index],
                   "sha256": sha(audio["path"]), "asr": asr.get(str(Path(audio["path"]).resolve())),
                   "conditioning": "reference" if "clone" in identifier else
                   "none" if label.endswith("-plain") else "inherited session reference"}
            if row["asr"]:
                expected = read(f"phase19-gguf-{label}-requests.json")[index]["text"]
                row["asr_text_match"] = normalize(row["asr"]["text"]) == normalize(expected)
            rows.append(row)
        longs = [row for row in rows if "clone-long" in row["path"]]
        native[label] = {
            "requests": rows, "resources": report["resources"], "whole_batch_seconds": report["wall_seconds"],
            "median_long_request_seconds": statistics.median(row["request_seconds"] for row in longs)
            if longs else None,
        }
    parity = [
        {"file": Path(a["path"]).name, "wav_equal": a["sha256"] == b["sha256"]}
        for a, b in zip(native["q4-quality"]["requests"], native["q4-quality-compact"]["requests"])
    ]
    assert len(parity) == 4 and all(row["wav_equal"] for row in parity)
    assert native["q4-quality-compact"]["requests"][1]["asr_text_match"]
    assert native["q4-quality-compact-plain"]["requests"][0]["asr_text_match"]
    quant = read("quant-phase19-w4a8-benchmark.json")["results"]
    quant_rows = []
    for row in quant:
        path = row["output"]
        transcript = asr.get(str(Path(path).resolve())) if path else None
        quant_rows.append({
            "case": row["benchmark_case"], "status": row["status"], "output": path,
            "audio": row["audio"], "request_seconds": row["total_seconds"],
            "decode_seconds": sum(chunk["decode_seconds"] for chunk in row["chunks"]),
            "peak_allocated_bytes": max(chunk["peak_allocated_bytes"] for chunk in row["chunks"]),
            "peak_reserved_bytes": max(chunk["peak_reserved_bytes"] for chunk in row["chunks"]),
            "process": row["memory"]["process"], "asr": transcript,
            "asr_text_match": normalize(transcript["text"]) == normalize(row["text"]) if transcript else None,
        })
    assert quant_rows[1]["asr_text_match"]
    assert quant_rows[4]["status"] == "incomplete" and not quant_rows[4]["asr_text_match"]
    comfy = read("phase19-comfy-collected.json")["rows"]
    for row in comfy:
        row["cached_without_generation"] = row["status"] == "success" and not any(
            node["class_type"] == "Drama3Generate" for node in row["node_times"])
        for audio in row["audio"]:
            audio["asr"] = asr.get(str(Path(audio["wav"]).resolve()))
    result = {
        "scope": "Windows RTX4070Laptop8GB/7940H/48GB. No training; all selected experts retained. "
                 "Different formats/runtimes can change audio. Native PCM parity only compares the same GGUF "
                 "before/after head storage compaction, not against official FP8.",
        "native": native, "native_compact_wav_parity": parity, "w4a8": quant_rows,
        "comfy": comfy,
        "native_time_reduction_fraction": 1 - native["q4-quality-compact"]["median_long_request_seconds"]
        / native["q4-quality"]["median_long_request_seconds"],
        "rotation_fusion": read("phase19-rotation-fusion.json"),
        "limits": [
            "ASR checks text, not perceptual quality or speaker identity.",
            "Cold model loading is separate from per-request native metrics.",
            "Native batch prepares from the first request; later omitted voice inherits defaults. "
            "Only the fresh -plain process is counted as no-reference TTS.",
            "Cached Comfy executions without a Generate node are excluded from speed conclusions.",
            "Comfy INT8 measurements overlapped compilation/host pressure; no fair superiority claim.",
            "Native comparison backend is separate from the default PyTorch UI engine.",
        ],
    }
    (ROOT / "research/phase19-results-audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "native_long_medians": {name: study["median_long_request_seconds"] for name, study in native.items()},
        "native_head_time_reduction": result["native_time_reduction_fraction"],
        "all_four_native_wavs_equal": True, "w4a8_long_clones_complete": True,
        "comfy_cached_runs_excluded": sum(row["cached_without_generation"] for row in comfy),
    }, indent=2))


if __name__ == "__main__":
    main()
