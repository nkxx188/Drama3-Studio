"""Check mixed-backbone model discovery and bounded real clone generation."""

from __future__ import annotations

import json
import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".cache/drama3-backbone-q8-20261031/validation"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor/fish-speech")]
WORK.mkdir(parents=True, exist_ok=True)
os.environ.update(TEMP=str(WORK), TMP=str(WORK), TMPDIR=str(WORK))

import numpy as np
import soundfile as sf

from drama_low_vram.gguf_resources import inspect_gguf
from drama_low_vram.gguf_runtime import GGUFEngine
from drama_low_vram.gguf_workflows import GGUFWorkflows
from drama_low_vram.protocols import VoiceReference


def save(report):
    (WORK / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=("q4", "q8"), default=["q4", "q8"])
    args = parser.parse_args()
    recipe = json.loads((
        ROOT / "ui_data/jobs/e9825702930d423494b9a03318f32024/job.json"
    ).read_text(encoding="utf-8"))["recipe"]
    reference = recipe["reference_snapshot"][0]
    text = recipe["text"]
    refs = [VoiceReference(reference["audio"], reference["text"], 0)]
    report = {
        "scope": "Real short warmup and full clone smoke checks; not a controlled quality comparison",
        "sampling": {"seed": 123, "temperature": 1.0, "top_p": 0.9, "top_k": 0},
        "worker_policy": "Use the currently installed worker for both variants",
        "models": {},
    }
    for label in args.models:
        path = next((ROOT / f"checkpoints/drama-3-20261029-gguf-{label}-backbone-q8").glob("*.gguf"))
        row = report["models"][label] = {
            "model": str(path), "inspection": inspect_gguf(path), "tasks": {},
        }
        save(report)
        engine = GGUFEngine(path, root=ROOT)
        workflows = GGUFWorkflows(engine)
        try:
            print(f"LOAD {label}", flush=True)
            begin = time.monotonic()
            engine.load()
            row["cold_load_seconds"] = time.monotonic() - begin
            row["resource_plan"] = engine.plan.report()
            print(
                f"LOADED {label}: waveform={engine.plan.decoder_backend}, "
                f"cache={engine.plan.cache_experts}, load={row['cold_load_seconds']:.2f}s",
                flush=True,
            )
            row["reference_prewarm"] = engine.prewarm_references(refs)
            print(f"CLONE {label}", flush=True)
            result = workflows.run(
                "tts", text, WORK / f"{label}-clone.wav",
                references=refs, max_new_tokens=768, seed=123,
                temperature=1.0, top_p=0.9, top_k=0, silence_retry=True,
            )
            row["tasks"]["clone"] = result
            audio, rate = sf.read(result["output"], dtype="float32")
            if (
                result["status"] != "complete" or rate != 44100
                or not np.isfinite(audio).all() or np.max(np.abs(audio)) < 0.001
            ):
                raise RuntimeError("Clone did not produce complete non-silent finite audio")
            row["passed"] = True
            print(json.dumps({
                "event": "pass", "model": label,
                "request_seconds": result["request_seconds"],
                "audio_seconds": result["audio_seconds"],
                "decoder": result["resource_plan"]["decoder_backend"],
                "codec_seconds": result["timings"].get("drama3.codec_ms", 0) / 1000,
            }), flush=True)
        except Exception as error:
            row.update(passed=False, error=f"{type(error).__name__}: {error}")
            save(report)
            raise
        finally:
            engine.close()
            save(report)
    print("Both mixed-backbone models passed native clone generation.", flush=True)


if __name__ == "__main__":
    main()
