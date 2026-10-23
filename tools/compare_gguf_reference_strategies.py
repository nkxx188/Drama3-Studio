"""Generate listening samples with one GGUF and three reference encoders."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TEXT = (
    "\u4f60\u597d\uff0c\u8fd9\u662f\u540c\u4e00\u4e2a\u58f0\u97f3\u7684\u5bf9\u6bd4\u6d4b\u8bd5\u3002"
    "\u8bf7\u4ed4\u7ec6\u542c\u6211\u7684\u97f3\u8272\u3001\u505c\u987f\u548c\u8bed\u6c14\u3002"
    "Hello everyone. Take your time, and enjoy what you do."
)
STRATEGIES = (
    ("A-default-native", "native.safetensors", "Native CPU reference encoding, current default"),
    ("B-native-fp32", "native-f32.safetensors", "Native CPU FP32 reference encoding"),
    ("C-python-fp32", "python-f32.safetensors", "Official Python codec implementation, FP32"),
)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--codes-directory", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1629579822)
    parser.add_argument("--text", default=TEXT)
    args = parser.parse_args()
    import numpy as np
    import soundfile as sf
    from safetensors.numpy import load_file

    from drama_low_vram.gguf_runtime import GGUFEngine

    recipe = json.loads(args.job.read_text(encoding="utf-8"))["recipe"]
    if len(recipe["reference_snapshot"]) != 1:
        raise ValueError("This listening comparison requires exactly one reference")
    reference = recipe["reference_snapshot"][0]
    speaker = reference["speaker"]
    transcript = f"<|speaker:{speaker}|> " + reference["text"]
    speech = f"<|speaker:{speaker}|> " + args.text
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.codes_directory / "input.wav", folder / "reference.wav")
    baseline = load_file(str(args.codes_directory / "python-f32.safetensors"))["codes"]
    report = {
        "scope": "Listening comparison only; the GGUF, generation seed, text, sampler and waveform decoder are fixed",
        "reference_preparation": (
            "Reuse the three validated code arrays produced from identical mono float32 samples and "
            "the same GGUF codec weights. Preparation/load timing is not a strategy benchmark."
        ),
        "source_job": str(args.job.resolve()), "model": recipe["model"],
        "reference": reference, "canonical_reference": str(folder / "reference.wav"),
        "text": args.text, "seed": args.seed,
        "sampling": {"temperature": 1.0, "top_p": 0.9, "top_k": 30, "max_tokens": 768},
        "results": [], "status": "running",
    }
    save(folder / "comparison.json", report)
    engine = GGUFEngine(recipe["model"], root=args.runtime_root, context=1536)
    try:
        engine.load()
        report["worker_sha256"] = engine.build["sha256"]
        report["resource_plan"] = engine.plan.report()
        report["load_seconds"] = engine.load_seconds
        save(folder / "comparison.json", report)
        for label, filename, description in STRATEGIES:
            path = args.codes_directory / filename
            codes = load_file(str(path))["codes"]
            if codes.shape != baseline.shape or codes.ndim != 2 or codes.shape[0] != 10:
                raise ValueError("Reference encoding dimensions differ")
            if codes.dtype.kind not in "iu" or np.min(codes) < 0 or np.max(codes) >= 4096:
                raise ValueError("Invalid reference code IDs")
            shutil.copy2(path, folder / (label + ".safetensors"))
            output = folder / (label + ".wav")
            identifier = uuid.uuid4().hex
            request = {
                "text": speech, "seed": args.seed, **report["sampling"],
                "options": {
                    "text_chunk_size": "2000",
                    "references": json.dumps([{"text": transcript, "vq_codes": codes.tolist()}],
                                             ensure_ascii=False),
                },
            }
            save(folder / (label + ".request.json"), request)
            started = time.perf_counter()
            print(json.dumps({"stage": "generate", "strategy": label}), flush=True)
            # Send the session's supported VQ-code reference descriptor directly;
            # no production API or default configuration is changed.
            with engine._lock:
                engine._busy = True
                try:
                    engine._process.stdin.write(json.dumps({
                        "id": identifier, "output": str(output), "request": request,
                        "streaming": False, "chunk_frames": 16,
                    }, ensure_ascii=False) + "\n")
                    engine._process.stdin.flush()
                    result = engine._receive(request_id=identifier, timeout=600)
                finally:
                    engine._busy = False
            generation = result.get("generation") or {}
            if result.get("event") != "result" or generation.get("reached_limit", True):
                raise RuntimeError(f"Incomplete listening sample: {result}")
            audio, rate = sf.read(output, dtype="float32", always_2d=True)
            if rate != 44100 or audio.shape[1] != 1 or not len(audio) or not np.isfinite(audio).all():
                raise ValueError("Invalid output waveform")
            if float(np.max(np.abs(audio))) < 1e-4:
                raise ValueError("Output waveform is silent")
            take = json.loads(output.with_suffix(".take.json").read_text(encoding="utf-8"))
            if len(take["vq_codes"]) != 10 or take["text"] != speech:
                raise ValueError("Generated take does not match the requested text")
            row = {
                "strategy": label, "description": description, "output": str(output),
                "codes_sha256": hashlib.sha256(codes.astype("<i4").tobytes()).hexdigest(),
                "code_agreement_vs_python": float((codes == baseline).mean()),
                "main_code_agreement_vs_python": float((codes[0] == baseline[0]).mean()),
                "duration_seconds": len(audio) / rate, "generation_seconds": time.perf_counter() - started,
                "wav_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "result": result,
            }
            report["results"].append(row)
            save(folder / "comparison.json", report)
            print(json.dumps({key: row[key] for key in
                              ("strategy", "duration_seconds", "generation_seconds")}), flush=True)
        report["status"] = "complete"
        report["native_log"] = str(engine._directory / "native.log")
        save(folder / "comparison.json", report)
    except BaseException as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        save(folder / "comparison.json", report)
        raise
    finally:
        engine.close()


if __name__ == "__main__":
    main()
