"""Validate and benchmark the locally converted audio.cpp Q8 package."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
from benchmark_memory_pressure import GpuUsage, audio_details, environment
from benchmark_model_comparison import CASES, TARGET, FRAMES, SHORT
from run_drama_gguf import basic_settings


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host-storage", choices=("auto", "pin", "mapped"), default="auto")
    parser.add_argument("--codec", choices=("same", "cpu"), default="same")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cpu-layers", type=int, default=0)
    parser.add_argument("--cache-experts", type=int, default=8)
    args = parser.parse_args()
    if not 8 <= args.cache_experts <= 256:
        parser.error("cache-experts must be 8..256")
    target = args.output.resolve()
    target.mkdir(parents=True, exist_ok=True)
    if (target / "summary.json").exists():
        raise FileExistsError("Choose a fresh benchmark directory")
    model = ROOT / "checkpoints/drama-3-gguf-q8/drama3-experts-Q8_0-bf16.gguf"
    conversion = json.loads(model.with_suffix(".conversion.json").read_text(encoding="utf-8"))
    if conversion["status"] != "complete" or model.stat().st_size != conversion["output_bytes"]:
        raise ValueError("The converted Q8 model is incomplete")
    build = json.loads((ROOT / "research/phase19-audio-cpp-build.json").read_text())
    settings = basic_settings(model.stat().st_size, psutil.virtual_memory().available,
                              psutil.cpu_count(logical=False), storage=args.host_storage)
    cases = (("smoke", 8, False),) if args.smoke else CASES
    requests = []
    for case, maximum, clone in cases:
        request = {
            "id": case, "text": TARGET, "seed": 123, "max_tokens": maximum,
            "temperature": 0.7, "top_k": 30, "top_p": 0.7,
            "options": {"text_chunk_size": "1000"},
        }
        if clone:
            request["voice_ref"] = str(ROOT / "outputs/quality-official-fp8.wav")
            request["options"]["reference_text"] = SHORT
        requests.append(request)
    save(target / "requests.json", requests)
    command = [
        build["executable"], "--task", "tts", "--family", "fish_drama3",
        "--backend", "cuda", "--mode", "offline", "--model", str(model),
        "--threads", str(settings["threads"]),
        "--session-option", "fish_drama3.disk_experts=true",
        "--session-option", f"fish_drama3.expert_cache_size={args.cache_experts}",
        "--session-option", f"fish_drama3.cpu_expert_layers={args.cpu_layers}",
        "--session-option", f"fish_drama3.codec_backend={args.codec}",
        "--session-option", "fish_drama3.context_length=512",
        "--session-option", "fish_drama3.profile_first_audio=true",
        "--request-sequence", str(target / "requests.json"), "--out-dir", str(target),
        "--metrics", "--log",
    ]
    if settings["host_storage"] == "pin":
        command += ["--session-option", "fish_drama3.pin_experts=true"]
    env = environment()
    env["DRAMA_COMPACT_GGUF_HEAD"] = "1"
    env["PATH"] = str(Path(build["cuda_toolkit"]) / "bin") + os.pathsep + env["PATH"]
    report = {
        "status": "running", "model": str(model), "model_sha256": conversion["output_sha256"],
        "recipe": conversion["recipe"], "settings": settings, "command": command,
        "smoke": args.smoke, "requests": [],
        "scope": (
            "Local Q8 experts and BF16 backbone/tower; native FP16 codec. "
            "Same phase27 text/215-frame cap/sampling; one formal sample per task after "
            "explicit eight-frame warmups. Different precision, not an equal-quality proof."
        ),
    }
    save(target / "summary.json", report)
    samples = []
    gpu = GpuUsage()
    started = time.perf_counter()
    try:
        with (target / "native.log").open("wb") as log_file:
            child = subprocess.Popen(command, cwd=build["source"], env=env,
                                     stdout=log_file, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
            observed = psutil.Process(child.pid)
            last = time.perf_counter()
            while child.poll() is None:
                try:
                    memory = observed.memory_info()
                    samples.append({
                        "rss": memory.rss, "private": getattr(memory, "private", memory.rss),
                        "whole_gpu": gpu.used(),
                        "available_ram": psutil.virtual_memory().available,
                    })
                except (psutil.NoSuchProcess, OSError):
                    pass
                if time.perf_counter() - last >= 20:
                    print(json.dumps({"event": "progress", "seconds": time.perf_counter() - started}),
                          flush=True)
                    last = time.perf_counter()
                time.sleep(0.5)
            child.wait()
        log = (target / "native.log").read_text(encoding="utf-8", errors="replace")
        report.update(returncode=child.returncode, process_seconds=time.perf_counter() - started)
        for case, maximum, clone in cases:
            elapsed = re.search(rf"metrics\[{case}\]\.wall_ms=([\d.]+)", log)
            schema = re.search(rf"request_id={case}\n.*?custom_schema_output=(\{{[^\n]+\}})", log, re.S)
            if elapsed is None or schema is None:
                continue
            data = json.loads(schema.group(1))["data"]
            path = target / f"{case}.wav"
            audio = audio_details(path) if path.is_file() else None
            row = {
                "case": case, "warmup": case.startswith("warmup") or args.smoke,
                "total_seconds": float(elapsed.group(1)) / 1000,
                "frames": data["frames"], "audio": audio,
                "finish_reason": ["length" if data["reached_limit"] else "eos"],
                "status": "incomplete" if data["reached_limit"] else "complete",
            }
            if audio:
                row["rtf"] = row["total_seconds"] / audio["seconds"]
            report["requests"].append(row)
        report["status"] = (
            "complete" if child.returncode == 0 and len(report["requests"]) == len(cases) else "error")
        report["outside_request_seconds"] = report["process_seconds"] - sum(
            row["total_seconds"] for row in report["requests"])
        if samples:
            report["resources"] = {
                "peak_rss_bytes": max(s["rss"] for s in samples),
                "peak_private_bytes": max(s["private"] for s in samples),
                "sampled_whole_gpu_peak_bytes": max(s["whole_gpu"] for s in samples),
                "minimum_available_ram_bytes": min(s["available_ram"] for s in samples),
            }
        save(target / "summary.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        if report["status"] == "error":
            raise RuntimeError(f"Native Q8 test failed; see {target / 'native.log'}")
    finally:
        gpu.close()


if __name__ == "__main__":
    main()
