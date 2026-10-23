"""Sequential supported-model TTS/clone comparison with a fixed audio frame cap."""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from benchmark_memory_pressure import GpuUsage, SHORT, audio_details, environment

MODELS = {
    "fp8": "drama-3", "int4": "drama-3-int4", "int8": "drama-3-int8",
    "w6a8": "drama-3-w6a8", "w4a8": "drama-3-w4a8", "gguf-q4": "drama-3-gguf-q4",
}
FRAMES = 215
TEXT = "今天我们测试本地语音生成。希望声音清楚自然，停顿流畅，方便大家比较不同模型的速度和效果。"
TARGET = f"<|speaker:0|> {TEXT} [audio tokens: {FRAMES}]"
CASES = (("warmup-tts", 8, False), ("tts", FRAMES, False),
         ("warmup-clone", 8, True), ("clone", FRAMES, True))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def emit(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def torch_worker(label, destination):
    os.environ.update(environment())
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor/fish-speech")]
    import torch
    from drama_low_vram.engine import DramaEngine, EngineOptions

    torch.set_num_threads(1)
    report = {"label": label, "status": "running", "requests": []}
    path = destination / "summary.json"
    save(path, report)
    try:
        with DramaEngine(EngineOptions(
            checkpoint=ROOT / "checkpoints" / MODELS[label],
            expert_storage="auto", backbone_graph=True, combined_decode_bank=True,
            prefix_cache_mib=256, host_gather="auto", compute_policy="auto",
            compute_profile_dir=ROOT / ".cache/compute-profiles",
        )) as engine:
            started = time.perf_counter()
            engine.load()
            report["model_load_seconds"] = time.perf_counter() - started
            report["runtime"] = engine.runtime.as_dict()
            started = time.perf_counter()
            engine._prepare(512)
            report["initial_prepare_seconds"] = time.perf_counter() - started
            report["compute_policy"] = engine.compute_report
            save(path, report)
            emit("prepared", label=label, load=report["model_load_seconds"],
                 prepare=report["initial_prepare_seconds"], compute=engine.compute_report)
            for case, maximum, clone in CASES:
                target = destination / f"{case}.wav"
                emit("request_begin", label=label, case=case)
                result = engine.synthesize(
                    TARGET, target, seed=123, max_new_tokens=maximum, allow_partial=True,
                    reference_audio=ROOT / "outputs/quality-official-fp8.wav" if clone else None,
                    reference_text=SHORT if clone else None,
                    progress=lambda count, limit: emit(
                        "frames", label=label, case=case, frames=count, limit=limit)
                    if count == 1 or count % 64 == 0 else None,
                )
                audio = audio_details(target) if target.is_file() else None
                row = {
                    "case": case, "warmup": case.startswith("warmup"),
                    "total_seconds": result["total_seconds"],
                    "status": result["status"],
                    "finish_reason": [c["finish_reason"] for c in result["chunks"]],
                    "frames": sum(c["frames"] for c in result["chunks"]),
                    "prefill_seconds": sum(c["prefill_seconds"] for c in result["chunks"]),
                    "decode_seconds": sum(c["decode_seconds"] for c in result["chunks"]),
                    "codec_seconds": result["codec_seconds"],
                    "reference_seconds": result["reference_seconds"],
                    "reference_cache_hit": result["reference_cache_hit"],
                    "audio": audio, "full_report": str(target.with_suffix(".json")),
                    "runtime_plans": [c["runtime_plan"] for c in result["chunks"]],
                    "prefix_cache": [c["prefix_cache"] for c in result["chunks"]],
                    "lifecycle": result["lifecycle_delta"],
                    "peak_gpu_allocated_bytes": max(c["peak_allocated_bytes"] for c in result["chunks"]),
                    "process": result["memory"]["process"],
                }
                if audio:
                    row["rtf"] = row["total_seconds"] / audio["seconds"]
                report["requests"].append(row)
                save(path, report)
                emit("request_complete", label=label, case=case, seconds=row["total_seconds"],
                     audio_seconds=audio["seconds"] if audio else None,
                     finish=row["finish_reason"], frames=row["frames"])
            report["status"] = "complete"
    except BaseException as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc())
        raise
    finally:
        save(path, report)


def gguf_worker(destination):
    from run_drama_gguf import basic_settings

    build = json.loads((ROOT / "research/phase19-audio-cpp-build.json").read_text())
    pin = json.loads((ROOT / "research/phase19-gguf-q4-quality-download-pinned.json").read_text())
    model = Path(pin["destination"]) / pin["files"][0]["name"]
    settings = basic_settings(model.stat().st_size, psutil.virtual_memory().available,
                              psutil.cpu_count(logical=False))
    requests = []
    for case, maximum, clone in CASES:
        request = {"id": case, "text": TARGET, "seed": 123, "max_tokens": maximum,
                   "temperature": 0.7, "top_k": 30, "top_p": 0.7,
                   "options": {"text_chunk_size": "1000"}}
        if clone:
            request["voice_ref"] = str(ROOT / "outputs/quality-official-fp8.wav")
            request["options"]["reference_text"] = SHORT
        requests.append(request)
    sequence = destination / "requests.json"
    save(sequence, requests)
    command = [
        build["executable"], "--task", "tts", "--family", "fish_drama3",
        "--backend", "cuda", "--mode", "offline", "--model", str(model),
        "--threads", str(settings["threads"]),
        "--session-option", "fish_drama3.disk_experts=true",
        "--session-option", "fish_drama3.expert_cache_size=8",
        "--session-option", "fish_drama3.cpu_expert_layers=0",
        "--session-option", "fish_drama3.context_length=512",
        "--session-option", "fish_drama3.profile_first_audio=true",
        "--request-sequence", str(sequence), "--out-dir", str(destination),
        "--metrics", "--log",
    ]
    if settings["host_storage"] == "pin":
        command += ["--session-option", "fish_drama3.pin_experts=true"]
    env = environment()
    env["DRAMA_COMPACT_GGUF_HEAD"] = "1"
    env["PATH"] = str(Path(build["cuda_toolkit"]) / "bin") + os.pathsep + env["PATH"]
    report = {"label": "gguf-q4", "status": "running", "requests": [],
              "settings": settings, "command": command}
    save(destination / "summary.json", report)
    emit("native_begin", settings=settings)
    started = time.perf_counter()
    log_path = destination / "native.log"
    with log_path.open("wb") as log_file:
        result = subprocess.run(command, cwd=build["source"], env=env,
                                stdout=log_file, stderr=subprocess.STDOUT)
    log = log_path.read_text(encoding="utf-8", errors="replace")
    report["process_seconds"] = time.perf_counter() - started
    report["returncode"] = result.returncode
    for case, maximum, clone in CASES:
        elapsed = re.search(rf"metrics\[{case}\]\.wall_ms=([\d.]+)", log)
        schema = re.search(rf"request_id={case}\n.*?custom_schema_output=(\{{[^\n]+\}})", log, re.S)
        audio_path = destination / f"{case}.wav"
        if elapsed is None or schema is None:
            continue
        data = json.loads(schema.group(1))["data"]
        audio = audio_details(audio_path) if audio_path.is_file() else None
        row = {"case": case, "warmup": case.startswith("warmup"),
               "total_seconds": float(elapsed.group(1)) / 1000,
               "frames": data["frames"],
               "status": "incomplete" if data["reached_limit"] else "complete",
               "finish_reason": ["length" if data["reached_limit"] else "eos"],
               "audio": audio}
        if audio:
            row["rtf"] = row["total_seconds"] / audio["seconds"]
        report["requests"].append(row)
        emit("request_complete", label="gguf-q4", **row)
    report["outside_request_seconds"] = report["process_seconds"] - sum(
        row["total_seconds"] for row in report["requests"])
    report["status"] = "complete" if result.returncode == 0 and len(report["requests"]) == 4 else "error"
    save(destination / "summary.json", report)
    if report["status"] == "error":
        raise RuntimeError(f"Native comparison failed: see {log_path}")


def supervise(destination):
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "summary.json"
    if target.exists():
        raise FileExistsError("Choose a new output folder")
    report = {
        "status": "running", "started": datetime.datetime.now().isoformat(),
        "text": TEXT, "prompt_text": TARGET, "max_audio_frames": FRAMES,
        "duration_cap_seconds": FRAMES * 2048 / 44100,
        "sampling": {"seed": 123, "temperature": 0.7, "top_k": 30, "top_p": 0.7},
        "reference_audio": str(ROOT / "outputs/quality-official-fp8.wav"),
        "reference_text": SHORT, "models": [],
        "scope": (
            "One measured TTS and clone per model after explicit eight-frame warmups. "
            "215 audio-frame cap; early EOS may be shorter, frame-limit output is partial. "
            "Loading, initial compute calibration and warmups are separate. "
            "Fresh process per model; TTS always precedes cloning to avoid inherited voice. "
            "Different model precision/backend; approximate speed comparison, not equal quality."
        ),
    }
    save(target, report)
    gpu = None
    try:
        gpu = GpuUsage()
    except OSError:
        pass
    try:
        for label in MODELS:
            folder = destination / label
            folder.mkdir()
            row = {"label": label, "status": "running", "directory": str(folder)}
            report["models"].append(row)
            save(target, report)
            emit("model_begin", label=label)
            started = time.perf_counter()
            samples = []
            with (folder / "worker.log").open("wb") as log_file:
                process = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), "--worker", label,
                     "--output", str(folder)], cwd=ROOT, env=environment(),
                    stdout=log_file, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                observed = psutil.Process(process.pid)
                last = time.perf_counter()
                while process.poll() is None:
                    try:
                        processes = [observed, *observed.children(recursive=True)]
                        memories = [p.memory_info() for p in processes if p.is_running()]
                        sample = {"rss": sum(m.rss for m in memories),
                                  "private": sum(getattr(m, "private", m.rss) for m in memories),
                                  "available_ram": psutil.virtual_memory().available}
                        if gpu is not None:
                            sample["whole_gpu"] = gpu.used()
                        samples.append(sample)
                    except (psutil.NoSuchProcess, OSError):
                        pass
                    if time.perf_counter() - last >= 20:
                        emit("model_progress", label=label,
                             elapsed=time.perf_counter() - started)
                        last = time.perf_counter()
                    time.sleep(0.5)
                process.wait()
            row.update(
                status="complete" if process.returncode == 0 else "error",
                returncode=process.returncode, process_seconds=time.perf_counter() - started)
            if samples:
                row["resources"] = {
                    "peak_rss_bytes": max(s["rss"] for s in samples),
                    "peak_private_bytes": max(s["private"] for s in samples),
                    "minimum_available_ram_bytes": min(s["available_ram"] for s in samples),
                    "sampled_whole_gpu_peak_bytes": max(s.get("whole_gpu", 0) for s in samples),
                }
            worker_report = folder / "summary.json"
            if worker_report.exists():
                row["report"] = json.loads(worker_report.read_text(encoding="utf-8"))
            save(target, report)
            emit("model_complete", label=label, status=row["status"])
        report["status"] = "complete"
    finally:
        if gpu is not None:
            gpu.close()
        save(target, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=tuple(MODELS))
    args = parser.parse_args()
    destination = args.output.resolve()
    if args.worker == "gguf-q4":
        gguf_worker(destination)
    elif args.worker:
        torch_worker(args.worker, destination)
    else:
        supervise(destination)


if __name__ == "__main__":
    main()
