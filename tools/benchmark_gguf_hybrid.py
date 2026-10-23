"""Completed-request checks for the optional Q8 cooperative decoder."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import psutil

from benchmark_memory_pressure import GpuUsage, environment
from benchmark_model_comparison import SHORT, TARGET
from benchmark_q8_cache_sweep import parse_requests, save, emit, WddmMemory

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--policy", choices=("off", "0", "2", "4", "8", "auto"), default="auto")
    parser.add_argument("--scenario", choices=("diagnostic", "clone", "acceptance", "seed-check"), default="clone")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--keep-hits", choices=("true", "false"), default="true")
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("seed must be a 32-bit unsigned integer")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        raise FileExistsError("Use a fresh result directory")
    build = json.loads((ROOT / "research/phase33-hybrid-build.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(Path(build["executable"]).read_bytes()).hexdigest()
    if digest != build["sha256"]:
        raise RuntimeError("Cooperative executable changed since build")
    clone = {
        "text": TARGET, "seed": args.seed, "max_tokens": 384, "temperature": 0.7, "top_k": 30, "top_p": 0.7,
        "voice_ref": str(ROOT / "outputs/quality-official-fp8.wav"),
        "options": {"reference_text": SHORT, "text_chunk_size": "1000"},
    }
    if args.scenario == "diagnostic":
        requests = [{**clone, "id": "diagnostic", "max_tokens": 33}]
    else:
        requests = [{**clone, "id": "warmup", "max_tokens": 12},
                    {**clone, "id": "clone-1"}, {**clone, "id": "clone-2"}]
        if args.scenario in ("acceptance", "seed-check"):
            requests.extend([
                {"id": "tts", "text": "<|speaker:0|>你好，这是一次本地语音测试。",
                 "seed": args.seed, "max_tokens": 256, "temperature": 0.7, "top_k": 30, "top_p": 0.7},
                {**clone, "id": "clone-long", "max_tokens": 768,
                 "text": "<|speaker:0|>今天我们继续测试这套本地语音引擎。希望普通电脑也能保持清楚自然的声音，"
                         "读完一整句话，而不需要购买昂贵的专业显卡。接下来还会仔细检查语音克隆、"
                         "长句停顿和声音的稳定性。"},
            ])
        if args.scenario == "seed-check":
            requests = [request for request in requests if request["id"] != "clone-2"]
    save(out / "requests.json", requests)
    command = [
        build["executable"], "--task", "tts", "--family", "fish_drama3", "--backend", "cuda",
        "--mode", "offline", "--model",
        str(ROOT / "checkpoints/drama-3-gguf-q8/drama3-experts-Q8_0-bf16.gguf"),
        "--threads", "8", "--session-option", "fish_drama3.disk_experts=true",
        "--session-option", "fish_drama3.expert_cache_size=8",
        "--session-option", "fish_drama3.cpu_expert_layers=0",
        "--session-option", "fish_drama3.codec_backend=cpu",
        "--session-option", "fish_drama3.codec_decoder_backend=cuda",
        "--session-option", "fish_drama3.codec_decode_strategy=stream",
        "--session-option", "fish_drama3.context_length=" + ("512" if args.scenario == "diagnostic" else "1024"),
        "--request-sequence", str(out / "requests.json"), "--out-dir", str(out), "--metrics", "--log",
    ]
    if args.policy != "off":
        command.extend(["--session-option", f"fish_drama3.decode_cpu_experts={args.policy}",
                        "--session-option", f"fish_drama3.decode_keep_hits={args.keep_hits}"])
        if args.profile:
            command.extend(["--session-option", "fish_drama3.decode_profile=" + str(args.profile.resolve())])
    env = environment()
    env.update(DRAMA_COMPACT_GGUF_HEAD="1", DRAMA_HOST_GGUF_EMBEDDING="1")
    for name in ("DRAMA_Q8_STEP_RECORD", "DRAMA_Q8_STEP_REPLAY", "DRAMA_Q8_ROUTE_TRACE",
                 "DRAMA_Q8_ROUTE_TRACE_REQUEST", "DRAMA_Q8_RAW_CPU", "DRAMA_Q8_COMPATIBLE_QUANT"):
        env.pop(name, None)
    if args.record:
        env["DRAMA_Q8_STEP_RECORD"] = str(out / "steps.bin")
    if args.teacher:
        env["DRAMA_Q8_STEP_REPLAY"] = str(args.teacher.resolve())
    env["PATH"] = str(Path(build["cuda_toolkit"]) / "bin") + os.pathsep + env["PATH"]
    report = {"status": "running", "command": command, "policy": args.policy,
              "executable_sha256": digest, "diagnostic": bool(args.record or args.teacher),
              "scenario": args.scenario, "seed": args.seed}
    save(out / "summary.json", report)
    gpu, wddm = GpuUsage(), None
    samples = []
    try:
        try:
            wddm = WddmMemory()
        except OSError:
            pass
        started = last = time.perf_counter()
        emit("hybrid_begin", policy=args.policy, scenario=args.scenario, directory=str(out))
        with (out / "native.log").open("wb") as log:
            child = subprocess.Popen(
                command, cwd=build["source"], env=env, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW)
            observed = psutil.Process(child.pid)
            try:
                while child.poll() is None:
                    now = time.perf_counter()
                    try:
                        memory = observed.memory_info()
                        sample = {"seconds": now - started, "rss": memory.rss,
                                  "available_ram": psutil.virtual_memory().available,
                                  "gpu_total_used": gpu.used()}
                        if wddm:
                            sample.update(wddm.sample(child.pid))
                        samples.append(sample)
                    except (psutil.Error, OSError):
                        pass
                    if now - last >= 20:
                        emit("hybrid_progress", seconds=round(now - started, 1),
                             completed_wavs=len(list(out.glob("*.wav"))))
                        last = now
                    if now - started > args.timeout:
                        raise TimeoutError("Bounded native hybrid run timed out")
                    time.sleep(0.5)
            finally:
                if child.poll() is None:
                    child.terminate()
                    child.wait(timeout=20)
        report.update(returncode=child.returncode, process_seconds=time.perf_counter() - started)
        log_text = (out / "native.log").read_text(encoding="utf-8", errors="replace")
        report["statistics"] = {
            key: float(value) for key, value in re.findall(
                r"(drama3\.hybrid\.[a-z_0-9]+|disk_experts\.[a-z_]+) ([\d.]+)", log_text)
        }
        report["resources"] = {
            "sampled_peak_rss": max((r["rss"] for r in samples), default=0),
            "sampled_minimum_available_ram": min((r["available_ram"] for r in samples), default=0),
            "sampled_peak_gpu_total": max((r["gpu_total_used"] for r in samples), default=0),
            **{f"sampled_peak_{key}": max((r[key] for r in samples if r.get(key) is not None), default=None)
               for key in ("dedicated_bytes", "shared_bytes")},
        }
        if child.returncode:
            report["status"] = "error"
        else:
            report["requests"] = parse_requests(log_text, out)
            report["status"] = "complete" if len(report["requests"]) == len(requests) else "error"
            if args.scenario != "diagnostic" and any(
                r["finish_reason"] != "eos" for r in report["requests"] if not r["warmup"]
            ):
                report["status"] = "incomplete"
        save(out / "samples.json", samples)
        save(out / "summary.json", report)
        emit("hybrid_complete", status=report["status"], requests=[
            {"case": r["case"], "seconds": r["total_seconds"], "frames": r["frames"],
             "decode_ms_per_frame": r["decode_ms_per_frame"]}
            for r in report.get("requests", [])
        ])
        if report["status"] != "complete":
            raise RuntimeError(f"Native hybrid check failed; see {out}")
    finally:
        gpu.close()
        if wddm:
            wddm.close()


if __name__ == "__main__":
    main()
