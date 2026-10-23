"""Measure the selected quality-oriented Q4 GGUF in the native CUDA runtime."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import wave

import psutil
try:
    import pynvml
except ImportError:
    pynvml = None

ROOT = Path(__file__).resolve().parents[1]
SHORT = "<|speaker:0|>你好，这是官方 FP8 质量优先测试。"
LONG = ("<|speaker:0|>今天我们继续测试这套本地语音引擎。希望普通电脑也能保持清楚自然的声音，"
        "读完一整句话，而不需要购买昂贵的专业显卡。接下来还会仔细检查语音克隆、"
        "长句停顿和声音的稳定性。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="q4-quality")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--pin-experts", action="store_true")
    parser.add_argument("--cpu-layers", type=int, default=0)
    parser.add_argument("--cache-experts", type=int, default=8)
    parser.add_argument("--codec-cpu", action="store_true")
    parser.add_argument("--compact-head", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cases", nargs="+", choices=("clone-short", "clone-long", "tts-short"),
                        default=["clone-short", "clone-long", "clone-long", "tts-short"])
    args = parser.parse_args()
    build = json.loads((ROOT / "research/phase19-audio-cpp-build.json").read_text())
    pinned = json.loads((ROOT / "research/phase19-gguf-q4-quality-download-pinned.json").read_text())
    model = Path(pinned["destination"]) / pinned["files"][0]["name"]
    assert model.is_file()
    output = ROOT / "outputs" / f"phase19-gguf-{args.label}"
    output.mkdir(exist_ok=True)
    cases = ["clone-short"] if args.smoke else args.cases
    requests = []
    for index, case in enumerate(cases):
        request = {
            "id": f"{index}-{case}", "text": LONG if case == "clone-long" else SHORT,
            "seed": 123, "max_tokens": 4 if args.smoke else 768,
            "temperature": 1.0, "top_k": 30, "top_p": 0.9,
        }
        if case.startswith("clone"):
            request["voice_ref"] = str(ROOT / "outputs/quality-official-fp8.wav")
            request["options"] = {"reference_text": SHORT, "text_chunk_size": "1000"}
        requests.append(request)
    sequence = ROOT / "research" / f"phase19-gguf-{args.label}-requests.json"
    sequence.write_text(json.dumps(requests, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    command = [
        build["executable"], "--task", "tts", "--family", "fish_drama3",
        "--backend", "cuda", "--mode", "offline", "--model", str(model),
        "--threads", str(args.threads), "--session-option", "fish_drama3.context_length=1024",
        "--session-option", "fish_drama3.disk_experts=true",
        "--session-option", f"fish_drama3.expert_cache_size={args.cache_experts}",
        "--session-option", f"fish_drama3.cpu_expert_layers={args.cpu_layers}",
        "--session-option", "fish_drama3.profile_first_audio=true",
        "--request-sequence", str(sequence), "--out-dir", str(output), "--metrics", "--log",
    ]
    if args.pin_experts:
        command.extend(["--session-option", "fish_drama3.pin_experts=true"])
    if args.codec_cpu:
        command.extend(["--session-option", "fish_drama3.codec_backend=cpu"])
    environment = dict(os.environ)
    environment["PATH"] = str(Path(build["cuda_toolkit"]) / "bin") + os.pathsep + environment["PATH"]
    environment.update(TEMP=str(ROOT / ".cache/tmp"), TMP=str(ROOT / ".cache/tmp"))
    if args.compact_head:
        environment["DRAMA_COMPACT_GGUF_HEAD"] = "1"
    else:
        environment.pop("DRAMA_COMPACT_GGUF_HEAD", None)
    log_path = ROOT / "research" / f"phase19-gguf-{args.label}.log"
    report = {"scope": "Native GGUF backend. Batch requests reuse one model/session. "
              "Native sampling and FP16 codec differ from the PyTorch path.",
              "command": command, "build": build, "model": pinned, "samples": [],
              "compact_head": args.compact_head}
    gpu = None
    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
        except pynvml.NVMLError:
            pass
    print(json.dumps({"phase": "gguf-start", "label": args.label,
                      "cache_experts": args.cache_experts, "cpu_layers": args.cpu_layers}), flush=True)
    started, last = time.perf_counter(), 0
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, cwd=build["source"], env=environment,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        observed = psutil.Process(process.pid)
        while process.poll() is None:
            elapsed = time.perf_counter() - started
            try:
                memory = observed.memory_info()
                sample = {"seconds": elapsed, "rss": memory.rss,
                          "private": memory.private, "available_ram": psutil.virtual_memory().available}
                if gpu is not None:
                    sample["whole_gpu_bytes"] = pynvml.nvmlDeviceGetMemoryInfo(gpu).used
                report["samples"].append(sample)
            except psutil.NoSuchProcess:
                break
            if elapsed - last >= 20:
                print(json.dumps({"phase": "gguf-progress", "label": args.label,
                                  "seconds": elapsed,
                                  "log_tail": log_path.read_text(encoding="utf-8", errors="replace")[-650:]}),
                      flush=True)
                last = elapsed
            if elapsed > (240 if args.smoke else 1200):
                process.terminate()
                report["timeout"] = True
                break
            time.sleep(1)
        process.wait(timeout=20)
    report.update(returncode=process.returncode, wall_seconds=time.perf_counter() - started,
                  log=str(log_path), audio=[])
    for path in sorted(output.rglob("*.wav")):
        with wave.open(str(path), "rb") as wav:
            report["audio"].append({"path": str(path), "sample_rate": wav.getframerate(),
                                     "samples": wav.getnframes(),
                                     "seconds": wav.getnframes() / wav.getframerate()})
    if report["samples"]:
        report["resources"] = {
            "peak_rss": max(row["rss"] for row in report["samples"]),
            "peak_private": max(row["private"] for row in report["samples"]),
            "minimum_available_ram": min(row["available_ram"] for row in report["samples"]),
            "sampled_whole_gpu_peak_bytes": max(row.get("whole_gpu_bytes", 0)
                                                for row in report["samples"]),
        }
    target = ROOT / "research" / f"phase19-gguf-{args.label}.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key in (
        "returncode", "wall_seconds", "audio", "resources", "timeout")}), flush=True)
    if process.returncode != 0:
        raise RuntimeError(f"Native GGUF failed; see {log_path}")


if __name__ == "__main__":
    main()
