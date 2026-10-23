"""Run real FP8/Q4 requests with a verified, physically locked RAM balloon.

Windows still reports the installed RAM. Only the FP8 worker's total-RAM
snapshot is adjusted; available RAM and commit always come from the real OS.
The balloon is released on exit. No boot, paging-file or other-process settings
are changed. Disk counters cover the whole physical drive, including background
activity; process page faults are not labelled as physical disk traffic.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes as w
from dataclasses import replace
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
import wave

import psutil

ROOT = Path(__file__).resolve().parents[1]
GIB, MIB = 1024**3, 1024**2
SHORT = "<|speaker:0|>你好，这是官方 FP8 质量优先测试。"
LONG = (
    "<|speaker:0|>今天我们继续测试这套本地语音引擎。希望普通电脑也能保持清楚自然的声音，"
    "读完一整句话，而不需要购买昂贵的专业显卡。接下来还会仔细检查语音克隆、"
    "长句停顿和声音的稳定性。"
)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def emit(**value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def environment():
    env = dict(os.environ)
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    for name, directory in (
        ("TEMP", "tmp"), ("TMP", "tmp"), ("TRITON_CACHE_DIR", "triton"),
        ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
        ("TORCH_EXTENSIONS_DIR", "torch_extensions"), ("HF_HOME", "huggingface"),
    ):
        location = ROOT / ".cache" / directory
        location.mkdir(parents=True, exist_ok=True)
        env[name] = str(location)
    return env


class LockedBalloon:
    def __init__(self, size):
        self.size = size
        self.regions = []
        self.kernel = k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.GetCurrentProcess.restype = w.HANDLE
        k.SetProcessWorkingSetSize.argtypes = [w.HANDLE, ctypes.c_size_t, ctypes.c_size_t]
        k.SetProcessWorkingSetSize.restype = w.BOOL
        k.GetProcessWorkingSetSize.argtypes = [
            w.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        k.GetProcessWorkingSetSize.restype = w.BOOL
        k.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, w.DWORD, w.DWORD]
        k.VirtualAlloc.restype = ctypes.c_void_p
        k.VirtualLock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        k.VirtualLock.restype = w.BOOL
        k.VirtualUnlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        k.VirtualUnlock.restype = w.BOOL
        k.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, w.DWORD]
        k.VirtualFree.restype = w.BOOL
        self.handle = k.GetCurrentProcess()
        self.original_min, self.original_max = ctypes.c_size_t(), ctypes.c_size_t()
        if not k.GetProcessWorkingSetSize(
            self.handle, ctypes.byref(self.original_min), ctypes.byref(self.original_max)
        ):
            raise ctypes.WinError(ctypes.get_last_error())

    def __enter__(self):
        if not self.size:
            return self
        if psutil.virtual_memory().available < self.size + int(1.5 * GIB):
            raise RuntimeError("Not enough current RAM to establish this simulation safely")
        if not self.kernel.SetProcessWorkingSetSize(
            self.handle, self.size + 128 * MIB, self.size + 512 * MIB
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            remaining = self.size
            while remaining:
                size = min(256 * MIB, remaining)
                address = self.kernel.VirtualAlloc(None, size, 0x3000, 0x04)
                if not address:
                    raise ctypes.WinError(ctypes.get_last_error())
                self.regions.append((address, size))
                if not self.kernel.VirtualLock(address, size):
                    raise ctypes.WinError(ctypes.get_last_error())
                remaining -= size
            check = self.check()
            if check["sampled_locked_pages"] != check["sampled_pages"]:
                raise RuntimeError("RAM balloon is not fully locked in sampled physical pages")
            return self
        except BaseException:
            self.close()
            raise

    def check(self):
        class Page(ctypes.Structure):
            _fields_ = [("address", ctypes.c_void_p), ("attributes", ctypes.c_size_t)]

        points = [
            address + offset for address, size in self.regions
            for offset in (0, size // 2, size - 4096)
        ]
        if not points:
            return {"bytes": 0, "sampled_pages": 0, "sampled_locked_pages": 0}
        pages = (Page * len(points))()
        for index, address in enumerate(points):
            pages[index].address = address
        query = ctypes.WinDLL("psapi", use_last_error=True).QueryWorkingSetEx
        query.argtypes = [w.HANDLE, ctypes.c_void_p, w.DWORD]
        query.restype = w.BOOL
        if not query(self.handle, pages, ctypes.sizeof(pages)):
            raise ctypes.WinError(ctypes.get_last_error())
        return {
            "bytes": self.size,
            "sampled_pages": len(points),
            "sampled_valid_pages": sum(bool(page.attributes & 1) for page in pages),
            "sampled_locked_pages": sum(
                bool(page.attributes & 1 and page.attributes & (1 << 22)) for page in pages),
            "rss": psutil.Process().memory_info().rss,
        }

    def close(self):
        errors = []
        for address, size in reversed(self.regions):
            self.kernel.VirtualUnlock(address, size)
            if not self.kernel.VirtualFree(address, 0, 0x8000):
                errors.append(ctypes.get_last_error())
        self.regions.clear()
        self.kernel.SetProcessWorkingSetSize(
            self.handle, self.original_min.value, self.original_max.value)
        if errors:
            raise RuntimeError(f"VirtualFree failed for balloon regions: {errors}")

    def __exit__(self, *exception):
        self.close()


def disk_snapshot():
    return {name: value._asdict()
            for name, value in psutil.disk_io_counters(perdisk=True).items()}


def disk_difference(before, after):
    return {
        name: {key: value[key] - before[name][key]
               for key in ("read_bytes", "write_bytes", "read_count", "write_count")}
        for name, value in after.items() if name in before
    }


class GpuUsage:
    """Use the installed NVIDIA driver's NVML without an optional Python wheel."""

    class Memory(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                    ("used", ctypes.c_ulonglong)]

    def __init__(self):
        self.dll = ctypes.WinDLL("nvml.dll")
        if self.dll.nvmlInit_v2() != 0:
            raise OSError("NVML initialization failed")
        self.handle = ctypes.c_void_p()
        self.dll.nvmlDeviceGetHandleByIndex_v2.argtypes = [
            ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
        self.dll.nvmlDeviceGetMemoryInfo.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(self.Memory)]
        if self.dll.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)) != 0:
            raise OSError("NVML could not open GPU 0")

    def used(self):
        memory = self.Memory()
        if self.dll.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(memory)) != 0:
            raise OSError("NVML memory sample failed")
        return memory.used

    def close(self):
        self.dll.nvmlShutdown()


def audio_details(path):
    with wave.open(str(path), "rb") as wav:
        pcm = wav.readframes(wav.getnframes())
        return {
            "path": str(path), "sample_rate": wav.getframerate(),
            "samples": wav.getnframes(), "channels": wav.getnchannels(),
            "seconds": wav.getnframes() / wav.getframerate(),
            "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
            "wav_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }


def worker(args):
    runtime_src = args.runtime_src or ROOT / "src"
    sys.path[:0] = [str(runtime_src), str(ROOT / "vendor/fish-speech")]
    import torch
    autotune_module = importlib.import_module("drama_low_vram.autotune")
    engine_module = importlib.import_module("drama_low_vram.engine")

    original_inspection = autotune_module.inspect_hardware

    def inspection(device=None):
        actual = original_inspection(device)
        return replace(actual, ram_total_bytes=actual.ram_total_bytes - args.reserved_bytes)

    autotune_module.inspect_hardware = inspection
    engine_module.inspect_hardware = inspection
    torch.set_num_threads(1)
    report = {"status": "loading", "requests": [], "runtime_src": str(runtime_src),
              "hardware_before": inspection("cuda:0").as_dict()}
    destination = args.output
    report_file = destination / "fp8-worker.json"
    save(report_file, report)
    started = time.perf_counter()
    try:
        with engine_module.DramaEngine(engine_module.EngineOptions(
            checkpoint=ROOT / "checkpoints/drama-3",
            cache_experts=args.cache_experts,
            expert_storage="auto", backbone_graph=True, combined_decode_bank=True,
            host_gather="auto", prefix_cache_mib=256, context_bucket_size=512,
            compute_policy=args.compute_policy,
            compute_profile_dir=ROOT / ".cache" / "compute-profiles-memory-pressure" / destination.name,
        )) as engine:
            engine.status_callback = lambda event: emit(event="runtime", details=event)
            engine.load()
            report.update(
                load_seconds=time.perf_counter() - started,
                host_plan=engine.host_plan.as_dict(), runtime=engine.runtime.as_dict())
            sources = getattr(engine.model, "_drama_low_vram_weight_sources", None)
            if sources is not None:
                report["mapped_weight_sources"] = sources.as_dict()
            save(report_file, report)
            emit(event="loaded", seconds=report["load_seconds"],
                 storage=engine.storage, pin_memory=engine.runtime.pin_memory)
            for index in range(args.repeats):
                emit(event="request_begin", index=index)
                result = engine.synthesize(
                    LONG if args.text_case == "long" else SHORT,
                    destination / f"clone-{index}.wav", seed=123,
                    max_new_tokens=768, reference_audio=ROOT / "outputs/quality-official-fp8.wav",
                    reference_text=SHORT,
                    progress=lambda frame, maximum: emit(
                        event="frames", frame=frame, maximum=maximum)
                    if frame == 1 or frame % 32 == 0 else None,
                )
                row = {
                    "index": index, "status": result["status"],
                    "total_seconds": result["total_seconds"],
                    "generation_seconds": sum(c["generation_seconds"] for c in result["chunks"]),
                    "decode_seconds": sum(c["decode_seconds"] for c in result["chunks"]),
                    "frames": sum(c["frames"] for c in result["chunks"]),
                    "semantic_codes_sha256": [c.get("semantic_codes_sha256") for c in result["chunks"]],
                    "runtime_plans": [c.get("runtime_plan") for c in result["chunks"]],
                    "audio": audio_details(destination / f"clone-{index}.wav"),
                    "reference_cache_hit": result["reference_cache_hit"],
                    "reference_prefix_cache": result.get("reference_prefix_cache"),
                    "prefix_admission": getattr(engine, "_prefix_admission_report", None),
                    "compute_report": engine.compute_report,
                    "parking_mirrors": {
                        "total_bytes": sum(t.numel() * t.element_size()
                                           for t in engine._cpu_mirrors.values()),
                        "private_bytes": sum(
                            t.numel() * t.element_size()
                            for key, t in engine._cpu_mirrors.items()
                            if sources is None or not sources.is_source(key, t)),
                    },
                }
                report["requests"].append(row)
                save(report_file, report)
                emit(event="request_complete", index=index, seconds=row["total_seconds"],
                     frames=row["frames"], audio_seconds=row["audio"]["seconds"])
            report["status"] = "complete"
    except BaseException as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc())
        raise
    finally:
        report["worker_wall_seconds"] = time.perf_counter() - started
        save(report_file, report)


def stop_process(process):
    try:
        descendants = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        descendants = []
    for child in reversed(descendants):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def build_command(backend, destination, reserved, repeats, env, *,
                  compute_policy="auto", text_case="long", runtime_src=None,
                  cache_experts=None):
    if backend == "fp8":
        command = [
            str(ROOT / ".venv/Scripts/python.exe"), str(Path(__file__).resolve()),
            "--worker", "--output", str(destination),
            "--reserved-bytes", str(reserved), "--repeats", str(repeats),
            "--compute-policy", compute_policy,
            "--text-case", text_case,
        ]
        if runtime_src is not None:
            command += ["--runtime-src", str(runtime_src)]
        if cache_experts is not None:
            command += ["--cache-experts", str(cache_experts)]
        return command, ROOT, {
            "host_storage": "mmap", "compute_policy": compute_policy,
            "runtime_src": str(runtime_src) if runtime_src else None,
            "cache_experts": cache_experts,
        }
    build = json.loads((ROOT / "research/phase19-audio-cpp-build.json").read_text())
    pin = json.loads((ROOT / "research/phase19-gguf-q4-quality-download-pinned.json").read_text())
    model = Path(pin["destination"]) / pin["files"][0]["name"]
    requests = [
        {"id": f"clone-{index}", "text": LONG if text_case == "long" else SHORT,
         "seed": 123, "max_tokens": 768,
         "temperature": 1.0, "top_k": 30, "top_p": 0.9,
         "voice_ref": str(ROOT / "outputs/quality-official-fp8.wav"),
         "options": {"reference_text": SHORT, "text_chunk_size": "1000"}}
        for index in range(repeats)
    ]
    request_file = destination / "requests.json"
    save(request_file, requests)
    available = psutil.virtual_memory().available
    pinned = available >= 30 * GIB
    env["DRAMA_COMPACT_GGUF_HEAD"] = "1"
    env["PATH"] = str(Path(build["cuda_toolkit"]) / "bin") + os.pathsep + env["PATH"]
    command = [
        build["executable"], "--task", "tts", "--family", "fish_drama3",
        "--backend", "cuda", "--mode", "offline", "--model", str(model),
        "--threads", "8", "--session-option", "fish_drama3.disk_experts=true",
        "--session-option", "fish_drama3.expert_cache_size=8",
        "--session-option", "fish_drama3.cpu_expert_layers=0",
        "--session-option", "fish_drama3.context_length=1024",
        "--session-option", "fish_drama3.profile_first_audio=true",
        "--request-sequence", str(request_file), "--out-dir", str(destination),
        "--metrics", "--log",
    ]
    if pinned:
        command += ["--session-option", "fish_drama3.pin_experts=true"]
    return command, Path(build["source"]), {
        "host_storage": "pin" if pinned else "mapped",
        "available_at_storage_selection": available, "model": str(model)}


def gguf_requests(destination, log):
    rows = []
    for match in re.finditer(r"metrics\[(clone-\d+)\]\.wall_ms=([\d.]+)", log):
        name, milliseconds = match.groups()
        audio = destination / f"{name}.wav"
        artifact = destination / name / "drama3_take.json"
        token_data = json.loads(artifact.read_text(encoding="utf-8")) if artifact.is_file() else None
        rows.append({
            "id": name, "total_seconds": float(milliseconds) / 1000,
            "audio": audio_details(audio) if audio.is_file() else None,
            "token_artifact": token_data,
        })
    return rows


def supervise(args):
    if os.name != "nt":
        raise RuntimeError("This simulation requires native Windows")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "summary.json").exists():
        raise FileExistsError("Choose a fresh --output directory to preserve earlier measurements")
    initial = psutil.virtual_memory()
    report = {
        "started": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        "method": "Nonpageable VirtualLock balloon; original GPU and SSD; total-RAM-only "
                  "snapshot adjustment in FP8; real OS available RAM/commit throughout.",
        "limits": "Same CPU/GPU/SSD/pagefile/background apps. Not a separate physical "
                  "computer. No cache purge: first/repeated requests have natural OS cache history.",
        "physical_disk_scope": "Whole-system physical drive counters; includes background I/O.",
        "host_installed_gib": args.host_installed_gib,
        "text_case": args.text_case, "compute_policy": args.compute_policy,
        "windows_visible_bytes": initial.total, "initial_available_bytes": initial.available,
        "cases": [],
    }
    summary = args.output / "summary.json"
    save(summary, report)
    gpu = None
    try:
        gpu = GpuUsage()
    except Exception:
        pass
    for target in args.ram_gib:
        reserved = int((args.host_installed_gib - target) * GIB)
        if reserved < 0:
            raise ValueError("Target capacity exceeds the host's installed RAM")
        for backend in args.backends:
            destination = args.output / f"{target}g-{backend}"
            destination.mkdir(exist_ok=True)
            case = {
                "target_installed_gib": target, "backend": backend,
                "withheld_physical_bytes": reserved,
                "effective_visible_bytes": initial.total - reserved,
                "status": "preparing", "samples": [], "output": str(destination),
            }
            report["cases"].append(case)
            save(summary, report)
            process = None
            emit(event="balloon_start", target_gib=target, backend=backend,
                 reserve_gib=reserved / GIB,
                 current_available_gib=psutil.virtual_memory().available / GIB)
            try:
                with LockedBalloon(reserved) as balloon:
                    case["balloon_start"] = balloon.check()
                    case["available_after_balloon"] = psutil.virtual_memory().available
                    env = environment()
                    command, cwd, options = build_command(
                        backend, destination, reserved, args.repeats, env,
                        compute_policy=args.compute_policy, text_case=args.text_case,
                        runtime_src=args.runtime_src, cache_experts=args.cache_experts)
                    case.update(command=command, options=options, status="running")
                    save(summary, report)
                    before_disk = disk_snapshot()
                    started = time.perf_counter()
                    last_notice, last_check, critically_low = 0.0, 0.0, 0
                    log_file = destination / "runtime.log"
                    with log_file.open("wb") as log:
                        process = subprocess.Popen(
                            command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                            creationflags=subprocess.CREATE_NO_WINDOW)
                        observed = psutil.Process(process.pid)
                        while process.poll() is None:
                            elapsed = time.perf_counter() - started
                            vm = psutil.virtual_memory()
                            sample = {"seconds": elapsed, "available_ram": vm.available}
                            rss = private = faults = 0
                            try:
                                members = [observed] + observed.children(recursive=True)
                            except psutil.NoSuchProcess:
                                members = []
                            for member in members:
                                try:
                                    memory = member.memory_info()
                                    rss += memory.rss
                                    private += getattr(memory, "private", 0)
                                    faults += getattr(memory, "num_page_faults", 0)
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    pass
                            sample.update(rss=rss, private=private, page_faults=faults)
                            if gpu is not None:
                                try:
                                    sample["whole_gpu_bytes"] = gpu.used()
                                except Exception:
                                    pass
                            physical_disk = disk_snapshot()
                            sample["physical_disk_delta"] = disk_difference(before_disk, physical_disk)
                            case["samples"].append(sample)
                            critically_low = critically_low + 1 if vm.available < 128 * MIB else 0
                            if elapsed - last_check >= 10:
                                check = balloon.check()
                                case["balloon_latest"] = check
                                if check["sampled_locked_pages"] != check["sampled_pages"]:
                                    case["abort_reason"] = "Balloon no longer verified as locked"
                                    break
                                last_check = elapsed
                                save(summary, report)
                            if elapsed - last_notice >= 20:
                                tail = log_file.read_text(encoding="utf-8", errors="replace")[-900:]
                                emit(event="progress", target_gib=target, backend=backend,
                                     seconds=round(elapsed, 1), available_gib=round(vm.available/GIB, 3),
                                     process_rss_gib=round(rss/GIB, 3), log_tail=tail)
                                last_notice = elapsed
                            if critically_low >= 5:
                                case["abort_reason"] = "Available physical RAM below 128MiB for 5 samples"
                                break
                            if elapsed > args.timeout:
                                case["abort_reason"] = f"Cell timeout at {args.timeout} seconds"
                                break
                            time.sleep(1)
                        if process.poll() is None:
                            stop_process(process)
                        process.wait(timeout=10)
                    case.update(returncode=process.returncode,
                                wall_seconds=time.perf_counter() - started,
                                physical_disk_delta=disk_difference(before_disk, disk_snapshot()),
                                log=str(log_file),
                                status="complete" if process.returncode == 0 else "failed")
                    case["balloon_final"] = balloon.check()
                    log_text = log_file.read_text(encoding="utf-8", errors="replace")
                    if backend == "gguf":
                        case["requests"] = gguf_requests(destination, log_text)
                    elif (destination / "fp8-worker.json").is_file():
                        worker_report = json.loads((destination / "fp8-worker.json").read_text())
                        case["worker"] = {key: value for key, value in worker_report.items()
                                          if key != "requests"}
                        case["requests"] = worker_report.get("requests", [])
                    samples = case["samples"]
                    if samples:
                        case["resources"] = {
                            "peak_rss": max(s["rss"] for s in samples),
                            "peak_private": max(s["private"] for s in samples),
                            "min_available_ram": min(s["available_ram"] for s in samples),
                            "sampled_whole_gpu_peak": max(
                                (s["whole_gpu_bytes"] for s in samples if "whole_gpu_bytes" in s),
                                default=None),
                        }
            except Exception as error:
                case.update(status="error", error=f"{type(error).__name__}: {error}",
                            traceback=traceback.format_exc())
            finally:
                if process is not None and process.poll() is None:
                    stop_process(process)
                case["available_after_release"] = psutil.virtual_memory().available
                save(destination / "result.json", case)
                save(summary, report)
                emit(event="cell_complete", target_gib=target, backend=backend,
                     status=case["status"], seconds=case.get("wall_seconds"),
                     requests=[{k: row.get(k) for k in ("index", "id", "status", "total_seconds")}
                               for row in case.get("requests", [])],
                     resources=case.get("resources"), error=case.get("error"))
    report["finished"] = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    save(summary, report)
    if gpu is not None:
        gpu.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ram-gib", type=int, nargs="+", choices=(16, 24, 32, 48),
                        default=[32, 24, 16])
    parser.add_argument("--backends", choices=("fp8", "gguf"), nargs="+", default=["gguf", "fp8"])
    parser.add_argument("--host-installed-gib", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--compute-policy", choices=("auto", "manual"), default="auto")
    parser.add_argument("--text-case", choices=("long", "short"), default="long")
    parser.add_argument("--runtime-src", type=Path)
    parser.add_argument("--cache-experts", type=int)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output", type=Path, default=ROOT / "research/memory-pressure-20261016")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--reserved-bytes", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 3 or not 30 <= args.timeout <= 1200:
        parser.error("Use 1..3 repetitions and a 30..1200 second cell timeout")
    if args.worker:
        worker(args)
    else:
        supervise(args)


if __name__ == "__main__":
    main()
