"""Measure RAM versus mmap engine requests and sampled Windows expert residency."""

from __future__ import annotations

import argparse
import ctypes
import datetime
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor/fish-speech")]
for key, directory in (
    ("TRITON_CACHE_DIR", "triton"), ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
    ("TORCH_EXTENSIONS_DIR", "torch_extensions"), ("HF_HOME", "huggingface"),
    ("TEMP", "tmp"), ("TMP", "tmp"),
):
    directory = ROOT / ".cache" / directory
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(directory)

from drama_low_vram.autotune import inspect_hardware
from drama_low_vram.engine import DramaEngine, EngineOptions, _process_memory
from drama_low_vram.offload import EXPERT_BUFFER_NAMES


def available_memory():
    from ctypes import wintypes

    class Memory(ctypes.Structure):
        _fields_ = [
            ("length", wintypes.DWORD), ("load", wintypes.DWORD),
            ("total", ctypes.c_ulonglong), ("available", ctypes.c_ulonglong),
            ("commit_limit", ctypes.c_ulonglong), ("commit_available", ctypes.c_ulonglong),
            ("virtual_total", ctypes.c_ulonglong), ("virtual_available", ctypes.c_ulonglong),
            ("extended", ctypes.c_ulonglong),
        ]

    memory = Memory()
    memory.length = ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise ctypes.WinError()
    return {"ram_available_bytes": int(memory.available),
            "commit_available_bytes": int(memory.commit_available)}


def expert_residency(manager, stride_bytes=64 * 1024):
    """Query sampled page state; this does not read weights or cause page faults."""
    from ctypes import wintypes

    class Page(ctypes.Structure):
        _fields_ = [("address", ctypes.c_void_p), ("attributes", ctypes.c_size_t)]

    get_process = ctypes.windll.kernel32.GetCurrentProcess
    get_process.restype = wintypes.HANDLE
    query = ctypes.windll.psapi.QueryWorkingSetEx
    query.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    query.restype = wintypes.BOOL
    pages = (Page * 4096)()
    process = get_process()
    count = valid = sampled_bytes = resident_bytes = 0
    buffered = []

    def flush():
        nonlocal count, valid, sampled_bytes, resident_bytes
        if not buffered:
            return
        if not query(process, pages, len(buffered) * ctypes.sizeof(Page)):
            raise ctypes.WinError()
        for index, represented in enumerate(buffered):
            is_valid = bool(pages[index].attributes & 1)
            count += 1
            valid += int(is_valid)
            sampled_bytes += represented
            resident_bytes += represented * is_valid
        buffered.clear()

    total_bytes = 0
    for module in manager.modules.values():
        for name in EXPERT_BUFFER_NAMES:
            tensor = getattr(module, name)
            size = tensor.numel() * tensor.element_size()
            total_bytes += size
            for offset in range(0, size, stride_bytes):
                index = len(buffered)
                pages[index].address = tensor.data_ptr() + offset
                pages[index].attributes = 0
                buffered.append(min(stride_bytes, size - offset))
                if len(buffered) == len(pages):
                    flush()
    flush()
    return {
        "scope": "QueryWorkingSetEx valid-bit samples, weighted by represented range; not locked memory or physical disk I/O.",
        "stride_bytes": stride_bytes, "expert_bytes": total_bytes,
        "sampled_pages": count, "resident_sampled_pages": valid,
        "estimated_resident_bytes": resident_bytes,
        "estimated_resident_fraction": resident_bytes / sampled_bytes,
    }


def snapshot(engine):
    return {
        **available_memory(), "process": _process_memory(),
        "experts": expert_residency(engine.manager),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", choices=("ram", "mmap"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--cache-experts", type=int, default=2)
    parser.add_argument("--warm-clone", action="store_true")
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("This observation tool currently requires Windows")
    short = "<|speaker:0|>你好，这是官方 FP8 质量优先测试。"
    long = "<|speaker:0|>今天我们继续测试这套本地语音引擎。希望普通电脑也能保持清楚自然的声音，读完一整句话，而不需要购买昂贵的专业显卡。接下来还会仔细检查语音克隆、长句停顿和声音的稳定性。"
    rows = [
        ("short-1", short, 256, False),
        ("short-2", short, 256, False),
        ("clone", long, 768, True),
    ]
    if args.warm_clone:
        rows.append(("clone-warm", long, 768, True))
    report = {
        "date": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        "storage": args.storage,
        "hardware_before": inspect_hardware("cuda:0").as_dict(),
        "configuration": {
            "staged_experts": 8, "pair_rows": 64, "prefill_chunk": 32,
            "cache_experts": args.cache_experts, "backbone_graph": True,
            "combined_decode_bank": True, "packed_staging": False,
            "codec_policy": "auto",
        },
        "requests": [], "snapshots": {}, "samples": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    stop = threading.Event()
    phase = "load"
    started = time.perf_counter()

    def sample():
        while not stop.is_set():
            report["samples"].append({
                "elapsed_seconds": time.perf_counter() - started,
                "phase": phase, **available_memory(), "process": _process_memory(),
            })
            stop.wait(1)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    try:
        with DramaEngine(EngineOptions(
            checkpoint=ROOT / "checkpoints/drama-3", expert_storage=args.storage,
            staged_experts=8, pair_rows=64, prefill_chunk=32,
            cache_experts=args.cache_experts, backbone_graph=True,
            combined_decode_bank=True,
        )) as engine:
            load_started = time.perf_counter()
            engine.load()
            report["explicit_load_seconds"] = time.perf_counter() - load_started
            report["host_plan"] = engine.host_plan.as_dict()
            report["snapshots"]["loaded"] = snapshot(engine)
            print(json.dumps({"phase": "loaded", "seconds": report["explicit_load_seconds"],
                              "memory": report["snapshots"]["loaded"]}), flush=True)
            save()
            for name, text, maximum, cloned in rows:
                phase = name
                result = engine.synthesize(
                    text, ROOT / "outputs" / f"{args.prefix}-{name}.wav",
                    seed=123, max_new_tokens=maximum,
                    reference_audio=ROOT / "outputs/quality-official-fp8.wav" if cloned else None,
                    reference_text=short if cloned else None,
                    progress=lambda frame, limit: print(f"{name}: {frame}/{limit}", flush=True)
                    if frame == 1 or frame % 64 == 0 else None,
                )
                output = Path(result["output"]) if result["output"] else None
                report["requests"].append({
                    "name": name, "report": str((ROOT / "outputs" / f"{args.prefix}-{name}.json").resolve()),
                    "status": result["status"], "audio": result["audio"],
                    "generation_seconds": sum(chunk["generation_seconds"] for chunk in result["chunks"]),
                    "runtime_prepare_seconds": sum(chunk["runtime_prepare_seconds"] for chunk in result["chunks"]),
                    "total_seconds": result["total_seconds"],
                    "reference_cache_hit": result["reference_cache_hit"],
                    "semantic_codes_sha256": [chunk.get("semantic_codes_sha256") for chunk in result["chunks"]],
                    "wav_sha256": hashlib.sha256(output.read_bytes()).hexdigest() if output else None,
                    "lifecycle_delta": result["lifecycle_delta"],
                })
                report["snapshots"][name] = snapshot(engine)
                print(json.dumps({"phase": name, **report["requests"][-1],
                                  "memory": report["snapshots"][name]}), flush=True)
                save()
            report["status"] = "complete"
    except BaseException as error:
        report["status"] = "error"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        stop.set()
        monitor.join()
        report["total_wall_seconds"] = time.perf_counter() - started
        save()
    return 0 if all(row["status"] == "complete" for row in report["requests"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
