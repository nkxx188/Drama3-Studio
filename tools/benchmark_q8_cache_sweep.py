"""Bounded, same-backend Q8 cache comparison with Windows residency counters.

All candidates use one warmup and the same measured requests. Native expert
bytes_read counts mapped-source copies, not physical SSD traffic; the separate
drive counters include background activity. WDDM shared memory includes normal
CUDA host allocations and is evidence of pressure only when it grows above the
candidate's baseline. Process sampling is diagnostic, not an exact peak.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes as w
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

import psutil

from benchmark_memory_pressure import (
    GpuUsage, audio_details, disk_difference, disk_snapshot, environment,
)
from benchmark_model_comparison import TARGET, SHORT

ROOT = Path(__file__).resolve().parents[1]
MIB = 1024**2


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def emit(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


class WddmMemory:
    """Query the installed WDDM per-process dedicated/shared usage via PDH."""

    class Number(ctypes.Union):
        _fields_ = [("number", ctypes.c_double), ("integer", ctypes.c_longlong)]

    class Value(ctypes.Structure):
        pass

    class Item(ctypes.Structure):
        pass

    Value._fields_ = [("status", w.DWORD), ("value", Number)]
    Item._fields_ = [("name", w.LPWSTR), ("value", Value)]

    def __init__(self):
        self.pdh = p = ctypes.WinDLL("pdh.dll")
        self.query = ctypes.c_void_p()
        p.PdhOpenQueryW.argtypes = [w.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
        p.PdhAddEnglishCounterW.argtypes = [
            ctypes.c_void_p, w.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
        p.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
        p.PdhGetFormattedCounterArrayW.argtypes = [
            ctypes.c_void_p, w.DWORD, ctypes.POINTER(w.DWORD),
            ctypes.POINTER(w.DWORD), ctypes.c_void_p,
        ]
        p.PdhCloseQuery.argtypes = [ctypes.c_void_p]
        self._check(p.PdhOpenQueryW(None, 0, ctypes.byref(self.query)))
        self.counters = {}
        try:
            for label, name in (("dedicated_bytes", "Dedicated Usage"),
                                ("shared_bytes", "Shared Usage")):
                handle = ctypes.c_void_p()
                self._check(p.PdhAddEnglishCounterW(
                    self.query, rf"\GPU Process Memory(*)\{name}", 0, ctypes.byref(handle)))
                self.counters[label] = handle
            self._check(p.PdhCollectQueryData(self.query))
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _check(status):
        if status:
            raise OSError(f"PDH status {status & 0xffffffff:#x}")

    def sample(self, pid):
        self._check(self.pdh.PdhCollectQueryData(self.query))
        result = {}
        for label, counter in self.counters.items():
            size, count = w.DWORD(), w.DWORD()
            status = self.pdh.PdhGetFormattedCounterArrayW(
                counter, 0x200, ctypes.byref(size), ctypes.byref(count), None)
            if status & 0xffffffff != 0x800007d2:  # PDH_MORE_DATA
                self._check(status)
            if not size.value:
                result[label] = None
                continue
            buffer = ctypes.create_string_buffer(size.value)
            self._check(self.pdh.PdhGetFormattedCounterArrayW(
                counter, 0x200, ctypes.byref(size), ctypes.byref(count), buffer))
            items = ctypes.cast(buffer, ctypes.POINTER(self.Item))
            values = [
                int(items[i].value.value.number) for i in range(count.value)
                if items[i].name.startswith(f"pid_{pid}_") and items[i].value.status in (0, 1)
            ]
            result[label] = sum(values) if values else None
        return result

    def close(self):
        if self.query:
            self.pdh.PdhCloseQuery(self.query)
            self.query = None


def parse_requests(log, destination):
    boundaries = list(re.finditer(r"request_index=\d+\s+request_id=([^\r\n]+)", log))
    rows = []
    previous = 0
    for index, boundary in enumerate(boundaries):
        case = boundary.group(1)
        before = log[previous:boundary.start()]
        after = log[boundary.end():boundaries[index + 1].start()
                    if index + 1 < len(boundaries) else len(log)]
        previous = boundary.end()
        elapsed = re.search(rf"metrics\[{re.escape(case)}\]\.wall_ms=([\d.]+)", after)
        schema = re.search(r"custom_schema_output=(\{[^\r\n]+\})", after)
        if not elapsed or not schema:
            raise ValueError(f"Missing completed native metrics for {case}")
        data = json.loads(schema.group(1))["data"]
        components = {
            key: float(value) / 1000
            for key, value in re.findall(r"drama3\.([a-z_]+)_ms ([\d.]+)", before)
        }
        path = destination / f"{case}.wav"
        row = {
            "case": case, "warmup": case.startswith("warmup"),
            "total_seconds": float(elapsed.group(1)) / 1000,
            "frames": data["frames"], "finish_reason": "length" if data["reached_limit"] else "eos",
            "components_seconds": components, "audio": audio_details(path),
        }
        # At a cap the last sampled frame does not enter the backbone; at EOS
        # it already has, before the following semantic sample selects EOS.
        row["backbone_decode_steps"] = data["frames"] - int(data["reached_limit"])
        row["decode_ms_per_frame"] = (
            components["backbone_step"] * 1000 / max(1, row["backbone_decode_steps"]))
        row["rtf"] = row["total_seconds"] / row["audio"]["seconds"]
        artifact = destination / case / "drama3_take.json"
        payload = json.loads(bytes.fromhex(json.loads(artifact.read_text())["payload_hex"]))
        codes = payload["vq_codes"]
        row["codes_sha256"] = hashlib.sha256(
            json.dumps(codes, separators=(",", ":")).encode()).hexdigest()
        row["semantic_codes_sha256"] = hashlib.sha256(
            json.dumps(codes[0], separators=(",", ":")).encode()).hexdigest()
        row["artifact"] = str(artifact)
        rows.append(row)
    return rows


def run_candidate(slots, destination, args, build, model, gpu, wddm):
    destination.mkdir()
    requests = []
    for case, maximum in [("warmup", 8), *[(f"clone-{i + 1}", args.frames)
                                           for i in range(args.repeats)]]:
        requests.append({
            "id": case, "text": TARGET, "seed": 123, "max_tokens": maximum,
            "temperature": 0.7, "top_k": 30, "top_p": 0.7,
            "voice_ref": str(ROOT / "outputs/quality-official-fp8.wav"),
            "options": {"text_chunk_size": "1000", "reference_text": SHORT},
        })
    save(destination / "requests.json", requests)
    command = [
        build["executable"], "--task", "tts", "--family", "fish_drama3",
        "--backend", "cuda", "--mode", "offline", "--model", str(model),
        "--threads", str(args.threads), "--session-option", "fish_drama3.disk_experts=true",
        "--session-option", f"fish_drama3.expert_cache_size={slots}",
        "--session-option", "fish_drama3.cpu_expert_layers=0",
        "--session-option", f"fish_drama3.codec_backend={args.codec}",
        "--session-option", f"fish_drama3.context_length={args.context}",
        "--session-option", "fish_drama3.profile_first_audio=true",
        "--request-sequence", str(destination / "requests.json"),
        "--out-dir", str(destination), "--metrics", "--log",
    ]
    if args.codec_decode != "full" or args.decoder != "same":
        command += [
            "--session-option", f"fish_drama3.codec_decode_strategy={args.codec_decode}",
            "--session-option", f"fish_drama3.codec_decoder_backend={args.decoder}",
        ]
    env = environment()
    env["DRAMA_COMPACT_GGUF_HEAD"] = "1"
    env.pop("DRAMA_HOST_GGUF_EMBEDDING", None)
    if args.host_embedding:
        env["DRAMA_HOST_GGUF_EMBEDDING"] = "1"
    env["PATH"] = str(Path(build["cuda_toolkit"]) / "bin") + os.pathsep + env["PATH"]
    report = {"status": "running", "cache_slots": slots, "command": command}
    save(destination / "summary.json", report)
    samples = []
    disk_before = disk_snapshot()
    started = time.perf_counter()
    log_path = destination / "native.log"
    emit("candidate_begin", slots=slots, directory=str(destination))
    with log_path.open("wb") as output:
        child = subprocess.Popen(
            command, cwd=build["source"], env=env, stdout=output, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        observed = psutil.Process(child.pid)
        last = started
        try:
            while child.poll() is None:
                now = time.perf_counter()
                try:
                    memory = observed.memory_info()
                    sample = {
                        "seconds": now - started, "rss_bytes": memory.rss,
                        "private_bytes": getattr(memory, "private", memory.rss),
                        "available_ram_bytes": psutil.virtual_memory().available,
                        "whole_gpu_bytes": gpu.used(),
                    }
                    if wddm:
                        sample.update(wddm.sample(child.pid))
                    samples.append(sample)
                except (psutil.NoSuchProcess, OSError) as error:
                    report.setdefault("sample_errors", []).append(str(error))
                if now - last >= 20:
                    emit("progress", slots=slots, seconds=round(now - started, 1),
                         completed_wavs=len(list(destination.glob("*.wav"))))
                    save(destination / "samples.json", samples)
                    last = now
                if now - started > args.timeout:
                    report["timeout"] = True
                    child.terminate()
                    break
                time.sleep(0.5)
            child.wait(timeout=20)
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=20)
    report.update(returncode=child.returncode, process_seconds=time.perf_counter() - started)
    log = log_path.read_text(encoding="utf-8", errors="replace")
    report["drive_io_delta_including_background"] = disk_difference(disk_before, disk_snapshot())
    report["native_lifetime_stats"] = {
        key: float(value) for key, value in re.findall(r"disk_experts\.([a-z_]+) ([\d.]+)", log)
    }
    save(destination / "samples.json", samples)
    report["resources"] = {
        "peak_rss_bytes": max((s["rss_bytes"] for s in samples), default=0),
        "peak_private_bytes": max((s["private_bytes"] for s in samples), default=0),
        "minimum_available_ram_bytes": min((s["available_ram_bytes"] for s in samples), default=0),
        "sampled_whole_gpu_peak_bytes": max((s["whole_gpu_bytes"] for s in samples), default=0),
        "sampled_process_dedicated_peak_bytes": max(
            (s["dedicated_bytes"] for s in samples if s.get("dedicated_bytes") is not None), default=None),
        "sampled_process_shared_peak_bytes": max(
            (s["shared_bytes"] for s in samples if s.get("shared_bytes") is not None), default=None),
    }
    if child.returncode:
        report["status"] = "error"
    else:
        report["requests"] = parse_requests(log, destination)
        report["status"] = "complete" if len(report["requests"]) == len(requests) else "error"
        formal = [row for row in report["requests"] if not row["warmup"]]
        report["formal_median_seconds"] = statistics.median(row["total_seconds"] for row in formal)
        report["formal_median_decode_ms_per_frame"] = statistics.median(
            row["decode_ms_per_frame"] for row in formal)
    save(destination / "summary.json", report)
    emit("candidate_complete", slots=slots, status=report["status"],
         median_seconds=report.get("formal_median_seconds"),
         requests=[{"case": row["case"], "seconds": row["total_seconds"],
                    "frames": row["frames"], "finish": row["finish_reason"]}
                   for row in report.get("requests", [])],
         resources=report["resources"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--caches", type=int, nargs="+", default=[8, 10, 12, 14, 16])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--frames", type=int, default=215)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=360)
    parser.add_argument("--host-embedding", action="store_true")
    parser.add_argument("--codec", choices=("cpu", "same"), default="cpu")
    parser.add_argument("--codec-decode", choices=("full", "stream"), default="full")
    parser.add_argument("--decoder", choices=("same", "cpu", "cuda"), default="same")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--build-receipt", type=Path,
                        default=ROOT / "research/phase19-audio-cpp-build.json")
    args = parser.parse_args()
    if any(not 8 <= slots <= 256 for slots in args.caches) or args.repeats < 1:
        parser.error("Caches must be 8..256 and repeats must be positive")
    if not 8 <= args.frames <= 215 or args.context < 512 or args.timeout < 30:
        parser.error("Use 8..215 frames, context >= 512, timeout >= 30")
    if not 1 <= args.threads <= 32:
        parser.error("threads must be 1..32")
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "summary.json").exists():
        raise FileExistsError("Choose a fresh sweep directory")
    build = json.loads(args.build_receipt.read_text())
    executable_sha = hashlib.sha256(Path(build["executable"]).read_bytes()).hexdigest()
    model = ROOT / "checkpoints/drama-3-gguf-q8/drama3-experts-Q8_0-bf16.gguf"
    conversion = json.loads(model.with_suffix(".conversion.json").read_text())
    if model.stat().st_size != conversion["output_bytes"]:
        raise ValueError("Q8 model size changed")
    report = {
        "status": "running", "scope": (
            "One unchanged native Q8 binary/model/seed/text/reference/codec/threads/context. "
            "Each process starts with an eight-frame clone warmup, then repeated capped clones. "
            "Independent requests invalidate native expert placement. OS page-cache history "
            "and background GPU use are observed, not reset. EOS length may vary; compare "
            "decode per frame as well as wall time. Lifetime stats include warmup."
        ),
        "model_sha256_from_conversion": conversion["output_sha256"],
        "executable_sha256": executable_sha, "build_receipt_sha256": build["sha256"],
        "parameters": vars(args) | {
            "output": str(destination), "build_receipt": str(args.build_receipt.resolve()),
        }, "candidates": [],
    }
    save(destination / "summary.json", report)
    gpu, wddm = GpuUsage(), None
    try:
        try:
            wddm = WddmMemory()
        except OSError as error:
            report["wddm_unavailable"] = str(error)
        for index, slots in enumerate(args.caches):
            folder = destination / f"{index + 1:02d}-cache{slots}"
            candidate = run_candidate(slots, folder, args, build, model, gpu, wddm)
            report["candidates"].append(candidate)
            save(destination / "summary.json", report)
        report["status"] = "complete" if all(c["status"] == "complete"
                                             for c in report["candidates"]) else "error"
        save(destination / "summary.json", report)
        emit("sweep_complete", status=report["status"], output=str(destination))
        if report["status"] != "complete":
            raise RuntimeError(f"At least one native candidate failed; see {destination}")
    finally:
        gpu.close()
        if wddm:
            wddm.close()


if __name__ == "__main__":
    main()
