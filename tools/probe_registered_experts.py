"""Bounded read-only mmap registration feasibility and complete transfer costs."""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from benchmark_engine_decode import torch
from drama_low_vram.expert_cache_file import ExpertCacheFile
from drama_low_vram.offload import EXPERT_BUFFER_NAMES, ExpertStore, OffloadStatistics, _StagingPool
from drama_low_vram.native_backend import NativeLibrary


def merged_regions(store, ids):
    regions = []
    for name in EXPERT_BUFFER_NAMES:
        source = getattr(store, name)
        row_bytes = source.numel() // store.num_experts * source.element_size()
        for index in ids.tolist():
            start = source.data_ptr() + index * row_bytes
            end = start + row_bytes
            regions.append((start // 4096 * 4096, (end + 4095) // 4096 * 4096))
    merged = []
    for start, end in sorted(regions):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    runtime = ctypes.CDLL(str(Path(torch.__file__).parent / "lib/cudart64_12.dll"))
    runtime.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    runtime.cudaHostRegister.restype = ctypes.c_int
    runtime.cudaHostUnregister.argtypes = [ctypes.c_void_p]
    runtime.cudaHostUnregister.restype = ctypes.c_int
    runtime.cudaGetErrorString.argtypes = [ctypes.c_int]
    runtime.cudaGetErrorString.restype = ctypes.c_char_p
    cache = ExpertCacheFile.open(ROOT / ".cache/drama3-fp8-experts")
    store = ExpertStore(**cache.layer_buffers(0))
    stats = OffloadStatistics()
    pool = _StagingPool(8, True, False, stats)
    pool.native_gather = NativeLibrary().gather_experts
    target = torch.device("cuda:0")
    torch.cuda.init()
    report = {
        "scope": "Research only, first-layer warm file pages. At most eight experts and their rounded source pages registered at a time. Read-only original bank; no full-model pinning or production change.",
        "device": torch.cuda.get_device_name(), "rows": [],
        "registered_source_supported": False,
    }
    rng = torch.Generator().manual_seed(925)
    try:
        for count in (2, 6, 8):
            routes = [torch.randperm(store.num_experts, generator=rng)[:count] for _ in range(8)]
            with pool.session(store, target):
                for ids in routes:
                    pool.stage(store, ids)
                    # session() owns the pool's non-reentrant lock.
                    torch.cuda.synchronize(target)
            pool.synchronize()
            samples = {"staging": [], "registration_copy_unregistration": []}
            register_samples, copy_samples, unregister_samples = [], [], []
            maximum_bytes = maximum_regions = 0
            for repetition in range(3):
                for ids in routes:
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    with pool.session(store, target):
                        pool.stage(store, ids)
                    pool.synchronize()
                    samples["staging"].append(time.perf_counter() - started)
                    regions = merged_regions(store, ids)
                    maximum_bytes = max(maximum_bytes, sum(end - start for start, end in regions))
                    maximum_regions = max(maximum_regions, len(regions))
                    registered = []
                    total_started = time.perf_counter()
                    try:
                        registration_started = time.perf_counter()
                        for start, end in regions:
                            # CUDA's read-only registration flag is required for
                            # this ACCESS_READ mapping. No GPU writes are made.
                            status = runtime.cudaHostRegister(ctypes.c_void_p(start), end - start, 8)
                            if status:
                                text = runtime.cudaGetErrorString(status).decode("utf-8")
                                report["unsupported_reason"] = {"status": status, "message": text}
                                raise NotImplementedError(text)
                            registered.append(start)
                        register_samples.append(time.perf_counter() - registration_started)
                        report["registered_source_supported"] = True
                        copy_started = time.perf_counter()
                        for slot, expert in enumerate(ids.tolist()):
                            for name in EXPERT_BUFFER_NAMES:
                                source = getattr(store, name)[expert]
                                destination = pool._device[name][slot]
                                if source.dtype == torch.float8_e4m3fn:
                                    source = source.view(torch.uint8)
                                    destination = destination.view(torch.uint8)
                                destination.copy_(source, non_blocking=True)
                        torch.cuda.synchronize()
                        copy_samples.append(time.perf_counter() - copy_started)
                    finally:
                        # Unregister only after every GPU read completed.
                        torch.cuda.synchronize()
                        unregister_started = time.perf_counter()
                        for start in reversed(registered):
                            status = runtime.cudaHostUnregister(ctypes.c_void_p(start))
                            if status:
                                raise RuntimeError(runtime.cudaGetErrorString(status).decode("utf-8"))
                        unregister_samples.append(time.perf_counter() - unregister_started)
                    samples["registration_copy_unregistration"].append(time.perf_counter() - total_started)
                    # Byte comparisons include GPU readback and must remain
                    # outside the complete registration/transfer timing.
                    if repetition == 0:
                        for name in EXPERT_BUFFER_NAMES:
                            expected = getattr(store, name).view(torch.uint8).index_select(0, ids)
                            actual = pool._device[name][:count].view(torch.uint8).cpu()
                            if not torch.equal(actual, expected):
                                raise RuntimeError("Registered source transfer changed bytes")
            row = {
                "experts": count, "source_bytes_equal": True,
                "maximum_registered_bytes": maximum_bytes, "maximum_regions": maximum_regions,
                "median_ms": {key: statistics.median(values) * 1000 for key, values in samples.items()},
                "registration_median_ms": statistics.median(register_samples) * 1000,
                "copy_median_ms": statistics.median(copy_samples) * 1000,
                "unregistration_median_ms": statistics.median(unregister_samples) * 1000,
                "samples_seconds": samples,
            }
            report["rows"].append(row)
            print(json.dumps(row), flush=True)
    except NotImplementedError:
        print(json.dumps(report), flush=True)
    finally:
        pool.close()
        cache.close()
        (ROOT / "research/registered-experts-probe.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
