"""Consecutive full clones with explicit prefix-key and codec lifecycle records."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from benchmark_memory_pressure import LONG, SHORT, audio_details, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", choices=("tuned", "auto"), default="auto")
    parser.add_argument("--runtime-src", type=Path, default=ROOT / "src")
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 2 <= args.requests <= 8:
        parser.error("Use 2..8 consecutive requests")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "summary.json"
    if report_path.exists():
        raise FileExistsError("Choose a new output directory")
    import os

    os.environ.update(environment())
    sys.path[:0] = [str(args.runtime_src.resolve()), str(ROOT / "vendor/fish-speech")]
    import torch
    from drama_low_vram import engine as engine_module
    from drama_low_vram.prefix_cache import prefix_execution_key

    torch.set_num_threads(1)
    options = dict(
        checkpoint=ROOT / "checkpoints/drama-3",
        expert_storage="mmap", backbone_graph=True, combined_decode_bank=True,
        prefix_cache_mib=256, host_gather="auto", context_bucket_size=512,
        compute_policy="auto",
        compute_profile_dir=ROOT / ".cache/compute-profiles-stability",
    )
    if args.configuration == "tuned":
        options.update(
            cache_experts=1, staged_experts=8, pair_rows=64, prefill_chunk=32,
            cpu_experts=2, native_threads=4, context_bucket_size=0,
            compute_policy="manual",
        )
    report = {
        "status": "running", "configuration": args.configuration,
        "runtime_src": str(args.runtime_src.resolve()),
        "options": {key: str(value) if isinstance(value, Path) else value
                    for key, value in options.items()},
        "requests": [],
        "scope": "Consecutive complete clones; first request retained, no hidden warmups.",
    }

    def save():
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def emit(event, **values):
        print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)

    prefix_records = []
    original_generate = engine_module.generate

    def traced_generate(model, prompt, **kwargs):
        cache = kwargs.get("prefix_cache")
        execution = prefix_execution_key(model)
        key, boundary = cache.key(
            model, prompt, kwargs["prefill_chunk_size"], execution) if cache else (None, 0)
        prefix_records.append({
            "execution_key": list(execution), "boundary": boundary,
            "prompt_digest": key[-1] if key else None,
            "stored_keys": [
                {"chunk": stored[1], "boundary": stored[2],
                 "execution_key": list(stored[3]), "prompt_digest": stored[4]}
                for stored in cache.entries
            ] if cache else [],
        })
        return original_generate(model, prompt, **kwargs)

    engine_module.generate = traced_generate
    save()
    try:
        with engine_module.DramaEngine(engine_module.EngineOptions(**options)) as engine:
            engine.status_callback = lambda event: emit("runtime", details=event)
            started = time.perf_counter()
            engine.load()
            report["load_seconds"] = time.perf_counter() - started
            report["runtime"] = engine.runtime.as_dict()
            codec_records = []
            original_codec = engine._codec_on_gpu

            @contextmanager
            def traced_codec(frames, **kwargs):
                stage_started = time.perf_counter()
                item = {
                    "frames": frames, "operation": kwargs.get("operation", "all"),
                    "free_before": int(torch.cuda.mem_get_info(engine.device)[0]),
                    "allocated_before": int(torch.cuda.memory_allocated(engine.device)),
                    "reserved_before": int(torch.cuda.memory_reserved(engine.device)),
                }
                try:
                    with original_codec(frames, **kwargs) as codec:
                        item["admission_seconds"] = time.perf_counter() - stage_started
                        item["parked"] = engine._parked
                        allocated = int(torch.cuda.memory_allocated(engine.device))
                        item["allocated_after_staging"] = allocated
                        torch.cuda.reset_peak_memory_stats(engine.device)
                        try:
                            yield codec
                        finally:
                            item["peak_operation_extra_bytes"] = (
                                int(torch.cuda.max_memory_allocated(engine.device)) - allocated)
                finally:
                    item["total_seconds"] = time.perf_counter() - stage_started
                    codec_records.append(item)

            engine._codec_on_gpu = traced_codec
            for index in range(args.requests):
                prefix_records.clear()
                codec_records.clear()
                emit("request_begin", index=index)
                target = args.output / f"clone-{index}.wav"
                result = engine.synthesize(
                    LONG, target, seed=123, max_new_tokens=768,
                    reference_audio=ROOT / "outputs/quality-official-fp8.wav",
                    reference_text=SHORT,
                    progress=lambda frame, maximum: emit(
                        "frames", frame=frame, maximum=maximum)
                    if frame == 1 or frame % 64 == 0 else None,
                )
                assert result["status"] == "complete"
                assert all(chunk["complete"] for chunk in result["chunks"])
                item = {
                    "index": index, "total_seconds": result["total_seconds"],
                    "audio": audio_details(target), "chunks": result["chunks"],
                    "codec_seconds": result["codec_seconds"],
                    "reference_seconds": result["reference_seconds"],
                    "lifecycle": result["lifecycle_delta"],
                    "memory": result["memory"], "compute": engine.compute_report,
                    "prefix_keys": list(prefix_records), "codec_stages": list(codec_records),
                }
                report["requests"].append(item)
                save()
                emit("request_complete", index=index, seconds=item["total_seconds"],
                     audio_seconds=item["audio"]["seconds"],
                     prefix_hit=all(c["prefix_cache"]["hit"] for c in item["chunks"]))
            report["status"] = "complete"
    except BaseException as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        engine_module.generate = original_generate
        save()


if __name__ == "__main__":
    main()
