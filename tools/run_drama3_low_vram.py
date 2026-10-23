"""Generate Drama 3 audio with bounded FP8 expert offload on Windows CUDA."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import random
import sys
import time
import weakref
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "fish-speech"
SRC = ROOT / "src"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drama_low_vram.codec import (  # noqa: E402
    decode_to_audio,
    encode_audio,
    load_codec,
    release_codec,
    write_wav,
)
from drama_low_vram.autotune import autotune  # noqa: E402
from drama_low_vram.model import load_offloaded_drama3, release_drama3  # noqa: E402
from fish_speech.models.text2semantic.fish_omni.generate import (  # noqa: E402
    build_drama3_prompt,
    generate_drama3,
)
from fish_speech.models.text2semantic.inference import resolve_codec_path  # noqa: E402
from fish_speech.models.text2semantic.sampling import (  # noqa: E402
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)
from drama_low_vram.sampling import DEFAULT_TOP_K  # noqa: E402


def load_reference_codes(
    checkpoint: Path,
    audio: Path,
    device: str,
    dtype: torch.dtype,
    max_frames: int,
) -> torch.Tensor:
    codec_path = resolve_codec_path(checkpoint)
    print("Loading codec for reference encoding...", flush=True)
    codec = load_codec(codec_path, device=device, dtype=dtype, max_frames=max_frames)
    try:
        return encode_audio(audio, codec, device).cpu()
    finally:
        release_codec(codec)


def write_audio(
    checkpoint: Path,
    codes: torch.Tensor,
    output: Path,
    device: str,
    dtype: torch.dtype,
) -> dict[str, int | float]:
    codec_path = resolve_codec_path(checkpoint)
    print("Loading codec for waveform decoding...", flush=True)
    codec = load_codec(
        codec_path,
        device=device,
        dtype=dtype,
        max_frames=max(1, int(codes.shape[-1])),
    )
    try:
        audio = decode_to_audio(codes, codec, device=device)
        if not bool(torch.isfinite(audio).all().item()):
            raise RuntimeError("Codec returned a waveform containing NaN or infinity")
        write_wav(audio, output, sample_rate=codec.sample_rate)
        sample_count = int(audio.shape[-1])
        return {
            "sample_rate": int(codec.sample_rate),
            "samples": sample_count,
            "duration_seconds": sample_count / int(codec.sample_rate),
        }
    finally:
        release_codec(codec)


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "drama-3")
    parser.add_argument("--text", required=True)
    parser.add_argument("--prompt-audio", type=Path)
    parser.add_argument("--prompt-text", action="append", default=[])
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "drama3.wav")
    parser.add_argument(
        "--report",
        type=Path,
        help="JSON run report; defaults to the output path with a .json extension.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Disable hardware auto-tuning and use conservative manual defaults.",
    )
    parser.add_argument(
        "--lm-head-device",
        choices=("auto", "cpu", "cuda"),
        default=None,
        help="Keep lm_head on CPU (8GB-safe) or place it on the compute GPU.",
    )
    parser.add_argument(
        "--max-new-tokens", type=_positive_int, default=DEFAULT_MAX_NEW_TOKENS
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--top-k", type=_nonnegative_int, default=DEFAULT_TOP_K)
    parser.add_argument("--seed", type=_nonnegative_int, default=123)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write audio that reaches the token limit before EOS; report remains incomplete.",
    )
    parser.add_argument("--backend", choices=("auto", "triton", "torch"), default=None)
    parser.add_argument("--max-staged-experts", type=_positive_int)
    parser.add_argument("--max-pair-rows", type=_positive_int)
    parser.add_argument(
        "--cache-experts",
        type=_nonnegative_int,
        default=None,
        help="Persistent FP8 hot-cache capacity per MoE layer; 0 disables it.",
    )
    parser.add_argument("--prefill-chunk-size", type=_positive_int)
    parser.add_argument("--no-pin-memory", action="store_true")
    return parser


def resolve_runtime_options(
    args: argparse.Namespace, runtime: Any | None
) -> dict[str, Any]:
    """Resolve explicit options over the hardware recommendation."""

    def choose(explicit: Any, key: str, manual_default: Any) -> Any:
        if explicit is not None:
            return explicit
        return getattr(runtime, key) if runtime is not None else manual_default

    return {
        "device": runtime.device if runtime is not None else args.device,
        "backend": choose(
            args.backend if args.backend != "auto" else None, "backend", "triton"
        ),
        "lm_head_device": choose(
            args.lm_head_device if args.lm_head_device != "auto" else None,
            "lm_head_device",
            "cpu",
        ),
        "max_staged_experts": choose(args.max_staged_experts, "max_staged_experts", 8),
        "max_pair_rows": choose(args.max_pair_rows, "max_pair_rows", 64),
        "cache_experts": choose(args.cache_experts, "cache_experts", 0),
        "prefill_chunk_size": choose(args.prefill_chunk_size, "prefill_chunk_size", 16),
        "codec_max_frames": getattr(runtime, "codec_max_frames", 2048),
        "pin_memory": bool(
            not args.no_pin_memory and getattr(runtime, "pin_memory", True)
        ),
    }


def _configure_loaded_runtime(
    model: Any,
    manager: Any,
    *,
    sequence_capacity: int,
    runtime_config: Any | None,
    cache_experts: int | None,
    prefill_chunk_size: int,
    allow_staging_reduction: bool,
) -> Any:
    from drama_low_vram.model import configure_model_runtime

    return configure_model_runtime(
        model,
        manager,
        sequence_capacity=sequence_capacity,
        runtime_config=runtime_config,
        cache_experts=cache_experts,
        prefill_chunk_size=prefill_chunk_size,
        allow_staging_reduction=allow_staging_reduction,
    )


def _seed_generation(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _synchronize(device: str) -> None:
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _reset_gpu_peak(device: str) -> None:
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _process_memory() -> dict[str, int | None]:
    result: dict[str, int | None] = {
        "resident_bytes": None,
        "peak_resident_bytes": None,
        "private_committed_bytes": None,
    }
    if os.name != "nt":
        return result
    try:
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process.restype = wintypes.HANDLE
        read_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        read_memory.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        )
        read_memory.restype = wintypes.BOOL
        if read_memory(get_process(), ctypes.byref(counters), counters.cb):
            result.update(
                resident_bytes=int(counters.WorkingSetSize),
                peak_resident_bytes=int(counters.PeakWorkingSetSize),
                private_committed_bytes=int(counters.PrivateUsage),
            )
    except (AttributeError, OSError):
        pass
    return result


def _memory_snapshot(device: str) -> dict[str, Any]:
    result: dict[str, Any] = {"process": _process_memory(), "cuda": None}
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        return result
    free, total = torch.cuda.mem_get_info(device)
    result["cuda"] = {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "driver_free_bytes": int(free),
        "driver_total_bytes": int(total),
    }
    return result


def _write_report(report: dict[str, Any], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _expert_release_references(manager: Any) -> list[tuple[Any, int]]:
    """Keep weak references only, so release diagnostics cannot retain weights."""
    return [
        (
            weakref.ref(buffer),
            buffer.numel() * buffer.element_size(),
        )
        for module in getattr(manager, "modules", {}).values()
        for name, buffer in module.named_buffers(recurse=False)
        if name in ("w13_weight", "w13_scale_inv", "w2_weight", "w2_scale_inv")
    ]


def _expert_release_summary(references: list[tuple[Any, int]]) -> dict[str, int]:
    alive = [size for reference, size in references if reference() is not None]
    return {
        "tracked_expert_buffers": len(references),
        "tracked_expert_bytes": sum(size for _, size in references),
        "live_expert_buffers_after_release": len(alive),
        "live_expert_bytes_after_release": sum(alive),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.checkpoint.is_dir():
        parser.error(f"checkpoint directory does not exist: {args.checkpoint}")
    if args.prompt_audio and len(args.prompt_text) != 1:
        parser.error("--prompt-audio currently requires exactly one --prompt-text")
    if args.prompt_text and not args.prompt_audio:
        parser.error("--prompt-text requires --prompt-audio")
    if not 0 < args.temperature < 2:
        parser.error("--temperature must be in (0, 2)")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    if args.seed > 2**64 - 1:
        parser.error("--seed must be smaller than 2**64")
    report_path = args.report or args.output.with_suffix(".json")
    if report_path.resolve() == args.output.resolve():
        parser.error("--report and --output must use different paths")
    os.environ["FISH_DRAMA3_CUDA_GRAPH"] = "0"
    total_started = time.perf_counter()
    probe_started = time.perf_counter()
    try:
        runtime = None if args.manual else autotune(device=args.device)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    if runtime is not None and not runtime.can_run:
        parser.error("; ".join(runtime.warnings) or "No usable CUDA device is available")
    options = resolve_runtime_options(args, runtime)
    device = options["device"]
    if runtime is not None:
        print(
            "Auto runtime profile: "
            + json.dumps(runtime.as_dict(include_hardware=False), ensure_ascii=False),
            flush=True,
        )
    dtype = torch.bfloat16
    report: dict[str, Any] = {
        "status": "error",
        "finish_reason": None,
        "frames": 0,
        "seed": args.seed,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
        },
        "checkpoint": str(args.checkpoint.resolve()),
        "requested_output": str(args.output.resolve()),
        "output": None,
        "allow_partial": args.allow_partial,
        "requested_runtime": options,
        "auto_runtime": runtime.as_dict() if runtime is not None else None,
        "runtime_plan": None,
        "sequence_capacity": None,
        "offload": None,
        "audio": None,
        "phase_seconds": {
            "hardware_probe": time.perf_counter() - probe_started,
        },
        "memory": {},
        "precision": {
            "routed_experts": str(torch.float8_e4m3fn),
            "expert_block_scales": str(torch.float32),
            "dense_and_audio_tower": str(dtype),
            "codec": str(dtype),
            "expert_backend": options["backend"],
            "all_routed_experts_preserved": True,
            "additional_weight_quantization": False,
        },
        "environment": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": device,
        },
        "reproducibility": (
            "The seed is reset immediately before sampling. Matching results require "
            "the same model, device, backend, and runtime settings."
        ),
    }
    phase_seconds = report["phase_seconds"]
    memory = report["memory"]
    model, manager = None, None
    exit_code = 1
    try:
        memory["before_load"] = _memory_snapshot(device)
        prompt_tokens = []
        if args.prompt_audio:
            _reset_gpu_peak(device)
            started = time.perf_counter()
            prompt_tokens.append(
                load_reference_codes(
                    args.checkpoint,
                    args.prompt_audio,
                    device,
                    dtype,
                    options["codec_max_frames"],
                )
            )
            _synchronize(device)
            phase_seconds["reference_encode"] = time.perf_counter() - started
            memory["reference_encode"] = _memory_snapshot(device)

        _reset_gpu_peak(device)
        started = time.perf_counter()
        model, manager = load_offloaded_drama3(
            args.checkpoint,
            device=device,
            dtype=dtype,
            lm_head_device=options["lm_head_device"],
            expert_backend=options["backend"],
            max_staged_experts=options["max_staged_experts"],
            max_pair_rows=options["max_pair_rows"],
            cache_experts=0,
            pin_memory=options["pin_memory"],
        )
        _synchronize(device)
        phase_seconds["model_load"] = time.perf_counter() - started
        memory["model_load"] = _memory_snapshot(device)
        print(
            f"Loaded offloaded model in {phase_seconds['model_load']:.1f}s; "
            f"embedding=CPU, experts=CPU FP8, backbone=tower GPU, "
            f"lm_head={options['lm_head_device'].upper()}, backend={options['backend']}",
            flush=True,
        )

        started = time.perf_counter()
        prompt = build_drama3_prompt(
            args.text,
            model.tokenizer,
            model.config,
            prompt_texts=args.prompt_text,
            prompt_tokens=prompt_tokens,
        )
        report["prompt_rows"] = len(prompt)
        report["sequence_capacity"] = len(prompt) + args.max_new_tokens
        plan = _configure_loaded_runtime(
            model,
            manager,
            sequence_capacity=report["sequence_capacity"],
            runtime_config=runtime,
            cache_experts=options["cache_experts"] if args.manual else args.cache_experts,
            prefill_chunk_size=options["prefill_chunk_size"],
            allow_staging_reduction=(
                not args.manual
                and args.max_staged_experts is None
                and args.max_pair_rows is None
                and args.prefill_chunk_size is None
            ),
        )
        _synchronize(device)
        phase_seconds["runtime_setup"] = time.perf_counter() - started
        report["runtime_plan"] = plan.as_dict()
        report["runtime_allocation"] = getattr(
            model, "_drama_low_vram_allocation", None
        )
        report["effective_runtime"] = {
            "device": device,
            "backend": options["backend"],
            "lm_head_device": options["lm_head_device"],
            "cache_experts": plan.cache_experts,
            "max_staged_experts": plan.max_staged_experts,
            "max_pair_rows": plan.max_pair_rows,
            "prefill_chunk_size": plan.prefill_chunk,
            "pin_memory": getattr(
                getattr(manager, "pool", None), "pin_memory", options["pin_memory"]
            ),
            "pin_memory_error": getattr(
                getattr(manager, "pool", None), "pin_memory_error", None
            ),
        }
        report["model_cache_capacity"] = int(model.cache_capacity)
        memory["runtime_setup"] = _memory_snapshot(device)
        print(
            "Effective runtime plan: "
            + json.dumps(report["runtime_plan"], ensure_ascii=False),
            flush=True,
        )

        _seed_generation(args.seed)
        _reset_gpu_peak(device)
        _synchronize(device)
        started = time.perf_counter()
        frames, reason = generate_drama3(
            model,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            prefill_chunk_size=plan.prefill_chunk,
        )
        _synchronize(device)
        elapsed = time.perf_counter() - started
        phase_seconds["generation_including_prefill"] = elapsed
        memory["generation"] = _memory_snapshot(device)
        report["finish_reason"] = reason
        report["frames"] = len(frames)
        report["frames_per_second_including_prefill"] = len(frames) / max(elapsed, 1e-6)
        report["offload"] = manager.stats.as_dict()
        if reason not in ("eos", "length"):
            raise RuntimeError(f"Unknown Drama 3 finish reason: {reason!r}")
        if not len(frames):
            raise RuntimeError(f"Drama 3 returned no audio frames ({reason})")
        codes = frames[:, 1:].T.contiguous()
        report["status"] = "complete" if reason == "eos" else "incomplete"
        print(
            f"Generated {len(frames)} frames in {elapsed:.1f}s "
            f"({report['frames_per_second_including_prefill']:.2f} frames/s "
            f"including prefill), finish={reason}; offload={report['offload']}",
            flush=True,
        )
        release_started = time.perf_counter()
        release_references = _expert_release_references(manager)
        release_drama3(model, manager)
        model, manager = None, None
        phase_seconds["model_release"] = time.perf_counter() - release_started
        report["model_release"] = _expert_release_summary(release_references)
        memory["after_model_release"] = _memory_snapshot(device)

        if reason != "eos" and not args.allow_partial:
            report["error"] = (
                "Reached --max-new-tokens before EOS; audio is incomplete. "
                "Increase the token budget or use --allow-partial to write it."
            )
            print(report["error"], file=sys.stderr, flush=True)
            exit_code = 2
        else:
            _reset_gpu_peak(device)
            started = time.perf_counter()
            report["audio"] = write_audio(
                args.checkpoint, codes, args.output, device, dtype
            )
            _synchronize(device)
            phase_seconds["codec_decode_and_write"] = time.perf_counter() - started
            memory["codec_decode"] = _memory_snapshot(device)
            report["output"] = str(args.output.resolve())
            duration = report["audio"]["duration_seconds"]
            report["generation_rtf_including_prefill"] = elapsed / max(duration, 1e-6)
            print(f"Wrote {args.output} ({report['status']})", flush=True)
            exit_code = 0
    except Exception as exc:
        report["status"] = "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(report["error"], file=sys.stderr, flush=True)
    finally:
        if manager is not None:
            report["offload"] = manager.stats.as_dict()
        if model is not None:
            release_drama3(model, manager)
        phase_seconds["total"] = time.perf_counter() - total_started
        if report["audio"] is not None:
            report["end_to_end_rtf"] = phase_seconds["total"] / max(
                report["audio"]["duration_seconds"], 1e-6
            )
        _write_report(report, report_path)
        print(f"Wrote run report {report_path}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
