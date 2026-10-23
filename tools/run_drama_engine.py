"""Run reusable Drama 3 engine requests, optionally from a JSONL batch."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor/fish-speech")]
for key, directory in (
    ("TRITON_CACHE_DIR", "triton"),
    ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
    ("TORCH_EXTENSIONS_DIR", "torch_extensions"),
    ("HF_HOME", "huggingface"),
    ("TEMP", "tmp"),
    ("TMP", "tmp"),
):
    path = ROOT / ".cache" / directory
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)

import torch

from drama_low_vram.engine import DramaEngine, EngineOptions
from fish_speech.models.text2semantic.sampling import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)
from drama_low_vram.sampling import DEFAULT_TOP_K


REQUEST_KEYS = frozenset(
    {
        "text",
        "output",
        "reference_audio",
        "reference_text",
        "seed",
        "max_new_tokens",
        "max_chunk_chars",
        "temperature",
        "top_k",
        "top_p",
        "allow_partial",
    }
)


def _strict_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _strict_float(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value <= minimum:
        raise ValueError(f"{name} must be greater than {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _output_for_batch(default: Path, index: int) -> Path:
    suffix = default.suffix or ".wav"
    return default.with_name(f"{default.stem}-{index:03d}{suffix}")


def _validate_request(
    value: Any,
    *,
    index: int,
    default_output: Path,
    default_seed: int,
    default_max_new_tokens: int,
    default_max_chunk_chars: int,
    default_allow_partial: bool,
    batched: bool,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"request {index} must be a JSON object")
    unknown = sorted(set(value) - REQUEST_KEYS)
    if unknown:
        names = ", ".join(repr(name) for name in unknown)
        raise ValueError(f"request {index} has unknown field(s): {names}")
    request: dict[str, Any] = {}
    request["text"] = _nonempty_string(value.get("text"), f"request {index}.text")
    raw_output = value.get("output")
    if raw_output is None:
        request["output"] = (
            _output_for_batch(default_output, index) if batched else default_output
        )
    else:
        request["output"] = Path(_nonempty_string(raw_output, f"request {index}.output"))

    for field in ("reference_audio", "reference_text"):
        raw = value.get(field)
        if raw is not None:
            request[field] = _nonempty_string(raw, f"request {index}.{field}")
        else:
            request[field] = None
    if bool(request["reference_audio"]) != bool(request["reference_text"]):
        raise ValueError(
            f"request {index} requires both reference_audio and reference_text"
        )

    request["seed"] = _strict_int(
        value.get("seed", default_seed), f"request {index}.seed", minimum=0
    )
    if request["seed"] >= 2**64:
        raise ValueError(f"request {index}.seed must be less than 2**64")
    request["max_new_tokens"] = _strict_int(
        value.get("max_new_tokens", default_max_new_tokens),
        f"request {index}.max_new_tokens",
        minimum=1,
    )
    request["max_chunk_chars"] = _strict_int(
        value.get("max_chunk_chars", default_max_chunk_chars),
        f"request {index}.max_chunk_chars",
        minimum=1,
    )
    request["temperature"] = _strict_float(
        value.get("temperature", DEFAULT_TEMPERATURE),
        f"request {index}.temperature",
        minimum=0,
        maximum=2,
    )
    if request["temperature"] >= 2:
        raise ValueError(f"request {index}.temperature must be less than 2")
    request["top_k"] = _strict_int(
        value.get("top_k", DEFAULT_TOP_K), f"request {index}.top_k", minimum=0
    )
    request["top_p"] = _strict_float(
        value.get("top_p", DEFAULT_TOP_P), f"request {index}.top_p", minimum=0, maximum=1
    )
    allow_partial = value.get("allow_partial", default_allow_partial)
    if not isinstance(allow_partial, bool):
        raise ValueError(f"request {index}.allow_partial must be a boolean")
    request["allow_partial"] = allow_partial
    return request


def _cuda_resource_failure(error: BaseException) -> bool:
    if isinstance(error, MemoryError):
        return True
    import torch

    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "cuda out of memory",
            "cuda error",
            "cublas",
            "device-side assert",
            "illegal memory access",
            "driver shutting down",
            "offset increment outside graph",
            "graph capture",
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/drama-3")
    parser.add_argument("--text")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/engine.wav")
    parser.add_argument("--requests", type=Path, help="JSONL requests, executed with one resident model.")
    parser.add_argument("--prompt-audio", type=Path)
    parser.add_argument("--prompt-text")
    parser.add_argument("--device", default="auto",
                        help="Default: supported NVIDIA GPU with the most currently free VRAM")
    parser.add_argument("--expert-storage", choices=("auto", "ram", "mmap"), default="auto")
    parser.add_argument("--expert-decode", choices=("auto", "official"), default="auto",
                        help="Use automatic batch-one FP8 decoding or retain the official general kernel.")
    parser.add_argument("--cpu-experts", type=int, default=0,
                        help="Experimental SM89 CPU/GPU decode cooperation: 0 to 8 experts per layer.")
    parser.add_argument("--native-threads", type=int, default=4,
                        help="Shared native CPU worker count for experimental cooperation.")
    parser.add_argument("--host-gather", choices=("auto", "torch", "native"), default="torch",
                        help="CPU expert byte gathering: auto uses the optional DLL for larger contiguous groups.")
    parser.add_argument("--compute-policy", choices=("manual", "auto"), default="manual",
                        help="Measure this machine once and cache an exact-output compute choice.")
    parser.add_argument("--compute-profile-dir", type=Path)
    parser.add_argument("--context-bucket-size", type=int, default=512,
                        help="Round sequence caches for reuse across requests; 0 uses exact capacities.")
    parser.add_argument("--host-threads", type=int,
                        help="Optional process-wide PyTorch CPU thread count for reproducible tuning.")
    parser.add_argument("--prefix-cache-mib", type=int, default=0,
                        help="Experimental bounded CPU reference-prefix state cache; 0 disables it.")
    parser.add_argument("--expert-cache", type=Path)
    parser.add_argument("--host-resident-gib", type=float, default=0,
                        help="Experimental partial RAM expert copy; 0 uses the OS file cache.")
    parser.add_argument("--profile-transfers", action="store_true",
                        help="Measure CUDA transfer event elapsed time (adds profiling overhead).")
    parser.add_argument("--codec-policy", choices=("auto", "park"), default="auto")
    parser.add_argument("--codec-placement", choices=("auto", "full"), default="auto",
                        help="Stage only the official codec branches required by encode/decode, or stage the full codec.")
    parser.add_argument("--cache-experts", type=int)
    parser.add_argument("--staged-experts", type=int, default=None)
    parser.add_argument("--pair-rows", type=int, default=None)
    parser.add_argument("--prefill-chunk", type=int, default=None)
    parser.add_argument("--no-compact-head", action="store_true")
    parser.add_argument("--no-tower-graph", action="store_true")
    parser.add_argument("--backbone-graph", action="store_true")
    parser.add_argument("--combined-decode-bank", action="store_true",
                        help="Experimental merged hot/cold expert execution for one-token decode.")
    parser.add_argument("--packed-staging", action="store_true",
                        help="Experimental single-copy expert-major staging arena.")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-chunk-chars", type=int, default=160)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.device == "auto":
        from drama_low_vram.diagnostics import select_python_device
        args.device = select_python_device()
    if bool(args.text) == bool(args.requests):
        parser.error("Provide either --text or --requests")
    for name in ("staged_experts", "pair_rows", "prefill_chunk"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.cache_experts is not None and args.cache_experts < 0:
        parser.error("--cache-experts cannot be negative")
    if not 0 <= args.cpu_experts <= 8 or not 1 <= args.native_threads <= 64:
        parser.error("--cpu-experts must be 0..8 and --native-threads must be 1..64")
    if args.host_threads is not None and not 1 <= args.host_threads <= 64:
        parser.error("--host-threads must be 1..64")
    if args.prefix_cache_mib < 0:
        parser.error("--prefix-cache-mib cannot be negative")
    if not 0 <= args.context_bucket_size <= 4096:
        parser.error("--context-bucket-size must be 0..4096")
    if not math.isfinite(args.host_resident_gib) or args.host_resident_gib < 0:
        parser.error("--host-resident-gib must be finite and non-negative")
    if args.requests:
        try:
            lines = [
                (line_number, line)
                for line_number, line in enumerate(
                    args.requests.read_text(encoding="utf-8-sig").splitlines(), 1
                )
                if line.strip()
            ]
        except (OSError, UnicodeError) as error:
            parser.error(f"Cannot read requests file: {error}")
        if not lines:
            parser.error("Requests JSONL file is empty")
        raw_requests = []
        for line_number, line in lines:
            try:
                raw_requests.append(json.loads(line))
            except json.JSONDecodeError as error:
                parser.error(f"Invalid JSON on requests line {line_number}: {error.msg}")
        try:
            requests = [
                _validate_request(
                    value,
                    index=index,
                    default_output=args.output,
                    default_seed=args.seed,
                    default_max_new_tokens=args.max_new_tokens,
                    default_max_chunk_chars=args.max_chunk_chars,
                    default_allow_partial=args.allow_partial,
                    batched=True,
                )
                for index, value in enumerate(raw_requests, 1)
            ]
        except ValueError as error:
            parser.error(str(error))
        outputs = [str(request["output"].resolve()).casefold() for request in requests]
        if len(outputs) != len(set(outputs)):
            parser.error("Requests must use distinct output files")
    else:
        try:
            requests = [
                _validate_request(
                    {
                        "text": args.text,
                        "output": str(args.output),
                        "reference_audio": str(args.prompt_audio)
                        if args.prompt_audio
                        else None,
                        "reference_text": args.prompt_text,
                        "seed": args.seed,
                        "max_new_tokens": args.max_new_tokens,
                        "max_chunk_chars": args.max_chunk_chars,
                        "allow_partial": args.allow_partial,
                    },
                    index=1,
                    default_output=args.output,
                    default_seed=args.seed,
                    default_max_new_tokens=args.max_new_tokens,
                    default_max_chunk_chars=args.max_chunk_chars,
                    default_allow_partial=args.allow_partial,
                    batched=False,
                )
            ]
        except ValueError as error:
            parser.error(str(error))
    if args.host_threads is not None:
        torch.set_num_threads(args.host_threads)
    options = EngineOptions(
        args.checkpoint, device=args.device, expert_storage=args.expert_storage,
        expert_decode=args.expert_decode,
        cpu_experts=args.cpu_experts, native_threads=args.native_threads,
        host_gather=args.host_gather,
        compute_policy=args.compute_policy, compute_profile_dir=args.compute_profile_dir,
        context_bucket_size=args.context_bucket_size,
        prefix_cache_mib=args.prefix_cache_mib,
        expert_cache_dir=args.expert_cache, compact_head=not args.no_compact_head,
        tower_graph=not args.no_tower_graph, backbone_graph=args.backbone_graph,
        combined_decode_bank=args.combined_decode_bank,
        packed_staging=args.packed_staging,
        profile_transfers=args.profile_transfers,
        host_resident_gib=args.host_resident_gib,
        codec_policy=args.codec_policy,
        codec_placement=args.codec_placement,
        cache_experts=args.cache_experts, staged_experts=args.staged_experts,
        pair_rows=args.pair_rows, prefill_chunk=args.prefill_chunk,
    )
    exit_code = 0
    with DramaEngine(options) as engine:
        engine.status_callback = lambda event: print(
            json.dumps({"runtime": event}, ensure_ascii=False), flush=True)
        for index, request in enumerate(requests, 1):
            def progress(frame, maximum):
                if frame == 1 or frame % 32 == 0:
                    print(f"Audio frames: {frame}/{maximum}", flush=True)

            try:
                report = engine.synthesize(
                    request["text"],
                    request["output"],
                    reference_audio=Path(request["reference_audio"])
                    if request["reference_audio"]
                    else None,
                    reference_text=request["reference_text"],
                    seed=request["seed"],
                    max_new_tokens=request["max_new_tokens"],
                    max_chunk_chars=request["max_chunk_chars"],
                    temperature=request["temperature"],
                    top_k=request["top_k"],
                    top_p=request["top_p"],
                    allow_partial=request["allow_partial"],
                    progress=progress,
                )
            except Exception as error:
                error_report = {
                    "status": "error",
                    "request_index": index,
                    "output": str(request["output"]),
                    "error": f"{type(error).__name__}: {error}",
                }
                print(json.dumps(error_report, ensure_ascii=False), file=sys.stderr, flush=True)
                exit_code = 1
                if _cuda_resource_failure(error):
                    return exit_code
                continue
            print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
            if report["status"] != "complete" and not request["allow_partial"]:
                if exit_code != 1:
                    exit_code = 2
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
