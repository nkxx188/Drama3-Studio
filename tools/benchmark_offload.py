"""Benchmark bounded Drama 3 expert offload without loading the full checkpoint.

The default synthetic benchmark exercises the same ``ExpertOffloadManager`` used
by the Windows runtime, but allocates a small deterministic FP8 expert bank.
This keeps iteration fast and makes the transfer/profile numbers reproducible.
Use ``--full-model`` only when a complete Drama 3 checkpoint is available and a
real generation measurement is desired.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
VENDOR = ROOT / "vendor" / "fish-speech"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from drama_low_vram.offload import (  # noqa: E402
    FP8_BLOCK_SIZE,
    FP8_DTYPE,
    OffloadStatistics,
    install_expert_offload,
)


COUNTER_FIELDS = (
    "forward_calls",
    "staging_calls",
    "expert_loads",
    "expert_cache_hits",
    "hot_cache_hits",
    "hot_cache_loads",
    "staged_bytes",
    "h2d_bytes",
    "host_staging_seconds",
    "h2d_seconds",
)


class SyntheticExperts(nn.Module):
    """A small FP8 bank matching Fish Omni's routed-expert buffer layout."""

    def __init__(
        self,
        *,
        experts: int,
        hidden: int,
        intermediate: int,
        seed: int,
    ) -> None:
        super().__init__()
        if experts < 1:
            raise ValueError("experts must be positive")
        if hidden < FP8_BLOCK_SIZE or intermediate < FP8_BLOCK_SIZE:
            raise ValueError("hidden and intermediate must be at least 128")
        if hidden % FP8_BLOCK_SIZE or intermediate % FP8_BLOCK_SIZE:
            raise ValueError("hidden and intermediate must be divisible by 128")
        self.fp8 = True
        generator = torch.Generator(device="cpu").manual_seed(seed)

        def weight(shape: tuple[int, ...]) -> torch.Tensor:
            # Small integer values keep the synthetic GEMM numerically stable
            # while preserving the byte layout of the real block-FP8 weights.
            return torch.randint(
                -3, 4, shape, generator=generator, dtype=torch.int8
            ).to(FP8_DTYPE)

        def scales(shape: tuple[int, ...]) -> torch.Tensor:
            return (
                torch.rand(shape, generator=generator, dtype=torch.float32) * 0.02
                + 0.01
            )

        self.register_buffer(
            "w13_weight", weight((experts, 2 * intermediate, hidden))
        )
        self.register_buffer(
            "w13_scale_inv",
            scales((experts, 2 * intermediate // FP8_BLOCK_SIZE, hidden // FP8_BLOCK_SIZE)),
        )
        self.register_buffer(
            "w2_weight", weight((experts, hidden, intermediate))
        )
        self.register_buffer(
            "w2_scale_inv",
            scales((experts, hidden // FP8_BLOCK_SIZE, intermediate // FP8_BLOCK_SIZE)),
        )

    def forward(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> torch.Tensor:
        raise AssertionError("Synthetic experts must be wrapped by offload manager")


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    return device


def _resolve_backend(value: str, device: torch.device) -> str:
    if value == "auto":
        return "triton" if device.type == "cuda" else "torch"
    return value


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _memory_snapshot(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {"allocated_bytes": 0, "reserved_bytes": 0, "peak_allocated_bytes": 0}
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def _stats_snapshot(stats: OffloadStatistics) -> dict[str, int | float]:
    return stats.as_dict()


def _stats_delta(
    before: dict[str, int | float], after: dict[str, int | float]
) -> dict[str, int | float]:
    return {
        field: after[field] - before[field]
        for field in COUNTER_FIELDS
        if field in before and field in after
    }


def _reset_stats(stats: OffloadStatistics) -> None:
    """Reset counters in place; the staging pool retains this object by reference."""
    for name in (*COUNTER_FIELDS, "peak_staged_experts"):
        value = getattr(stats, name)
        setattr(stats, name, 0 if isinstance(value, int) else 0.0)


def _routes(
    *,
    tokens: int,
    experts: int,
    top_k: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randint(
        0, experts, (tokens, top_k), generator=generator, dtype=torch.int64
    )
    weights = torch.softmax(torch.randn((tokens, top_k), generator=generator), dim=-1)
    return ids.to(device), weights.to(device)


def _phase_result(
    *,
    name: str,
    elapsed: float,
    frames: int,
    rows: int,
    before: dict[str, int | float],
    after: dict[str, int | float],
    memory: dict[str, int] | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "name": name,
        "elapsed_seconds": elapsed,
        "frames": frames,
        "rows": rows,
        "frames_per_second": frames / max(elapsed, 1e-9) if frames else 0.0,
        "rows_per_second": rows / max(elapsed, 1e-9) if rows else 0.0,
        "stats": _stats_delta(before, after),
    }
    if memory is not None:
        metrics["memory"] = memory
    h2d_seconds = float(metrics["stats"].get("h2d_seconds", 0.0))
    h2d_bytes = int(metrics["stats"].get("h2d_bytes", 0))
    metrics["h2d_gib_per_second"] = (
        h2d_bytes / (1024**3) / h2d_seconds if h2d_seconds > 0 else 0.0
    )
    return metrics


def run_synthetic(args: argparse.Namespace) -> dict[str, Any]:
    """Run a deterministic prefill/decode benchmark against a small expert bank."""
    device = _resolve_device(args.device)
    backend = _resolve_backend(args.backend, device)
    if args.cache < 0:
        raise ValueError("--cache cannot be negative")
    if args.top_k_routes < 1 or args.top_k_routes > args.experts:
        raise ValueError("--top-k-routes must be between 1 and --experts")
    if args.max_new_tokens < 1 or args.prefill_tokens < 1:
        raise ValueError("prefill and decode token counts must be positive")

    torch.manual_seed(args.seed)
    experts = SyntheticExperts(
        experts=args.experts,
        hidden=args.hidden,
        intermediate=args.intermediate,
        seed=args.seed,
    )
    backbone = nn.Module()
    backbone.add_module("experts", experts)
    manager = install_expert_offload(
        backbone,
        backend=backend,
        max_staged_experts=args.max_staged_experts,
        max_pair_rows=args.max_pair_rows,
        cache_experts=args.cache,
        pin_memory=not args.no_pin_memory,
        profile_transfers=True,
    )

    # Allocate activations once per phase so timing reflects routing/staging and
    # expert compute rather than repeated random tensor construction.
    dtype = torch.bfloat16 if backend == "triton" else torch.float32
    prefill_x = torch.randn(
        (args.prefill_tokens, args.hidden), device=device, dtype=dtype
    )
    decode_x = torch.randn((1, args.hidden), device=device, dtype=dtype)

    try:
        # A short warmup absorbs one-time CUDA/Triton initialization. Counters are
        # reset afterwards while preserving the manager's allocated buffers/cache.
        for warmup_index in range(args.warmup):
            warm_ids, warm_weights = _routes(
                tokens=args.prefill_tokens,
                experts=args.experts,
                top_k=args.top_k_routes,
                device=device,
                seed=args.seed + 101 + warmup_index,
            )
            with torch.inference_mode():
                experts(prefill_x, warm_ids, warm_weights)
        manager.synchronize()
        _synchronize(device)
        _reset_stats(manager.stats)
        _reset_peak_memory(device)

        prefill_ids, prefill_weights = _routes(
            tokens=args.prefill_tokens,
            experts=args.experts,
            top_k=args.top_k_routes,
            device=device,
            seed=args.seed + 1001,
        )
        before = _stats_snapshot(manager.stats)
        started = time.perf_counter()
        with torch.inference_mode():
            prefill_out = experts(prefill_x, prefill_ids, prefill_weights)
        manager.synchronize()
        _synchronize(device)
        prefill_elapsed = time.perf_counter() - started
        prefill_memory = _memory_snapshot(device)
        after = _stats_snapshot(manager.stats)
        prefill = _phase_result(
            name="prefill",
            elapsed=prefill_elapsed,
            frames=0,
            rows=args.prefill_tokens,
            before=before,
            after=after,
            memory=prefill_memory,
        )

        before = after
        _reset_peak_memory(device)
        started = time.perf_counter()
        decode_checksum_tensor = torch.zeros((), device=device, dtype=torch.float32)
        for step in range(args.max_new_tokens):
            decode_ids, decode_weights = _routes(
                tokens=1,
                experts=args.experts,
                top_k=args.top_k_routes,
                device=device,
                seed=args.seed + 2001 + step,
            )
            with torch.inference_mode():
                decode_out = experts(decode_x, decode_ids, decode_weights)
            # Keep a scalar only for a final correctness signal; synchronizing
            # once after the loop avoids timing a host read on every frame.
            decode_checksum_tensor += decode_out.detach().float().sum()
        manager.synchronize()
        _synchronize(device)
        decode_elapsed = time.perf_counter() - started
        decode_memory = _memory_snapshot(device)
        decode_checksum = float(decode_checksum_tensor.cpu())
        after = _stats_snapshot(manager.stats)
        decode = _phase_result(
            name="decode",
            elapsed=decode_elapsed,
            frames=args.max_new_tokens,
            rows=args.max_new_tokens,
            before=before,
            after=after,
            memory=decode_memory,
        )

        final_stats = _stats_snapshot(manager.stats)
        result = {
            "schema_version": 1,
            "kind": "synthetic",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "cpu"
            ),
            "config": {
                "text": args.text,
                "seed": args.seed,
                "experts": args.experts,
                "hidden": args.hidden,
                "intermediate": args.intermediate,
                "top_k_routes": args.top_k_routes,
                "prefill_tokens": args.prefill_tokens,
                "max_new_tokens": args.max_new_tokens,
                "backend": backend,
                "max_staged_experts": args.max_staged_experts,
                "max_pair_rows": args.max_pair_rows,
                "cache_experts": args.cache,
                "warmup": args.warmup,
                "pin_memory": not args.no_pin_memory,
            },
            "phases": {"prefill": prefill, "decode": decode},
            "checksum": decode_checksum + float(
                prefill_out.detach().float().sum().cpu()
            ),
            "peak_vram": {
                key: max(
                    int(prefill_memory.get(key, 0)),
                    int(decode_memory.get(key, 0)),
                )
                for key in prefill_memory
            },
            "final_stats": final_stats,
        }
        return result
    finally:
        manager.close()


def run_full_model(args: argparse.Namespace) -> dict[str, Any]:
    """Measure one real generation; phase counters cover the full generation."""
    device = _resolve_device(args.device)
    if device.type != "cuda":
        raise ValueError("--full-model requires a CUDA device")
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {args.checkpoint}")
    backend = _resolve_backend(args.backend, device)
    os.environ["FISH_DRAMA3_CUDA_GRAPH"] = "0"

    from drama_low_vram.model import load_offloaded_drama3, release_drama3
    from fish_speech.models.text2semantic.fish_omni.generate import (
        build_drama3_prompt,
        generate_drama3,
    )

    started = time.perf_counter()
    model, manager = load_offloaded_drama3(
        args.checkpoint,
        device=device,
        dtype=torch.bfloat16,
        lm_head_device=args.lm_head_device,
        expert_backend=backend,
        max_staged_experts=args.max_staged_experts,
        max_pair_rows=args.max_pair_rows,
        cache_experts=args.cache,
        pin_memory=not args.no_pin_memory,
        profile_transfers=True,
    )
    load_elapsed = time.perf_counter() - started
    try:
        prompt = build_drama3_prompt(args.text, model.tokenizer, model.config)
        manager.synchronize()
        _synchronize(device)
        _reset_stats(manager.stats)
        _reset_peak_memory(device)
        before = _stats_snapshot(manager.stats)
        started = time.perf_counter()
        frames, finish_reason = generate_drama3(
            model,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            prefill_chunk_size=args.prefill_chunk_size,
        )
        manager.synchronize()
        _synchronize(device)
        elapsed = time.perf_counter() - started
        generation_memory = _memory_snapshot(device)
        after = _stats_snapshot(manager.stats)
        generation = _phase_result(
            name="generation",
            elapsed=elapsed,
            frames=len(frames),
            rows=len(prompt),
            before=before,
            after=after,
            memory=generation_memory,
        )
        return {
            "schema_version": 1,
            "kind": "full-model",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "config": {
                "checkpoint": str(args.checkpoint),
                "text": args.text,
                "max_new_tokens": args.max_new_tokens,
                "backend": backend,
                "max_staged_experts": args.max_staged_experts,
                "max_pair_rows": args.max_pair_rows,
                "cache_experts": args.cache,
                "lm_head_device": args.lm_head_device,
                "prefill_chunk_size": args.prefill_chunk_size,
                "pin_memory": not args.no_pin_memory,
            },
            "load_seconds": load_elapsed,
            "finish_reason": finish_reason,
            "phases": {"generation": generation},
            "peak_vram": generation_memory,
            "final_stats": _stats_snapshot(manager.stats),
        }
    finally:
        release_drama3(model, manager)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-model",
        action="store_true",
        help="load the complete Drama 3 checkpoint (off by default)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "drama-3",
    )
    parser.add_argument("--text", default="<|speaker:0|>benchmark")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lm-head-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--backend", choices=("auto", "torch", "triton"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--prefill-tokens", type=int, default=32)
    parser.add_argument("--prefill-chunk-size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--top-k-routes", dest="top_k_routes", type=int, default=2)
    parser.add_argument("--max-staged-experts", type=int, default=8)
    parser.add_argument("--max-pair-rows", type=int, default=64)
    parser.add_argument(
        "--cache",
        type=int,
        default=0,
        help="synthetic per-layer hot-cache capacity in experts",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        help="write JSON to this path in addition to printing it",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.warmup < 0:
        _parser().error("--warmup cannot be negative")
    try:
        result = run_full_model(args) if args.full_model else run_synthetic(args)
    except Exception as exc:
        _parser().error(str(exc))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
