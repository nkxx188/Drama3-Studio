"""Compare the compact audio head with the official BF16 checkpoint head.

No model or checkpoint download is performed. The default reference stays on CPU
as in the low-VRAM loader. ``--full-gpu-reference`` additionally allocates the
approximately 1 GB full head on the selected GPU for same-device comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / ".cache" / "triton"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / ".cache" / "torchinductor"))

import torch
from safetensors import safe_open
from torch.nn import functional as F

from drama_low_vram.semantic_head import SemanticHead, semantic_candidate_ids


def compare(reference: torch.Tensor, actual: torch.Tensor) -> dict:
    reference = reference.float().cpu()
    actual = actual.float().cpu()
    difference = (actual - reference).abs()
    magnitude = reference.abs()
    relative = difference / magnitude.clamp_min(1e-5)
    return {
        "shape": list(reference.shape),
        "bitwise_equal": bool(torch.equal(reference, actual)),
        "max_absolute_error": float(difference.max()),
        "mean_absolute_error": float(difference.mean()),
        "changed_values": int(torch.count_nonzero(difference)),
        "total_values": int(difference.numel()),
        "max_relative_error": float(relative.max()),
        "relative_l2_error": float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(reference).clamp_min(1e-5)
        ),
        "top1_matches": int((reference.argmax(-1) == actual.argmax(-1)).sum()),
        "rows": int(reference.reshape(-1, reference.shape[-1]).shape[0]),
        "finite": bool(torch.isfinite(reference).all() and torch.isfinite(actual).all()),
    }


def _timed(operation, device: torch.device) -> tuple[torch.Tensor, float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = operation()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return result, time.perf_counter() - start


def benchmark_single_rows(
    weight: torch.Tensor, hidden: torch.Tensor, *, iterations: int
) -> dict:
    """Measure the batch-one GEMM used for each generated audio frame."""
    device = weight.device
    for index in range(min(len(hidden), 3)):
        F.linear(hidden[index : index + 1], weight)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        events = [
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in range(iterations)
        ]
        with torch.cuda.device(device):
            start_wall = time.perf_counter()
            for index, (start, end) in enumerate(events):
                row = index % len(hidden)
                start.record()
                F.linear(hidden[row : row + 1], weight)
                end.record()
            torch.cuda.synchronize(device)
            host_seconds = time.perf_counter() - start_wall
        times_ms = [start.elapsed_time(end) for start, end in events]
        clock = "cuda_events"
    else:
        times_ms = []
        start_wall = time.perf_counter()
        for index in range(iterations):
            row = index % len(hidden)
            start = time.perf_counter()
            F.linear(hidden[row : row + 1], weight)
            times_ms.append((time.perf_counter() - start) * 1000)
        host_seconds = time.perf_counter() - start_wall
        clock = "perf_counter"
    return {
        "batch_size": 1,
        "iterations": iterations,
        "clock": clock,
        "median_ms": statistics.median(times_ms),
        "mean_ms": statistics.mean(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "host_total_seconds": host_seconds,
        "excludes_input_transfer": True,
    }


def validate(args: argparse.Namespace) -> dict:
    checkpoint = args.checkpoint.resolve()
    config_bytes = (checkpoint / "config.json").read_bytes()
    config = json.loads(config_bytes)
    index_bytes = (checkpoint / "model.safetensors.index.json").read_bytes()
    index = json.loads(index_bytes)
    head_key = "lm_head.weight"
    shard = checkpoint / index["weight_map"][head_key]
    vocab = int(config["text_config"]["vocab_size"])
    hidden_size = int(config["text_config"]["hidden_size"])
    layout = {
        "eos_token_id": int(config["eos_token_id"]),
        "semantic_start_token_id": int(config["semantic_start_token_id"]),
        "semantic_end_token_id": int(config["semantic_end_token_id"]),
    }
    candidates = semantic_candidate_ids(vocab_size=vocab, **layout)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu")
    torch.set_num_threads(args.cpu_threads)
    if args.hidden_tensor is None:
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        hidden = torch.randn(args.rows, hidden_size, generator=generator, dtype=torch.bfloat16)
        hidden_source = {"kind": "seeded_random", "seed": args.seed}
    else:
        hidden = torch.load(args.hidden_tensor, map_location="cpu", weights_only=True)
        if isinstance(hidden, dict):
            hidden = hidden["hidden"]
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("--hidden-tensor must contain a Tensor or {'hidden': Tensor}")
        if hidden.ndim < 1 or hidden.shape[-1] != hidden_size:
            raise ValueError("Captured hidden states must end with the configured hidden width")
        hidden = hidden.reshape(-1, hidden_size).to(torch.bfloat16)
        hidden_source = {"kind": "captured_hidden", "path": str(args.hidden_tensor.resolve())}
    if hidden.numel() == 0:
        raise ValueError("At least one hidden row is required")
    with torch.inference_mode(), safe_open(str(shard), framework="pt", device="cpu") as reader:
        source = reader.get_slice(head_key)
        if source.get_shape() != [vocab, hidden_size]:
            raise ValueError("Checkpoint head shape does not match config")
        eos = source[layout["eos_token_id"] : layout["eos_token_id"] + 1]
        band = source[
            layout["semantic_start_token_id"] : layout["semantic_end_token_id"] + 1
        ]
        # EOS lies outside the band and can sort before or after it.
        rows = (
            torch.cat((eos, band))
            if layout["eos_token_id"] < layout["semantic_start_token_id"]
            else torch.cat((band, eos))
        )
        if rows.dtype != torch.bfloat16:
            raise TypeError("This validation requires official BF16 head weights")
        subset_hash = hashlib.sha256(rows.view(torch.uint8).numpy().tobytes()).hexdigest()
        compact_cpu = SemanticHead(rows.clone(), vocab_size=vocab, **layout)
        compact = SemanticHead(rows.to(device), vocab_size=vocab, **layout)
        full_cpu = reader.get_tensor(head_key)
        hidden_gpu = hidden.to(device)
        if device.type == "cuda":
            # Warm up CUDA libraries outside the recorded GEMM time.
            compact.compact_logits(hidden_gpu)
            torch.cuda.synchronize(device)
        reference_cpu, cpu_seconds = _timed(lambda: F.linear(hidden, full_cpu), torch.device("cpu"))
        cpu_compact, cpu_compact_seconds = _timed(
            lambda: compact_cpu.compact_logits(hidden), torch.device("cpu")
        )
        optimized, optimized_seconds = _timed(
            lambda: compact.compact_logits(hidden_gpu), device
        )
        expected = reference_cpu[:, candidates]
        single_row_reference = torch.cat(
            [F.linear(row[None], full_cpu)[:, candidates] for row in hidden]
        )
        single_row_cpu = torch.cat(
            [compact_cpu.compact_logits(row[None]) for row in hidden]
        )
        single_row_selected = torch.cat(
            [compact.compact_logits(row[None]) for row in hidden_gpu]
        )
        result = {
            "created_local": datetime.now().astimezone().isoformat(),
            "checkpoint": str(checkpoint),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
            "head_shard": shard.name,
            "head_key": head_key,
            "weight_dtype": str(rows.dtype),
            "candidate_rows_sha256": subset_hash,
            "candidate_count": int(candidates.numel()),
            "candidate_ids": {
                "eos": layout["eos_token_id"],
                "semantic_start": layout["semantic_start_token_id"],
                "semantic_end": layout["semantic_end_token_id"],
            },
            "full_weight_bytes": int(full_cpu.numel() * full_cpu.element_size()),
            "compact_weight_bytes": compact.weight_bytes,
            "device": str(device),
            "hidden": {**hidden_source, "shape": list(hidden.shape)},
            "cpu_full_vs_cpu_compact": compare(expected, cpu_compact),
            "cpu_full_vs_selected_compact": compare(expected, optimized),
            "single_row_logits": {
                "cpu_full_vs_cpu_compact": compare(single_row_reference, single_row_cpu),
                "cpu_full_vs_selected_compact": compare(
                    single_row_reference, single_row_selected
                ),
            },
            "timings_seconds": {
                "cpu_full": cpu_seconds,
                "cpu_compact": cpu_compact_seconds,
                "selected_compact": optimized_seconds,
            },
            "single_row_gemm": {
                "cpu_full": benchmark_single_rows(
                    full_cpu, hidden, iterations=args.benchmark_iterations
                ),
                "cpu_compact": benchmark_single_rows(
                    compact_cpu.weight, hidden, iterations=args.benchmark_iterations
                ),
                "selected_compact": benchmark_single_rows(
                    compact.weight, hidden_gpu, iterations=args.benchmark_iterations
                ),
            },
            "torch_version": torch.__version__,
            "cpu_threads": args.cpu_threads,
            "note": (
                "Exact weight-row copies are used. Moving GEMM between CPU/CUDA or "
                "changing matrix shape can change floating-point rounding. timings_seconds "
                "contains single batch samples; single_row_gemm contains repeated, warmed "
                "batch-one samples. Neither is an end-to-end audio benchmark."
            ),
        }
        if args.full_gpu_reference:
            if device.type != "cuda":
                raise ValueError("--full-gpu-reference requires a CUDA --device")
            full_gpu = full_cpu.to(device)
            F.linear(hidden_gpu, full_gpu)
            torch.cuda.synchronize(device)
            reference_gpu, gpu_seconds = _timed(
                lambda: F.linear(hidden_gpu, full_gpu), device
            )
            result["gpu_full_vs_gpu_compact"] = compare(
                reference_gpu[:, candidates.to(device)], optimized
            )
            single_row_gpu = torch.cat(
                [F.linear(row[None], full_gpu)[:, candidates.to(device)] for row in hidden_gpu]
            )
            result["single_row_logits"]["gpu_full_vs_gpu_compact"] = compare(
                single_row_gpu, single_row_selected
            )
            result["timings_seconds"]["gpu_full"] = gpu_seconds
            result["single_row_gemm"]["gpu_full"] = benchmark_single_rows(
                full_gpu, hidden_gpu, iterations=args.benchmark_iterations
            )
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "drama-3")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--seed", type=int, default=217)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--benchmark-iterations", type=int, default=20)
    parser.add_argument("--hidden-tensor", type=Path)
    parser.add_argument("--full-gpu-reference", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rows < 1 or args.cpu_threads < 1 or args.benchmark_iterations < 1:
        parser.error("--rows, --cpu-threads and --benchmark-iterations must be positive")
    result = validate(args)
    document = json.dumps(result, indent=2, ensure_ascii=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(document + "\n", encoding="utf-8")
    print(document)


if __name__ == "__main__":
    main()
