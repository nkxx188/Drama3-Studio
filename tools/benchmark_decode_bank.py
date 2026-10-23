"""Measure optional mixed-residency single-token expert execution."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor" / "fish-speech")]
for key, directory in (("TRITON_CACHE_DIR", "triton"), ("TEMP", "tmp"), ("TMP", "tmp")):
    path = ROOT / ".cache" / directory
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)

import torch  # noqa: E402
from torch import nn  # noqa: E402

from drama_low_vram.expert_cache_file import ExpertCacheFile  # noqa: E402
from drama_low_vram.offload import (  # noqa: E402
    EXPERT_BUFFER_NAMES,
    FP8_DTYPE,
    install_expert_offload,
)
from fish_speech.models.text2semantic.fish_omni.fp8_moe import fp8_moe_forward  # noqa: E402


class Experts(nn.Module):
    def __init__(self, buffers):
        super().__init__()
        self.fp8 = True
        self.num_experts = buffers["w13_weight"].shape[0]
        for name, tensor in buffers.items():
            self.register_buffer(name, tensor)

    def forward(self, *_args):
        raise AssertionError("Expert manager must replace this forward")


def synthetic_buffers(experts: int, hidden: int, intermediate: int):
    generator = torch.Generator().manual_seed(20261015)
    result = {}
    for name, rows, columns in (
        ("w13", 2 * intermediate, hidden),
        ("w2", hidden, intermediate),
    ):
        result[f"{name}_weight"] = torch.randint(
            -3, 4, (experts, rows, columns), generator=generator
        ).to(FP8_DTYPE)
        result[f"{name}_scale_inv"] = torch.rand(
            experts, rows // 128, columns // 128, generator=generator
        ) * 0.02 + 0.01
    return result


def selected_reference(buffers, x, ids, weights):
    unique = torch.unique(ids.cpu(), sorted=True)
    selected = []
    for name in EXPERT_BUFFER_NAMES:
        source = buffers[name]
        if source.dtype == FP8_DTYPE:
            values = source.view(torch.uint8).index_select(0, unique).view(FP8_DTYPE)
        else:
            values = source.index_select(0, unique)
        selected.append(values.to(x.device))
    lookup = torch.full((buffers["w13_weight"].shape[0],), -1, dtype=torch.int64)
    lookup[unique] = torch.arange(unique.numel())
    local = lookup[ids.cpu()].to(x.device)
    result = fp8_moe_forward(x, local, weights, *selected)
    torch.cuda.synchronize(x.device)
    return result


def run_benchmark(args):
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Decode-bank benchmark requires CUDA")
    cache = None
    if args.real_cache is not None:
        cache = ExpertCacheFile.open(args.real_cache)
        buffers = cache.layer_buffers(args.layer)
    else:
        buffers = synthetic_buffers(args.experts, args.hidden, args.intermediate)
    experts, _, hidden = buffers["w13_weight"].shape
    generator = torch.Generator().manual_seed(args.seed)
    cases = [
        torch.tensor([[1, 3, 5, 7, 9, 11, 13, 15]]),
        torch.tensor([[5, 2, 3, 4, 1, 6, 8, 10]]),
        torch.tensor([[3, 2, 4, 6, 8, 10, 12, 14]]),
        torch.tensor([[2, 1, 4, 6, 9, 8, 11, 10]]),
        torch.tensor([[3, 2, 1, 2, 3, 6, 6, 8]]),
    ]
    if experts < 16:
        raise ValueError("The parity route cases require at least 16 experts")
    x = torch.randn((1, hidden), generator=generator).to(device, torch.bfloat16)
    weights = torch.softmax(torch.randn((1, 8), generator=generator), -1).to(device)
    device_cases = [ids.to(device) for ids in cases]
    expected = [selected_reference(buffers, x, ids, weights) for ids in device_cases]
    results = {}
    try:
        for enabled in (False, True):
            module = Experts(buffers)
            backbone = nn.Module()
            backbone.add_module("experts", module)
            manager = install_expert_offload(
                backbone,
                backend="triton",
                max_staged_experts=args.staged_experts,
                max_pair_rows=args.pair_rows,
                cache_experts=args.cache_experts,
                pin_memory=args.pin_memory,
                combined_decode_bank=enabled,
                packed_staging=args.packed_staging,
            )
            if args.host_resident_experts:
                per_expert = sum(
                    tensor[0].numel() * tensor.element_size()
                    for tensor in buffers.values()
                )
                manager.configure_host_residency(per_expert * args.host_resident_experts)
            parity = []
            for ids, official in zip(device_cases, expected):
                actual = module(x, ids, weights)
                torch.cuda.synchronize(device)
                parity.append(
                    {
                        "routes": ids.cpu().tolist()[0],
                        "bitwise_equal": bool(torch.equal(actual, official)),
                        "max_abs_error": float((actual.float() - official.float()).abs().max()),
                        "finite": bool(torch.isfinite(actual).all()),
                    }
                )
            for _ in range(args.warmup):
                for ids in device_cases:
                    module(x, ids, weights)
            torch.cuda.synchronize(device)
            before = manager.stats.as_dict()
            timings = []
            for _ in range(args.repeats):
                started = time.perf_counter()
                for ids in device_cases:
                    module(x, ids, weights)
                torch.cuda.synchronize(device)
                timings.append((time.perf_counter() - started) / len(device_cases))
            after = manager.stats.as_dict()
            results["combined" if enabled else "partitioned"] = {
                "mean_ms_per_call": statistics.mean(timings) * 1000,
                "median_ms_per_call": statistics.median(timings) * 1000,
                "min_ms_per_call": min(timings) * 1000,
                "max_ms_per_call": max(timings) * 1000,
                "parity_cases": parity,
                "stats_delta": {
                    key: after[key] - before[key]
                    for key in (
                        "combined_decode_calls",
                        "staging_calls",
                        "expert_loads",
                        "hot_cache_hits",
                        "hot_cache_loads",
                        "h2d_bytes",
                        "host_staging_seconds",
                        "host_gather_seconds",
                        "h2d_submit_seconds",
                        "h2d_copy_calls",
                        "packed_staging_calls",
                    )
                },
            }
            manager.close()
    finally:
        if cache is not None:
            cache.close()
    partitioned = results["partitioned"]["median_ms_per_call"]
    combined = results["combined"]["median_ms_per_call"]
    return {
        "source": "real_official_fp8" if args.real_cache else "synthetic_fp8",
        "source_layer": args.layer if args.real_cache else None,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "experts": experts,
        "hidden": hidden,
        "intermediate": buffers["w13_weight"].shape[1] // 2,
        "cache_experts": args.cache_experts,
        "staged_experts": args.staged_experts,
        "pair_rows": args.pair_rows,
        "pin_memory": args.pin_memory,
        "packed_staging": args.packed_staging,
        "host_resident_experts": args.host_resident_experts,
        "calls_per_repeat": len(device_cases),
        "repeats": args.repeats,
        "median_speed_ratio": partitioned / combined,
        "all_cases_bitwise_equal_to_official": all(
            case["bitwise_equal"] and case["finite"]
            for row in results.values()
            for case in row["parity_cases"]
        ),
        "timing_scope": "One expert module only, host staging + H2D + expert execution; no full-model FPS claim.",
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--real-cache", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--cache-experts", type=int, default=3)
    parser.add_argument("--staged-experts", type=int, default=8)
    parser.add_argument("--pair-rows", type=int, default=64)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--packed-staging", action="store_true")
    parser.add_argument("--host-resident-experts", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.cache_experts, args.staged_experts, args.pair_rows, args.repeats) < 1:
        parser.error("Capacities and repeats must be positive")
    if args.host_resident_experts < 0:
        parser.error("Resident expert count cannot be negative")
    if args.cache_experts >= 8 or args.staged_experts < 8:
        parser.error("Need a partial cache (<8) and an eight-expert staging bank")
    result = run_benchmark(args)
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        print(f"Report: {args.output.resolve()}")
    return 0 if result["all_cases_bitwise_equal_to_official"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
