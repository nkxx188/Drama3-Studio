"""Isolate native/GPU expert arithmetic using real captured inputs and injections."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor/fish-speech")]
for key, name in (("TRITON_CACHE_DIR", "triton"), ("TEMP", "tmp"), ("TMP", "tmp")):
    directory = ROOT / ".cache" / name
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(directory)

import torch
import triton
import triton.language as tl

from drama_low_vram.expert_cache_file import ExpertCacheFile
from drama_low_vram.fp8_cpu_reference import quantize_fp8_blocks
from drama_low_vram.fp8_decode import build_gpu_silu_table
from drama_low_vram.native_backend import NativeExpertExecutor, NativeLibrary
from drama_low_vram.offload import (
    ExpertStore, OffloadStatistics, TritonExpertBackend, _StagingPool,
)


@triton.jit
def silu_trace_kernel(
    gu, activation, normalized, quantized, scales,
    HALF: tl.constexpr, GROUP: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    offsets = group * GROUP + tl.arange(0, GROUP)
    gate = tl.load(gu + row * 2 * HALF + offsets).to(tl.float32)
    up = tl.load(gu + row * 2 * HALF + HALF + offsets).to(tl.float32)
    y = gate * tl.sigmoid(gate) * up
    scale = tl.maximum(tl.max(tl.abs(y)), 1e-10) / 448.0
    q = tl.clamp(y / scale, -448.0, 448.0)
    tl.store(activation + row * HALF + offsets, y)
    tl.store(normalized + row * HALF + offsets, q)
    tl.store(quantized + row * HALF + offsets, q.to(quantized.dtype.element_ty))
    tl.store(scales + row * (HALF // GROUP) + group, scale)


@triton.jit
def division_trace_kernel(values, scales, out, COLS: tl.constexpr, GROUP: tl.constexpr):
    row = tl.program_id(0)
    group = tl.program_id(1)
    offsets = group * GROUP + tl.arange(0, GROUP)
    value = tl.load(values + row * COLS + offsets).to(tl.float32)
    scale = tl.load(scales + row * (COLS // GROUP) + group)
    tl.store(out + row * COLS + offsets, value / scale)


@triton.jit
def block_dot_trace_kernel(
    x, weights, out, ROWS: tl.constexpr, COLS: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    expert, tile, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m = tl.arange(0, 16)
    n = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    k = block * BLOCK_K + tl.arange(0, BLOCK_K)
    a = tl.load(x + k[None, :] + tl.zeros((16, 1), tl.int32))
    b = tl.load(weights + expert * ROWS * COLS + n[None, :] * COLS + k[:, None],
                mask=n[None, :] < ROWS, other=0.0)
    dot = tl.dot(a, b)
    ptr = out + expert * ROWS * (COLS // BLOCK_K) + n[None, :] * (COLS // BLOCK_K) + block
    tl.store(ptr + tl.zeros((16, 1), tl.int32), dot,
             mask=(m[:, None] == 0) & (n[None, :] < ROWS))


def difference(actual, expected):
    actual, expected = actual.float(), expected.float()
    delta = actual - expected
    return {
        "equal": bool(torch.equal(actual, expected)),
        "different_elements": int((actual != expected).sum()),
        "elements": actual.numel(),
        "relative_l2": float(delta.norm() / expected.norm().clamp_min(1e-30)),
        "max_abs": float(delta.abs().max()),
    }


def gpu_stages(backend, x, weights, staged):
    k = backend.kernel
    count, hidden = weights.numel(), x.shape[-1]
    gu_rows = staged.w13_weight.shape[1]
    ids = torch.arange(count, device=x.device).reshape(1, count)
    sorted_ids, expert_ids, padded = k.moe_align(ids, staged.count, 16)
    q, scales = k.quantize_groups(k.quant_kernel, x, hidden)
    gu = torch.empty((count, gu_rows), device=x.device, dtype=torch.bfloat16)
    gu32 = torch.empty_like(gu, dtype=torch.float32)
    for output in (gu, gu32):
        k.fused_moe_gemm(
            q, scales, staged.w13_weight, staged.w13_scale_inv, output, None,
            sorted_ids, expert_ids, padded, top_k=count, block_m=16, group_m=1,
        )
    hq, hs = k.quantize_groups(k.silu_mul_quant_kernel, gu, gu_rows // 2)
    activation = torch.empty_like(hq, dtype=torch.float32)
    normalized = torch.empty_like(activation)
    debug_hq, debug_hs = torch.empty_like(hq), torch.empty_like(hs)
    silu_trace_kernel[(count, gu_rows // 2 // 128)](
        gu, activation, normalized, debug_hq, debug_hs, HALF=gu_rows // 2,
        GROUP=128, num_warps=1,
    )
    input_normalized = torch.empty_like(x, dtype=torch.float32)
    division_trace_kernel[(1, hidden // 128)](
        x, scales, input_normalized, COLS=hidden, GROUP=128, num_warps=1,
    )
    down, partials = (
        torch.empty((count, hidden), device=x.device, dtype=torch.float32)
        for _ in range(2)
    )
    for output, route in ((down, None), (partials, weights)):
        k.fused_moe_gemm(
            hq, hs, staged.w2_weight, staged.w2_scale_inv, output, route,
            sorted_ids, expert_ids, padded, top_k=1, block_m=16, group_m=1,
        )
    block_dots = torch.empty(
        (count, gu_rows, hidden // 128), device=x.device, dtype=torch.float32)
    block_dot_trace_kernel[(count, triton.cdiv(gu_rows, 128), hidden // 128)](
        q, staged.w13_weight, block_dots, ROWS=gu_rows, COLS=hidden,
        BLOCK_N=128, BLOCK_K=128, num_warps=4,
    )
    return {
        "xq": q, "x_scales": scales, "input_normalized": input_normalized,
        "gate_up": gu32, "gate_up_bf16": gu, "activation": activation,
        "hidden_normalized": normalized, "hidden_q": hq, "hidden_scales": hs,
        "down": down, "partials": partials, "block_dots": block_dots,
        "trace_matches_official_quant": bool(torch.equal(hq.view(torch.uint8), debug_hq.view(torch.uint8))),
        "trace_matches_official_scales": bool(torch.equal(hs, debug_hs)),
    }


@torch.inference_mode()
def diagnose(args):
    data = torch.load(args.cases, map_location="cpu", weights_only=True)
    library, backend = NativeLibrary(), TritonExpertBackend()
    silu_table = build_gpu_silu_table() if args.gpu_silu_table else None
    rows = []
    artifacts = []
    source = ExpertCacheFile.open(ROOT / ".cache/drama3-fp8-experts")
    store = ExpertStore(**source.layer_buffers(data["layer"]))
    try:
        with NativeExpertExecutor(
            store, threads=1, arithmetic=args.arithmetic, silu_table=silu_table,
            implementation=args.implementation, library=library,
        ) as executor:
            for index, case in enumerate(data["cases"][:args.cases_count]):
                x = case["x"].contiguous()
                ids = case["ids"].reshape(-1).contiguous()
                weights = case["weights"].reshape(-1).float().contiguous()
                device = torch.device("cuda:0")
                pool = _StagingPool(ids.numel(), True, False, OffloadStatistics())
                try:
                    with pool.session(store, device):
                        staged = pool.stage(store, ids)
                        gpu = gpu_stages(backend, x.to(device), weights.to(device), staged)
                    gpu = {name: value.cpu() if isinstance(value, torch.Tensor) else value
                           for name, value in gpu.items()}
                finally:
                    pool.close()
                q, scales = gpu["xq"], gpu["x_scales"]
                native = executor.trace_quantized_pairs(q, scales, ids, weights)
                shared_gu = executor.trace_quantized_pairs(
                    q, scales, ids, weights, shared_gate_up=gpu["gate_up_bf16"],
                )
                shared_hidden = executor.trace_quantized_pairs(
                    q, scales, ids, weights, shared_hidden_q=gpu["hidden_q"],
                    shared_hidden_scales=gpu["hidden_scales"],
                )
                cpu_q, cpu_scales = library.quantize(x)
                cpu_normalized = x.float().reshape(-1, 128) / cpu_scales[:, None]
                q_from_gpu_division = gpu["input_normalized"].to(torch.float8_e4m3fn)
                gu_rounded = gpu["gate_up"].to(torch.bfloat16)
                cpu_hidden_q, cpu_hidden_scales = quantize_fp8_blocks(gpu["activation"])
                shared_gpu_y_native_div = {
                    "hidden_q": difference(cpu_hidden_q.view(torch.uint8), gpu["hidden_q"].view(torch.uint8)),
                    "hidden_scales": difference(cpu_hidden_scales, gpu["hidden_scales"].reshape(-1)),
                }
                stage_names = ("gate_up", "gate_up_bf16", "activation", "hidden_q",
                               "hidden_scales", "down", "partials")
                result = {
                    "case": index,
                    "input_quant": {
                        "raw_bytes": difference(cpu_q, gpu["xq"].view(torch.uint8)),
                        "scales": difference(cpu_scales, scales.reshape(-1)),
                        "division": difference(cpu_normalized, gpu["input_normalized"].reshape(-1, 128)),
                        "cpu_rounding_gpu_division": difference(q_from_gpu_division.view(torch.uint8), gpu["xq"].view(torch.uint8)),
                    },
                    "gpu_trace_quant_exact": gpu["trace_matches_official_quant"],
                    "gpu_trace_scales_exact": gpu["trace_matches_official_scales"],
                    "gpu_fp32_gu_cast_vs_direct_bf16": difference(gu_rounded, gpu["gate_up_bf16"]),
                    "shared_xq": {
                        name: difference(
                            native[name], gpu[name].view(torch.uint8) if name == "hidden_q" else gpu[name]
                        ) for name in stage_names
                    },
                    "shared_gu": {
                        name: difference(
                            shared_gu[name], gpu[name].view(torch.uint8) if name == "hidden_q" else gpu[name]
                        ) for name in ("activation", "hidden_q", "hidden_scales", "down", "partials")
                    },
                    "shared_gpu_activation_cpu_quant": shared_gpu_y_native_div,
                    "shared_hq": {name: difference(shared_hidden[name], gpu[name])
                                  for name in ("down", "partials")},
                    "gpu_sum_vs_captured": difference(
                        gpu["partials"].sum(0).bfloat16(), case["output"].reshape(-1)),
                    "native_trace_vs_forward": difference(
                        native["partials"], executor.forward_quantized_pairs(q, scales, ids, weights)),
                }
                reconstructed, exact_dots = [], []
                for route, expert in enumerate(ids.tolist()):
                    matrix = store.w13_weight[expert].float().reshape(
                        store.w13_weight.shape[1], store.hidden // 128, 128)
                    exact = (matrix * q.float().reshape(1, -1, 128)).sum(-1)
                    exact_dots.append(exact)
                    scaled = gpu["block_dots"][route] * scales.reshape(1, -1)
                    block_scales = store.w13_scale_inv[expert].repeat_interleave(128, 0)
                    acc = torch.zeros(matrix.shape[0])
                    for block in range(store.hidden // 128):
                        acc = (scaled[:, block].double() * block_scales[:, block].double() +
                               acc.double()).float()
                    reconstructed.append(acc)
                result["isolated_tensor_core_dot_vs_cpu"] = difference(
                    gpu["block_dots"], torch.stack(exact_dots))
                result["gpu_block_dot_cpu_scale_vs_gpu_gu"] = difference(
                    torch.stack(reconstructed), gpu["gate_up"])
                mismatch = torch.nonzero(cpu_q.reshape(-1) != gpu["xq"].view(torch.uint8).reshape(-1)).reshape(-1)
                result["input_boundary_examples"] = [
                    {"offset": int(n), "cpu_div": float(cpu_normalized.reshape(-1)[n]),
                     "gpu_div": float(gpu["input_normalized"].reshape(-1)[n]),
                     "cpu_byte": int(cpu_q.reshape(-1)[n]),
                     "gpu_byte": int(gpu["xq"].view(torch.uint8).reshape(-1)[n])}
                    for n in mismatch[:8]
                ]
                rows.append(result)
                artifacts.append({"gpu": gpu, "native": native, "shared_gu": shared_gu,
                                  "shared_hq": shared_hidden, "ids": ids, "weights": weights})
                print(json.dumps({
                    "case": index, "input_quant_mismatches": int(mismatch.numel()),
                    "gu_fp32_rel": result["shared_xq"]["gate_up"]["relative_l2"],
                    "gu_bf16_mismatches": result["shared_xq"]["gate_up_bf16"]["different_elements"],
                    "shared_xq_partial_rel": result["shared_xq"]["partials"]["relative_l2"],
                    "shared_gu_partial_rel": result["shared_gu"]["partials"]["relative_l2"],
                    "shared_hq_partial_rel": result["shared_hq"]["partials"]["relative_l2"],
                }), flush=True)
        args.tensors.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"source_layer": data["layer"], "cases": artifacts}, args.tensors)
        return {
            "scope": "Stage isolation on fixed real layer-zero MoE inputs; not speech quality.",
            "cases": rows, "tensor_artifact": str(args.tensors),
            "library_sha256": hashlib.sha256(library.path.read_bytes()).hexdigest(),
            "source_cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
            "default_engine_changed": False,
            "native_arithmetic": args.arithmetic,
            "gpu_silu_table": args.gpu_silu_table,
            "cpu_implementation": args.implementation,
        }
    finally:
        source.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "research/native-expert-cases.pt")
    parser.add_argument("--cases-count", type=int, default=11)
    parser.add_argument("--arithmetic", choices=("reference", "sm89"), default="reference")
    parser.add_argument("--implementation", choices=("auto", "scalar", "avx2", "avx512"), default="auto")
    parser.add_argument("--gpu-silu-table", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "research/native-stage-diagnosis.json")
    parser.add_argument("--tensors", type=Path, default=ROOT / "research/native-stage-tensors.pt")
    args = parser.parse_args()
    if args.cases_count < 1:
        parser.error("Positive case count required")
    torch.set_num_threads(1)
    result = diagnose(args)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
