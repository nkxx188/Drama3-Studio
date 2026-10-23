"""Verify the native CUDA and Fish FP8 kernels with small synthetic inputs."""

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
import triton
import triton.language as tl
from torch.nn import functional as F


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, count, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < count
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def device_info(device):
    props = torch.cuda.get_device_properties(device)
    driver = None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
                "-i",
                str(device.index),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        driver = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "device": str(device),
        "name": props.name,
        "compute_capability": [props.major, props.minor],
        "total_vram_bytes": props.total_memory,
        "driver_version": driver,
        "bf16_supported": torch.cuda.is_bf16_supported(),
    }


def check_torch(device):
    a = torch.randn((128, 128), device=device, dtype=torch.bfloat16)
    b = torch.randn((128, 128), device=device, dtype=torch.bfloat16)
    actual = a @ b
    expected = (a.float() @ b.float()).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0.015, atol=0.08)
    torch.cuda.synchronize(device)
    return {
        "passed": True,
        "dtype": str(actual.dtype),
        "finite": bool(torch.isfinite(actual).all()),
        "max_abs_error": (actual.float() - expected.float()).abs().max().item(),
    }


def check_triton(device):
    x = torch.randn(4099, device=device)
    y = torch.randn_like(x)
    actual = torch.empty_like(x)
    started = time.perf_counter()
    add_kernel[(triton.cdiv(x.numel(), 256),)](
        x, y, actual, x.numel(), BLOCK_SIZE=256
    )
    torch.cuda.synchronize(device)
    torch.testing.assert_close(actual, x + y, rtol=0.0, atol=0.0)
    return {
        "passed": True,
        "first_compile_and_launch_seconds": time.perf_counter() - started,
        "max_abs_error": (actual - (x + y)).abs().max().item(),
    }


def load_fish_fp8(path):
    spec = importlib.util.spec_from_file_location("drama3_fp8_smoke", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Fish FP8 module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def quantize_reference(x, group=128):
    rows, width = x.shape
    if width % group:
        raise ValueError("Reference expects widths divisible by the FP8 block size")
    blocks = x.float().reshape(rows, width // group, group)
    scales = blocks.abs().amax(dim=-1).clamp_min(1e-10) / 448.0
    q = (blocks / scales.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return (q.float() * scales.unsqueeze(-1)).reshape(rows, width)


def dequantize_reference(weight, scales):
    return weight.float() * scales.repeat_interleave(128, dim=1).repeat_interleave(
        128, dim=2
    )


def reference_moe(x, ids, route, w13, s13, w2, s2):
    # Match W8A8 activation quantization and the BF16 gate/up intermediate.
    x_dequant = quantize_reference(x)
    w13_dequant = dequantize_reference(w13, s13)
    w2_dequant = dequantize_reference(w2, s2)
    output = torch.zeros_like(x, dtype=torch.float32)
    intermediate = w2.shape[-1]
    for expert in range(w13.shape[0]):
        token, slot = torch.where(ids == expert)
        if not token.numel():
            continue
        gate_up = F.linear(x_dequant[token], w13_dequant[expert]).to(x.dtype)
        hidden = F.silu(gate_up[:, :intermediate].float()) * gate_up[
            :, intermediate:
        ].float()
        hidden_dequant = quantize_reference(hidden)
        down = F.linear(hidden_dequant, w2_dequant[expert])
        output.index_add_(0, token, down * route[token, slot, None])
    return output.to(x.dtype)


def check_fish_fp8(device, path, tokens):
    fish = load_fish_fp8(path)
    if not fish.fp8_gemm_supported([device]):
        raise RuntimeError("Fish requires native SM89+ CUDA with Triton")
    # Exact Drama 3 expert dimensions, with only eight resident experts.
    experts, hidden, intermediate, top_k = 8, 2048, 512, 8
    x = torch.randn((tokens, hidden), device=device, dtype=torch.bfloat16)
    ids = torch.stack(
        [torch.randperm(experts, device=device)[:top_k] for _ in range(tokens)]
    )
    route = torch.softmax(torch.randn((tokens, top_k), device=device), dim=-1)
    w13 = (
        torch.randn((experts, intermediate * 2, hidden), device=device)
        .mul_(64)
        .clamp_(-448, 448)
        .to(torch.float8_e4m3fn)
    )
    s13 = torch.full(
        (experts, intermediate * 2 // 128, hidden // 128),
        0.00035,
        device=device,
    )
    w2 = (
        torch.randn((experts, hidden, intermediate), device=device)
        .mul_(64)
        .clamp_(-448, 448)
        .to(torch.float8_e4m3fn)
    )
    s2 = torch.full(
        (experts, hidden // 128, intermediate // 128), 0.00035, device=device
    )
    args = (x, ids, route, w13, s13, w2, s2)
    started = time.perf_counter()
    actual = fish.fp8_moe_forward(*args)
    torch.cuda.synchronize(device)
    first_seconds = time.perf_counter() - started
    reference = reference_moe(*args)
    delta = actual.float() - reference.float()
    relative_rms = (
        delta.square().mean().sqrt()
        / reference.float().square().mean().sqrt().clamp_min(1e-10)
    ).item()
    max_abs = delta.abs().max().item()
    if not bool(torch.isfinite(actual).all()):
        raise AssertionError("Fish FP8 kernel produced nonfinite outputs")
    if relative_rms > 0.03:
        raise AssertionError(f"W8A8 reference relative RMS error {relative_rms:.6g}")
    # An independent PyTorch reference may differ slightly at FP8 rounding edges.
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    repeats = 10
    start_event.record()
    for _ in range(repeats):
        fish.fp8_moe_forward(*args)
    end_event.record()
    end_event.synchronize()
    return {
        "passed": True,
        "source": str(path.resolve()),
        "tokens": tokens,
        "resident_experts": experts,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "top_k": top_k,
        "weight_dtype": str(w13.dtype),
        "output_dtype": str(actual.dtype),
        "first_compile_and_launch_seconds": first_seconds,
        "mean_gpu_milliseconds": start_event.elapsed_time(end_event) / repeats,
        "relative_rms_error": relative_rms,
        "max_abs_error": max_abs,
        "reference": "Independent PyTorch W8A8, with BF16 gate/up intermediate",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--fish-source",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "vendor/fish-speech/fish_speech/models/text2semantic/fish_omni/fp8_moe.py",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "cuda_available": torch.cuda.is_available(),
        "full_model_or_audio_verified": False,
        "checks": {},
    }
    if not torch.cuda.is_available():
        report["error"] = "No CUDA device is available"
    else:
        device = torch.device(f"cuda:{args.device}")
        torch.cuda.set_device(device)
        torch.manual_seed(20261014)
        torch.cuda.reset_peak_memory_stats(device)
        report["gpu"] = device_info(device)
        checks = [
            ("torch_bf16", lambda: check_torch(device)),
            ("triton_jit", lambda: check_triton(device)),
            (
                "fish_fp8_decode",
                lambda: check_fish_fp8(device, args.fish_source, 1),
            ),
            (
                "fish_fp8_prefill",
                lambda: check_fish_fp8(device, args.fish_source, 16),
            ),
        ]
        with torch.inference_mode():
            for name, check in checks:
                try:
                    report["checks"][name] = check()
                except Exception as exc:
                    report["checks"][name] = {
                        "passed": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
        report["peak_vram_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    report["passed"] = bool(report["checks"]) and all(
        item["passed"] for item in report["checks"].values()
    )
    rendered = json.dumps(report, ensure_ascii=True, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered, flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
