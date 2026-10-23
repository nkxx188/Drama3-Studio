"""Probe raw SM89 FP8 dot behavior before choosing a CPU arithmetic contract."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor/fish-speech"), str(ROOT / "tools")]
for key, name in (("TRITON_CACHE_DIR", "triton"), ("TEMP", "tmp"), ("TMP", "tmp")):
    path = ROOT / ".cache" / name
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)

import torch
import triton
import triton.language as tl


def emulate_sm89_dot(x, weights):
    """Empirical SM89 grouping; a validation model, not a portable ISA promise."""
    x, weights = x.float(), weights.float()
    products = weights * x
    operand_exponents = (
        weights.abs().clamp_min(1e-30).log2().floor()
        + x.abs().clamp_min(1e-30).log2().floor()
    )
    acc = torch.zeros(weights.shape[:-1])
    for start in range(0, weights.shape[-1], 16):
        exponent = torch.maximum(
            operand_exponents[..., start:start + 16].amax(-1),
            acc.abs().clamp_min(1e-30).log2().floor())
        step = 2. ** (exponent - 13)
        acc = (
            (acc / step).trunc() * step
            + (products[..., start:start + 16] / step[..., None]).trunc().sum(-1) * step
        )
        acc = (acc.view(torch.int32) & -1024).view(torch.float32)
    return acc


@triton.jit
def raw_dots(x, weights, out, ROWS: tl.constexpr, K: tl.constexpr,
             IMPRECISE: tl.constexpr):
    n = tl.program_id(0) * 128 + tl.arange(0, 128)
    m = tl.arange(0, 16)
    k = tl.arange(0, K)
    a = tl.load(x + k[None, :] + tl.zeros((16, 1), tl.int32))
    b = tl.load(weights + n[None, :] * K + k[:, None],
                mask=n[None, :] < ROWS, other=0.)
    if IMPRECISE < 0:
        acc = tl.dot(a, b)
    else:
        acc = tl.dot(a, b, max_num_imprecise_acc=IMPRECISE)
    tl.store(out + n[None, :] + tl.zeros((16, 1), tl.int32), acc,
             mask=(m[:, None] == 0) & (n[None, :] < ROWS))


def metrics(actual, expected):
    d = actual.float() - expected.float()
    return {"different": int((actual != expected).sum()), "elements": actual.numel(),
            "relative_l2": float(d.norm() / expected.float().norm().clamp_min(1e-30)),
            "max_abs": float(d.abs().max())}


@torch.inference_mode()
def run():
    g = torch.Generator().manual_seed(92834)
    rows, k = 256, 128
    scenarios = []
    finite = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    x = torch.ones(k).to(torch.float8_e4m3fn)
    w = torch.zeros(rows, k)
    w[:127, 0] = finite
    w[127:254, 0] = -finite
    scenarios.append(("one_product", x, w.to(torch.float8_e4m3fn)))
    small = torch.cat((finite, -finite, torch.zeros(2)))
    for second_index in (1, 32):
        for sign in (1, -1):
            w = torch.zeros(rows, k)
            w[:, 0] = sign * 256
            w[:, second_index] = small
            scenarios.append((
                f"two_products_{second_index}_{sign}", x, w.to(torch.float8_e4m3fn)))
    for spread in (0, 2, 8, 14):
        exponent = torch.randint(-spread, spread + 1, (rows, k), generator=g)
        w = torch.ldexp(torch.randn(rows, k, generator=g), exponent).clamp(-448, 448).to(torch.float8_e4m3fn)
        x = torch.randn(k, generator=g).mul(80).clamp(-448, 448).to(torch.float8_e4m3fn)
        scenarios.append((f"random_exponent_{spread}", x, w))
    results, artifacts = [], []
    for name, x, w in scenarios:
        ref = (w.double() * x.double()).sum(-1).float()
        xg, wg = x.cuda(), w.cuda()
        outputs = {}
        tensors = {}
        for precision in (-1, 0, 32):
            out = torch.empty(rows, device="cuda", dtype=torch.float32)
            try:
                kernel = raw_dots[(triton.cdiv(rows, 128),)](
                    xg, wg, out, ROWS=rows, K=k, IMPRECISE=precision, num_warps=4)
                value = out.cpu()
                tensors[str(precision)] = value
                outputs[str(precision)] = {"error": metrics(value, ref),
                                           "sm89_emulation_error": metrics(emulate_sm89_dot(x, w), value),
                                           "first_values": value[:8].tolist()}
            except (RuntimeError, triton.CompilationError) as error:
                outputs[str(precision)] = {"exception": str(error)[:1000]}
        result = {"scenario": name, "results": outputs}
        results.append(result)
        artifacts.append({"scenario": name, "x": x, "weight": w,
                          "reference": ref, "gpu": tensors})
        print(json.dumps(result), flush=True)
    return {"device": torch.cuda.get_device_name(), "torch": torch.__version__,
            "scope": "Raw FP8 dots, FP64 exact-product reference; no scaled expert or model.",
            "results": results}, artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "research/fp8-dot-contract-probe.json")
    parser.add_argument("--tensors", type=Path, default=ROOT / "research/fp8-dot-contract-probe.pt")
    args = parser.parse_args()
    torch.set_num_threads(1)
    report, artifacts = run()
    torch.save(artifacts, args.tensors)
    report["tensor_artifact"] = str(args.tensors)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
