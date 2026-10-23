"""Bounded GPU microbenchmarks at Drama3's actual expert matrix dimensions."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor/fish-speech")]

import torch
import triton.testing
from drama_low_vram.quantized_kernels import projection


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    torch.manual_seed(318)
    rows = []
    for mode in range(3):
        for name, n, k, shared in (("gate-up", 1024, 2048, True),
                                    ("down", 2048, 512, False)):
            for pairs in (8, 64):
                bank = torch.randint(-127, 127, (
                    8, n, k // 8 if mode == 0 else k if mode == 1 else k * 3 // 4),
                    device="cuda", dtype=torch.int32 if mode == 0 else torch.int8)
                if mode == 0:
                    scales = torch.full((8, n, k // 128), 0.02, device="cuda", dtype=torch.float16)
                elif mode == 1:
                    scales = torch.full((8, n, 1), 0.02, device="cuda")
                else:
                    relative = torch.ones((8, n, k // 16), device="cuda").to(
                        torch.float8_e4m3fn).view(torch.uint8)
                    channel = torch.full((8, n, 1), 0.02, device="cuda")
                    scales = torch.cat((relative, channel.view(torch.uint8)), -1)
                x = torch.randn(1 if shared else pairs, k, device="cuda", dtype=torch.bfloat16)
                ids = torch.arange(pairs, device="cuda") % 8
                weights = torch.full((pairs,), 0.125, device="cuda")
                args = (x, ids, weights, bank, scales)
                kwargs = dict(mode=mode, shared=shared, weighted=not shared)
                reference = projection(*args, **kwargs)
                candidates = []
                for block_n in (32, 64, 128):
                    for block_k in (64, 128, 256):
                        tile = block_n, block_k
                        try:
                            actual = projection(*args, **kwargs, tile=tile)
                        except triton.runtime.errors.OutOfResources as exc:
                            candidates.append(dict(tile=tile, error=str(exc), close=False))
                            continue
                        error = float((actual.float() - reference.float()).abs().max())
                        close = bool(torch.allclose(actual.float(), reference.float(),
                                                    atol=0.02, rtol=0.01))
                        milliseconds = triton.testing.do_bench(
                            lambda: projection(*args, **kwargs, tile=tile),
                            warmup=10, rep=50, return_mode="median")
                        candidates.append(dict(tile=tile, milliseconds=milliseconds,
                                               close=close, max_absolute_error=error))
                row = dict(mode=mode, projection=name, pairs=pairs, candidates=candidates)
                rows.append(row)
                best = min((c for c in candidates if c["close"]), key=lambda c: c["milliseconds"])
                print(json.dumps(dict(mode=mode, projection=name, pairs=pairs, best=best)), flush=True)
                (ROOT / "research/quantized-kernel-tuning.json").write_text(
                    json.dumps({"scope": "Synthetic weights at real matrix shapes; includes "
                                "rotation, activation quantization, allocation and GEMM. "
                                "This is not complete-request speed.", "rows": rows},
                               ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
