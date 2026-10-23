"""Run a tiny CUDA/Triton kernel to verify the portable compiler and DLLs."""

import torch
import triton
import triton.language as tl


@triton.jit
def _add_one(source, target, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(source + index, index < N, 0)
    tl.store(target + index, value + 1, index < N)


def main():
    source = torch.arange(257, device="cuda", dtype=torch.float32)
    target = torch.empty_like(source)
    _add_one[(triton.cdiv(source.numel(), 128),)](source, target, N=source.numel(), BLOCK=128)
    torch.cuda.synchronize()
    assert torch.equal(target, source + 1)
    print("Portable CUDA/Triton compilation and execution passed:", torch.cuda.get_device_name(), flush=True)


if __name__ == "__main__":
    main()
