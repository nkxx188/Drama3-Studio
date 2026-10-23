"""Small capability policy shared by Python and native Windows entry points."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class NvidiaPolicy:
    capability: tuple[int, int]
    supported: bool
    generation: str
    compute_dtype: str
    fp8_tensor_cores: bool
    experimental_hybrid: bool
    experimental_pipeline: bool
    note: str

    def report(self):
        return asdict(self)


def policy_for(capability):
    cc = tuple(int(v) for v in capability)
    value = cc[0] * 10 + cc[1]
    supported = value >= 75
    generation = {75: "RTX20/Turing", 86: "RTX30/Ampere", 89: "RTX40/Ada",
                  120: "RTX50/Blackwell"}.get(value, "CUDA")
    return NvidiaPolicy(
        cc, supported, generation, "float16" if value < 80 else "bfloat16",
        value >= 89, value == 89, value == 89,
        "Generic staged compute; model storage precision and all routed experts are preserved."
        if value < 89 else
        "Use supported native GPU kernels; architecture-specific experimental paths remain gated.",
    )


def uses_quantized_torch_fallback(capability, mode):
    cc = tuple(capability)
    return cc < (8, 0) or (cc < (8, 9) and mode in (2, 3))
