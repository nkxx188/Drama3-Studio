"""Compare bounded expert offload against Fish's kernel using real Drama 3 weights.

Only sixteen experts from the first layer are materialized. The full model,
tokenizer, tower and codec are not loaded, and no checkpoint is downloaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor" / "fish-speech"))

from drama_low_vram.offload import (  # noqa: E402
    EXPERT_BUFFER_NAMES,
    FP8_DTYPE,
    TritonExpertBackend,
    install_expert_offload,
)
from fish_speech.models.text2semantic.fish_omni import fp8_moe  # noqa: E402

EXPERT_IDS = (0, 1, 3, 7, 17, 23, 42, 55, 81, 99, 123, 151, 177, 201, 231, 255)


class CheckpointExperts(nn.Module):
    def __init__(self, weights: dict[str, Tensor]) -> None:
        super().__init__()
        self.fp8 = True
        for name, tensor in weights.items():
            self.register_buffer(name, tensor)

    def forward(self, x: Tensor, ids: Tensor, weights: Tensor) -> Tensor:
        raise AssertionError("Real expert subset must be wrapped by the offload manager")


class AuditedTritonBackend(TritonExpertBackend):
    def __init__(self) -> None:
        super().__init__()
        self.routed_pairs = 0

    def forward(self, x, ids, weights, staged):
        self.routed_pairs += ids.numel()
        return super().forward(x, ids, weights, staged)

    def forward_pairs(self, x, ids, weights, staged):
        self.routed_pairs += ids.numel()
        return super().forward_pairs(x, ids, weights, staged)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_expert_subset(checkpoint: Path) -> tuple[CheckpointExperts, dict]:
    """Copy exact FP8 bytes/scales from indexed tensors into the normal expert layout."""
    index_path = checkpoint / "model.safetensors.index.json"
    index_bytes = index_path.read_bytes()
    index = json.loads(index_bytes)
    weight_map = index["weight_map"]
    config_bytes = (checkpoint / "config.json").read_bytes()
    config = json.loads(config_bytes)
    text = config["text_config"]
    hidden = int(text["hidden_size"])
    intermediate = int(text["moe_intermediate_size"])
    count = len(EXPERT_IDS)
    buffers = {
        "w13_weight": torch.empty(count, intermediate * 2, hidden, dtype=FP8_DTYPE),
        "w13_scale_inv": torch.empty(
            count, intermediate * 2 // 128, hidden // 128, dtype=torch.float32
        ),
        "w2_weight": torch.empty(count, hidden, intermediate, dtype=FP8_DTYPE),
        "w2_scale_inv": torch.empty(
            count, hidden // 128, intermediate // 128, dtype=torch.float32
        ),
    }
    tensor_records = []
    subset_hash = hashlib.sha256()
    shard_names = set()
    with ExitStack() as stack:
        shards = {}
        for local, original in enumerate(EXPERT_IDS):
            prefix = f"model.language_model.layers.0.mlp.experts.{original}"
            for projection, target, half in (
                ("gate_proj", "w13", slice(0, intermediate)),
                ("up_proj", "w13", slice(intermediate, intermediate * 2)),
                ("down_proj", "w2", slice(None)),
            ):
                for suffix, target_suffix in (
                    ("weight", "weight"), ("weight_scale_inv", "scale_inv")
                ):
                    name = f"{prefix}.{projection}.{suffix}"
                    shard_name = weight_map[name]
                    shard_names.add(shard_name)
                    if shard_name not in shards:
                        shards[shard_name] = stack.enter_context(
                            safe_open(
                                str(checkpoint / shard_name), framework="pt", device="cpu"
                            )
                        )
                    tensor = shards[shard_name].get_tensor(name)
                    buffer = buffers[f"{target}_{target_suffix}"]
                    row_slice = half
                    if suffix == "weight_scale_inv" and target == "w13":
                        row_slice = (
                            slice(0, intermediate // 128)
                            if projection == "gate_proj"
                            else slice(intermediate // 128, intermediate * 2 // 128)
                        )
                    destination = buffer[local, row_slice]
                    if tensor.dtype != destination.dtype or tensor.shape != destination.shape:
                        raise ValueError(
                            f"{name}: checkpoint {tensor.shape}/{tensor.dtype} does not "
                            f"match {destination.shape}/{destination.dtype}"
                        )
                    destination.copy_(tensor)
                    raw_bytes = tensor.view(torch.uint8).numpy().tobytes()
                    tensor_hash = _sha256(raw_bytes)
                    subset_hash.update(name.encode("ascii"))
                    subset_hash.update(raw_bytes)
                    tensor_records.append(
                        {
                            "name": name,
                            "shard": shard_name,
                            "shape": list(tensor.shape),
                            "dtype": str(tensor.dtype),
                            "sha256": tensor_hash,
                        }
                    )
    verification_path = checkpoint / "download-verification.json"
    verification = (
        json.loads(verification_path.read_text(encoding="utf-8"))
        if verification_path.exists()
        else {}
    )
    recorded_files = {
        record["name"]: record
        for record in verification.get("files", [])
    }
    report = {
        "path": str(checkpoint.resolve()),
        "model": verification.get("model"),
        "source": verification.get("source"),
        "revision": verification.get("revision"),
        "equivalent_huggingface_weight_revision": verification.get(
            "equivalent_huggingface_weight_revision"
        ),
        "index_sha256": _sha256(index_bytes),
        "config_sha256": _sha256(config_bytes),
        "shard_sha256_from_prior_download_verification": {
            name: recorded_files.get(name, {}).get("sha256")
            for name in sorted(shard_names)
        },
        "full_shards_rehashed_this_run": False,
        "selected_expert_tensors_rehashed_this_run": True,
        "subset_sha256": subset_hash.hexdigest(),
        "layer": 0,
        "local_to_checkpoint_expert_ids": list(EXPERT_IDS),
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "subset_bytes": sum(t.numel() * t.element_size() for t in buffers.values()),
        "weight_dtype": str(FP8_DTYPE),
        "scale_dtype": str(torch.float32),
        "weight_requantization": False,
        "tensor_records": tensor_records,
    }
    return CheckpointExperts(buffers), report


def make_cases(hidden: int, seed: int, device: torch.device) -> list[dict]:
    generator = torch.Generator().manual_seed(seed)
    cases = []
    decode_routes = (
        ("decode_cold", [15, 3, 0, 8, 4, 10, 1, 6]),
        ("decode_repeated", [15, 3, 0, 8, 4, 10, 1, 6]),
        ("decode_partial_overlap", [15, 3, 0, 8, 5, 12, 9, 7]),
        ("decode_duplicate_pairs", [15, 3, 0, 0, 8, 4, 10, 1]),
    )
    for name, routes in decode_routes:
        ids = torch.tensor([routes], dtype=torch.int64)
        cases.append({"name": name, "ids": ids})
    for tokens in (8, 16):
        ids = torch.stack(
            [
                torch.randperm(len(EXPERT_IDS), generator=generator)[:8]
                for _ in range(tokens)
            ]
        )
        cases.append({"name": f"prefill_{tokens}", "ids": ids})
    for case in cases:
        tokens = case["ids"].shape[0]
        case["x"] = torch.randn(
            tokens, hidden, generator=generator, dtype=torch.bfloat16
        ).to(device)
        case["weights"] = torch.softmax(
            torch.randn(tokens, 8, generator=generator, dtype=torch.float32), dim=-1
        ).to(device=device, dtype=torch.bfloat16)
        case["ids"] = case["ids"].to(device)
    return cases


@torch.inference_mode()
def validate(checkpoint: Path, device: torch.device, seed: int) -> dict:
    if not fp8_moe.fp8_gemm_supported([device]):
        raise RuntimeError("Validation requires the official Triton FP8 kernel and SM89+")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    experts, checkpoint_report = load_expert_subset(checkpoint)
    cases = make_cases(checkpoint_report["hidden_size"], seed, device)
    resident = {
        name: getattr(experts, name).to(device) for name in EXPERT_BUFFER_NAMES
    }
    for case in cases:
        case["reference"] = fp8_moe.fp8_moe_forward(
            case["x"], case["ids"], case["weights"], **resident
        )
    configurations = []
    for pair_limit in (32, 3):
        for cache_capacity in (0, 2, 4, 8):
            backbone = nn.Module()
            backbone.add_module("experts", experts)
            backend = AuditedTritonBackend()
            manager = install_expert_offload(
                backbone,
                backend=backend,
                max_staged_experts=4,
                max_pair_rows=pair_limit,
                cache_experts=cache_capacity,
                pin_memory=True,
                profile_transfers=True,
            )
            checks = []
            try:
                manager.prepare_buffers(device)
                for case in cases:
                    before = manager.stats.as_dict()
                    pairs_before = backend.routed_pairs
                    actual = experts(case["x"], case["ids"], case["weights"])
                    manager.synchronize()
                    expected = case["reference"]
                    delta = (actual.float() - expected.float()).abs()
                    retained_pairs = backend.routed_pairs - pairs_before
                    bitwise_equal = torch.equal(actual, expected)
                    finite = bool(torch.isfinite(actual).all())
                    all_pairs = retained_pairs == case["ids"].numel()
                    checks.append(
                        {
                            "case": case["name"],
                            "tokens": case["x"].shape[0],
                            "top_k": case["ids"].shape[1],
                            "expected_route_pairs": case["ids"].numel(),
                            "evaluated_route_pairs": retained_pairs,
                            "all_routes_retained": all_pairs,
                            "weight_dtype": str(FP8_DTYPE),
                            "input_dtype": str(case["x"].dtype),
                            "routing_weight_dtype": str(case["weights"].dtype),
                            "output_dtype": str(actual.dtype),
                            "bitwise_equal_to_official_fp8": bitwise_equal,
                            "max_abs_difference": delta.max().item(),
                            "mean_abs_difference": delta.mean().item(),
                            "finite": finite,
                            "passed": bitwise_equal and finite and all_pairs,
                            "stats_delta": {
                                name: value - before[name]
                                for name, value in manager.stats.as_dict().items()
                            },
                        }
                    )
                configurations.append(
                    {
                        "cache_experts": cache_capacity,
                        "max_staged_experts": 4,
                        "max_pair_rows": pair_limit,
                        "checks": checks,
                        "passed": all(check["passed"] for check in checks),
                        "statistics": manager.stats.as_dict(),
                    }
                )
            finally:
                manager.close()
    torch.cuda.synchronize(device)
    props = torch.cuda.get_device_properties(device)
    return {
        "schema_version": 1,
        "kind": "official_checkpoint_expert_fp8_parity",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": props.name,
        "device": str(device),
        "compute_capability": [props.major, props.minor],
        "seed": seed,
        "reference": "Unmodified Fish fp8_moe_forward with selected experts resident on GPU",
        "checkpoint": checkpoint_report,
        "configurations": configurations,
        "tested_check_count": sum(len(config["checks"]) for config in configurations),
        "passed": all(config["passed"] for config in configurations),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "full_model_or_audio_quality_verified": False,
        "notes": [
            "Actual first-layer expert weights/scales are preserved byte-for-byte.",
            "Local expert IDs only remap the selected checkpoint experts.",
            "Every original top-8 route is computed, including duplicate route pairs.",
            "Parity covers this expert subset and input set, not whole-model audio quality.",
            "Full shard SHA values are prior download records, not hashes recomputed here.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "checkpoints" / "drama-3"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "research" / "official-fp8-parity.json"
    )
    args = parser.parse_args()
    report = validate(args.checkpoint, torch.device(args.device), args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "checks": report["tested_check_count"],
                "subset_bytes": report["checkpoint"]["subset_bytes"],
                "elapsed_seconds": report["elapsed_seconds"],
                "peak_vram_allocated_bytes": report["peak_vram_allocated_bytes"],
                "output": str(args.output.resolve()),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
