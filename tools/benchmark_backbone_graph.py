"""Benchmark one real Drama 3 decoder layer against eager execution.

Only layer zero is materialized. Routed FP8 bytes stay in host RAM and are
staged through the normal expert offload manager, so this does not load the
30-billion-byte full backbone.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor" / "fish-speech"))
os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / ".cache" / "triton"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / ".cache" / "torchinductor"))

from drama_low_vram.backbone_graph import BackboneDecodeGraph  # noqa: E402
from drama_low_vram.offload import install_expert_offload  # noqa: E402
from fish_speech.models.text2semantic.fish_omni.backbone import (  # noqa: E402
    FishOmniBackbone,
    load_backbone_weights,
)
from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig  # noqa: E402


def _load_one_layer(checkpoint: Path, device: torch.device) -> FishOmniBackbone:
    config = FishOmniConfig.from_dict(
        json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    ).text
    with torch.device("meta"):
        backbone = FishOmniBackbone(config, fp8_experts=True, dtype=torch.bfloat16)
    # Keep just one decoder layer; use the real layer shape and expert buffers.
    backbone.model.layers = torch.nn.ModuleList([backbone.model.layers[0]])
    backbone.model.embed_tokens = torch.nn.Identity()
    backbone.lm_head = torch.nn.Identity()
    expert = backbone.model.layers[0].mlp.experts
    for module in backbone.modules():
        if list(module.parameters(recurse=False)) or list(module.buffers(recurse=False)):
            target = torch.device("cpu") if module is expert else device
            module.to_empty(device=target, recurse=False)
    index = json.loads(
        (checkpoint / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    names = [
        name for name in index
        if name.startswith("model.language_model.layers.0.")
        or name == "model.language_model.norm.weight"
    ]

    def weights():
        with ExitStack() as stack:
            readers = {}
            for name in names:
                shard = index[name]
                if shard not in readers:
                    readers[shard] = stack.enter_context(
                        safe_open(str(checkpoint / shard), framework="pt", device="cpu")
                    )
                yield name, readers[shard].get_tensor(name)

    load_backbone_weights(
        backbone,
        weights(),
    )
    return backbone.eval()


@torch.no_grad()
def benchmark(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    checkpoint = args.checkpoint.resolve()
    eager = _load_one_layer(checkpoint, device)
    graph_model = copy.deepcopy(eager)
    for model in (eager, graph_model):
        model.setup_caches(1, args.capacity)
    eager_manager = install_expert_offload(
        eager, backend=args.backend, max_staged_experts=args.staged, max_pair_rows=args.pairs
    )
    graph_manager = install_expert_offload(
        graph_model, backend=args.backend, max_staged_experts=args.staged, max_pair_rows=args.pairs
    )
    eager_manager.prepare_buffers(device)
    graph_manager.prepare_buffers(device)
    graph = BackboneDecodeGraph(
        graph_model, graph_manager, sequence_capacity=args.capacity, device=device, strict=True
    )
    torch.manual_seed(args.seed)
    if args.hidden_tensor is None:
        inputs = torch.randn(
            args.rows, eager.config.hidden_size, dtype=torch.bfloat16, device=device
        )
        hidden_source = "seeded_random"
    else:
        inputs = torch.load(args.hidden_tensor, weights_only=True, map_location="cpu")
        if isinstance(inputs, dict):
            inputs = inputs["hidden"]
        inputs = inputs.reshape(-1, eager.config.hidden_size).to(device)
        if len(inputs) < args.rows:
            raise ValueError("Captured hidden tensor has fewer rows than --rows")
        inputs = inputs[: args.rows].to(torch.bfloat16)
        hidden_source = str(args.hidden_tensor.resolve())
    if len(inputs) >= args.capacity:
        raise ValueError("--capacity must exceed the number of teacher-forced inputs")
    eager_records = []
    original_expert_forward = eager.model.layers[0].mlp.experts.forward

    def recorded_expert_forward(x, ids, weights):
        eager_records.append(
            {"input": x.clone(), "ids": ids.clone(), "weights": weights.clone()}
        )
        return original_expert_forward(x, ids, weights)

    eager.model.layers[0].mlp.experts.forward = recorded_expert_forward
    parity = []
    torch.cuda.synchronize(device)
    for position, row in enumerate(inputs, start=1):
        hidden = row.view(1, 1, -1)
        eager_out = eager.run_layers(
            hidden,
            torch.tensor([position], device=device),
            kv_len=position + 1,
            continue_state=True,
            use_cache=True,
        )
        graph_out = graph(hidden, position)
        diff = (eager_out.detach() - graph_out.detach()).float().abs()
        record = eager_records[-1]
        segment = graph._layers[0]
        parity.append(
            {
                "position": position,
                "bitwise_equal": bool(torch.equal(eager_out, graph_out)),
                "max_abs": float(diff.max()),
                "mean_abs": float(diff.mean()),
                "finite": bool(torch.isfinite(graph_out).all()),
                "top_k_ids_equal": bool(torch.equal(record["ids"], segment.topk_ids)),
                "top_k_weights_equal": bool(torch.equal(record["weights"], segment.topk_weights)),
                "expert_input_equal": bool(
                    torch.equal(record["input"], segment.mlp_input.reshape(1, -1))
                ),
            }
        )
    eager.model.layers[0].mlp.experts.forward = original_expert_forward
    state_snapshot = {}
    for label, model in (("eager", eager), ("graph", graph_model)):
        state = model.model.layers[0].linear_attn
        state_snapshot[label] = {
            name: getattr(state, name).clone()
            for name in ("conv_state", "recurrent_state")
        }
    timings = {}
    position = args.rows
    hidden = inputs[-1:].view(1, 1, -1)
    for label, fn in (
        ("eager", lambda: eager.run_layers(
            hidden, torch.tensor([position], device=device), kv_len=position + 1,
            continue_state=True, use_cache=True
        )),
        ("graph", lambda: graph(hidden, position)),
    ):
        # Resetting caches is outside timed region; this is a short steady replay.
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(args.iterations):
            fn()
        torch.cuda.synchronize(device)
        timings[label] = (time.perf_counter() - start) / args.iterations
    result = {
        "checkpoint": str(checkpoint),
        "device": str(device),
        "backend": args.backend,
        "layer_count": 1,
        "hidden_size": eager.config.hidden_size,
        "num_experts": eager.config.num_experts,
        "top_k": eager.config.num_experts_per_tok,
        "description": graph.description,
        "hidden_source": hidden_source,
        "parity": {
            "all_bitwise_equal": all(record["bitwise_equal"] for record in parity),
            "all_top_k_ids_equal": all(record["top_k_ids_equal"] for record in parity),
            "all_top_k_weights_equal": all(record["top_k_weights_equal"] for record in parity),
            "all_expert_inputs_equal": all(record["expert_input_equal"] for record in parity),
            "max_abs": max(record["max_abs"] for record in parity),
            "finite": all(record["finite"] for record in parity),
            "steps": parity,
            "conv_state_equal": bool(torch.equal(
                state_snapshot["eager"]["conv_state"], state_snapshot["graph"]["conv_state"]
            )),
            "recurrent_state_equal": bool(torch.equal(
                state_snapshot["eager"]["recurrent_state"],
                state_snapshot["graph"]["recurrent_state"],
            )),
        },
        "timings_seconds_per_call": timings,
        "speedup": timings["eager"] / max(timings["graph"], 1e-12),
        "iterations": args.iterations,
        "note": "One-layer component benchmark; expert transfer and dynamic staging are included.",
    }
    eager_manager.close()
    graph_manager.close()
    graph.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "drama-3")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", choices=("torch", "triton"), default="triton")
    parser.add_argument("--capacity", type=int, default=8)
    parser.add_argument("--staged", type=int, default=4)
    parser.add_argument("--pairs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--seed", type=int, default=411)
    parser.add_argument("--hidden-tensor", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.capacity < 2 or args.iterations < 1:
        parser.error("--capacity must be >=2 and --iterations >=1")
    result = benchmark(args)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
