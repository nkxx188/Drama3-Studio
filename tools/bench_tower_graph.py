"""Measure tower-only CUDA graphs; this is not an end-to-end speech benchmark.

The default is a tiny official tower. ``--checkpoint`` loads only the checkpoint
audio tower (about 2.28 GiB on the official BF16 checkpoint), never the backbone.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for source in (ROOT / "src", ROOT / "vendor" / "fish-speech"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
for variable, directory in (
    ("TRITON_CACHE_DIR", "triton"),
    ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
    ("TORCH_EXTENSIONS_DIR", "torch_extensions"),
    ("TEMP", "tmp"),
    ("TMP", "tmp"),
):
    os.environ.setdefault(variable, str(ROOT / ".cache" / directory))
(ROOT / ".cache" / "tmp").mkdir(parents=True, exist_ok=True)

import torch  # noqa: E402
from torch import Tensor  # noqa: E402

from drama_low_vram.tower_graph import TowerDecodeGraph  # noqa: E402
from fish_speech.models.text2semantic.fish_omni.backbone import (  # noqa: E402
    CHECKPOINT_BACKBONE_PREFIX,
    CHECKPOINT_LM_HEAD_KEY,
)
from fish_speech.models.text2semantic.fish_omni.config import (  # noqa: E402
    AudioDecoderConfig,
    FishOmniConfig,
)
from fish_speech.models.text2semantic.fish_omni.model import (  # noqa: E402
    iter_checkpoint_tensors,
)
from fish_speech.models.text2semantic.fish_omni.tower import (  # noqa: E402
    FishOmniAudioDecoder,
    load_audio_decoder_weights,
)


def load_tower(checkpoint: Path | None, device: torch.device):
    if checkpoint is None:
        config = SimpleNamespace(
            audio_decoder=AudioDecoderConfig(
                dim=32,
                intermediate_size=64,
                n_head=4,
                n_local_heads=2,
                head_dim=8,
                norm_eps=1e-6,
                rope_base=10000,
                rotary_percent=0.5,
                vocab_size=16,
                num_codebooks=4,
                shared_enrichment_layers=2,
                vq_backbone_codebooks=2,
            ),
            semantic_start_token_id=7,
            semantic_end_token_id=22,
            eos_token_id=3,
        )
        tower = FishOmniAudioDecoder(
            config, dtype=torch.bfloat16, device=device
        ).eval()
        return tower, 30
    checkpoint = checkpoint.resolve()
    config = FishOmniConfig.from_dict(
        json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    )
    with torch.device("meta"):
        tower = FishOmniAudioDecoder(config, dtype=torch.bfloat16)
    tower.to_empty(device=device)
    load_audio_decoder_weights(
        tower,
        iter_checkpoint_tensors(
            checkpoint,
            skip_prefixes=(CHECKPOINT_BACKBONE_PREFIX, CHECKPOINT_LM_HEAD_KEY),
        ),
    )
    return tower.eval(), config.text.vocab_size


def clear_cache(tower: FishOmniAudioDecoder) -> None:
    for layer in (*tower.shared_layers, *tower.residual_layers):
        attention = layer.transformer_layer.attention
        attention.k_cache.zero_()
        attention.v_cache.zero_()


def seed_cache(tower: FishOmniAudioDecoder) -> None:
    """Give every KV row finite nonzero values so parity exercises attended history."""
    generator = torch.Generator(device=tower.device).manual_seed(791)
    for layer in (*tower.shared_layers, *tower.residual_layers):
        attention = layer.transformer_layer.attention
        attention.k_cache.normal_(generator=generator)
        attention.v_cache.normal_(generator=generator)


def make_inputs(
    tower: FishOmniAudioDecoder,
    vocab_size: int,
    position: int,
    hidden_rows: Tensor | None = None,
):
    device = tower.device
    hidden = (
        torch.randn(
            1, tower.config.dim, dtype=tower.codebook_embeddings.weight.dtype, device=device
        )
        if hidden_rows is None
        else hidden_rows[position % len(hidden_rows) : position % len(hidden_rows) + 1]
    )
    logits = torch.zeros(
        1, vocab_size, dtype=tower.codebook_embeddings.weight.dtype, device=device
    )
    # A definite main token avoids RAS divergence when comparing numerical paths.
    token = tower.semantic_begin + position % tower.config.vocab_size
    logits[:, token] = 100
    frame = torch.zeros(
        1, 1 + tower.config.num_codebooks, dtype=torch.long, device=device
    )
    frame[:, 0] = token
    frame[:, 1:] = torch.randint(
        0, tower.config.vocab_size, frame[:, 1:].shape, device=device
    )
    frame[:, 1] = token - tower.semantic_begin
    return logits, hidden, frame


@torch.inference_mode()
def check_parity(
    tower: FishOmniAudioDecoder,
    graph: TowerDecodeGraph,
    vocab_size: int,
    hidden_rows: Tensor | None = None,
):
    checks = []
    positions = sorted(
        {0, min(graph.capacity - 1, 17), *(length - 1 for length in graph.buckets)}
    )
    for position in positions:
        seed_cache(tower)
        logits, hidden, frame = make_inputs(tower, vocab_size, position, hidden_rows)
        expected_heads = []
        handles = [
            head.register_forward_hook(
                lambda _module, _inputs, output: expected_heads.append(output.detach().clone())
            )
            for head in tower.codebook_heads
        ]
        try:
            expected = tower.decode_step(
                logits,
                hidden,
                torch.tensor([position], device=tower.device),
                frame,
                torch.tensor([-1], device=tower.device),
                temperature=0.8,
                top_k=1,
                top_p=1.0,
            )
        finally:
            for handle in handles:
                handle.remove()
        seed_cache(tower)
        # A distinct prompt main token prevents RAS from overriding greedy input.
        prompt_frame = frame.clone()
        prompt_frame[:, 0] = tower.eos_id
        graph.start(prompt_frame, 0.8, 1, 1.0)
        actual, _ = graph.step(logits, hidden, position, frame)
        eager_heads = torch.cat(expected_heads).float()
        actual_heads = graph.codebook_logits.float()
        difference = (actual_heads - eager_heads).abs()
        checks.append(
            {
                "position": position,
                "frames_equal": bool(torch.equal(expected, actual)),
                "head_logits_equal": bool(torch.equal(eager_heads, actual_heads)),
                "max_abs_difference": float(difference.max()),
                "relative_l2_difference": float(
                    torch.linalg.vector_norm(difference)
                    / torch.linalg.vector_norm(eager_heads).clamp_min(1e-12)
                ),
            }
        )
    return checks


@torch.inference_mode()
def measure(
    tower: FishOmniAudioDecoder,
    graph: TowerDecodeGraph,
    vocab_size: int,
    steps: int,
    *,
    use_graph: bool,
    hidden_rows: Tensor | None = None,
) -> dict[str, Any]:
    clear_cache(tower)
    logits, hidden, frame = make_inputs(tower, vocab_size, 0, hidden_rows)
    history = torch.full((16,), -1, dtype=torch.long, device=tower.device)
    if use_graph:
        prompt_frame = frame.clone()
        prompt_frame[:, 0] = tower.eos_id
        graph.start(prompt_frame, 0.8, 1, 1.0)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(tower.device)
    start_event.record()
    started = time.perf_counter()
    for step in range(steps):
        position = step % graph.capacity
        # Vary main tokens to avoid the official RAS retry's fixed top-k=30.
        logits.zero_()
        logits[:, tower.semantic_begin + step % tower.config.vocab_size] = 100
        if hidden_rows is not None:
            hidden = hidden_rows[step % len(hidden_rows) : step % len(hidden_rows) + 1]
        if use_graph:
            frame, _ = graph.step(logits, hidden, position, frame)
        else:
            frame = tower.decode_step(
                logits,
                hidden,
                torch.tensor([position], device=tower.device),
                frame,
                history,
                temperature=0.8,
                top_k=1,
                top_p=1.0,
            )
            frame[0, 0].item()
            history.copy_(torch.cat((history[1:], frame[0, :1])))
    end_event.record()
    torch.cuda.synchronize(tower.device)
    wall_seconds = time.perf_counter() - started
    return {
        "steps": steps,
        "wall_seconds": wall_seconds,
        "wall_ms_per_frame": 1000 * wall_seconds / steps,
        "stream_ms_per_frame": start_event.elapsed_time(end_event) / steps,
        "note": "tower + sampling only; no backbone, codec, reference, or disk I/O",
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--capacity", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument(
        "--hidden-file",
        type=Path,
        default=None,
        help="real backbone hidden rows [rows,hidden_size] saved with torch.save",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    device = torch.device(args.device)
    torch.manual_seed(1234)
    tower, vocab_size = load_tower(args.checkpoint, device)
    hidden_rows = None
    if args.hidden_file:
        hidden_rows = torch.load(
            args.hidden_file, weights_only=True, map_location="cpu"
        )
        if (
            not isinstance(hidden_rows, Tensor)
            or hidden_rows.ndim != 2
            or not len(hidden_rows)
            or hidden_rows.shape[1] != tower.config.dim
        ):
            raise ValueError("hidden file must contain nonempty [rows,hidden_size] tensor")
        hidden_rows = hidden_rows.to(
            device=device, dtype=tower.codebook_embeddings.weight.dtype
        )
    tower.setup_caches(args.capacity)
    torch.cuda.synchronize(device)
    before = torch.cuda.memory_allocated(device)
    started = time.perf_counter()
    graph = TowerDecodeGraph(
        tower, args.capacity, logits_size=vocab_size, record_logits=True
    )
    torch.cuda.synchronize(device)
    capture_seconds = time.perf_counter() - started
    after = torch.cuda.memory_allocated(device)
    parity = check_parity(tower, graph, vocab_size, hidden_rows)
    # Both paths are warmed before measurement; order is recorded explicitly.
    measure(
        tower, graph, vocab_size, min(4, args.steps), use_graph=False, hidden_rows=hidden_rows
    )
    measure(
        tower, graph, vocab_size, min(4, args.steps), use_graph=True, hidden_rows=hidden_rows
    )
    eager = measure(
        tower, graph, vocab_size, args.steps, use_graph=False, hidden_rows=hidden_rows
    )
    captured = measure(
        tower, graph, vocab_size, args.steps, use_graph=True, hidden_rows=hidden_rows
    )
    payload = {
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "synthetic": args.checkpoint is None,
        "hidden_file": str(args.hidden_file) if args.hidden_file else None,
        "parity_scope": "fixed real/synthetic hidden; synthetic valid frames and nonzero KV history",
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "graph": graph.description(),
        "capture_seconds": capture_seconds,
        "allocated_bytes_before_capture": before,
        "graph_allocated_bytes": after - before,
        "parity": parity,
        "measurement_order": ["eager", "cuda_graph"],
        "eager": eager,
        "cuda_graph": captured,
        "tower_speedup": eager["wall_seconds"] / captured["wall_seconds"],
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
