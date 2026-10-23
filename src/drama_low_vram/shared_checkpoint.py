"""Read packed Drama3 safetensors without ComfyUI or a second expert bank.

Only explicit local descriptors enable this format. The official model math is
unchanged: fused source weights are split into views for the original modules.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import torch

from .checkpoint_reader import _read_header, iter_readonly_checkpoint_tensors
from .offload import ExpertStore

DESCRIPTOR = "shared-model.json"
EXPERT_NAMES = {
    "w13_weight": "gate_up_weight", "w13_scale_inv": "gate_up_scale",
    "w2_weight": "down_weight", "w2_scale_inv": "down_scale",
}


def descriptor(directory):
    path = Path(directory).resolve() / DESCRIPTOR
    if not path.is_file():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    if (document.get("format") != "drama3-packed-shared-v1"
            or not isinstance(document.get("model"), str)
            or not isinstance(document.get("size"), int)
            or not re.fullmatch(r"[a-f0-9]{64}", document.get("sha256", ""))):
        raise ValueError("Invalid shared model descriptor")
    model = (path.parent / document["model"]).resolve()
    if not model.is_file() or model.stat().st_size != document["size"]:
        raise ValueError(f"Shared model missing or size changed: {model}")
    return document | {"model": str(model)}


def is_shared(directory):
    return (Path(directory) / DESCRIPTOR).is_file()


def iter_packed(directory, *, include_prefixes=None, skip_prefixes=()):
    entry = descriptor(directory)
    if entry is None:
        raise ValueError("Shared model descriptor not found")
    # Reader accepts a single file through a virtual model.safetensors index.
    # No symlink or full copy is needed: explicit file iteration is supported.
    from .checkpoint_reader import iter_readonly_file_tensors
    yield from iter_readonly_file_tensors(
        entry["model"], include_prefixes=include_prefixes, skip_prefixes=skip_prefixes)


def virtual_tensors(directory):
    """Map converted dense/tower names back to original checkpoint names."""
    for name, tensor in iter_packed(directory):
        if ".mlp.experts." in name:
            continue
        if name == "audio_head.weight":
            # This compact head has a dedicated loading path, not a full head.
            continue
        if name.startswith("model."):
            key = "model.language_model." + name[len("model."):]
            if ".shared_expert.gate_up_proj." in key:
                half = tensor.shape[0] // 2
                yield key.replace(".gate_up_proj.", ".gate_proj."), tensor[:half]
                yield key.replace(".gate_up_proj.", ".up_proj."), tensor[half:]
            else:
                yield key, tensor
        elif name == "tower.codebook_embeddings.weight":
            yield "audio_decoder.codebook_embeddings.weight", tensor
        elif name.startswith("tower.heads."):
            yield "audio_decoder_mtp.heads." + name[len("tower.heads."):], tensor
        elif name.startswith(("tower.shared_layers.", "tower.residual_layers.")):
            shared = name.startswith("tower.shared_layers.")
            source = "tower.shared_layers." if shared else "tower.residual_layers."
            target = "audio_decoder_mtp_shared.layers." if shared else "audio_decoder_mtp.layers."
            suffix = name[len(source):]
            number, suffix = suffix.split(".", 1)
            if suffix.startswith("embedding."):
                suffix = ("semantic_embedding." if shared else "codebook_embedding.") + suffix[len("embedding."):]
            elif suffix.startswith(("attention.", "attention_norm.", "ffn_norm.", "feed_forward.")):
                suffix = "transformer_layer." + suffix
            key = target + number + "." + suffix
            if ".feed_forward.w13." in key:
                half = tensor.shape[0] // 2
                yield key.replace(".w13.", ".w1."), tensor[:half]
                yield key.replace(".w13.", ".w3."), tensor[half:]
            else:
                yield key, tensor
        else:
            raise ValueError(f"Unknown packed checkpoint weight: {name}")


class SharedFP8Experts:
    """Direct immutable views of the experts in the shared safetensors file."""

    def __init__(self, directory, config):
        self.directory = Path(directory).resolve()
        self.entry = descriptor(directory)
        self.model_fingerprint = self.entry["sha256"]
        self.num_layers = config.text.num_hidden_layers
        self.expert_module_names = tuple(
            f"model.layers.{layer}.mlp.experts" for layer in range(self.num_layers))
        self.expert_checkpoint_prefixes = ()
        self.closed = False
        header = _read_header(Path(self.entry["model"]))
        self.nbytes = sum(entry.nbytes for name, entry in header.entries.items() if ".mlp.experts." in name)
        # Validate every bank before allocating the GPU backbone.
        text = config.text
        expected = {
            "gate_up_weight": ((text.num_experts, 2 * text.moe_intermediate_size, text.hidden_size), torch.float8_e4m3fn),
            "down_weight": ((text.num_experts, text.hidden_size, text.moe_intermediate_size), torch.float8_e4m3fn),
            "gate_up_scale": ((text.num_experts, 2 * text.moe_intermediate_size // 128, text.hidden_size // 128), torch.float32),
            "down_scale": ((text.num_experts, text.hidden_size // 128, text.moe_intermediate_size // 128), torch.float32),
        }
        actual = {name for name in header.entries if ".mlp.experts." in name}
        wanted = set()
        for index in range(self.num_layers):
            for name, (shape, dtype) in expected.items():
                key = f"model.layers.{index}.mlp.experts.{name}"
                wanted.add(key)
                entry = header.entries.get(key)
                if entry is None or entry.shape != shape or entry.dtype != dtype:
                    raise ValueError(f"Packed FP8 expert shape/dtype mismatch: {key}")
        if actual != wanted:
            raise ValueError("Unexpected or missing packed FP8 expert tensors")

    def assign_to_backbone(self, backbone):
        values = dict(iter_packed(
            self.directory, include_prefixes=self.expert_module_names))
        for index, module_name in enumerate(self.expert_module_names):
            module = backbone.get_submodule(module_name)
            selected = {target: values[f"{module_name}.{source}"] for target, source in EXPERT_NAMES.items()}
            ExpertStore(**selected).validate()
            for name, tensor in selected.items():
                module._buffers[name] = tensor
            module._drama_expert_storage = "mmap"
        return self.num_layers

    def close(self):
        self.closed = True
