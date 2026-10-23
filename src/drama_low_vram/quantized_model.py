"""Load published W4A16/INT8/W6A8 banks into the existing DramaEngine."""

from dataclasses import dataclass
import json
from pathlib import Path
import time

import torch
from torch import nn
from tokenizers import Tokenizer

from fish_speech.models.text2semantic.fish_omni.backbone import (
    FishOmniBackbone, CHECKPOINT_BACKBONE_PREFIX, CHECKPOINT_LM_HEAD_KEY, backbone_name,
)
from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig
from fish_speech.models.text2semantic.fish_omni.model import FishOmniModel, AUDIO_DECODER_PREFIXES
from fish_speech.models.text2semantic.fish_omni.tower import FishOmniAudioDecoder, load_audio_decoder_weights

from .checkpoint_reader import iter_readonly_checkpoint_tensors
from .offload import materialize_offloaded_backbone, install_expert_offload, EXPERT_BUFFER_NAMES
from .quantized_kernels import QuantizedExpertBackend, projection, rotate_quantize
from .semantic_head import SemanticHead
from .shared_checkpoint import descriptor, iter_packed, virtual_tensors


@dataclass(frozen=True)
class QuantizedStore:
    w13_weight: torch.Tensor
    w13_scale_inv: torch.Tensor
    w2_weight: torch.Tensor
    w2_scale_inv: torch.Tensor
    hidden: int
    intermediate: int
    mode: int

    @property
    def num_experts(self):
        return self.w13_weight.shape[0]

    @property
    def nbytes(self):
        return sum(getattr(self, name).numel() * getattr(self, name).element_size()
                   for name in EXPERT_BUFFER_NAMES)

    def validate(self):
        for name in EXPERT_BUFFER_NAMES:
            value = getattr(self, name)
            if value.device.type != "cpu" or value.ndim != 3 or not value.is_contiguous():
                raise ValueError(f"Quantized bank must be contiguous three-dimensional CPU storage: {name}")
            if value.shape[0] != self.num_experts:
                raise ValueError("Quantized bank expert counts differ")
        if self.mode not in (0, 1, 2, 3):
            raise ValueError("Unknown quantized bank format")
        for bank, scales, n, k in (
            (self.w13_weight, self.w13_scale_inv, 2 * self.intermediate, self.hidden),
            (self.w2_weight, self.w2_scale_inv, self.hidden, self.intermediate),
        ):
            columns = (k // 8 if self.mode == 0 else k if self.mode == 1 else
                       k * 3 // 4 if self.mode == 2 else k // 2)
            scale_columns = (k // 128 if self.mode == 0 else 1 if self.mode == 1 else
                             k // 16 + (8 if self.mode == 3 else 4))
            if bank.shape[1:] != (n, columns) or scales.shape[1:] != (n, scale_columns):
                raise ValueError("Quantized weight/scale geometry mismatch")
            if bank.dtype != (torch.int32 if self.mode == 0 else torch.int8):
                raise ValueError("Quantized weight dtype mismatch")
            if scales.dtype != (torch.float16 if self.mode == 0 else
                                torch.float32 if self.mode == 1 else torch.uint8):
                raise ValueError("Quantized scale dtype mismatch")


class GPTQLinear(nn.Module):
    """Keep dense W4A16 weights compressed on GPU; no whole-matrix expansion."""

    def __init__(self, weight, scales, *, device, dtype=torch.bfloat16):
        super().__init__()
        self.in_features, self.out_features = weight.shape[0] * 8, weight.shape[1]
        self.register_buffer("qweight", weight.T.contiguous().to(device))
        self.register_buffer("scales", scales.T.contiguous().to(device))
        self.register_buffer("_anchor", torch.empty(0, device=device, dtype=dtype))

    @property
    def weight(self):
        return self._anchor

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).contiguous()
        ids = torch.zeros(flat.shape[0], device=x.device, dtype=torch.int64)
        weights = torch.ones_like(ids, dtype=torch.float32)
        out = projection(flat, ids, weights, self.qweight[None], self.scales[None], mode=0)
        return out.reshape(*shape[:-1], self.out_features)


def quantization_mode(directory):
    entry = descriptor(directory) if (Path(directory) / "shared-model.json").is_file() else None
    if entry:
        return {"int8_convrot": 1, "w6a8": 2, "w4a8": 3}.get(entry.get("quantization"))
    config_path = Path(directory) / "config.json"
    if config_path.is_file():
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        if raw.get("quantization_config", {}).get("quant_method") == "drama3-int4":
            return 0
    return None


def _bind(module, store):
    store.validate()
    for name in EXPERT_BUFFER_NAMES:
        module._buffers[name] = getattr(store, name)
    module._drama_quant_store = store
    module._drama_expert_storage = "quantized-cpu"


def _comfy_store(values, prefix, text, mode):
    tensors = {}
    for target, source, k in (
        ("w13", "gate_up_proj", text.hidden_size),
        ("w2", "down_proj", text.moe_intermediate_size),
    ):
        base = prefix + "." + source
        tensors[target + "_weight"] = values.pop(base + ".weight")
        config = json.loads(bytes(values.pop(base + ".comfy_quant").tolist()).decode("utf-8"))
        if (config.get("convrot_groupsize") != 256
                or config.get("num_experts") != text.num_experts):
            raise ValueError("Unsupported ConvRot group size or expert count")
        if mode == 1:
            if config.get("format") != "int8_tensorwise" or not config.get("convrot"):
                raise ValueError("Expected INT8 ConvRot expert bank")
            scale = values.pop(base + ".weight_scale")
            tensors[target + "_scale_inv"] = scale.reshape(scale.shape[0], scale.shape[1], 1)
        else:
            expected = "asym_w4a8_int8" if mode == 3 else "w6a8_int8"
            if config.get("format") != expected or config.get("group_size") != 16:
                raise ValueError("Expected published W4A8/W6A8 group16 expert bank")
            relative = values.pop(base + ".weight_s_rel")
            channel = values.pop(base + ".weight_s_channel").contiguous().view(torch.uint8)
            relative = relative.reshape(tensors[target + "_weight"].shape[:2] + (k // 16,))
            channel = channel.reshape(relative.shape[:2] + (4,))
            if mode == 3:
                codebook = values.pop(base + ".weight_codebook")
                if codebook.shape != (16,) or codebook.dtype != torch.float32:
                    raise ValueError("W4A8 requires the published 16-level FP32 codebook")
                # One codebook float in each of the first 16 auxiliary rows.
                # This adds four bytes per row, not 64 replicated bytes per row.
                aux = torch.zeros((*relative.shape[:2], k // 16 + 8), dtype=torch.uint8)
                aux[..., :k // 16] = relative
                aux[..., k // 16:k // 16 + 4] = channel
                aux[:, :16, k // 16 + 4:] = codebook.contiguous().view(torch.uint8).reshape(16, 4)
                tensors[target + "_scale_inv"] = aux
            else:
                tensors[target + "_scale_inv"] = torch.cat((relative, channel), dim=-1)
    return QuantizedStore(**tensors, hidden=text.hidden_size,
                          intermediate=text.moe_intermediate_size, mode=mode)


def _gptq_store(values, prefix, text):
    banks = {}
    for target, names in (("w13", ("gate_proj", "up_proj")), ("w2", ("down_proj",))):
        weights, scales = [], []
        for expert in range(text.num_experts):
            parts = [values.pop(f"{prefix}.{expert}.{name}.qweight").T for name in names]
            scale_parts = [values.pop(f"{prefix}.{expert}.{name}.scales").T for name in names]
            weights.append(torch.cat(parts, dim=0) if len(parts) > 1 else parts[0])
            scales.append(torch.cat(scale_parts, dim=0) if len(parts) > 1 else scale_parts[0])
        banks[target + "_weight"] = torch.stack(weights)
        banks[target + "_scale_inv"] = torch.stack(scales)
    return QuantizedStore(**banks, hidden=text.hidden_size,
                          intermediate=text.moe_intermediate_size, mode=0)


def load_quantized_drama3(
    directory, compute_device, *, dtype, embedding_device,
    max_staged_experts, max_pair_rows, cache_experts, pin_memory,
    profile_transfers, combined_decode_bank, packed_staging, semantic_head,
):
    directory = Path(directory).resolve()
    mode = quantization_mode(directory)
    if mode is None:
        raise ValueError("Unsupported quantized checkpoint")
    if not semantic_head:
        raise ValueError("Quantized model loading uses the compact semantic head")
    raw = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    # Config describes the unchanged architecture; actual precision is owned by
    # the validated bank/linear types below, not the upstream FP8 constructor.
    architecture = dict(raw)
    architecture.pop("quantization_config", None)
    config = FishOmniConfig.from_dict(architecture)
    with torch.device("meta"):
        backbone = FishOmniBackbone(config.text, fp8_experts=True, dtype=dtype)
        tower = FishOmniAudioDecoder(config, dtype=dtype)
    if mode == 0:
        values = dict(iter_readonly_checkpoint_tensors(directory))
        for name in list(values):
            if not name.endswith(".qweight") or ".mlp.experts." in name:
                continue
            base = name[:-len(".qweight")]
            quantized = GPTQLinear(values.pop(name), values.pop(base + ".scales"),
                                   device=compute_device, dtype=dtype)
            zeros, groups = values.pop(base + ".qzeros"), values.pop(base + ".g_idx")
            if not bool((zeros == -2004318072).all()) or not torch.equal(
                    groups, torch.arange(quantized.in_features, dtype=torch.int32) // 128):
                raise ValueError(f"Unsupported asymmetric or permuted GPTQ dense weight: {base}")
            destination = backbone_name(base + ".weight")[:-len(".weight")]
            parent, _, child = destination.rpartition(".")
            setattr(backbone.get_submodule(parent), child, quantized)
    else:
        values = dict(iter_packed(directory))
    # Materialize only remaining meta tensors. Quantized dense linears are
    # already placed and must not be overwritten by to_empty.
    placement = {}
    expert_ids = {id(layer.mlp.experts) for layer in backbone.model.layers}
    for name, module in backbone.named_modules():
        if id(module) in expert_ids or isinstance(module, GPTQLinear):
            continue
        device = torch.device(embedding_device) if name == "model.embed_tokens" else (
            torch.device("meta") if name == "lm_head" else compute_device)
        if any(t.is_meta for t in [*module.parameters(recurse=False), *module.buffers(recurse=False)]):
            module.to_empty(device=device, recurse=False)
        placement[name] = str(device)
    started = time.perf_counter()
    for index, layer in enumerate(backbone.model.layers):
        prefix = (f"model.language_model.layers.{index}.mlp.experts" if mode == 0 else
                  f"model.layers.{index}.mlp.experts")
        store = _gptq_store(values, prefix, config.text) if mode == 0 else _comfy_store(
            values, prefix, config.text, mode)
        _bind(layer.mlp.experts, store)
        if index % 10 == 9:
            print(json.dumps({"phase": "quantized-bank-load", "layers": index + 1,
                              "mode": mode, "seconds": time.perf_counter() - started}), flush=True)
    if mode == 0:
        dense = {backbone_name(name): value for name, value in values.items()
                 if name.startswith(CHECKPOINT_BACKBONE_PREFIX)}
        targets = {name: value for name, value in backbone.named_parameters()
                   if ".mlp.experts." not in name and name != "lm_head.weight"}
        # Quantized modules expose only compressed buffers; all remaining
        # floating parameters must be supplied, exactly once.
        if set(dense) != set(targets):
            raise ValueError(f"INT4 dense slots differ: {sorted(set(dense) ^ set(targets))[:8]}")
        for name, target in targets.items():
            value = dense[name]
            if target.shape != value.shape:
                raise ValueError(f"INT4 remaining dense shape mismatch: {name}")
            target.data.copy_(value)
        head = values[CHECKPOINT_LM_HEAD_KEY]
        backbone.lm_head = SemanticHead.from_weight(
            head, eos_token_id=config.eos_token_id,
            semantic_start_token_id=config.semantic_start_token_id,
            semantic_end_token_id=config.semantic_end_token_id, device=compute_device, dtype=dtype)
        tower_values = ((name, value) for name, value in values.items() if name.startswith(AUDIO_DECODER_PREFIXES))
    else:
        from .model import _load_dense_backbone
        _load_dense_backbone(backbone, ((name, value) for name, value in virtual_tensors(directory)
                                       if name.startswith(CHECKPOINT_BACKBONE_PREFIX)), skip_head=True)
        backbone.lm_head = SemanticHead.from_candidate_rows(
            values["audio_head.weight"], vocab_size=config.text.vocab_size,
            eos_token_id=config.eos_token_id, semantic_start_token_id=config.semantic_start_token_id,
            semantic_end_token_id=config.semantic_end_token_id, device=compute_device, dtype=dtype)
        tower_values = ((name, value) for name, value in virtual_tensors(directory)
                        if name.startswith(AUDIO_DECODER_PREFIXES))
    tower.to_empty(device=compute_device)
    load_audio_decoder_weights(tower, tower_values)
    model = FishOmniModel(config, backbone, tower, Tokenizer.from_file(str(directory / "tokenizer.json"))).eval()
    backend = QuantizedExpertBackend(mode)
    backend.quantized_mode = ("W4A16-GPTQ", "INT8-ConvRot/BF16", "W6A8/BF16",
                              "W4A8-Codebook/BF16")[mode]
    manager = install_expert_offload(
        backbone, backend=backend, max_staged_experts=max_staged_experts,
        max_pair_rows=max_pair_rows, cache_experts=cache_experts, pin_memory=pin_memory,
        profile_transfers=profile_transfers, combined_decode_bank=combined_decode_bank,
        packed_staging=packed_staging)
    model._drama_low_vram_placement = placement
    model._drama_low_vram_manager = manager
    model._drama_low_vram_expert_cache = None
    model._drama_quantized_mode = backend.quantized_mode
    return model, manager
