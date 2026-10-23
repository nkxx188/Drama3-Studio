"""Memory-aware Drama 3 model loading on native Windows CUDA."""

from __future__ import annotations

import gc
import json
import os
from dataclasses import replace
from pathlib import Path

import torch
from tokenizers import Tokenizer

from fish_speech.models.text2semantic.fish_omni.backbone import (
    CHECKPOINT_BACKBONE_PREFIX,
    CHECKPOINT_LM_HEAD_KEY,
    FishOmniBackbone,
    backbone_name,
    load_backbone_weights,
)
from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig
from fish_speech.models.text2semantic.fish_omni.model import (
    AUDIO_DECODER_PREFIXES,
    FishOmniModel,
)
from fish_speech.models.text2semantic.fish_omni.tower import (
    FishOmniAudioDecoder,
    TOWER_CHECKPOINT_PREFIXES,
    load_audio_decoder_weights,
)

from .autotune import PostLoadPlan, RuntimeConfig, plan_post_load
from .checkpoint_reader import iter_readonly_checkpoint_tensors as iter_checkpoint_tensors
from .offload import (
    ExpertBackend,
    ExpertOffloadManager,
    ExpertStore,
    TorchExpertBackend,
    install_expert_offload,
)
from .offload import materialize_offloaded_backbone


def _require_cuda(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError("Drama 3 low-VRAM runtime currently requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if resolved.index is None:
        resolved = torch.device("cuda:0")
    return resolved


def _resolve_placement_device(
    device: str | torch.device, compute_device: torch.device
) -> torch.device:
    requested = torch.device(device)
    if requested.type == "cuda" and requested.index is None:
        return compute_device
    return requested


def load_offloaded_drama3(
    checkpoint_dir: str | Path,
    *,
    device: str | torch.device = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    embedding_device: str | torch.device = "cpu",
    lm_head_device: str | torch.device = "cpu",
    expert_backend: str | ExpertBackend = "triton",
    max_staged_experts: int = 8,
    max_pair_rows: int = 64,
    cache_experts: int = 0,
    pin_memory: bool = True,
    profile_transfers: bool = True,
    semantic_head: bool = False,
    expert_cache_dir: str | Path | None = None,
    combined_decode_bank: bool = False,
    packed_staging: bool = False,
) -> tuple[FishOmniModel, ExpertOffloadManager]:
    """Load Drama 3 with FP8 experts kept in CPU RAM and staged on demand.

    The model is constructed on ``meta`` and materialized directly into its final
    placement. This avoids the upstream loader's temporary full-GPU expert layout.
    ``checkpoint_dir`` must contain the complete eight-shard checkpoint.
    """
    checkpoint_dir = Path(checkpoint_dir).resolve()
    if not (checkpoint_dir / "config.json").is_file():
        raise FileNotFoundError(f"Drama 3 config not found: {checkpoint_dir / 'config.json'}")
    compute_device = _require_cuda(device)
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Drama 3 compute dtype must be float16 or bfloat16")
    os.environ["FISH_DRAMA3_CUDA_GRAPH"] = "0"
    from .quantized_model import quantization_mode, load_quantized_drama3
    if quantization_mode(checkpoint_dir) is not None:
        return load_quantized_drama3(
            checkpoint_dir, compute_device, dtype=dtype, embedding_device=embedding_device,
            max_staged_experts=max_staged_experts, max_pair_rows=max_pair_rows,
            cache_experts=cache_experts, pin_memory=pin_memory,
            profile_transfers=profile_transfers, combined_decode_bank=combined_decode_bank,
            packed_staging=packed_staging, semantic_head=semantic_head)
    from .shared_checkpoint import is_shared
    if is_shared(checkpoint_dir):
        return _load_shared_drama3(
            checkpoint_dir, compute_device, dtype=dtype,
            embedding_device=embedding_device, expert_backend=expert_backend,
            max_staged_experts=max_staged_experts, max_pair_rows=max_pair_rows,
            cache_experts=cache_experts, pin_memory=pin_memory,
            profile_transfers=profile_transfers, combined_decode_bank=combined_decode_bank,
            packed_staging=packed_staging, semantic_head=semantic_head)

    with (checkpoint_dir / "config.json").open(encoding="utf-8") as handle:
        config = FishOmniConfig.from_dict(json.load(handle))
    with torch.device("meta"):
        backbone = FishOmniBackbone(config.text, fp8_experts=True, dtype=dtype)
        audio_decoder = FishOmniAudioDecoder(config, dtype=dtype)

    expert_cache = None
    if expert_cache_dir is not None:
        from .expert_cache_file import ExpertCacheFile

        expert_cache = ExpertCacheFile.open(
            expert_cache_dir, checkpoint_dir=checkpoint_dir
        )
    placement = materialize_offloaded_backbone(
        backbone,
        compute_device,
        embedding_device=_resolve_placement_device(embedding_device, compute_device),
        lm_head_device=(
            "meta" if expert_cache is not None and semantic_head
            else _resolve_placement_device(lm_head_device, compute_device)
        ),
        expert_device="meta" if expert_cache is not None else "cpu",
    )
    if expert_cache is not None:
        expert_cache.assign_to_backbone(backbone)
        _load_dense_backbone(
            backbone,
            iter_checkpoint_tensors(
                checkpoint_dir,
                skip_prefixes=(
                    AUDIO_DECODER_PREFIXES + tuple(expert_cache.expert_checkpoint_prefixes)
                    + ((CHECKPOINT_LM_HEAD_KEY,) if semantic_head else ())
                ),
            ),
            skip_head=semantic_head,
        )
        for name in expert_cache.expert_module_names:
            placement[name] = "mmap"
    else:
        load_backbone_weights(
            backbone,
            iter_checkpoint_tensors(checkpoint_dir, skip_prefixes=AUDIO_DECODER_PREFIXES),
        )
    if semantic_head:
        from .semantic_head import SemanticHead, install_semantic_head

        if backbone.lm_head.weight.is_meta:
            head_weights = iter_checkpoint_tensors(
                checkpoint_dir, include_prefixes=(CHECKPOINT_LM_HEAD_KEY,)
            )
            weight = None
            try:
                key, weight = next(head_weights)
                if key != CHECKPOINT_LM_HEAD_KEY or weight.shape != backbone.lm_head.weight.shape:
                    raise ValueError("Official semantic output head shape/key mismatch")
                backbone.lm_head = SemanticHead.from_weight(
                    weight, eos_token_id=config.eos_token_id,
                    semantic_start_token_id=config.semantic_start_token_id,
                    semantic_end_token_id=config.semantic_end_token_id,
                    device=compute_device, dtype=dtype,
                )
                if next(head_weights, None) is not None:
                    raise ValueError("Duplicate semantic output head")
            finally:
                head_weights.close()
                del weight
        else:
            install_semantic_head(backbone, config, device=compute_device)
        placement["lm_head"] = str(compute_device)
    audio_decoder.to_empty(device=compute_device)
    load_audio_decoder_weights(
        audio_decoder,
        iter_checkpoint_tensors(
            checkpoint_dir,
            skip_prefixes=(CHECKPOINT_BACKBONE_PREFIX, CHECKPOINT_LM_HEAD_KEY),
        ),
    )
    tokenizer = Tokenizer.from_file(str(checkpoint_dir / "tokenizer.json"))
    model = FishOmniModel(config, backbone, audio_decoder, tokenizer).eval()
    manager = install_expert_offload(
        model.backbone,
        backend=expert_backend,
        max_staged_experts=max_staged_experts,
        max_pair_rows=max_pair_rows,
        cache_experts=cache_experts,
        pin_memory=pin_memory,
        profile_transfers=profile_transfers,
        combined_decode_bank=combined_decode_bank,
        packed_staging=packed_staging,
    )
    model._drama_low_vram_placement = placement
    model._drama_low_vram_manager = manager
    model._drama_low_vram_expert_cache = expert_cache
    return model, manager


def _load_shared_drama3(
    directory, compute_device, *, dtype, embedding_device, expert_backend,
    max_staged_experts, max_pair_rows, cache_experts, pin_memory,
    profile_transfers, combined_decode_bank, packed_staging, semantic_head,
):
    from .shared_checkpoint import SharedFP8Experts, iter_packed, virtual_tensors
    from .semantic_head import SemanticHead
    from .weight_sources import ImmutableWeightSources
    if not semantic_head:
        raise ValueError("Packed shared model contains the compact audio head; compact_head is required")
    config = FishOmniConfig.from_dict(json.loads((directory / "config.json").read_text(encoding="utf-8")))
    experts = SharedFP8Experts(directory, config)
    with torch.device("meta"):
        backbone = FishOmniBackbone(config.text, fp8_experts=True, dtype=dtype)
        audio_decoder = FishOmniAudioDecoder(config, dtype=dtype)
    sources = ImmutableWeightSources()
    mapped_embedding = torch.device(embedding_device).type == "cpu"
    placement = materialize_offloaded_backbone(
        backbone, compute_device,
        embedding_device="meta" if mapped_embedding else embedding_device,
        lm_head_device="meta", expert_device="meta")
    experts.assign_to_backbone(backbone)
    try:
        _load_dense_backbone(
            backbone, ((name, tensor) for name, tensor in virtual_tensors(directory)
                       if name.startswith(CHECKPOINT_BACKBONE_PREFIX)), skip_head=True,
            sources=sources, mapped_embedding=mapped_embedding)
        weights = iter_packed(directory, include_prefixes=("audio_head.weight",))
        try:
            name, weight = next(weights)
            if name != "audio_head.weight":
                raise ValueError("Packed semantic head missing")
            backbone.lm_head = SemanticHead.from_candidate_rows(
                weight, vocab_size=config.text.vocab_size,
                eos_token_id=config.eos_token_id,
                semantic_start_token_id=config.semantic_start_token_id,
                semantic_end_token_id=config.semantic_end_token_id,
                device=compute_device, dtype=dtype)
            sources.remember(backbone, "lm_head.weight", weight)
        finally:
            weights.close()
        audio_decoder.to_empty(device=compute_device)

        def tower_weights():
            for key, tensor in virtual_tensors(directory):
                if not key.startswith(AUDIO_DECODER_PREFIXES):
                    continue
                name = next(
                    (new + key[len(old):] for old, new in TOWER_CHECKPOINT_PREFIXES.items()
                     if key.startswith(old)), None)
                if name is not None:
                    sources.remember(audio_decoder, name, tensor)
                yield key, tensor

        load_audio_decoder_weights(audio_decoder, tower_weights())
        tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        model = FishOmniModel(config, backbone, audio_decoder, tokenizer).eval()
        manager = install_expert_offload(
            backbone, backend=expert_backend, max_staged_experts=max_staged_experts,
            max_pair_rows=max_pair_rows, cache_experts=cache_experts, pin_memory=pin_memory,
            profile_transfers=profile_transfers, combined_decode_bank=combined_decode_bank,
            packed_staging=packed_staging)
    except BaseException:
        experts.close()
        raise
    placement.update({name: "mmap-shared" for name in experts.expert_module_names})
    placement["lm_head"] = str(compute_device)
    model._drama_low_vram_placement = placement
    model._drama_low_vram_manager = manager
    model._drama_low_vram_expert_cache = experts
    model._drama_low_vram_weight_sources = sources
    if sources.cpu_embedding_bytes:
        placement["model.embed_tokens"] = "mmap-cpu"
    return model, manager


@torch.no_grad()
def _load_dense_backbone(backbone, weights, *, skip_head=False, sources=None,
                         mapped_embedding=False):
    """Validate every non-expert slot while keeping read-only mapped banks intact."""
    targets = dict(backbone.named_parameters())
    targets.update(backbone.named_buffers())
    targets = {
        name: tensor for name, tensor in targets.items()
        if ".mlp.experts." not in name and not (skip_head and name == "lm_head.weight")
    }
    loaded = set()
    for key, tensor in weights:
        name = backbone_name(key)
        if name not in targets or name in loaded:
            raise ValueError(f"Unknown or duplicate dense backbone weight: {key}")
        target = targets[name]
        if target.shape != tensor.shape:
            raise ValueError(f"Dense backbone shape mismatch for {key}")
        if sources is not None:
            remembered = sources.remember(backbone, name, tensor)
            if mapped_embedding and name == "model.embed_tokens.weight":
                module = backbone.model.embed_tokens
                if remembered:
                    module._parameters["weight"] = sources.get(module, "parameter", "weight")
                    sources.cpu_embedding_bytes = tensor.numel() * tensor.element_size()
                else:
                    # Other supported compute dtypes still use the original
                    # CPU conversion; never bind differently typed source bytes.
                    module._parameters["weight"] = torch.nn.Parameter(
                        tensor.to(device="cpu", dtype=target.dtype, copy=True),
                        requires_grad=False)
                loaded.add(name)
                continue
        target.copy_(tensor)
        loaded.add(name)
    if missing := targets.keys() - loaded:
        raise ValueError(f"Dense backbone weights missing: {sorted(missing)[:8]}")


def configure_model_runtime(
    model: FishOmniModel,
    manager: ExpertOffloadManager,
    *,
    sequence_capacity: int,
    runtime_config: RuntimeConfig | None = None,
    cache_experts: int | None = None,
    prefill_chunk_size: int | None = None,
    allow_staging_reduction: bool = True,
) -> PostLoadPlan:
    """Plan and allocate FP8 caching after actual weights and sequence caches.

    Only placement and batching change. The checkpoint, router, top-k and
    sampling remain intact. Automatic cache allocation can shrink on OOM before
    sampling begins; an explicit capacity is either honored or fails clearly.
    """
    from fish_speech.models.text2semantic.fish_omni.generate import ensure_drama3_caches

    if (
        isinstance(sequence_capacity, bool)
        or not isinstance(sequence_capacity, int)
        or not 1 <= sequence_capacity <= model.config.text.max_position_embeddings
    ):
        raise ValueError("sequence_capacity is outside the model's supported context")
    if not manager.modules:
        raise ValueError("Expert offload manager has no attached layers")
    device = _require_cuda(model.audio_decoder.device)
    stores = [ExpertStore.from_module(module) for module in manager.modules.values()]
    if len({store.num_experts for store in stores}) != 1:
        raise ValueError("Post-load planning requires equal expert counts per layer")
    requested_staging = manager.max_staged_experts
    requested_pairs = manager.max_pair_rows
    prefill = (
        prefill_chunk_size
        if prefill_chunk_size is not None
        else (runtime_config.prefill_chunk if runtime_config is not None else 16)
    )
    auto_cache_limit = getattr(runtime_config, "max_auto_cache_experts", None)
    request = (cache_experts, prefill, allow_staging_reduction,
               requested_staging, requested_pairs, auto_cache_limit)
    previous = getattr(model, "_drama_low_vram_runtime_plan", None)
    if (
        previous is not None and model.cache_capacity >= sequence_capacity
        and getattr(model, "_drama_low_vram_plan_request", None) == request
        and getattr(manager, "runtime_plan", None) is previous
        and manager.stats.device_buffer_bytes > 0
    ):
        return previous
    # Reclaim only for a changed plan. Existing runtime allocations must not be
    # mistaken for fixed model weights by the post-load budget calculation.
    manager.release_runtime_buffers()
    with torch.cuda.device(device):
        try:
            ensure_drama3_caches(model, sequence_capacity)
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(
                "The requested context does not fit in current free VRAM. "
                "Close GPU applications or shorten the reference/generation limit."
            ) from exc
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(device)
        allocated = int(torch.cuda.memory_allocated(device))
        plan = plan_post_load(
            fixed_gpu_bytes=allocated,
            vram_free_bytes=int(free),
            vram_total_bytes=int(total),
            vram_allocated_bytes=allocated,
            vram_reserved_bytes=int(torch.cuda.memory_reserved(device)),
            expert_total_bytes=sum(store.nbytes for store in stores),
            num_layers=len(stores),
            num_experts_per_layer=stores[0].num_experts,
            max_staged_experts=requested_staging,
            max_pair_rows=requested_pairs,
            prefill_chunk=prefill,
            cache_experts=cache_experts,
            max_auto_cache_experts=auto_cache_limit,
            backend="torch" if isinstance(manager.backend, TorchExpertBackend) else "triton",
            allow_staging_reduction=allow_staging_reduction,
        )
        if not plan.can_run:
            raise RuntimeError(" ".join(plan.warnings))
        retries = 0
        while True:
            manager.configure_staging(plan.max_staged_experts, plan.max_pair_rows)
            manager.configure_hot_cache(plan.cache_experts)
            try:
                manager.prepare_buffers(device)
                torch.cuda.synchronize(device)
                break
            except torch.cuda.OutOfMemoryError:
                manager.configure_hot_cache(0)
                manager.release_runtime_buffers()
                gc.collect()
                torch.cuda.empty_cache()
                if cache_experts is not None:
                    raise
                retries += 1
                if plan.cache_experts:
                    reduced = plan.cache_experts // 2
                    recovered = (plan.cache_experts - reduced) * plan.cache_slot_bytes
                    plan = replace(
                        plan,
                        cache_experts=reduced,
                        cache_bytes=reduced * plan.cache_slot_bytes,
                        gpu_headroom_bytes=plan.gpu_headroom_bytes + recovered,
                        warnings=plan.warnings + (
                            f"Cache allocation encountered VRAM pressure; reduced to {reduced} experts per layer before sampling.",
                        ),
                    )
                elif allow_staging_reduction and plan.max_staged_experts > 1:
                    reduced = max(1, plan.max_staged_experts // 2)
                    recovered = (plan.max_staged_experts - reduced) * plan.staged_expert_bytes
                    plan = replace(
                        plan, max_staged_experts=reduced,
                        max_pair_rows=max(1, plan.max_pair_rows // 2),
                        prefill_chunk=max(1, plan.prefill_chunk // 2),
                        staging_bytes=reduced * plan.staged_expert_bytes,
                        available_cache_bytes=plan.available_cache_bytes + recovered,
                        gpu_headroom_bytes=plan.gpu_headroom_bytes + recovered,
                        warnings=plan.warnings + (
                            f"Staging allocation encountered VRAM pressure; reduced to {reduced} experts before sampling.",
                        ),
                    )
                else:
                    raise
        model._drama_low_vram_runtime_plan = plan
        model._drama_low_vram_plan_request = (
            cache_experts, prefill, allow_staging_reduction,
            plan.max_staged_experts, plan.max_pair_rows, auto_cache_limit,
        )
        model._drama_low_vram_allocation = {
            "sequence_capacity": sequence_capacity,
            "allocation_retries": retries,
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "free_bytes": int(torch.cuda.mem_get_info(device)[0]),
        }
        manager.runtime_plan = plan
    return plan


def release_drama3(model: torch.nn.Module, manager: ExpertOffloadManager | None = None) -> None:
    """Release staged buffers and model references without copying GPU weights to CPU."""
    if manager is None:
        manager = getattr(model, "_drama_low_vram_manager", None)
    if manager is not None:
        manager.close()
    # ``model.cpu()`` would duplicate every GPU tensor, including the large
    # non-expert backbone, before Python can collect it. Detach submodules first.
    if isinstance(model, torch.nn.Module):
        for name in ("backbone", "audio_decoder"):
            if hasattr(model, name):
                setattr(model, name, None)
        if hasattr(model, "tokenizer"):
            model.tokenizer = None
        if hasattr(model, "decode_graphs"):
            model.decode_graphs = None
        if hasattr(model, "cache_capacity"):
            model.cache_capacity = 0
        model._drama_released = True
        sources = getattr(model, "_drama_low_vram_weight_sources", None)
        if sources is not None:
            sources.clear()
            model._drama_low_vram_weight_sources = None
        expert_cache = getattr(model, "_drama_low_vram_expert_cache", None)
        if expert_cache is not None:
            expert_cache.close()
            model._drama_low_vram_expert_cache = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


__all__ = ["configure_model_runtime", "load_offloaded_drama3", "release_drama3"]
