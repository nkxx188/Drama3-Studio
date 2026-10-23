"""Retain the official output rows that Drama 3's audio sampler can select.

The upstream tower indexes EOS and the semantic band before any normalization or
sampling. Other vocabulary rows therefore need not be multiplied. ``forward``
keeps the full-vocabulary shape for the upstream generation code; its zero-filled
non-candidate entries are placeholders, not text-token logits.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def semantic_candidate_ids(
    *,
    vocab_size: int,
    eos_token_id: int,
    semantic_start_token_id: int,
    semantic_end_token_id: int,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Return the sorted candidate IDs used by the official audio tower."""
    values = {
        "vocab_size": vocab_size,
        "eos_token_id": eos_token_id,
        "semantic_start_token_id": semantic_start_token_id,
        "semantic_end_token_id": semantic_end_token_id,
    }
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values.values()):
        raise TypeError("Vocabulary size and token IDs must be integers")
    if vocab_size < 1:
        raise ValueError("vocab_size must be positive")
    if not 0 <= semantic_start_token_id <= semantic_end_token_id < vocab_size:
        raise ValueError("The semantic band must lie inside the text vocabulary")
    if not 0 <= eos_token_id < vocab_size:
        raise ValueError("eos_token_id must lie inside the text vocabulary")
    if semantic_start_token_id <= eos_token_id <= semantic_end_token_id:
        raise ValueError("eos_token_id must lie outside the semantic band")
    eos = torch.tensor([eos_token_id], dtype=torch.long, device=device)
    band = torch.arange(
        semantic_start_token_id,
        semantic_end_token_id + 1,
        dtype=torch.long,
        device=device,
    )
    return torch.cat((eos, band)).sort().values


class SemanticHead(nn.Module):
    """Inference head with exact copies of the EOS and semantic weight rows.

    Moving a GEMM to another device or changing its matrix shape can change
    floating-point rounding. No weights, candidate IDs, or sampling probabilities
    are intentionally approximated here.
    """

    def __init__(
        self,
        weight: Tensor,
        *,
        vocab_size: int,
        eos_token_id: int,
        semantic_start_token_id: int,
        semantic_end_token_id: int,
        bias: Tensor | None = None,
    ) -> None:
        super().__init__()
        candidates = semantic_candidate_ids(
            vocab_size=vocab_size,
            eos_token_id=eos_token_id,
            semantic_start_token_id=semantic_start_token_id,
            semantic_end_token_id=semantic_end_token_id,
            device=weight.device,
        )
        if weight.ndim != 2 or weight.shape[0] != candidates.numel():
            raise ValueError("weight must contain one row for each sorted audio candidate")
        if not weight.is_floating_point():
            raise TypeError("Semantic head weights must retain a floating-point dtype")
        if bias is not None:
            if bias.shape != (candidates.numel(),):
                raise ValueError("bias must contain one value for each audio candidate")
            if bias.device != weight.device or bias.dtype != weight.dtype:
                raise ValueError("bias and weight must have the same device and dtype")
        self.in_features = int(weight.shape[1])
        self.out_features = vocab_size
        self.eos_token_id = eos_token_id
        self.semantic_start_token_id = semantic_start_token_id
        self.semantic_end_token_id = semantic_end_token_id
        self.weight = nn.Parameter(weight.detach(), requires_grad=False)
        self.bias = (
            None if bias is None else nn.Parameter(bias.detach(), requires_grad=False)
        )
        self.register_buffer("candidate_ids", candidates)

    @property
    def device(self) -> torch.device:
        return self.weight.device

    @property
    def candidate_count(self) -> int:
        return int(self.weight.shape[0])

    @property
    def compact_size(self) -> int:
        return self.candidate_count

    @property
    def full_vocab_size(self) -> int:
        return self.out_features

    @property
    def weight_bytes(self) -> int:
        size = self.weight.numel() * self.weight.element_size()
        if self.bias is not None:
            size += self.bias.numel() * self.bias.element_size()
        return size

    @classmethod
    @torch.no_grad()
    def from_candidate_rows(
        cls,
        weight: Tensor,
        *,
        vocab_size: int,
        eos_token_id: int,
        semantic_start_token_id: int,
        semantic_end_token_id: int,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        bias: Tensor | None = None,
        bias_is_compact: bool = False,
    ) -> "SemanticHead":
        """Build from rows already restricted to the audio candidate IDs.

        This is the loader-facing constructor: a safetensors slice containing
        only the EOS and semantic rows can be copied to its final device without
        ever materializing the approximately 1 GB full vocabulary head.
        ``weight`` must be ordered by the sorted IDs returned in
        :attr:`candidate_ids`.
        """
        if not isinstance(weight, Tensor):
            raise TypeError("weight must be a torch.Tensor")
        if weight.is_meta:
            raise ValueError("Candidate rows must be loaded before materialization")
        if weight.ndim != 2:
            raise ValueError("weight must have shape [candidate_rows, hidden_size]")
        if not weight.is_floating_point():
            raise TypeError("Semantic head weights must retain a floating-point dtype")
        candidates = semantic_candidate_ids(
            vocab_size=vocab_size,
            eos_token_id=eos_token_id,
            semantic_start_token_id=semantic_start_token_id,
            semantic_end_token_id=semantic_end_token_id,
            device=weight.device,
        )
        if weight.shape[0] != candidates.numel():
            raise ValueError(
                "weight must contain one row for each sorted audio candidate"
            )
        target = weight.device if device is None else torch.device(device)
        target_dtype = weight.dtype if dtype is None else dtype
        if not isinstance(target_dtype, torch.dtype):
            raise TypeError("dtype must be a torch.dtype")
        if not torch.empty((), dtype=target_dtype).is_floating_point():
            raise TypeError("dtype must be floating point")
        # Clone the small slice so the compact parameter never retains a view of
        # a larger mmap/full-head allocation.
        compact_weight = weight.to(
            device=target,
            dtype=target_dtype,
            copy=True,
            memory_format=torch.contiguous_format,
        )
        compact_bias = None
        if bias is not None:
            if not isinstance(bias, Tensor) or bias.ndim != 1:
                raise ValueError("bias must be a one-dimensional tensor")
            if bias_is_compact:
                if bias.shape[0] != candidates.numel():
                    raise ValueError(
                        "compact bias must contain one value per audio candidate"
                    )
                selected_bias = bias
            else:
                if bias.shape[0] != vocab_size:
                    raise ValueError("full bias must contain one value per vocabulary row")
                selected_bias = bias.index_select(0, candidates.to(bias.device))
            if not selected_bias.is_floating_point():
                raise TypeError("Semantic head bias must be floating point")
            compact_bias = (
                selected_bias.to(
                    device=target,
                    dtype=target_dtype,
                    copy=True,
                    memory_format=torch.contiguous_format,
                )
            )
        return cls(
            compact_weight,
            vocab_size=vocab_size,
            eos_token_id=eos_token_id,
            semantic_start_token_id=semantic_start_token_id,
            semantic_end_token_id=semantic_end_token_id,
            bias=compact_bias,
        ).eval()

    @classmethod
    @torch.no_grad()
    def from_weight(
        cls,
        weight: Tensor,
        *,
        eos_token_id: int,
        semantic_start_token_id: int,
        semantic_end_token_id: int,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        bias: Tensor | None = None,
    ) -> "SemanticHead":
        """Build from a full ``[vocab, hidden]`` output weight.

        Candidate rows are selected before any device or dtype conversion. For a
        loader that already has a safetensors row slice, use
        :meth:`from_candidate_rows` to avoid allocating the full head.
        """
        if not isinstance(weight, Tensor):
            raise TypeError("weight must be a torch.Tensor")
        if weight.is_meta:
            raise ValueError("Load the official output head before selecting its rows")
        if weight.ndim != 2:
            raise ValueError("weight must have shape [vocab_size, hidden_size]")
        vocab_size = int(weight.shape[0])
        candidates = semantic_candidate_ids(
            vocab_size=vocab_size,
            eos_token_id=eos_token_id,
            semantic_start_token_id=semantic_start_token_id,
            semantic_end_token_id=semantic_end_token_id,
            device=weight.device,
        )
        rows = weight.index_select(0, candidates)
        selected_bias = None
        if bias is not None:
            if not isinstance(bias, Tensor) or bias.ndim != 1:
                raise ValueError("bias must be a one-dimensional tensor")
            if bias.shape[0] != vocab_size:
                raise ValueError("full bias must contain one value per vocabulary row")
            selected_bias = bias.index_select(0, candidates.to(bias.device))
        return cls.from_candidate_rows(
            rows,
            vocab_size=vocab_size,
            eos_token_id=eos_token_id,
            semantic_start_token_id=semantic_start_token_id,
            semantic_end_token_id=semantic_end_token_id,
            device=device,
            dtype=dtype,
            bias=selected_bias,
            bias_is_compact=True,
        )

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        eos_token_id: int,
        semantic_start_token_id: int,
        semantic_end_token_id: int,
        device: str | torch.device | None = None,
    ) -> "SemanticHead":
        """Copy only candidate rows, preserving their dtype and exact values."""
        if not isinstance(linear, nn.Linear):
            raise TypeError("from_linear requires the original nn.Linear output head")
        if linear.weight.is_meta:
            raise ValueError("Load the official output head before selecting its rows")
        return cls.from_weight(
            linear.weight,
            eos_token_id=eos_token_id,
            semantic_start_token_id=semantic_start_token_id,
            semantic_end_token_id=semantic_end_token_id,
            device=device,
            bias=linear.bias,
        ).train(linear.training)

    def compact_logits(self, hidden: Tensor) -> Tensor:
        """Compute logits in ``candidate_ids`` order without full-vocab padding."""
        return F.linear(hidden.to(self.device), self.weight, self.bias)

    def forward(self, hidden: Tensor) -> Tensor:
        """Return candidate logits at their original IDs for the upstream tower."""
        compact = self.compact_logits(hidden)
        full = compact.new_zeros((*compact.shape[:-1], self.out_features))
        return full.index_copy_(-1, self.candidate_ids, compact)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"candidate_count={self.candidate_count}, dtype={self.weight.dtype}"
        )


def install_semantic_head(
    backbone: nn.Module,
    config: Any,
    *,
    device: str | torch.device | None = None,
) -> SemanticHead:
    """Replace an already-loaded backbone head for audio-only generation.

    Callers can leave the original head installed for a full-vocabulary reference
    run. This function does not retain it, so its approximately 1 GB CPU/GPU
    allocation can be released after installation.
    """
    original = backbone.lm_head
    layout = {
        "eos_token_id": config.eos_token_id,
        "semantic_start_token_id": config.semantic_start_token_id,
        "semantic_end_token_id": config.semantic_end_token_id,
    }
    if isinstance(original, SemanticHead):
        if original.out_features != config.text.vocab_size or any(
            getattr(original, key) != value for key, value in layout.items()
        ):
            raise ValueError("Installed semantic head does not match the model config")
        return original if device is None else original.to(device)
    if original.out_features != config.text.vocab_size:
        raise ValueError("Output head vocabulary does not match the model config")
    compact = SemanticHead.from_linear(original, device=device, **layout)
    backbone.lm_head = compact
    return compact
