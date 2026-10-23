"""Small exact output rows for task controls, separate from the base audio GEMM."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .checkpoint_reader import iter_readonly_checkpoint_tensors
from .semantic_head import SemanticHead


class ControlHead(nn.Module):
    def __init__(self, ids, weights):
        super().__init__()
        if (ids.ndim != 1 or ids.dtype != torch.int64 or weights.ndim != 2
                or len(ids) != len(weights) or len(ids) < 1
                or len(set(ids.cpu().tolist())) != len(ids)):
            raise ValueError("Control rows require unique int64 IDs and matching weights")
        self.register_buffer("ids", ids.detach().clone())
        self.weight = nn.Parameter(weights.detach().clone(), requires_grad=False)

    @classmethod
    def load(cls, checkpoint, tokenizer, head, tokens=("<|audio_separator|>",)):
        if not isinstance(head, SemanticHead):
            raise ValueError("Task control head requires the compact semantic head")
        values = [tokenizer.token_to_id(token) for token in tokens]
        if any(value is None for value in values):
            raise ValueError("Tokenizer lacks a required output control token")
        if any(value < 0 or value >= head.out_features for value in values):
            raise ValueError("Output control ID outside model vocabulary")
        base = set(head.candidate_ids.cpu().tolist())
        if any(value in base for value in values):
            raise ValueError("Control output rows must be outside base audio candidates")
        selected = torch.tensor(sorted(set(values)), dtype=torch.int64)
        from .shared_checkpoint import is_shared
        if is_shared(checkpoint):
            from .checkpoint_reader import iter_readonly_file_tensors
            from pathlib import Path
            weights = dict(iter_readonly_file_tensors(Path(checkpoint) / "control-head.safetensors"))
            ids = weights.get("ids")
            rows = weights.get("weight")
            if ids is None or rows is None:
                raise ValueError("Shared checkpoint lacks preserved task control rows")
            lookup = {int(token): index for index, token in enumerate(ids.tolist())}
            if any(int(token) not in lookup for token in selected):
                raise ValueError("Shared checkpoint lacks requested task control rows")
            rows = rows[[lookup[int(token)] for token in selected]].to(
                device=head.device, dtype=head.weight.dtype, copy=True)
            return cls(selected.to(head.device), rows)
        iterator = iter_readonly_checkpoint_tensors(checkpoint, include_prefixes=("lm_head.weight",))
        try:
            key, weight = next(iterator)
            if key != "lm_head.weight" or weight.shape != (head.out_features, head.in_features):
                raise ValueError("Official output head geometry does not match")
            rows = weight.index_select(0, selected).to(
                device=head.device, dtype=head.weight.dtype, copy=True)
            if next(iterator, None) is not None:
                raise ValueError("Duplicate official output head")
        finally:
            iterator.close()
        return cls(selected.to(head.device), rows)

    def candidates(self, head):
        return torch.cat((head.candidate_ids, self.ids)).sort().values

    def logits(self, hidden, head):
        # Preserve the original 4097-row GEMM exactly; add only the control GEMM.
        base = head.compact_logits(hidden)
        extra = F.linear(hidden.to(self.weight.device), self.weight)
        ids = torch.cat((head.candidate_ids, self.ids))
        return torch.cat((base, extra), dim=-1).index_select(-1, ids.argsort())
