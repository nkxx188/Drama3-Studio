"""Additional exact output rows leave every base candidate logit unchanged."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from drama_low_vram.control_head import ControlHead
from drama_low_vram.semantic_head import SemanticHead


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_control_logits_retain_base_gemm_bytes(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(81)
    original = nn.Linear(128, 64, bias=False, device=device, dtype=torch.bfloat16)
    head = SemanticHead.from_linear(
        original, eos_token_id=3, semantic_start_token_id=8, semantic_end_token_id=15)
    extra = ControlHead(
        torch.tensor([27], device=device),
        original.weight[27:28].detach())
    x = torch.randn(1, 128, device=device, dtype=torch.bfloat16)
    before = head.compact_logits(x)
    with_controls = extra.logits(x, head)
    ids = extra.candidates(head)
    selection = torch.nonzero(torch.isin(ids, head.candidate_ids)).reshape(-1)
    assert torch.equal(before.view(torch.uint8), with_controls[:, selection].contiguous().view(torch.uint8))
    assert ids.tolist() == [3, *range(8, 16), 27]
    assert torch.equal(extra.weight, original.weight[27:28])
    assert torch.equal(before, head.compact_logits(x))


def test_duplicate_control_rows_are_rejected():
    with pytest.raises(ValueError):
        ControlHead(torch.tensor([1, 1]), torch.zeros(2, 8))
