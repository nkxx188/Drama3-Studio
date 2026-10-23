"""The compact head must retain every audio candidate and official sampling."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from drama_low_vram.semantic_head import (
    SemanticHead,
    install_semantic_head,
    semantic_candidate_ids,
)


def make_head(*, eos=3, dtype=torch.bfloat16, bias=False):
    torch.manual_seed(217)
    original = nn.Linear(32, 64, bias=bias, dtype=dtype).eval()
    compact = SemanticHead.from_linear(
        original,
        eos_token_id=eos,
        semantic_start_token_id=40,
        semantic_end_token_id=47,
    )
    return original, compact


@pytest.mark.parametrize("eos,expected", [(3, [3, *range(40, 48)]), (63, [*range(40, 48), 63])])
def test_candidates_match_official_tower_sort_order(eos, expected):
    _, compact = make_head(eos=eos)
    assert compact.candidate_ids.tolist() == expected
    assert compact.candidate_count == 9


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("leading_shape", [(), (1,), (4,), (2, 3)])
def test_exact_candidate_rows_and_cpu_logits(bias, leading_shape):
    original, compact = make_head(bias=bias)
    hidden = torch.randn(*leading_shape, 32, dtype=original.weight.dtype)
    expected = original(hidden)
    actual = compact(hidden)
    ids = compact.candidate_ids
    assert compact.weight.dtype == original.weight.dtype
    assert torch.equal(compact.weight, original.weight[ids])
    assert compact.weight.untyped_storage().data_ptr() != original.weight.untyped_storage().data_ptr()
    assert compact.weight.numel() < original.weight.numel()
    assert compact.weight.requires_grad is False
    if bias:
        assert torch.equal(compact.bias, original.bias[ids])
    torch.testing.assert_close(actual[..., ids], expected[..., ids], rtol=0, atol=0)
    torch.testing.assert_close(
        compact.compact_logits(hidden), actual[..., ids], rtol=0, atol=0
    )
    outside = torch.ones(original.out_features, dtype=torch.bool)
    outside[ids] = False
    assert actual.shape == expected.shape
    assert torch.count_nonzero(actual[..., outside]).item() == 0
    assert torch.isfinite(actual).all()
    assert compact.device == original.weight.device
    assert compact.in_features == original.in_features
    assert compact.out_features == original.out_features
    assert compact.training is False


def test_upstream_tower_sampling_uses_identical_candidates():
    from fish_speech.models.text2semantic.fish_omni.tower import (
        FishOmniAudioDecoder,
    )

    # The real decode_step samples the main token; empty layer lists isolate that
    # contract without requiring a complete tower or a large checkpoint.
    tower = object.__new__(FishOmniAudioDecoder)
    nn.Module.__init__(tower)
    tower.config = SimpleNamespace(vocab_size=8)
    tower.semantic_begin = 40
    tower.semantic_end = 47
    tower.eos_id = 3
    tower.main_candidates = None
    tower.codebook_embeddings = nn.Embedding(8, 32, dtype=torch.bfloat16)
    tower.shared_layers = nn.ModuleList()
    tower.residual_layers = nn.ModuleList()
    tower.codebook_heads = nn.ModuleList()
    tower.run_shared = lambda hidden, *args: hidden
    original, compact = make_head()
    hidden = torch.randn(4, 32, dtype=torch.bfloat16)
    args = (
        hidden,
        torch.arange(4),
        torch.zeros(4, 2, dtype=torch.long),
        torch.tensor([40, 42, 44]),
    )
    sampling = {"temperature": 0.8, "top_k": 6, "top_p": 0.95}
    torch.manual_seed(913)
    expected = tower.decode_step(original(hidden), *args, **sampling)
    torch.manual_seed(913)
    actual = tower.decode_step(compact(hidden), *args, **sampling)
    assert torch.equal(actual, expected)
    assert torch.equal(tower.main_candidates, compact.candidate_ids)


def test_install_releases_full_head_and_is_idempotent():
    original, _ = make_head()
    backbone = nn.Module()
    backbone.lm_head = original
    config = SimpleNamespace(
        text=SimpleNamespace(vocab_size=64),
        eos_token_id=3,
        semantic_start_token_id=40,
        semantic_end_token_id=47,
    )
    compact = install_semantic_head(backbone, config, device="cpu")
    assert backbone.lm_head is compact
    assert install_semantic_head(backbone, config) is compact
    assert "lm_head.weight" in backbone.state_dict()
    assert len(list(backbone.parameters())) == 1
    assert compact.weight_bytes == 9 * 32 * 2
    config.eos_token_id = 4
    with pytest.raises(ValueError, match="does not match"):
        install_semantic_head(backbone, config)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"vocab_size": 0}, ValueError),
        ({"semantic_start_token_id": -1}, ValueError),
        ({"semantic_end_token_id": 64}, ValueError),
        ({"semantic_start_token_id": 48}, ValueError),
        ({"eos_token_id": 44}, ValueError),
        ({"eos_token_id": 64}, ValueError),
        ({"eos_token_id": True}, TypeError),
    ],
)
def test_invalid_candidate_layouts_fail_before_sampling(kwargs, error):
    layout = {
        "vocab_size": 64,
        "eos_token_id": 3,
        "semantic_start_token_id": 40,
        "semantic_end_token_id": 47,
    }
    layout.update(kwargs)
    with pytest.raises(error):
        semantic_candidate_ids(**layout)


def test_unloaded_or_nonfloating_head_is_rejected():
    layout = {
        "eos_token_id": 3,
        "semantic_start_token_id": 40,
        "semantic_end_token_id": 47,
    }
    with pytest.raises(ValueError, match="Load the official"):
        SemanticHead.from_linear(nn.Linear(32, 64, device="meta"), **layout)
    with pytest.raises(TypeError, match="floating-point"):
        SemanticHead(torch.zeros(9, 32, dtype=torch.int8), vocab_size=64, **layout)


@pytest.mark.parametrize("dtype", [None, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("eos", [3, 63])
def test_from_weight_selects_before_dtype_conversion_and_preserves_ids(dtype, eos):
    torch.manual_seed(201)
    full_weight = torch.randn(64, 32, dtype=torch.bfloat16)
    full_bias = torch.randn(64, dtype=torch.bfloat16)
    head = SemanticHead.from_weight(
        full_weight,
        eos_token_id=eos,
        semantic_start_token_id=40,
        semantic_end_token_id=47,
        dtype=dtype,
        bias=full_bias,
    )
    final_dtype = full_weight.dtype if dtype is None else dtype
    assert head.weight.dtype == final_dtype
    assert head.bias.dtype == final_dtype
    assert torch.equal(head.weight, full_weight[head.candidate_ids].to(final_dtype))
    assert torch.equal(head.bias, full_bias[head.candidate_ids].to(final_dtype))
    assert head.weight.untyped_storage().data_ptr() != full_weight.untyped_storage().data_ptr()
    assert head.training is False
    assert head.weight.shape == (9, 32)


@pytest.mark.parametrize("bias_is_compact", [False, True])
def test_from_candidate_rows_detaches_mmap_style_views_and_bias(bias_is_compact):
    full_weight = torch.arange(64 * 32, dtype=torch.float32).reshape(64, 32)
    full_bias = torch.arange(64, dtype=torch.float32)
    rows = torch.cat((full_weight[3:4], full_weight[40:48]))
    bias = full_bias[torch.tensor([3, *range(40, 48)])] if bias_is_compact else full_bias
    head = SemanticHead.from_candidate_rows(
        rows,
        vocab_size=64,
        eos_token_id=3,
        semantic_start_token_id=40,
        semantic_end_token_id=47,
        device="cpu",
        dtype=torch.bfloat16,
        bias=bias,
        bias_is_compact=bias_is_compact,
    )
    torch.testing.assert_close(head.weight.float(), rows.to(torch.bfloat16).float(), rtol=0, atol=0)
    assert torch.equal(head.bias, full_bias[head.candidate_ids].to(torch.bfloat16))
    assert head.weight.untyped_storage().data_ptr() != rows.untyped_storage().data_ptr()
    assert head.bias.untyped_storage().data_ptr() != bias.untyped_storage().data_ptr()
    assert head.weight_bytes == (9 * 32 + 9) * 2


def test_from_candidate_rows_noncontiguous_view_becomes_independent_parameter():
    full_storage = torch.randn(9, 64, dtype=torch.bfloat16)
    rows = full_storage[:, ::2]
    assert not rows.is_contiguous()
    head = SemanticHead.from_candidate_rows(
        rows,
        vocab_size=64,
        eos_token_id=3,
        semantic_start_token_id=40,
        semantic_end_token_id=47,
    )
    assert head.weight.is_contiguous()
    assert torch.equal(head.weight, rows)
    full_storage.zero_()
    assert head.weight.count_nonzero() > 0


def test_from_linear_retains_training_mode_without_sharing_parameter_storage():
    linear = nn.Linear(32, 64, dtype=torch.bfloat16).train()
    compact = SemanticHead.from_linear(
        linear, eos_token_id=3, semantic_start_token_id=40, semantic_end_token_id=47
    )
    assert compact.training
    assert compact.weight.requires_grad is False
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.zero_()
    assert compact.weight.count_nonzero() > 0
    assert compact.bias.count_nonzero() > 0


@pytest.mark.parametrize(
    "weight,bias,dtype,error",
    [
        (torch.randn(8, 32), None, None, ValueError),
        (torch.randn(9, 32), torch.randn(10), None, ValueError),
        (torch.randn(9, 32), None, torch.int8, TypeError),
        (torch.randn(9, 32), None, "bfloat16", TypeError),
        (torch.empty(9, 32, device="meta"), None, None, ValueError),
        (torch.zeros(9, 32, dtype=torch.int8), None, None, TypeError),
    ],
)
def test_from_candidate_rows_rejects_invalid_materialization_inputs(weight, bias, dtype, error):
    with pytest.raises(error):
        SemanticHead.from_candidate_rows(
            weight,
            vocab_size=64,
            eos_token_id=3,
            semantic_start_token_id=40,
            semantic_end_token_id=47,
            bias=bias,
            dtype=dtype,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_placement_preserves_rows_and_finite_full_vocab_contract():
    original, _ = make_head()
    device = torch.device("cuda", torch.cuda.current_device())
    compact = SemanticHead.from_linear(
        original,
        eos_token_id=3,
        semantic_start_token_id=40,
        semantic_end_token_id=47,
        device=device,
    )
    hidden = torch.randn(3, 32, dtype=torch.bfloat16, device=device)
    full_gpu = original.to(device)
    actual = compact(hidden)
    assert compact.device == device
    assert compact.candidate_ids.device == device
    torch.testing.assert_close(
        actual[:, compact.candidate_ids],
        full_gpu(hidden)[:, compact.candidate_ids],
        rtol=0,
        atol=0,
    )
    assert torch.isfinite(actual).all()
