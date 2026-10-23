import json
import gc
import ctypes
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from drama_low_vram.expert_cache_file import (
    CACHE_FORMAT,
    ExpertCacheFile,
    build_expert_cache,
    inspect_expert_cache_source,
)
from drama_low_vram.offload import EXPERT_BUFFER_NAMES, FP8_DTYPE


def _fixture_checkpoint(
    tmp_path: Path, *, num_layers: int = 2, num_experts: int = 3
) -> Path:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    hidden = intermediate = 128
    tensors = {}
    weight_map = {}
    for layer in range(num_layers):
        for expert in range(num_experts):
            for projection_index, (projection, shape) in enumerate(
                (
                    ("gate_proj", (intermediate, hidden)),
                    ("up_proj", (intermediate, hidden)),
                    ("down_proj", (hidden, intermediate)),
                )
            ):
                base = (
                    f"model.language_model.layers.{layer}.mlp.experts."
                    f"{expert}.{projection}"
                )
                generator = torch.Generator().manual_seed(
                    100 + layer * 20 + expert * 3 + projection_index
                )
                tensors[f"{base}.weight"] = torch.randint(
                    0, 255, shape, generator=generator, dtype=torch.uint8
                ).view(FP8_DTYPE)
                scale_shape = (shape[0] // 128, shape[1] // 128)
                tensors[f"{base}.weight_scale_inv"] = torch.rand(
                    scale_shape, generator=generator, dtype=torch.float32
                )
    for shard_index in range(2):
        shard_name = f"model-{shard_index + 1:05d}-of-00002.safetensors"
        shard_tensors = {
            name: tensor
            for index, (name, tensor) in enumerate(tensors.items())
            if index % 2 == shard_index
        }
        save_file(shard_tensors, str(checkpoint / shard_name))
        weight_map.update({name: shard_name for name in shard_tensors})
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "model_type": "fish_omni",
                "text_config": {
                    "num_hidden_layers": num_layers,
                    "num_experts": num_experts,
                    "hidden_size": hidden,
                    "moe_intermediate_size": intermediate,
                },
            }
        ),
        encoding="utf-8",
    )
    return checkpoint


class _Experts(nn.Module):
    def __init__(self, num_experts=3) -> None:
        super().__init__()
        self.fp8 = True
        self.num_experts = num_experts
        self.register_buffer(
            "w13_weight",
            torch.empty(num_experts, 256, 128, dtype=FP8_DTYPE, device="meta"),
        )
        self.register_buffer(
            "w13_scale_inv",
            torch.empty(num_experts, 2, 1, dtype=torch.float32, device="meta"),
        )
        self.register_buffer(
            "w2_weight",
            torch.empty(num_experts, 128, 128, dtype=FP8_DTYPE, device="meta"),
        )
        self.register_buffer(
            "w2_scale_inv",
            torch.empty(num_experts, 1, 1, dtype=torch.float32, device="meta"),
        )


def _backbone(num_layers=2, num_experts=3) -> nn.Module:
    root = nn.Module()
    layers = nn.ModuleList()
    for _ in range(num_layers):
        layer = nn.Module()
        layer.mlp = nn.Module()
        layer.mlp.experts = _Experts(num_experts)
        layers.append(layer)
    root.model = nn.Module()
    root.model.layers = layers
    return root


def test_inspect_and_build_keep_official_bytes(tmp_path: Path):
    checkpoint = _fixture_checkpoint(tmp_path)
    source = inspect_expert_cache_source(checkpoint)
    assert source.num_layers == 2
    assert source.num_experts == 3
    assert source.nbytes == 6 * (3 * 128 * 128 + 3 * 4)

    cache_dir = tmp_path / "cache"
    build_expert_cache(checkpoint, cache_dir, disk_reserve_bytes=0)
    cache = ExpertCacheFile.open(cache_dir, checkpoint_dir=checkpoint, verify_data=True)
    assert cache.model_fingerprint == source.model_fingerprint
    assert not cache.is_mapped
    buffers = cache.layer_buffers(0)
    assert buffers["w13_weight"].shape == (3, 256, 128)
    assert buffers["w2_weight"].shape == (3, 128, 128)
    assert buffers["w13_weight"].dtype == FP8_DTYPE

    rebuilt = _backbone()
    assert cache.assign_to_backbone(rebuilt) == 2
    assert rebuilt.model.layers[0].mlp.experts.w13_weight.data_ptr() == buffers[
        "w13_weight"
    ].data_ptr()
    assert rebuilt.model.layers[0].mlp.experts._drama_expert_storage == "mmap"
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    for layer in range(2):
        actual = cache.layer_buffers(layer)
        for expert in range(3):
            for projection, target, half in (
                ("gate_proj", "w13", slice(0, 128)),
                ("up_proj", "w13", slice(128, 256)),
                ("down_proj", "w2", slice(0, 128)),
            ):
                for kind, suffix in (
                    ("weight", "weight"),
                    ("weight_scale_inv", "scale_inv"),
                ):
                    name = (
                        f"model.language_model.layers.{layer}.mlp.experts."
                        f"{expert}.{projection}.{kind}"
                    )
                    with safe_open(
                        str(checkpoint / index["weight_map"][name]),
                        framework="pt",
                        device="cpu",
                    ) as handle:
                        expected = handle.get_tensor(name)
                    row_slice = half
                    if suffix == "scale_inv":
                        row_slice = slice(half.start // 128, half.stop // 128)
                    tensor = actual[f"{target}_{suffix}"].view(torch.uint8)[
                        expert, row_slice
                    ]
                    assert torch.equal(tensor, expected.view(torch.uint8))
    cache.close()
    del cache
    gc.collect()
    assert torch.equal(
        rebuilt.model.layers[0].mlp.experts.w13_weight.view(torch.uint8),
        buffers["w13_weight"].view(torch.uint8),
    )


def test_partial_or_wrong_source_is_rejected(tmp_path: Path):
    checkpoint = _fixture_checkpoint(tmp_path)
    cache_dir = tmp_path / "cache"
    build_expert_cache(checkpoint, cache_dir, disk_reserve_bytes=0)
    (cache_dir / "experts.bin").write_bytes(
        (cache_dir / "experts.bin").read_bytes()[:-1]
    )
    with pytest.raises(ValueError, match="truncated|extra"):
        ExpertCacheFile.open(cache_dir)


def test_build_does_not_overwrite_without_force(tmp_path: Path):
    checkpoint = _fixture_checkpoint(tmp_path)
    cache_dir = tmp_path / "cache"
    build_expert_cache(checkpoint, cache_dir, disk_reserve_bytes=0)
    with pytest.raises(FileExistsError):
        build_expert_cache(checkpoint, cache_dir, disk_reserve_bytes=0)


def test_failure_does_not_leave_an_openable_partial_cache(tmp_path: Path):
    checkpoint = _fixture_checkpoint(tmp_path)
    cache_dir = tmp_path / "cache"

    def stop(copied, total):
        raise OSError("simulated disk failure")

    with pytest.raises(OSError, match="simulated"):
        build_expert_cache(
            checkpoint, cache_dir, disk_reserve_bytes=0, progress=stop
        )
    assert not cache_dir.exists()
    assert list(tmp_path.glob(".cache.partial-*")) == []


def test_force_keeps_prior_cache_when_replacement_write_fails(tmp_path: Path):
    checkpoint = _fixture_checkpoint(tmp_path)
    cache_dir = tmp_path / "cache"
    build_expert_cache(checkpoint, cache_dir, disk_reserve_bytes=0)
    previous = (cache_dir / "experts.bin").read_bytes()

    def stop(copied, total):
        raise RuntimeError("injected failure")

    with pytest.raises(RuntimeError, match="injected"):
        build_expert_cache(
            checkpoint, cache_dir, force=True, disk_reserve_bytes=0, progress=stop
        )
    assert (cache_dir / "experts.bin").read_bytes() == previous
    build_expert_cache(checkpoint, cache_dir, force=True, disk_reserve_bytes=0)
    assert (cache_dir / "experts.bin").read_bytes() == previous


def test_source_identity_change_is_rejected(tmp_path: Path):
    checkpoint = _fixture_checkpoint(tmp_path)
    cache_dir = tmp_path / "cache"
    build_expert_cache(checkpoint, cache_dir, disk_reserve_bytes=0)
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    config["changed"] = True
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        ExpertCacheFile.open(cache_dir, checkpoint_dir=checkpoint)


def test_checksum_detects_same_size_data_corruption(tmp_path: Path):
    checkpoint = _fixture_checkpoint(tmp_path)
    cache_dir = tmp_path / "cache"
    build_expert_cache(checkpoint, cache_dir, disk_reserve_bytes=0)
    with (cache_dir / "experts.bin").open("r+b") as handle:
        first_byte = handle.read(1)
        handle.seek(0)
        handle.write(bytes([first_byte[0] ^ 1]))
    with pytest.raises(ValueError, match="checksum"):
        ExpertCacheFile.open(cache_dir, verify_data=True)


@pytest.mark.parametrize("change", ["dtype", "shape", "offset", "nbytes", "filename"])
def test_manifest_layout_tampering_is_rejected(tmp_path: Path, change: str):
    checkpoint = _fixture_checkpoint(tmp_path)
    cache_dir = tmp_path / "cache"
    build_expert_cache(checkpoint, cache_dir, disk_reserve_bytes=0)
    path = cache_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    if change == "filename":
        manifest["data"]["name"] = "../elsewhere.bin"
    else:
        spec = manifest["layers"][0]["buffers"]["w13_weight"]
        if change == "dtype":
            spec["dtype"] = "float32"
        elif change == "shape":
            spec["shape"][1] = 128
        elif change == "offset":
            spec["offset"] += 1
        else:
            spec["nbytes"] -= 1
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="layout|filename"):
        ExpertCacheFile.open(cache_dir)


def test_missing_scale_is_rejected_before_cache_creation(tmp_path: Path):
    checkpoint = _fixture_checkpoint(tmp_path)
    index_path = checkpoint / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"].pop(next(iter(index["weight_map"])))
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing expert"):
        build_expert_cache(checkpoint, tmp_path / "cache", disk_reserve_bytes=0)
    assert not (tmp_path / "cache").exists()


def test_insufficient_disk_is_rejected_before_copy(tmp_path: Path):
    checkpoint = _fixture_checkpoint(tmp_path)
    with patch(
        "drama_low_vram.expert_cache_file.shutil.disk_usage",
        return_value=SimpleNamespace(free=0),
    ):
        with pytest.raises(OSError, match="disk has"):
            build_expert_cache(checkpoint, tmp_path / "cache")
    assert not (tmp_path / "cache").exists()


def test_binding_validates_all_layers_before_mutation(tmp_path: Path):
    checkpoint = _fixture_checkpoint(tmp_path)
    cache_dir = tmp_path / "cache"
    build_expert_cache(checkpoint, cache_dir, disk_reserve_bytes=0)
    cache = ExpertCacheFile.open(cache_dir)
    backbone = _backbone()
    backbone.model.layers[1].mlp.experts._buffers["w2_weight"] = torch.empty(
        3, 128, 256, dtype=FP8_DTYPE, device="meta"
    )
    with pytest.raises(ValueError, match="shape/dtype"):
        cache.assign_to_backbone(backbone)
    assert backbone.model.layers[0].mlp.experts.w13_weight.device.type == "meta"
    assert not cache.is_mapped


def test_only_routed_expert_checkpoint_keys_are_skipped(tmp_path: Path):
    checkpoint = _fixture_checkpoint(tmp_path)
    cache_dir = tmp_path / "cache"
    build_expert_cache(checkpoint, cache_dir, disk_reserve_bytes=0)
    cache = ExpertCacheFile.open(cache_dir)
    assert cache.should_skip_checkpoint_key(
        "model.language_model.layers.1.mlp.experts.2.gate_proj.weight"
    )
    assert cache.should_skip_checkpoint_key(
        "model.layers.1.mlp.experts.2.gate_proj.weight_scale_inv"
    )
    for name in (
        "model.language_model.layers.2.mlp.experts.2.gate_proj.weight",
        "model.language_model.layers.0.mlp.experts.3.gate_proj.weight",
        "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
        "model.language_model.layers.0.mlp.gate.weight",
        "audio_decoder.layers.0.mlp.gate_proj.weight",
    ):
        assert not cache.should_skip_checkpoint_key(name)


def test_no_quantized_tensor_copy_is_allocated_while_mapping(tmp_path: Path):
    checkpoint = _fixture_checkpoint(tmp_path)
    cache_dir = tmp_path / "cache"
    build_expert_cache(checkpoint, cache_dir, disk_reserve_bytes=0)
    cache = ExpertCacheFile.open(cache_dir)
    with patch("torch.empty", side_effect=AssertionError("eager allocation")):
        buffers = cache.layer_buffers(0)
    assert buffers["w13_weight"].untyped_storage().nbytes() == cache.nbytes
    assert all(buffer.is_contiguous() for buffer in buffers.values())
    with pytest.raises(TypeError):
        cache._mmap[0] = 1


def test_unknown_expert_key_is_rejected_before_dense_loader_can_skip_it(tmp_path: Path):
    checkpoint = _fixture_checkpoint(tmp_path)
    index_path = checkpoint / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"][
        "model.language_model.layers.0.mlp.experts.0.gate_proj.zeros"
    ] = next(iter(index["weight_map"].values()))
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported routed expert"):
        inspect_expert_cache_source(checkpoint)


def test_source_bad_dtype_and_shape_are_rejected(tmp_path: Path):
    checkpoint = _fixture_checkpoint(tmp_path)
    path = next(checkpoint.glob("*.safetensors"))
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        tensors = {name: handle.get_tensor(name).clone() for name in handle.keys()}
    weight_name = next(name for name in tensors if name.endswith(".weight"))
    tensors[weight_name] = tensors[weight_name].float()
    save_file(tensors, str(path))
    with pytest.raises(ValueError, match="Wrong expert shape/dtype"):
        inspect_expert_cache_source(checkpoint)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows commit accounting")
def test_readonly_mapping_does_not_commit_the_whole_expert_bank(tmp_path: Path):
    from drama_low_vram.expert_cache_file import (
        CACHE_FORMAT_VERSION,
        _layout,
    )

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(Counters),
        ctypes.c_ulong,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int

    def private_usage():
        result = Counters()
        result.cb = ctypes.sizeof(result)
        if not psapi.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(result), result.cb
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return result.PrivateUsage

    layers, nbytes = _layout(1, 256, 256, 256)
    cache_dir = tmp_path / "mapping-cache"
    cache_dir.mkdir()
    with (cache_dir / "experts.bin").open("wb") as handle:
        handle.truncate(nbytes)
    manifest = {
        "format": CACHE_FORMAT,
        "format_version": CACHE_FORMAT_VERSION,
        "byte_order": "little",
        "model_fingerprint": "0" * 64,
        "model": {
            "model_type": "fish_omni",
            "num_layers": 1,
            "num_experts": 256,
            "hidden": 256,
            "intermediate": 256,
            "block_size": 128,
        },
        "data": {"name": "experts.bin", "size": nbytes, "sha256": "0" * 64},
        "layers": layers,
    }
    (cache_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    gc.collect()
    before = private_usage()
    cache = ExpertCacheFile.open(cache_dir)
    buffers = cache.layer_buffers(0)
    delta = private_usage() - before
    # Leave room for Python/torch metadata; a COW view would charge about 48MiB.
    assert nbytes > 48 * 1024**2
    assert delta < 8 * 1024**2, delta
    assert buffers["w13_weight"].device.type == "cpu"
    del buffers
    cache.close()
    del cache
    gc.collect()


def _dense_backbone():
    backbone = _backbone()
    backbone.model.embed_tokens = nn.Embedding(7, 128, device="meta")
    backbone.lm_head = nn.Linear(128, 7, bias=False, device="meta")
    for layer in backbone.model.layers:
        layer.input_layernorm = nn.LayerNorm(128, device="meta")
        layer.mlp.gate = nn.Linear(128, 3, bias=False, device="meta")
        layer.mlp.shared_expert = nn.Linear(128, 128, bias=False, device="meta")
    return backbone


def _dense_checkpoint_weights(backbone):
    generator = torch.Generator().manual_seed(913)
    result = []
    for name, parameter in backbone.named_parameters():
        checkpoint_name = (
            name.replace("model.", "model.language_model.", 1)
            if name.startswith("model.")
            else name
        )
        tensor = torch.randn(tuple(parameter.shape), generator=generator)
        result.append((checkpoint_name, tensor))
    return result


def test_mmap_and_dense_loader_match_official_complete_loader(tmp_path: Path):
    from drama_low_vram.model import _load_dense_backbone
    from drama_low_vram.offload import materialize_offloaded_backbone
    from fish_speech.models.text2semantic.fish_omni.backbone import load_backbone_weights

    checkpoint = _fixture_checkpoint(tmp_path)
    cache_dir = tmp_path / "cache"
    build_expert_cache(checkpoint, cache_dir, disk_reserve_bytes=0)
    mapped = _dense_backbone()
    baseline = _dense_backbone()
    dense_weights = _dense_checkpoint_weights(mapped)
    placement = materialize_offloaded_backbone(
        mapped, "cpu", embedding_device="cpu", expert_device="meta"
    )
    for layer in mapped.model.layers:
        assert layer.mlp.experts.w13_weight.is_meta
    assert placement["model.layers.0.mlp.experts"] == "meta"
    cache = ExpertCacheFile.open(cache_dir, checkpoint_dir=checkpoint)
    cache.assign_to_backbone(mapped)
    _load_dense_backbone(mapped, dense_weights)

    baseline.to_empty(device="cpu")
    all_weights = list(dense_weights)
    for path in checkpoint.glob("*.safetensors"):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            all_weights.extend((name, handle.get_tensor(name)) for name in handle.keys())
    load_backbone_weights(baseline, all_weights)
    for name, tensor in mapped.state_dict().items():
        assert torch.equal(
            tensor.view(torch.uint8), baseline.state_dict()[name].view(torch.uint8)
        ), name
    cache.close()


def test_dense_loader_can_leave_full_head_meta_for_direct_compact_loading():
    from drama_low_vram.model import _load_dense_backbone
    from drama_low_vram.offload import materialize_offloaded_backbone
    from drama_low_vram.semantic_head import SemanticHead

    backbone = _dense_backbone()
    weights = _dense_checkpoint_weights(backbone)
    original_head = next(value for name, value in weights if name == "lm_head.weight")
    materialize_offloaded_backbone(
        backbone, "cpu", lm_head_device="meta", expert_device="meta"
    )
    _load_dense_backbone(
        backbone, [(name, value) for name, value in weights if name != "lm_head.weight"],
        skip_head=True,
    )
    assert backbone.lm_head.weight.is_meta
    backbone.lm_head = SemanticHead.from_weight(
        original_head, eos_token_id=1, semantic_start_token_id=3,
        semantic_end_token_id=6,
    )
    assert torch.equal(backbone.lm_head.weight, original_head[[1, 3, 4, 5, 6]])


@pytest.mark.parametrize("invalid", ["missing", "duplicate", "unknown", "shape"])
def test_dense_loader_keeps_checkpoint_validation(tmp_path: Path, invalid):
    from drama_low_vram.model import _load_dense_backbone

    backbone = _dense_backbone()
    backbone.to_empty(device="cpu")
    weights = _dense_checkpoint_weights(backbone)
    if invalid == "missing":
        weights.pop()
        match = "missing"
    elif invalid == "duplicate":
        weights.append(weights[0])
        match = "duplicate"
    elif invalid == "unknown":
        weights.append(("model.language_model.unknown.weight", torch.zeros(1)))
        match = "Unknown"
    else:
        weights[0] = (weights[0][0], torch.zeros(1))
        match = "shape"
    with pytest.raises(ValueError, match=match):
        _load_dense_backbone(backbone, weights)
