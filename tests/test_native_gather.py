"""Byte-copy staging, failure atomicity and optional native-library fallback."""

from contextlib import closing
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from drama_low_vram.native_backend import NativeLibrary, default_library_path
from drama_low_vram.offload import (
    EXPERT_BUFFER_NAMES, ExpertOffloadManager, ExpertStore, FP8_DTYPE,
)


@pytest.fixture
def library():
    if not default_library_path().is_file():
        pytest.skip("Optional native library required")
    result = NativeLibrary()
    if result.gather_rows is None:
        pytest.skip("Optional native gather extension required")
    return result


@pytest.fixture
def store():
    rng = torch.Generator().manual_seed(237)
    # Every FP8 byte pattern, including NaNs and signed zeros, must survive.
    return ExpertStore(
        torch.randint(256, (4, 256, 128), dtype=torch.uint8, generator=rng).view(FP8_DTYPE),
        torch.randn(4, 2, 1, generator=rng),
        torch.randint(256, (4, 128, 128), dtype=torch.uint8, generator=rng).view(FP8_DTYPE),
        torch.randn(4, 1, 1, generator=rng),
    )


def targets_for(store, count):
    return {
        name: torch.zeros((count, *getattr(store, name).shape[1:]), dtype=getattr(store, name).dtype)
        for name in EXPERT_BUFFER_NAMES
    }


@pytest.mark.parametrize("routes", [[3], [2, 0], [3, 0, 3, 1], [3, 2, 1, 0, 3, 3, 2, 1]])
def test_gather_preserves_every_byte_and_duplicate_order(library, store, routes):
    ids = torch.tensor(routes)
    targets = targets_for(store, len(routes))
    originals = {name: getattr(store, name).view(torch.uint8).clone() for name in EXPERT_BUFFER_NAMES}
    library.gather_experts(store, ids, targets)
    for name, target in targets.items():
        assert torch.equal(target.view(torch.uint8), originals[name].index_select(0, ids))
        assert torch.equal(getattr(store, name).view(torch.uint8), originals[name])


@pytest.mark.parametrize("routes", [[0, -1], [0, 4]])
def test_invalid_last_id_leaves_every_target_unchanged(library, store, routes):
    targets = targets_for(store, 2)
    with pytest.raises(RuntimeError, match="outside bank"):
        library.gather_experts(store, torch.tensor(routes), targets)
    assert all(not value.view(torch.uint8).count_nonzero() for value in targets.values())


def test_bad_last_target_and_alias_are_rejected_before_any_write(library, store):
    ids = torch.tensor([0])
    targets = targets_for(store, 1)
    targets["w2_scale_inv"] = torch.zeros(2, 1, 1)
    with pytest.raises(ValueError, match="matching contiguous"):
        library.gather_experts(store, ids, targets)
    assert all(not value.view(torch.uint8).count_nonzero() for value in targets.values())
    targets = targets_for(store, 1)
    targets["w13_weight"] = store.w13_weight[:1]
    with pytest.raises(ValueError, match="overlap"):
        library.gather_experts(store, ids, targets)


@pytest.mark.parametrize("ids", [
    torch.tensor([0., 1.]), torch.tensor([[0, 1]]),
    torch.tensor([0, 1, 2, 3])[::2], torch.empty(0, dtype=torch.int64),
])
def test_invalid_id_layout_is_rejected_before_any_write(library, store, ids):
    targets = targets_for(store, 2)
    with pytest.raises(ValueError, match="contiguous CPU int64"):
        library.gather_experts(store, ids, targets)
    assert all(not value.view(torch.uint8).count_nonzero() for value in targets.values())


@pytest.mark.parametrize("packed", [False, True])
def test_auto_gather_keeps_small_and_packed_torch_paths_and_survives_resize(
    library, store, packed,
):
    with closing(ExpertOffloadManager(
        backend="torch", max_staged_experts=4, packed_staging=packed,
    )) as manager:
        status = manager.configure_host_gather("auto")
        assert status["native_available"]
        ids = torch.tensor([3, 0, 2, 1])
        with manager.pool.session(store, torch.device("cpu")):
            staged = manager.pool.stage(store, ids)
            for name in EXPERT_BUFFER_NAMES:
                assert torch.equal(getattr(staged, name).view(torch.uint8),
                                   getattr(store, name).view(torch.uint8).index_select(0, ids))
        assert manager.stats.native_gather_calls == (0 if packed else 1)
        assert manager.stats.torch_gather_calls == (1 if packed else 0)
        callback = manager.pool.native_gather
        manager.configure_staging(2, 64)
        assert manager.pool.native_gather == callback
        assert manager.pool.native_gather_min_experts == 4
        with manager.pool.session(store, torch.device("cpu")):
            manager.pool.stage(store, ids[:2])
        assert manager.stats.torch_gather_calls == (2 if packed else 1)


def test_missing_optional_library_falls_back_only_in_auto_mode():
    with closing(ExpertOffloadManager(backend="torch")) as manager:
        with patch("drama_low_vram.native_backend.NativeLibrary",
                   side_effect=FileNotFoundError("optional DLL missing")):
            status = manager.configure_host_gather("auto")
            assert not status["native_available"] and "optional DLL missing" in status["fallback"]
            with pytest.raises(FileNotFoundError):
                manager.configure_host_gather("native")
        with pytest.raises(ValueError, match="host_gather"):
            manager.configure_host_gather("unknown")


def test_older_abi_v1_library_without_gather_extension_keeps_torch_path():
    with closing(ExpertOffloadManager(backend="torch")) as manager:
        with patch("drama_low_vram.native_backend.NativeLibrary",
                   return_value=SimpleNamespace(gather_rows=None)):
            status = manager.configure_host_gather("auto")
            assert not status["native_available"]
            assert "extension unavailable" in status["fallback"]
            assert manager.pool.native_gather is None
            with pytest.raises(RuntimeError, match="extension unavailable"):
                manager.configure_host_gather("native")
