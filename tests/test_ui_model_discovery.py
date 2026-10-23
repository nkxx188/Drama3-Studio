"""Portable/shared model discovery is bounded, metadata-only and format-neutral."""

import json
from pathlib import Path

import pytest

import ui.server as server


@pytest.fixture
def make_state(tmp_path, monkeypatch):
    states = []

    def create(root=None, *, settings=None):
        root = root or tmp_path / "standalone package"
        root.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(server, "ROOT", root)
        if settings is not None:
            data = root / "ui_data"
            data.mkdir(exist_ok=True)
            (data / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
        state = server.AppState()
        states.append(state)
        return state

    yield create
    for state in states:
        state.shutdown()


def python_model(pool, name, *, model_type="fish_omni"):
    directory = pool / name
    directory.mkdir(parents=True, exist_ok=True)
    config = {} if model_type is None else {"model_type": model_type}
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return directory


def native_model(pool, name="drama3-q8.gguf"):
    path = pool / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"catalog fixture: never load these weights")
    return path


def listed(state):
    return {Path(row["directory"]) for row in state.available_models()}


def mark_application(root):
    (root / "ui").mkdir(parents=True, exist_ok=True)
    (root / "ui/server.py").write_text("# application marker\n", encoding="utf-8")
    (root / "src/drama_low_vram").mkdir(parents=True, exist_ok=True)


def test_nested_portable_defaults_to_its_own_checkpoints_only(tmp_path, make_state):
    source = tmp_path / "source checkout"
    mark_application(source)
    pool = source / "checkpoints"
    expected = {python_model(pool, name) for name in (
        "drama-3", "drama-3-int4", "drama-3-int8", "drama-3-shared",
        "drama-3-w4a8", "drama-3-w6a8",
    )}
    q4 = native_model(pool / "drama-3-gguf-q4", "drama3-q4.gguf")
    expected |= {q4, native_model(pool / "drama-3-gguf-q8")}
    python_model(pool, "Qwen3-ASR-0.6B-hf", model_type="qwen3_asr")
    portable = source / "dist/Drama3 Windows NVIDIA"
    (portable / "checkpoints").mkdir(parents=True)
    state = make_state(portable, settings={"checkpoint": str(q4)})

    assert listed(state) == {q4}
    assert [Path(row["path"]) for row in state.model_locations()] == [
        (portable / "checkpoints").resolve(),
    ]
    assert list((portable / "checkpoints").iterdir()) == []
    assert state._engine is None


def test_moved_portable_rebinds_stale_absolute_checkpoint_paths_without_rewriting_settings(
        tmp_path, make_state, monkeypatch):
    source = tmp_path / "source checkout"
    mark_application(source)
    portable = source / "dist/Drama3 Windows NVIDIA"
    local_model = native_model(
        portable / "checkpoints/drama-3-gguf-q4", "renamed-q4.gguf")
    old_model = source / "checkpoints/drama-3-gguf-q4/renamed-q4.gguf"
    old_pool = source / "checkpoints"
    saved_settings = {
        "checkpoint": str(old_model),
        "model_paths": [str(old_model)],
        "model_roots": [str(old_pool)],
    }
    monkeypatch.setattr(
        server, "inspect_model",
        lambda path: {"directory": str(path), "ready": Path(path) == local_model,
                      "runtime": "gguf"})
    state = make_state(portable, settings=saved_settings)

    model = state.model_status()

    assert state.checkpoint == local_model
    assert {Path(row["directory"]) for row in model["available_models"]} == {local_model}
    assert model["selected_ready"] is True
    assert model["model_locations"] == [
        {"path": str((portable / "checkpoints").resolve()), "removable": False},
    ]
    assert json.loads(state.settings_file.read_text(encoding="utf-8")) == saved_settings


def test_missing_external_checkpoint_is_not_rebound_to_a_same_named_portable_model(
        tmp_path, make_state):
    source = tmp_path / "source checkout"
    mark_application(source)
    portable = source / "dist/Drama3 Windows NVIDIA"
    local_model = python_model(portable / "checkpoints", "drama-3")
    external_model = tmp_path / "ComfyUI/checkpoints/drama-3"
    state = make_state(portable, settings={"checkpoint": str(external_model)})

    assert state.checkpoint == external_model
    assert state.checkpoint != local_model


@pytest.mark.parametrize("selected_format", ["python", "gguf"])
def test_standalone_package_does_not_infer_external_checkpoint_pool(
        tmp_path, make_state, selected_format):
    pool = tmp_path / "external drive/checkpoints"
    original = python_model(pool, "original-fp8")
    quantized = python_model(pool, "quantized-int4")
    q4 = native_model(pool / "q4", "renamed-q4.gguf")
    q8 = native_model(pool / "q8")
    selected = original if selected_format == "python" else q4
    state = make_state(settings={"checkpoint": str(selected)})

    assert listed(state) == {selected}
    assert [Path(row["path"]) for row in state.model_locations()] == [
        (server.ROOT / "checkpoints").resolve(),
    ]


def test_history_restores_other_model_pools_after_restart(tmp_path, make_state):
    root = tmp_path / "portable"
    local = python_model(root / "checkpoints", "local-fp8")
    external = tmp_path / "external/models"
    q4 = native_model(external / "q4", "old-selection.gguf")
    q8 = native_model(external / "q8")
    state = make_state(root, settings={
        "checkpoint": "checkpoints/local-fp8", "model_paths": [str(q4)],
    })
    expected = {local, q4}

    assert listed(state) == expected
    assert [Path(row["path"]) for row in state.model_locations()] == [
        (root / "checkpoints").resolve(),
    ]
    restored = make_state(root)
    assert listed(restored) == expected
    assert [Path(row["path"]) for row in restored.model_locations()] == [
        (root / "checkpoints").resolve(),
    ]
    assert restored.checkpoint == local


def test_selected_gguf_does_not_infer_a_model_location_or_sibling_models(tmp_path, make_state):
    folder = tmp_path / "arbitrary asset folder"
    q4 = native_model(folder, "quality-four.GGUF")
    q8 = native_model(folder, "quality-eight.gguf")
    state = make_state(settings={"checkpoint": str(q4)})

    assert listed(state) == {q4}
    assert {row["runtime"] for row in state.available_models()} == {"gguf"}
    assert [Path(row["path"]) for row in state.model_locations()] == [
        (server.ROOT / "checkpoints").resolve(),
    ]


def test_remembered_gguf_choices_stay_in_model_picker_not_location_list(tmp_path, make_state):
    pool = tmp_path / "checkpoints"
    q4 = native_model(pool / "drama-3-gguf-q4", "drama-q4.gguf")
    q8 = native_model(pool / "drama-3-gguf-q8", "drama-q8.gguf")
    state = make_state(settings={
        "checkpoint": str(q4), "model_paths": [str(q4), str(q8)],
    })

    assert listed(state) == {q4, q8}
    assert [Path(row["path"]) for row in state.model_locations()] == [
        (server.ROOT / "checkpoints").resolve(),
    ]


@pytest.mark.parametrize("config,include_directory", [
    ('{"model_type": "fish_omni"}', True),
    ('{"model_type": "qwen3_asr"}', False),
    ('{"model_type": "llama"}', False),
    ('{"unfinished":', False),
])
def test_gguf_files_are_found_independently_of_neighbor_config(
        make_state, config, include_directory):
    state = make_state()
    folder = server.ROOT / "checkpoints/mixed-distribution"
    first = native_model(folder, "first.gguf")
    second = native_model(folder, "second.GGUF")
    (folder / "config.json").write_text(config, encoding="utf-8")

    paths = listed(state)
    assert {first, second} <= paths
    assert (folder in paths) is include_directory


@pytest.mark.parametrize("model_type", ["qwen3_asr", "whisper", "llama"])
def test_automatic_catalog_excludes_unrelated_python_models(make_state, model_type):
    state = make_state()
    unrelated = python_model(server.ROOT / "checkpoints", "auxiliary", model_type=model_type)

    assert unrelated not in listed(state)


def test_legacy_config_without_model_type_remains_discoverable(make_state):
    state = make_state()
    legacy = python_model(server.ROOT / "checkpoints", "legacy-model", model_type=None)

    assert legacy in listed(state)


@pytest.mark.parametrize("config", ['{"invalid":', "[]", "42", "null"])
def test_malformed_metadata_does_not_hide_other_models(make_state, config):
    state = make_state()
    pool = server.ROOT / "checkpoints"
    valid = python_model(pool, "valid-model")
    broken = python_model(pool, "broken-model")
    (broken / "config.json").write_text(config, encoding="utf-8")

    paths = listed(state)
    assert valid in paths
    assert broken not in paths


def test_unavailable_pool_does_not_hide_other_models(tmp_path, monkeypatch, make_state):
    state = make_state()
    local = python_model(server.ROOT / "checkpoints", "local-model")
    blocked = tmp_path / "unavailable/checkpoints"
    selected = native_model(blocked / "selected", "chosen.gguf")
    native_model(blocked / "unreachable", "other.gguf")
    state.checkpoint = selected
    original = Path.iterdir

    def iterdir(path):
        if path == blocked:
            raise PermissionError("fixture pool is unavailable")
        return original(path)

    monkeypatch.setattr(Path, "iterdir", iterdir)
    assert listed(state) == {local, selected}


def test_denied_config_read_keeps_gguf_siblings_visible(monkeypatch, make_state):
    state = make_state()
    folder = python_model(server.ROOT / "checkpoints", "locked-config")
    native = native_model(folder)
    original = Path.read_text

    def read_text(path, *args, **kwargs):
        if path == folder / "config.json":
            raise PermissionError("fixture metadata is unavailable")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    paths = listed(state)
    assert native in paths
    assert folder not in paths


def test_missing_selected_model_stays_visible_without_resurrecting_missing_history(
        tmp_path, make_state):
    state = make_state()
    selected = tmp_path / "offline/checkpoints/not-downloaded.gguf"
    state.checkpoint = selected
    state.model_paths = [tmp_path / "removed/old-model"]

    assert listed(state) == {selected}


def test_duplicate_search_roots_and_history_paths_yield_one_row(make_state):
    state = make_state()
    selected = native_model(server.ROOT / "checkpoints/gguf", "chosen.gguf")
    state.checkpoint = selected
    state.model_paths = [
        selected,
        selected.parent / ".." / selected.parent.name / selected.name,
        selected,
    ]

    rows = state.available_models()
    assert len(rows) == 1
    assert rows[0] == {
        "label": selected.parent.name, "name": selected.parent.name,
        "directory": str(selected), "location": str(selected.parent), "runtime": "gguf",
    }
    roots = state.model_search_roots()
    assert len(roots) == len(set(roots))


def test_model_pool_scan_does_not_recurse_beyond_one_child_directory(
        monkeypatch, make_state):
    state = make_state()
    pool = server.ROOT / "checkpoints"
    visible = native_model(pool / "visible")
    hidden = native_model(pool / "nested/deeper", "too-deep.gguf")
    original = Path.iterdir

    def iterdir(path):
        assert path != hidden.parent, "Discovery must not recursively enumerate model trees"
        return original(path)

    monkeypatch.setattr(Path, "iterdir", iterdir)
    paths = listed(state)
    assert visible in paths
    assert hidden not in paths


def test_unmarked_application_ancestors_are_not_scanned(tmp_path, monkeypatch, make_state):
    unrelated = tmp_path / "unrelated parent"
    hidden = python_model(unrelated / "checkpoints", "not-a-shared-model")
    state = make_state(unrelated / "dist/standalone")
    original = Path.iterdir

    def iterdir(path):
        assert path != hidden.parent, "Unmarked ancestor pools must not be enumerated"
        return original(path)

    monkeypatch.setattr(Path, "iterdir", iterdir)
    assert listed(state) == {state.checkpoint}
    assert all(Path(row["path"]) != hidden.parent for row in state.model_locations())


def test_application_ancestor_search_is_limited_to_two_levels(tmp_path, make_state):
    source = tmp_path / "source"
    mark_application(source)
    hidden = python_model(source / "checkpoints", "too-distant")
    state = make_state(source / "one/two/three/portable")

    assert source / "checkpoints" not in state.model_search_roots()
    assert hidden not in listed(state)


def test_custom_model_location_scans_nested_drama_models_and_persists(tmp_path, make_state):
    root = tmp_path / "ComfyUI" / "models"
    python = python_model(root / "checkpoints", "Drama 3 FP8")
    gguf = native_model(root / "diffusion_models", "drama3-q4.gguf")
    safetensors = root / "text_encoders" / "drama3-custom.safetensors"
    safetensors.parent.mkdir(parents=True)
    safetensors.write_bytes(b"weight metadata fixture")
    unrelated_gguf = native_model(root / "diffusion_models", "unrelated.gguf")
    unrelated_python = python_model(root / "checkpoints", "other-model", model_type="llama")
    state = make_state()

    try:
        result = state.add_model_root(root)
        rows = {Path(row["directory"]): row for row in result["available_models"]}
        assert python in rows
        assert gguf in rows
        assert safetensors.parent in rows
        assert unrelated_gguf not in rows
        assert unrelated_python not in rows
        assert rows[gguf]["name"] == "drama3-q4"
        assert rows[gguf]["location"] == str(gguf.parent)
        assert rows[safetensors.parent]["name"] == "drama3-custom"
        assert rows[safetensors.parent]["location"] == str(safetensors.parent)
        assert not server.inspect_model(safetensors.parent)["ready"]
        assert any(row == {"path": str(root), "removable": True}
                   for row in result["model_locations"])

        restored = make_state()
        try:
            assert restored.model_roots == [root.resolve()]
            assert gguf in listed(restored)
            assert [Path(row["path"]) for row in restored.model_locations()] == [
                (server.ROOT / "checkpoints").resolve(), root.resolve(),
            ]
            restored.remove_model_root(root)
            assert restored.model_roots == []
            assert root.resolve() not in restored.model_search_roots()
        finally:
            restored.shutdown()
    finally:
        state.shutdown()


def test_custom_root_can_be_a_model_directory_itself(make_state, tmp_path):
    model = python_model(tmp_path, "Drama 3 release")
    state = make_state()
    try:
        state.add_model_root(model)
        assert model in listed(state)
    finally:
        state.shutdown()


def test_drive_root_is_never_enumerated_for_a_loose_gguf(
        tmp_path, monkeypatch, make_state):
    state = make_state()
    drive = Path(tmp_path.anchor)
    state.checkpoint = drive / "unavailable-drama3-fixture.gguf"
    original = Path.iterdir

    def iterdir(path):
        assert path != drive, "Discovery must not scan a drive root"
        return original(path)

    monkeypatch.setattr(Path, "iterdir", iterdir)
    assert drive not in state.model_search_roots()
    assert listed(state) == {state.checkpoint}


def test_arbitrary_selected_python_directory_does_not_expand_to_siblings(
        tmp_path, monkeypatch, make_state):
    folder = tmp_path / "arbitrary folder"
    selected = python_model(folder, "selected")
    hidden = python_model(folder, "unrelated-sibling")
    state = make_state(settings={"checkpoint": str(selected)})
    original = Path.iterdir

    def iterdir(path):
        assert path != folder, "Only explicit shared-pool names may expose sibling directories"
        return original(path)

    monkeypatch.setattr(Path, "iterdir", iterdir)
    assert listed(state) == {selected}
    assert hidden not in listed(state)


def test_shared_descriptor_does_not_expand_into_comfyui_weight_directory(
        tmp_path, monkeypatch, make_state):
    state = make_state()
    selected = python_model(server.ROOT / "checkpoints", "shared-fp8")
    external = tmp_path / "ComfyUI/models/diffusion_models"
    weights = external / "drama3.safetensors"
    hidden = native_model(external, "unrelated-model.gguf")
    weights.write_bytes(b"shared model weights")
    (selected / "shared-model.json").write_text(json.dumps({
        "format": "drama3-packed-shared-v1", "model": str(weights),
    }), encoding="utf-8")
    state.checkpoint = selected
    original = Path.iterdir

    def iterdir(path):
        assert path != external, "Shared descriptors must not infer arbitrary weight pools"
        return original(path)

    monkeypatch.setattr(Path, "iterdir", iterdir)
    assert listed(state) == {selected}
    assert hidden not in listed(state)


def test_discovery_never_loads_or_reads_model_payloads(monkeypatch, make_state):
    state = make_state()
    pool = server.ROOT / "checkpoints"
    original = python_model(pool, "original")
    (original / "model.safetensors").write_bytes(b"do not read")
    (original / "codec.pth").write_bytes(b"do not read")
    native = native_model(pool / "native")
    open_file = Path.open

    def metadata_only(path, *args, **kwargs):
        assert path.suffix.lower() not in (".gguf", ".safetensors", ".pth"), path
        return open_file(path, *args, **kwargs)

    def no_inference(*_args, **_kwargs):
        raise AssertionError("Model discovery must not inspect payloads or construct an engine")

    monkeypatch.setattr(Path, "open", metadata_only)
    for name in ("inspect_model", "inspect_checkpoint", "inspect_gguf", "DramaEngine", "GGUFEngine"):
        monkeypatch.setattr(server, name, no_inference)

    assert {original, native} <= listed(state)
    assert state._engine is None
