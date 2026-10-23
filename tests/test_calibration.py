"""Selection accepts repeatable small gains and rejects noise or wrong output."""

import pytest
from types import SimpleNamespace
from unittest.mock import Mock
import json
from threading import RLock

from drama_low_vram.calibration import (
    select_candidate, save_compute_profile, load_compute_profile, ensure_compute_policy,
    calibrate_compute_windows,
)


def row(jobs, seconds, *, equal=True, gather="auto"):
    return {
        "candidate": {"cpu_experts": jobs, "native_threads": 4, "host_gather": gather},
        "samples_seconds": seconds, "bytes_equal": equal,
    }


def windowed(jobs, samples, *, gather="auto"):
    measured = row(jobs, [value for window in samples for value in window], gather=gather)
    measured["windows"] = [
        {"offset": offset, "bytes_equal": True, "samples_seconds": values}
        for offset, values in zip((0, 16, 48), samples, strict=True)
    ]
    return measured


def test_gpu_only_remains_a_real_selection_when_candidates_do_not_pay_for_overhead():
    baseline = row(0, [1.00, 1.01, 0.99], gather="torch")
    result = select_candidate([
        baseline, row(2, [0.98, 0.99, 0.99]), row(4, [1.1, 1.2, 1.3])])
    assert result["candidate"] == baseline["candidate"]
    assert result["measured_time_reduction_fraction"] == 0


def test_numerically_wrong_fast_candidate_is_not_eligible():
    result = select_candidate([
        row(0, [1, 1, 1], gather="torch"),
        row(4, [0.4, 0.5, 0.4], equal=False),
        row(2, [0.9, 0.91, 0.92]),
    ])
    assert result["candidate"]["cpu_experts"] == 2
    assert result["median_seconds"] == 0.91


def test_host_gather_can_improve_without_enabling_cpu_expert_computation():
    result = select_candidate([
        row(0, [1, 1, 1], gather="torch"), row(0, [0.9, 0.89, 0.9])])
    assert result["candidate"]["cpu_experts"] == 0
    assert result["candidate"]["host_gather"] == "auto"


def test_repeatable_two_percent_gather_gain_is_accepted():
    result = select_candidate([
        row(0, [0.999, 1.001], gather="torch"),
        row(0, [0.979, 0.981]),
    ])
    assert result["candidate"]["host_gather"] == "auto"
    assert result["measured_time_reduction_fraction"] == pytest.approx(0.02)
    assert result["comparisons"][0]["gain_lower_bound_seconds"] > 0


def test_repeatable_cpu_gain_below_five_percent_is_accepted():
    result = select_candidate([
        row(0, [0.999, 1.001], gather="torch"),
        row(2, [0.951, 0.953]),
    ])
    assert result["candidate"]["cpu_experts"] == 2
    assert result["measured_time_reduction_fraction"] == pytest.approx(0.048)


def test_early_window_tie_does_not_veto_repeatable_total_saving():
    result = select_candidate([
        windowed(0, [[0.999, 1.001]] * 3, gather="torch"),
        windowed(2, [[0.998, 1.002], [0.959, 0.961], [0.919, 0.921]]),
    ])
    assert result["candidate"]["cpu_experts"] == 2
    assert result["comparisons"][0]["accepted"]


def test_noisy_apparent_gain_does_not_replace_the_gpu_baseline():
    result = select_candidate([
        row(0, [0.99, 1.01], gather="torch"), row(2, [0.90, 1.06]),
    ])
    assert result["candidate"]["cpu_experts"] == 0
    assert result["comparisons"][0]["reason"] == "repeat_ranges_overlap"


def test_only_repeated_equivalent_windows_can_replace_the_baseline():
    baseline = windowed(0, [[1, 1]] * 3, gather="torch")
    incomplete = windowed(2, [[0.8, 0.8]] * 3)
    incomplete["windows"].pop()
    incomplete["samples_seconds"] = [0.8] * 4
    result = select_candidate([baseline, incomplete])
    assert result["candidate"] == baseline["candidate"]
    assert result["comparisons"][0]["reason"] == "incomparable_windows"
    result = select_candidate([row(0, [1, 1], gather="torch"), row(2, [0.8])])
    assert result["candidate"]["cpu_experts"] == 0
    assert result["comparisons"][0]["reason"] == "insufficient_repeats"


def test_missing_baseline_cannot_produce_a_profile():
    with pytest.raises(ValueError, match="GPU-only"):
        select_candidate([row(2, [1, 1, 1])])


def test_cpu_work_must_improve_over_the_fastest_gpu_gather_option():
    result = select_candidate([
        row(0, [1, 1, 1], gather="torch"),
        row(0, [0.9, 0.9, 0.9]), row(2, [0.92, 0.92, 0.92]),
    ])
    assert result["candidate"]["cpu_experts"] == 0
    assert result["candidate"]["host_gather"] == "auto"


def test_cpu_can_win_with_small_repeatable_gain_over_best_gpu_gather():
    result = select_candidate([
        row(0, [1, 1], gather="torch"),
        row(0, [0.899, 0.901]), row(2, [0.879, 0.881]),
    ])
    assert result["candidate"]["cpu_experts"] == 2
    assert result["best_gpu_median_seconds"] == pytest.approx(0.9)
    assert result["comparisons"][-1]["reference"]["host_gather"] == "auto"


def test_a_fast_first_window_cannot_hide_cpu_slowdown_in_a_later_window():
    rows = [
        row(0, [1] * 6, gather="torch"),
        row(0, [0.9] * 6), row(2, [0.7, 0.7, 0.7, 0.7, 0.95, 0.95]),
    ]
    for measured in rows:
        measured["windows"] = [
            {"offset": offset, "bytes_equal": True,
             "samples_seconds": measured["samples_seconds"][index * 2:index * 2 + 2]}
            for index, offset in enumerate((0, 16, 48))
        ]
    result = select_candidate(rows)
    assert result["candidate"]["cpu_experts"] == 0
    rows[2]["windows"][-1]["samples_seconds"] = [0.6, 0.6]
    with pytest.raises(ValueError, match="Inconsistent"):
        select_candidate(rows)


def test_multiwindow_calibration_serializes_and_restores_the_recursion_guard(monkeypatch):
    engine = SimpleNamespace(_lock=RLock(), _computing_policy=False)
    offsets = []

    def measure(prepared, _prompt, **kwargs):
        assert prepared._computing_policy
        offsets.append(kwargs["offset"])
        rows = [row(0, [1, 1], gather="torch"), row(0, [0.9, 0.9])]
        for measured in rows:
            measured["fallback"] = None
        return {"rows": rows, "key": "abc", "identity": {"schema": 3}}

    monkeypatch.setattr("drama_low_vram.calibration.calibrate_compute", measure)
    result = calibrate_compute_windows(engine, object())
    assert offsets == [0, 16, 48]
    assert not engine._computing_policy
    assert len(result["rows"][0]["samples_seconds"]) == 6
    assert result["selection"]["candidate"]["cpu_experts"] == 0


def test_profile_identity_and_selection_are_revalidated(tmp_path, monkeypatch):
    identity = {"capability": [8, 9], "schema": 1}
    rows = [row(0, [1, 1, 1], gather="torch"), row(2, [0.8, 0.8, 0.8])]
    report = {"key": "abc", "identity": identity, "rows": rows,
              "selection": select_candidate(rows)}
    monkeypatch.setattr("drama_low_vram.calibration.calibration_key",
                        lambda _engine: ("abc", identity))
    target = save_compute_profile(report, tmp_path)
    assert load_compute_profile(object(), tmp_path)["selection"]["candidate"]["cpu_experts"] == 2
    changed = dict(report)
    changed["selection"] = {"candidate": {"cpu_experts": 8, "native_threads": 4, "host_gather": "auto"}}
    target.write_text(json.dumps(changed))
    assert load_compute_profile(object(), tmp_path) is None
    report["identity"] = {"capability": [8, 0]}
    save_compute_profile(report, tmp_path)
    assert load_compute_profile(object(), tmp_path) is None
    target.write_text("broken JSON")
    assert load_compute_profile(object(), tmp_path) is None


def test_switching_models_and_recreating_engines_reuses_each_saved_profile(tmp_path, monkeypatch):
    import drama_low_vram.calibration as module

    profiles = tmp_path / "profiles"
    monkeypatch.setattr(module.torch.cuda, "get_device_name", lambda _device: "test GPU")
    monkeypatch.setattr(module.torch.cuda, "get_device_capability", lambda _device: (8, 9))
    monkeypatch.setattr(module, "default_library_path", lambda: tmp_path / "native.dll")
    monkeypatch.setattr(
        "fish_speech.models.text2semantic.fish_omni.generate.build_drama3_prompt",
        Mock(return_value=object()))

    def make_engine(name):
        checkpoint = tmp_path / "checkpoints" / name
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / "config.json").write_text('{"same_architecture": true}')
        return SimpleNamespace(
            device="cuda:0", checkpoint=checkpoint,
            options=SimpleNamespace(compute_profile_dir=profiles, expert_decode="triton"),
            manager=SimpleNamespace(
                max_staged_experts=8, max_pair_rows=64, cache_experts=2,
                configure_hybrid_decode=Mock(), configure_host_gather=Mock()),
            model=SimpleNamespace(
                cache_capacity=512, tokenizer=object(), config=object(),
                _drama_low_vram_expert_cache=SimpleNamespace(model_fingerprint=f"weights-{name}")),
            graph=object(), backbone_graph=SimpleNamespace(enabled=True),
            _active_output_controls=(), _lock=RLock(),
        )

    measured_models = []

    def measure(engine, _prompt, **_kwargs):
        measured_models.append(engine.checkpoint.name)
        jobs = 2 if engine.checkpoint.name == "model-a" else 4
        rows = [row(0, [1, 1], gather="torch"), row(jobs, [0.8, 0.8])]
        key, identity = module.calibration_key(engine)
        return {"key": key, "identity": identity, "rows": rows,
                "selection": select_candidate(rows)}

    diagnostics = Mock(side_effect=measure)
    monkeypatch.setattr(module, "calibrate_compute_windows", diagnostics)
    first_a = ensure_compute_policy(make_engine("model-a"))
    saved_a = profiles / f"{first_a['key']}.json"
    original_a = saved_a.read_bytes()
    first_b = ensure_compute_policy(make_engine("model-b"))
    assert first_a["status"] == first_b["status"] == "calibrated"
    assert first_a["key"] != first_b["key"]
    assert saved_a.read_bytes() == original_a
    assert len(list(profiles.glob("*.json"))) == 2

    # Fresh engine objects exercise disk reuse after a model switch or restart.
    for name, original, jobs in (("model-a", first_a, 2), ("model-b", first_b, 4)):
        engine = make_engine(name)
        restored = ensure_compute_policy(engine)
        assert restored["status"] == "cached"
        assert restored["key"] == original["key"]
        assert restored["selection"] == original["selection"]
        engine.manager.configure_hybrid_decode.assert_called_with(
            jobs, threads=4, device="cuda:0")
    assert measured_models == ["model-a", "model-b"]
    assert diagnostics.call_count == 2


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, 0, True, "1"])
def test_nonfinite_or_invalid_calibration_times_cannot_select_a_policy(bad):
    with pytest.raises(ValueError, match="timing"):
        select_candidate([row(0, [1, bad, 1], gather="torch")])


@pytest.fixture
def auto_policy(tmp_path, monkeypatch):
    import drama_low_vram.calibration as module

    engine = SimpleNamespace(
        options=SimpleNamespace(compute_profile_dir=tmp_path / "profiles"),
        checkpoint=tmp_path / "checkpoints/model",
        graph=object(), backbone_graph=SimpleNamespace(enabled=True),
        _active_output_controls=(),
        model=SimpleNamespace(cache_capacity=512, tokenizer=object(), config=object()),
        _prepare=Mock(),
    )
    selection = select_candidate([
        row(0, [1, 1, 1], gather="torch"), row(2, [0.8, 0.8, 0.8])])
    measured = {"key": "abc", "selection": selection}
    apply = Mock()
    load = Mock(return_value=None)
    calibrate = Mock(return_value=measured)
    save = Mock(return_value=tmp_path / "profiles/abc.json")
    monkeypatch.setattr(module, "apply_compute_selection", apply)
    monkeypatch.setattr(module, "load_compute_profile", load)
    monkeypatch.setattr(module, "calibrate_compute_windows", calibrate)
    monkeypatch.setattr(module, "save_compute_profile", save)
    monkeypatch.setattr(
        "fish_speech.models.text2semantic.fish_omni.generate.build_drama3_prompt",
        Mock(return_value=object()))
    return SimpleNamespace(
        engine=engine, measured=measured, apply=apply, load=load,
        calibrate=calibrate, save=save)


def test_automatic_policy_uses_matching_profile_without_running_diagnostics(auto_policy):
    h = auto_policy
    h.load.return_value = h.measured
    result = ensure_compute_policy(h.engine)
    assert result["status"] == "cached"
    assert result["selection"] == h.measured["selection"]
    h.calibrate.assert_not_called()
    h.save.assert_not_called()


def test_automatic_policy_measures_cache_miss_and_applies_selection(auto_policy):
    h = auto_policy
    result = ensure_compute_policy(h.engine)
    assert result["status"] == "calibrated"
    h.calibrate.assert_called_once()
    h.apply.assert_called_with(h.engine, h.measured["selection"])
    h.save.assert_called_once()


def test_unavailable_graph_path_selects_gpu_without_attempting_calibration(auto_policy):
    h = auto_policy
    h.engine.backbone_graph.enabled = False
    result = ensure_compute_policy(h.engine)
    assert result["status"] == "gpu_fallback"
    assert result["selection"]["candidate"]["cpu_experts"] == 0
    h.calibrate.assert_not_called()


def test_profile_save_failure_keeps_the_measured_in_memory_selection(auto_policy):
    h = auto_policy
    h.save.side_effect = PermissionError("Read-only profile directory")
    result = ensure_compute_policy(h.engine)
    assert result["status"] == "calibrated"
    assert "profile_save_warning" in result
    h.apply.assert_called_with(h.engine, h.measured["selection"])


def test_unsupported_calibration_uses_gpu_and_restores_scene_sampler(auto_policy):
    h = auto_policy
    h.engine._active_output_controls = ("<|audio_separator|>",)

    def unsupported(*_args, **_kwargs):
        h.engine._active_output_controls = ()
        raise ValueError("Optional backend unavailable")

    h.calibrate.side_effect = unsupported
    result = ensure_compute_policy(h.engine)
    assert result["status"] == "gpu_fallback"
    assert result["selection"]["candidate"]["cpu_experts"] == 0
    h.engine._prepare.assert_called_once_with(512, output_controls=("<|audio_separator|>",))


def test_cuda_runtime_failure_propagates_instead_of_claiming_a_safe_fallback(auto_policy):
    h = auto_policy
    h.calibrate.side_effect = RuntimeError("CUDA device-side assert")
    with pytest.raises(RuntimeError, match="device-side"):
        ensure_compute_policy(h.engine)
