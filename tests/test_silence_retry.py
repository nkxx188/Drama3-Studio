from threading import Event

import numpy as np
import pytest

from drama_low_vram.gguf_runtime import GenerationCancelled
from drama_low_vram.silence_retry import (
    MAX_RETRIES, run_with_silence_retry, silent_span, silent_wav_span,
)


def tone(seconds, amplitude=.1, rate=8000):
    return amplitude * np.sin(np.arange(round(seconds * rate)) * 2 * np.pi * 220 / rate)


def test_only_continuous_near_silence_over_five_seconds_is_abnormal():
    assert silent_span(np.zeros(5 * 8000), 8000) is None
    assert silent_span(np.zeros(round(5.02 * 8000)), 8000)["seconds"] > 5
    assert silent_span(tone(6, amplitude=.001), 8000) is None
    # Pauses separated by speech cannot be added together.
    assert silent_span(np.concatenate((np.zeros(4 * 8000), tone(.5), np.zeros(4 * 8000))), 8000) is None
    assert silent_span(np.stack((np.zeros(6 * 8000), tone(6)), axis=1), 8000) is None


def test_quiet_noise_floor_and_brief_clicks_are_handled_conservatively():
    quiet = np.random.default_rng(7).uniform(-.0001, .0001, 6 * 8000)
    assert silent_span(quiet, 8000)["seconds"] == 6
    quiet[3 * 8000] = .01
    assert silent_span(quiet, 8000) is None
    assert silent_span(np.array([], dtype=np.float32), 8000) is None


def test_wav_inspection_limits_the_check_to_the_new_infill_span(tmp_path):
    import soundfile as sf
    path = tmp_path / "infill.wav"
    sf.write(path, np.concatenate((np.zeros(6 * 8000), tone(1), np.zeros(6 * 8000))), 8000)
    assert silent_wav_span(path)
    assert silent_wav_span(path, start_sample=6 * 8000, end_sample=7 * 8000) is None
    assert silent_wav_span(path, start_sample=6 * 8000, end_sample=6 * 8000) is None


def test_disabled_check_does_not_decode_or_change_the_seed():
    calls = []
    result, report = run_with_silence_retry(
        lambda seed: calls.append(seed) or "original",
        lambda _: pytest.fail("Disabled detection must not run"), seed=123)
    assert calls == [123] and result == "original" and report is None


def test_retry_changes_seed_only_and_stops_after_a_good_chunk():
    calls, events = [], []
    span = {"seconds": 6}
    result, report = run_with_silence_retry(
        lambda seed: calls.append(seed) or len(calls),
        lambda count: span if count == 1 else None,
        seed=2**32 - 1, enabled=True, on_retry=events.append)
    assert result == 2 and len(calls) == 2 and calls[0] != calls[1]
    assert all(0 <= seed < 2**32 for seed in calls)
    assert report["retries"] == 1 and not report["exhausted"]
    assert events == [{"stage": "silence_retry", "retry": 1,
                       "max_retries": MAX_RETRIES, "silence_seconds": 6}]


def test_retry_is_bounded_and_can_be_cancelled_before_another_generation():
    calls, cancel = [], Event()
    _, report = run_with_silence_retry(
        lambda seed: calls.append(seed), lambda _: {"seconds": 6}, seed=123, enabled=True)
    assert len(calls) == 3 and report["retries"] == 2 and report["exhausted"]
    calls.clear()
    with pytest.raises(GenerationCancelled):
        run_with_silence_retry(
            lambda seed: calls.append(seed), lambda _: {"seconds": 6}, seed=123,
            enabled=True, cancelled=cancel.is_set, on_retry=lambda _: cancel.set())
    assert len(calls) == 1
