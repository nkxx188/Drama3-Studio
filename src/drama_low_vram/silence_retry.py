"""Conservative, CPU-only silence checks on newly generated audio chunks."""

from pathlib import Path

import numpy as np


SILENCE_SECONDS = 5.0
MAX_RETRIES = 2
_WINDOW_SECONDS = .02
_RMS_LIMIT = 10 ** (-70 / 20)
_PEAK_LIMIT = 10 ** (-55 / 20)


def silent_span(audio, sample_rate=44100):
    """Find the longest near-zero run, preserving low-volume speech and breaths."""
    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu().numpy()
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.ndim != 2 or sample_rate <= 0:
        raise ValueError("Silence detection requires samples by channels and a positive sample rate")
    window = max(1, round(sample_rate * _WINDOW_SECONDS))
    longest_start = longest_end = run_start = 0
    for start in range(0, len(samples), window):
        block = samples[start:start + window]
        # Both limits must hold in every channel. A brief non-silent sound
        # resets the run; quiet speech is never accumulated across pauses.
        quiet = (np.isfinite(block).all()
                 and np.max(np.abs(block)) < _PEAK_LIMIT
                 and np.max(np.mean(block * block, axis=0)) < _RMS_LIMIT ** 2)
        end = start + len(block)
        if not quiet:
            run_start = end
        elif end - run_start > longest_end - longest_start:
            longest_start, longest_end = run_start, end
    seconds = (longest_end - longest_start) / sample_rate
    if seconds <= SILENCE_SECONDS:
        return None
    return {"start_seconds": longest_start / sample_rate,
            "end_seconds": longest_end / sample_rate, "seconds": seconds}


def silent_wav_span(path, *, start_sample=0, end_sample=None):
    import soundfile as sf

    with sf.SoundFile(str(Path(path))) as source:
        start = min(len(source), max(0, start_sample))
        end = min(len(source), end_sample) if end_sample is not None else len(source)
        if end <= start:
            return None
        source.seek(start)
        return silent_span(source.read(end - start, dtype="float32", always_2d=True),
                           source.samplerate)


def run_with_silence_retry(run, inspect, *, seed, enabled=False, cancelled=None, on_retry=None):
    """Retry only this chunk; callers commit its audio/history after acceptance."""
    if not isinstance(enabled, bool):
        raise ValueError("silence_retry must be a boolean")
    if not enabled:
        return run(seed), None
    attempts = []
    for attempt in range(MAX_RETRIES + 1):
        if cancelled and cancelled():
            from .gguf_runtime import GenerationCancelled
            raise GenerationCancelled("Generation cancelled during silence retry")
        # Separate retry seeds from adjacent chunk seeds, while keeping replay
        # deterministic and valid for both native and Python engines.
        current_seed = seed if not attempt else (seed + attempt * 0x9E3779B9) % 2**32
        value = run(current_seed)
        if cancelled and cancelled():
            from .gguf_runtime import GenerationCancelled
            raise GenerationCancelled("Generation cancelled during silence retry")
        span = inspect(value)
        attempts.append({"seed": current_seed, "silence": span})
        if span is None or attempt == MAX_RETRIES:
            return value, {
                "retries": attempt, "exhausted": span is not None,
                "threshold_seconds": SILENCE_SECONDS, "attempts": attempts,
            }
        if on_retry:
            on_retry({"stage": "silence_retry", "retry": attempt + 1,
                      "max_retries": MAX_RETRIES, "silence_seconds": span["seconds"]})


def retry_summary(records):
    reports = [row["silence_retry"] for row in records if row.get("silence_retry")]
    return ({"retries": sum(row["retries"] for row in reports),
             "exhausted": any(row["exhausted"] for row in reports)}
            if reports else None)
