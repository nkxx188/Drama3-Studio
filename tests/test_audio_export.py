"""Large exports preserve delivered PCM and reject incompatible sources atomically."""

import numpy as np
import pytest
import soundfile as sf

from drama_low_vram.codec import concatenate_wavs


def test_export_preserves_pcm16_across_block_and_file_boundaries(tmp_path):
    a = np.arange(70001, dtype=np.int32).astype(np.int16)
    b = np.asarray([-32768, -1, 0, 1, 32767], dtype=np.int16)
    first, second = tmp_path / "a.wav", tmp_path / "b.wav"
    sf.write(first, a, 44100, subtype="PCM_16")
    sf.write(second, b, 44100, subtype="PCM_16")
    output = concatenate_wavs([first, second], tmp_path / "joined.wav")
    actual, rate = sf.read(output, dtype="int16")
    assert rate == 44100 and np.array_equal(actual, np.concatenate((a, b)))


@pytest.mark.parametrize("rate,channels", [(22050, 1), (44100, 2)])
def test_incompatible_audio_does_not_overwrite_a_previous_export(tmp_path, rate, channels):
    source, output = tmp_path / "source.wav", tmp_path / "existing.wav"
    samples = np.ones(20 if channels == 1 else (20, channels), dtype=np.int16)
    sf.write(source, samples, rate, subtype="PCM_16")
    output.write_bytes(b"existing export")
    with pytest.raises(ValueError, match="mono PCM16"):
        concatenate_wavs([source], output)
    assert output.read_bytes() == b"existing export"
    assert not output.with_name(output.name + ".tmp").exists()
