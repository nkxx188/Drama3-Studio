"""Producer/consumer lookahead, bounded input, ordered delivery and cancellation."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
import torch

from drama_low_vram.codec import write_wav
from drama_low_vram.text_stream import TextStreamSession
from drama_low_vram.workflows import save_take


class Workflows:
    def __init__(self):
        self.calls = []
        self.engine = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(
            audio_decoder=SimpleNamespace(num_codebooks=2, vocab_size=8),
            semantic_start_token_id=100, semantic_end_token_id=107,
        )))

    def run(self, mode, text, output, **kwargs):
        self.calls.append((text, kwargs))
        codes = torch.tensor([[0, 1], [2, 3]])
        frames = torch.cat((codes[:1] + 100, codes)).T
        write_wav(torch.ones(4096) * 0.2, output)
        take_file = save_take(output, text, codes, frames, metadata={
            "mode": mode, "lookahead": kwargs["lookahead"]})
        return {"status": "complete", "output": str(output), "take_file": str(take_file),
                "audio": {"samples": 4096, "sample_rate": 44100}}


def test_lookahead_history_and_final_export_are_ordered(tmp_path):
    workflows = Workflows()
    stream = TextStreamSession(workflows, tmp_path / "parts")
    for text in ("One.", "Two.", "Three."):
        stream.push(text)
    stream.finish()
    rows = list(stream.run())
    assert [row["text"] for row in rows] == ["One.", "Two.", "Three."]
    assert [row["lookahead"] for row in rows] == ["Two.", "Three.", ""]
    assert [len(kwargs["previous_takes"]) for _, kwargs in workflows.calls] == [0, 1, 2]
    assert workflows.calls[1][1]["previous_takes"][0].lookahead == "Two."
    assert workflows.calls[2][1]["previous_takes"][-1].lookahead == "Three."
    assert stream.status == "complete"
    assert stream.export(tmp_path / "all.wav").is_file()
    import soundfile as sf
    samples, rate = sf.read(tmp_path / "all.wav")
    assert len(samples) == 4096 * 3 and rate == 44100
    with pytest.raises(RuntimeError):
        stream.push("Late.")
    with pytest.raises(RuntimeError):
        list(stream.run())


def test_consumer_waits_for_lookahead_and_cancellation_wakes_it(tmp_path):
    workflows = Workflows()
    stream = TextStreamSession(workflows, tmp_path)
    stream.push("One.")
    with ThreadPoolExecutor(max_workers=1) as executor:
        consumer = executor.submit(lambda: list(stream.run()))
        # Producer APIs and cancellation share a condition, with no polling sleep.
        stream.cancel()
        assert consumer.result(timeout=5) == []
    assert stream.status == "cancelled" and not workflows.calls
    with pytest.raises(RuntimeError):
        stream.export(tmp_path / "cancel.wav")


def test_finish_flushes_a_single_tail_and_queue_is_bounded(tmp_path):
    workflows = Workflows()
    stream = TextStreamSession(workflows, tmp_path, max_pending=2, max_chunk_chars=12)
    stream.push("One.")
    stream.push("Two.")
    with pytest.raises(BufferError):
        stream.push("Three.")
    stream.finish()
    rows = list(stream.run())
    assert len(rows) == 2 and rows[-1]["lookahead"] == ""
    single = TextStreamSession(workflows, tmp_path / "single")
    single.push("Tail.")
    single.finish()
    assert list(single.run())[0]["lookahead"] == ""


def test_bad_stream_inputs_are_rejected_before_the_engine(tmp_path):
    for kwargs in ({"max_pending": 1}, {"max_pending": True}, {"max_chunk_chars": 0},
                   {"options": {"cancelled": lambda: False}}):
        with pytest.raises(ValueError):
            TextStreamSession(Workflows(), tmp_path, **kwargs)
    stream = TextStreamSession(Workflows(), tmp_path, max_chunk_chars=5)
    with pytest.raises(ValueError):
        stream.push("Long input.")
    with pytest.raises(ValueError):
        stream.push("A<|endofturn|>B")
