"""Real process termination preserves delivered parts and frozen retry inputs."""

import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

import ui.server as ui
from test_gguf_workflows import Native
from test_ui_server import defer_workers


ROOT = Path(__file__).resolve().parents[1]
CHILD = r"""
import json
from pathlib import Path
import sys
from threading import RLock
import time
from types import SimpleNamespace

root, folder, kind = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path[:0] = [str(root), str(root / "src"), str(root / "vendor/fish-speech")]
import torch
import ui.server as ui
from drama_low_vram.codec import write_wav
from drama_low_vram.gguf_takes import save_native_take
from drama_low_vram.protocols import PreviousTake

ui.ROOT = folder
state = ui.AppState()
state.checkpoint = folder / "fixture.gguf"
state.settings_file.write_text(json.dumps({"checkpoint": str(state.checkpoint)}), encoding="utf-8")
state.capabilities = lambda: {
    "scene_scheduler": True, "streaming_text": True, "director_revision": True,
    "infill": True, "joint_dialogue": True, "voice_design": True,
}

class Engine:
    is_gguf = True
    model_path = state.checkpoint
    options = SimpleNamespace(codec_max_frames=2048)
    _lock = RLock()

    def generate(self, text, output, **kwargs):
        if "Third." in text:
            marker = folder / "ready.tmp"
            marker.write_text(
                json.dumps({"job_id": Path(output).relative_to(state.jobs_root).parts[0]}), encoding="utf-8")
            marker.replace(folder / "ready.json")
            while True:
                time.sleep(.1)
        codes = torch.ones(10, 4, dtype=torch.int64)
        saved = save_native_take(output, PreviousTake(
            text, codes, kwargs.get("lookahead") if kwargs.get("task_mode") == "stream" else None))
        write_wav(torch.zeros(8192), output)
        return {"status": "complete", "output": str(output), "take_file": str(saved),
                "audio": {"samples": 8192, "sample_rate": 44100}}

state._get_engine = lambda _identifier: Engine()
context = {"work_id": "interrupted-work", "segment_id": "part", "layout": "scene" if kind == "scene" else "single"}
if kind == "scene":
    identifier = state.create_job({
        "mode": "scene", "scene_strategy": "per_line", "scene": "A room.",
        "characters": [{"name": "Ada", "description": "Calm.", "speaker": 0}],
        "lines": [{"character": "Ada", "text": text} for text in ("First.", "Second.", "Third.")],
        "line_ids": ["first", "second", "third"], "studio": context,
    })
else:
    identifier = state.create_job({"mode": "stream", "chunks": ["First."], "live_stream": True, "studio": context})
    state.stream_input(identifier, text="Second.")
    state.stream_input(identifier, text="Third.")
    state.stream_input(identifier, finish=True)
while True:
    time.sleep(.1)
"""


@pytest.mark.parametrize("kind", ["scene", "stream"])
def test_forced_process_stop_recovers_completed_parts_and_all_received_text(tmp_path, monkeypatch, kind):
    process = subprocess.Popen(
        [sys.executable, "-X", "utf8", "-c", CHILD, str(ROOT), str(tmp_path), kind],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    marker = tmp_path / "ready.json"
    try:
        deadline = time.monotonic() + 60
        while not marker.is_file():
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                pytest.fail(f"Recovery fixture exited: {stdout.decode(errors='replace')}\n{stderr.decode(errors='replace')}")
            if time.monotonic() >= deadline:
                pytest.fail("Recovery fixture did not reach the third generation")
            time.sleep(.05)
        assert process.poll() is None
        identifier = json.loads(marker.read_text(encoding="utf-8"))["job_id"]
        persisted = json.loads((tmp_path / "ui_data/jobs" / identifier / "job.json").read_text(encoding="utf-8"))
        assert persisted["status"] == "running"
        delivered = [event for event in persisted["events"] if event.get("stage") == "segment"]
        assert len(delivered) == 2
        original_audio = [Path(row["path"]).read_bytes() for row in delivered]
        process.kill()
        process.wait(timeout=15)
        monkeypatch.setattr(ui, "ROOT", tmp_path)
        monkeypatch.setattr(ui, "load_native_runtime", lambda _root: {"protocol": 4, "sha256": "synthetic-runtime"})
        recovered = ui.AppState()
        defer_workers(recovered)
        recovered._get_engine = lambda _: Native()
        try:
            job = recovered.job(identifier)
            assert job["status"] == "interrupted" and job["output"] is None
            assert job["recipe"] == persisted["recipe"]
            assert len(job["result"]["segments"]) == 2
            for index in range(2):
                version = recovered.version({"job_id": identifier, "part_index": index})
                assert version["status"] == "complete" and version["take_file"]
                assert Path(version["output"]).read_bytes() == original_audio[index]
                exported = recovered.export_work({"versions": [{"job_id": identifier, "part_index": index}]})
                assert Path(exported["output"]).is_file()
            if kind == "scene":
                with pytest.raises(ValueError, match="Every dialogue line"):
                    recovered.export_work({"work_id": "interrupted-work"})
                assert recovered.version({"job_id": identifier, "part_index": 1})["line_id"] == "second"
            else:
                assert job["recipe"]["chunks"] == ["First."]
                assert job["stream_input"]["chunks"] == ["First.", "Second.", "Third."]
                assert recovered.version({"job_id": identifier})["text"] == "First.\nSecond.\nThird."
            retried = recovered.create_job({"source_version": {"job_id": identifier}, "operation": "retry", "seed": 789})
            recipe = recovered.job(retried)["recipe"]
            assert recipe["seed"] == 789 and recipe["source_version"] == {"job_id": identifier}
            if kind == "stream":
                assert recipe["chunks"] == ["First.", "Second.", "Third."]
                assert recipe["replay_stream"] and not recipe["live_stream"]
            function, arguments = recovered._executor.calls[-1]
            function(*arguments)
            assert recovered.job(retried)["status"] == "complete"
            assert recovered.job(identifier)["status"] == "interrupted"
            assert all(Path(row["path"]).read_bytes() == data for row, data in zip(delivered, original_audio))
        finally:
            recovered.shutdown()
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=15)
