"""Isolated browser acceptance: real local storage/API/audio, synthetic inference."""

from http.server import ThreadingHTTPServer
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from threading import RLock, Thread
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor/fish-speech"), str(ROOT)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-checks", action="store_true")
    parser.add_argument("--audio-editor-checks", action="store_true")
    parser.add_argument("--simplified-checks", action="store_true")
    parser.add_argument("--tag-checks", action="store_true")
    parser.add_argument("--library-layout-checks", action="store_true")
    parser.add_argument("--voice-design-checks", action="store_true")
    parser.add_argument("--dialogue-checks", action="store_true")
    parser.add_argument("--startup-checks", action="store_true")
    parser.add_argument("--reference-flow-checks", action="store_true")
    parser.add_argument("--reference-prewarm-checks", action="store_true")
    parser.add_argument("--polish-checks", action="store_true")
    parser.add_argument("--result-controls-checks", action="store_true")
    arguments = parser.parse_args()
    import numpy as np
    import soundfile as sf
    import torch
    import ui.server as ui
    from drama_low_vram.codec import write_wav
    from drama_low_vram.gguf_takes import save_native_take
    from drama_low_vram.protocols import PreviousTake

    ui.STATE.shutdown()
    folder = ROOT / ".cache/studio-ui-acceptance" / uuid.uuid4().hex[:10]
    state = ui.STATE = ui.AppState(data_root=folder / "ui_data")
    state.asr.model_dir = folder / "missing-asr-model"
    state.checkpoint = next((ROOT / "checkpoints/drama-3-gguf-q8").glob("*.gguf")).resolve()
    frames, rate = 128, 44100
    waveform = .2 * np.sin(np.arange(frames * 2048) / rate * 2 * np.pi * 220)
    source = folder / "reference.wav"
    sf.write(source, waveform, rate, subtype="FLOAT")
    for name in ("Fixture A", "Fixture B"):
        state.add_voice(name, "参考音频的台词。", 0, "reference.wav", source.read_bytes())

    class Synthetic:
        is_gguf = True
        model_path = state.checkpoint
        options = SimpleNamespace(codec_max_frames=2048)
        _lock = RLock()

        def generate(self, text, output, **kwargs):
            codes = torch.ones(10, frames, dtype=torch.int64)
            if kwargs.get("infill"):
                span = kwargs["infill"]
                codes = torch.cat((span.source[:, :span.start], codes[:, :16], span.source[:, span.end:]), dim=1)
            count = codes.shape[1]
            write_wav(torch.tensor(np.resize(waveform, count * 2048), dtype=torch.float32), output)
            saved = save_native_take(output, PreviousTake(text, codes))
            if kwargs.get("streaming_audio"):
                for index, first in enumerate(range(0, count, 16)):
                    last = min(first + 16, count)
                    part = Path(output).with_name(f"preview-{index}.wav")
                    write_wav(torch.tensor(np.resize(waveform, (last - first) * 2048), dtype=torch.float32), part)
                    kwargs["segment_ready"]({
                        "index": index, "path": str(part), "start_sample": first * 2048,
                        "end_sample": last * 2048, "samples": (last - first) * 2048, "sample_rate": rate,
                    })
                    time.sleep(.25)
            return {"status": "complete", "output": str(output), "take_file": str(saved),
                    "audio": {"samples": count * 2048, "sample_rate": rate}}

    engine = Synthetic()
    state._get_engine = lambda _identifier: engine

    class Handler(ui.Handler):
        def log_message(self, _format, *_arguments):
            pass

        def do_GET(self):
            if self.path.startswith("/files/ui_data/"):
                relative = self.path.removeprefix("/files/ui_data/")
                return self._file(ui._inside(state.data_root / ui.unquote(relative), state.data_root))
            return super().do_GET()

        def do_POST(self):
            if self.path in ("/__test__/long-result", "/__test__/interrupted"):
                payload = self._payload()
                record = ui.clone(state.job(payload["source"]))
                identifier = "fixture-" + uuid.uuid4().hex
                record.update(id=identifier, created_at=time.time(), updated_at=time.time(),
                              studio={"work_id": identifier, "segment_id": "part", "layout": "single"})
                record["recipe"]["studio"] = record["studio"]
                if self.path == "/__test__/long-result":
                    directory = state.jobs_root / identifier
                    directory.mkdir()
                    output = directory / "result.wav"
                    long_rate, seconds = 8000, 130
                    long_audio = .15 * np.sin(np.arange(long_rate * seconds) * 2 * np.pi * 220 / long_rate)
                    sf.write(output, long_audio, long_rate, subtype="FLOAT")
                    saved = save_native_take(
                        output, PreviousTake("Long reference fixture.", torch.ones(10, round(seconds * rate / 2048), dtype=torch.int64)))
                    record.update(mode="tts", status="complete", output=str(output), take_file=str(saved), events=[])
                    record["recipe"].update(mode="tts", text="Long reference fixture.", reference_snapshot=[])
                    record["result"] = {"status": "complete", "output": str(output), "take_file": str(saved),
                                        "segments": [], "audio": {"samples": len(long_audio), "sample_rate": long_rate,
                                                                  "duration_seconds": seconds}}
                else:
                    record.update(status="interrupted", output=None, take_file=None)
                    record["result"].update(status="interrupted", output=None, take_file=None)
                with state._jobs_lock:
                    state._jobs[identifier] = record
                    state._persist_job_locked(identifier)
                state.studio.register(record)
                state.studio.first_completed(record)
                return self._json(200, {"id": identifier})
            if self.path == "/__test__/archive":
                payload = self._payload()
                original = state.job(payload["source"])
                for index in range(106):
                    record = ui.clone(original)
                    record.update(id="archive-" + uuid.uuid4().hex, created_at=time.time() + index / 1000,
                                  events=[], studio={"work_id": "archive-filler", "segment_id": "one", "layout": "single"})
                    record["recipe"].update(studio=record["studio"], text=f"Archive fixture {index}.")
                    with state._jobs_lock:
                        state._jobs[record["id"]] = record
                        state._persist_job_locked(record["id"])
                    state.studio.register(record)
                    state.studio.first_completed(record)
                return self._json(200, {"archived": 106})
            return super().do_POST()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    environment = dict(os.environ, DRAMA_UI_URL=f"http://127.0.0.1:{server.server_port}",
                       DRAMA_STUDIO_FOLDER=str(folder), DRAMA_STUDIO_REFERENCE=str(source))
    try:
        result = subprocess.run(
            [shutil.which("node") or "node", str(ROOT / "tools" / (
                "validate_reference_prewarm_ui.cjs" if arguments.reference_prewarm_checks else
                "validate_result_controls_ui.cjs" if arguments.result_controls_checks else
                "validate_studio_polish_ui.cjs" if arguments.polish_checks else
                "validate_reference_flow_ui.cjs" if arguments.reference_flow_checks else
                "validate_startup_ui.cjs" if arguments.startup_checks else
                "validate_dialogue_ui.cjs" if arguments.dialogue_checks else
                "validate_voice_design_ui.cjs" if arguments.voice_design_checks else
                "validate_library_layout_ui.cjs" if arguments.library_layout_checks else
                "validate_tag_palette_ui.cjs" if arguments.tag_checks else
                "validate_simplified_studio_ui.cjs" if arguments.simplified_checks else
                "validate_audio_editor_ui.cjs" if arguments.audio_editor_checks else
                "validate_studio_recovery_ui.cjs" if arguments.recovery_checks else "validate_studio_ui.cjs"))],
            env=environment, cwd=ROOT, timeout=360)
        return result.returncode
    finally:
        server.shutdown()
        server.server_close()
        state.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
