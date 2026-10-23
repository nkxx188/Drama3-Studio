"""Real-browser waveform, crop and infill-boundary acceptance without inference."""

from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor/fish-speech"), str(ROOT)]


def main():
    import numpy as np
    import soundfile as sf
    import torch
    import ui.server as ui
    from drama_low_vram.gguf_takes import save_native_take
    from drama_low_vram.protocols import PreviousTake

    ui.STATE.shutdown()
    folder = ROOT / ".cache/waveform-ui-acceptance" / uuid.uuid4().hex[:10]
    state = ui.STATE = ui.AppState(data_root=folder / "ui_data")
    state.checkpoint = next((ROOT / "checkpoints/drama-3-gguf-q8").glob("*.gguf")).resolve()
    rate, frames = 44100, 128
    length = frames * 2048
    at = np.arange(length, dtype="float32") / rate
    speech = np.sin(at * 2 * np.pi * 220) * (0.05 + 0.3 * np.sin(at * 2.3)**2)
    source_dir = state.jobs_root / "waveform-fixture"
    source_dir.mkdir(parents=True, exist_ok=True)
    output = source_dir / "result.wav"
    sf.write(output, speech, rate, subtype="PCM_16")
    stereo = folder / "stereo-reference.wav"
    sf.write(stereo, np.stack((np.zeros_like(speech), speech), axis=1), rate, subtype="FLOAT")
    text = "你好，今天的天气很好，我们一起出去走走吧。"
    codes = torch.arange(10 * frames).reshape(10, frames) % 4096
    take = save_native_take(output, PreviousTake(text, codes), metadata={"spoken_text": text, "mode": "tts"})
    state._jobs["waveform-fixture"] = {
        "id": "waveform-fixture", "mode": "tts", "status": "complete",
        "created_at": time.time(), "updated_at": time.time(), "events": [],
        "payload_summary": {"text_length": len(text)}, "error": None,
        "output": str(output), "take_file": str(take),
        "result": {"status": "complete", "segments": [], "output": str(output), "take_file": str(take)},
    }
    class Handler(ui.Handler):
        def do_GET(self):
            if self.path.startswith("/files/ui_data/"):
                relative = self.path.removeprefix("/files/ui_data/")
                return self._file(ui._inside(state.data_root / ui.unquote(relative), state.data_root))
            return super().do_GET()

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    environment = dict(
        os.environ, DRAMA_UI_URL=f"http://127.0.0.1:{httpd.server_port}",
        DRAMA_WAVEFORM_AUDIO=str(stereo),
        DRAMA_WAVEFORM_TAKE=str(take),
        DRAMA_WAVEFORM_FOLDER=str(folder),
    )
    try:
        run = subprocess.run(
            [shutil.which("node") or "node", str(ROOT / "tools/validate_waveform_ui.cjs")],
            env=environment, cwd=ROOT, timeout=180)
        report_path = ROOT / "research/2026-10-19-waveform-creation-regression.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["checks"]["no_inference_model_loaded"] = state._engine is None
        if report.get("crop"):
            clip = report["crop"]
            cropped, cropped_rate = sf.read(
                state.uploads_root / clip["id"] / "reference.wav", dtype="float32", always_2d=True)
            original, _ = sf.read(stereo, dtype="float32", always_2d=True)
            first, last = round(clip["start"] * rate), round(clip["end"] * rate)
            report["checks"]["cropped_samples_exact"] = bool(np.array_equal(cropped, original[first:last]))
            report["checks"]["crop_preserves_stereo_and_rate"] = cropped_rate == rate and cropped.shape[1] == 2
        report["passed"] = not report["errors"] and all(report["checks"].values())
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return run.returncode or (0 if report["passed"] else 1)
    finally:
        httpd.shutdown()
        httpd.server_close()
        state.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
