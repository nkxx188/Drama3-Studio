"""Run real-browser UI checks with separate settings and no model inference.

Set NODE_PATH to an existing Playwright installation. A temporary localhost
server uses the actual model catalogue and selection APIs. It never touches
the user's running service, settings, audio or generation history.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import ui.server as ui

    ui.STATE.shutdown()
    folder = ROOT / ".cache" / os.environ.get("DRAMA_UI_TEST_FOLDER", "ui-localization-check")
    state = ui.STATE = ui.AppState(data_root=folder)
    # A synthetic running record exercises switch protection without loading a GPU model.
    class Handler(ui.Handler):
        def do_GET(self):
            if self.path.startswith("/files/ui_data/"):
                relative = self.path.removeprefix("/files/ui_data/")
                return self._file(ui._inside(state.data_root / ui.unquote(relative), state.data_root))
            return super().do_GET()

        def do_POST(self):
            if self.path == "/__test_busy":
                busy = self._payload()["busy"]
                with state._jobs_lock:
                    if busy:
                        state._jobs["ui-check-busy"] = {
                            "id": "ui-check-busy", "status": "running", "mode": "tts",
                            "created_at": time.time(), "events": [], "payload_summary": {},
                        }
                    else:
                        state._jobs.pop("ui-check-busy", None)
                return self._json(200, {"busy": busy})
            return super().do_POST()

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    environment = dict(os.environ, DRAMA_UI_URL=f"http://127.0.0.1:{httpd.server_port}")
    try:
        result = subprocess.run(
            [shutil.which("node") or "node", str(ROOT / "tools" / os.environ.get(
                "DRAMA_UI_BROWSER_SCRIPT", "validate_ui_localization.cjs"))],
            cwd=ROOT, env=environment, timeout=180,
        )
        restored = ui.AppState(data_root=folder)
        try:
            report_path = ROOT / os.environ.get("DRAMA_UI_REPORT", "research/phase38-ui-localization.json")
            with report_path.open(encoding="utf-8") as handle:
                report = json.load(handle)
            report["checks"]["selected_model_persisted_on_server_restart"] = (
                str(restored.checkpoint) == report.get("selected_model"))
            report["passed"] = not report["errors"] and all(report["checks"].values())
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        finally:
            restored.shutdown()
        return result.returncode or (0 if report["passed"] else 1)
    finally:
        httpd.shutdown()
        httpd.server_close()
        state.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
