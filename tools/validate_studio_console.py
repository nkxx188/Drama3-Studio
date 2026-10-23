"""Run a fresh isolated launcher and verify that routine HTTP traffic stays off its console."""

from __future__ import annotations

from http.client import HTTPException
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener
import uuid
import wave

import psutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    work = ROOT / ".cache/studio-console-acceptance" / uuid.uuid4().hex[:10]
    app = work / "application"
    app.mkdir(parents=True)
    ignore = shutil.ignore_patterns(
        "__pycache__", "*.pyc", ".git", "checkpoints", "outputs", ".cache",
        ".venv", "node_modules", "editor-source")
    for name in ("src", "ui", "vendor/fish-speech"):
        shutil.copytree(ROOT / name, app / name, ignore=ignore)
    (app / "tools").mkdir()
    shutil.copy2(ROOT / "tools/windows_launcher.py", app / "tools/windows_launcher.py")
    (app / "outputs").mkdir()
    with wave.open(str(app / "outputs/fixture.wav"), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\0\0" * 16000)
    output = work / "console.txt"
    report = {
        "scope": "Fresh launcher process and real HTTP/assets/draft/audio requests in an isolated copy; no model weights or inference jobs.",
        "work": str(work), "checks": {},
    }
    checks, process = report["checks"], None
    opener = build_opener(ProxyHandler({}))

    def request(url, path, *, data=None, headers=None, method=None):
        body = None if data is None else json.dumps(data).encode("utf-8")
        values = dict(headers or {})
        if body is not None:
            values["Content-Type"] = "application/json"
        return opener.open(Request(url + path.lstrip("/"), data=body, headers=values, method=method), timeout=5)

    try:
        with output.open("wb") as stream:
            process = subprocess.Popen(
                [sys.executable, "-X", "utf8", "-I", str(app / "tools/windows_launcher.py"),
                 "--no-browser", "--port", "0"],
                cwd=work, stdout=stream, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            deadline, info = time.monotonic() + 60, None
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("Isolated launcher exited: " + output.read_text(encoding="utf-8", errors="replace"))
                try:
                    info = json.loads((app / "diagnostics/ui-instance.json").read_text(encoding="utf-8"))
                    with request(info["url"], "api/instance") as response:
                        identity = json.load(response)
                    owned = [process.pid, *(child.pid for child in psutil.Process(process.pid).children(recursive=True))]
                    if identity["pid"] == info["pid"] and identity["pid"] in owned:
                        break
                except (OSError, HTTPException, URLError, ValueError, KeyError):
                    pass
                time.sleep(.1)
            else:
                raise TimeoutError("Isolated Studio did not start in 60 seconds")
            url = info["url"]
            checks["service_owned_by_this_validation"] = identity["pid"] in owned
            report["launcher_pid"], report["service_pid"] = process.pid, identity["pid"]
            report["url"] = url
            traffic = 0
            for _ in range(12):
                for path in ("api/jobs", "api/studio", "api/voices"):
                    with request(url, path) as response:
                        assert response.status == 200
                    traffic += 1
                with request(url, "api/studio/draft", data={"key": "single", "value": {"text": "Isolated log fixture."}}) as response:
                    assert response.status == 200
                traffic += 1
            for path in ("", "assets/app.js", "assets/studio.js", "assets/drama3-fish-mark.svg"):
                with request(url, path) as response:
                    assert response.status == 200
                traffic += 1
            for _ in range(8):
                with request(url, "files/outputs/fixture.wav", headers={"Range": "bytes=0-99"}) as response:
                    assert response.status == 206 and len(response.read()) == 100
                traffic += 1
            with request(url, "files/outputs/fixture.wav", method="HEAD") as response:
                assert response.status == 200
            traffic += 1
            try:
                request(url, "missing")
            except HTTPError as error:
                assert error.code == 404
            else:
                raise AssertionError("Missing page unexpectedly succeeded")
            traffic += 1
            report["routine_requests"] = traffic
            shown = output.read_text(encoding="utf-8", errors="replace")
            saved = (app / "diagnostics/studio.log").read_text(encoding="utf-8")
            checks["startup_address_visible"] = f"Ready: {url}" in shown
            checks["selected_model_and_log_location_visible"] = "Model:" in shown and "diagnostics/studio.log" in shown
            checks["window_has_only_a_few_lines"] = len(shown.splitlines()) <= 9
            checks["no_access_logs_or_draft_requests_in_console"] = all(
                text not in shown for text in ("HTTP/1.1", "GET /", "POST /", "fixture.wav"))
            checks["all_routine_requests_in_detailed_log"] = saved.count('"GET /api/jobs HTTP/1.1" 200') == 12 \
                and saved.count('"POST /api/studio/draft HTTP/1.1" 200') == 12
            checks["asset_and_audio_range_details_preserved"] = "/assets/app.js" in saved \
                and saved.count('"GET /files/outputs/fixture.wav HTTP/1.1" 206') == 8
            checks["no_gpu_generation_jobs_submitted"] = not any((app / "ui_data/jobs").iterdir())
            checks["no_model_weights_downloaded"] = not list(app.rglob("*.safetensors")) and not list(app.rglob("*.gguf"))
            report["console"] = shown
            report["passed"] = all(checks.values())
            if not report["passed"]:
                raise AssertionError([name for name, passed in checks.items() if not passed])
    except BaseException as error:
        report.update(passed=False, error=repr(error))
        raise
    finally:
        # This is the test's own isolated child, never a user's running Studio.
        if process is not None and process.poll() is None:
            children = psutil.Process(process.pid).children(recursive=True)
            for child in reversed(children):
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            process.terminate()
            psutil.wait_procs(children, timeout=10)
            process.wait(timeout=15)
        (ROOT / "research/2026-10-21-studio-console-acceptance.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
