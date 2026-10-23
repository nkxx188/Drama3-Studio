"""Exercise the real start.bat with isolated environment, port conflict and double clicks.

Does not submit generation jobs or alter model choices, voices or saved drafts.
Only processes created by this check are stopped afterward.
"""

from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from threading import Thread
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener
import uuid

import psutil

ROOT = Path(__file__).resolve().parents[1]


def api(url, path):
    with build_opener(ProxyHandler({})).open(url + path.lstrip("/"), timeout=3) as response:
        return json.loads(response.read())


def digest(path):
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run_batch(package, arguments="", *, environment, **options):
    return subprocess.Popen(
        f'cmd.exe /d /s /c ""{package / "start.bat"}" {arguments}"',
        cwd=ROOT.parent, env=environment,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), **options)


def stop_owned_tree(process):
    if process.poll() is not None:
        return
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        return
    for child in reversed(children):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    parent.terminate()
    psutil.wait_procs([parent, *children], timeout=10)
    process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=ROOT / "dist/Drama3 Windows NVIDIA")
    args = parser.parse_args()
    package = args.package_root.resolve()
    if os.name != "nt":
        raise RuntimeError("This acceptance check must run on Windows")
    work = ROOT / ".cache/windows-launcher-acceptance" / uuid.uuid4().hex[:10]
    work.mkdir(parents=True)
    environment = dict(os.environ, DRAMA_NO_PAUSE="1")
    for key in tuple(environment):
        if key.upper().startswith("PYTHON") or key.upper() in ("VIRTUAL_ENV", "CONDA_PREFIX"):
            environment.pop(key)
    system = Path(os.environ["SystemRoot"])
    environment["PATH"] = os.pathsep.join(map(str, (system / "System32", system)))
    checks = {}
    report = {"scope": "Real Windows batch/runtime/HTTP startup; no inference jobs.",
              "package": str(package), "checks": checks, "work": str(work)}
    protected = {str(path): digest(path) for path in (
        package / "ui_data/settings.json", package / "ui_data/studio.json",
    )}
    processes = []
    handle = work / "first-start.log"
    second_log = work / "second-start.log"
    unrelated = None
    unrelated_thread = None
    streams = []
    try:
        checks["one_top_level_batch_entry"] = {path.name for path in package.glob("*.bat")} == {"start.bat"}
        manifest = json.loads((package / "runtime/native/manifest.json").read_text())
        checks["current_native_protocol"] = manifest["protocol"] >= 5
        checks["native_reference_prewarm"] = "reference_prewarm" in manifest.get("features", [])

        class UnrelatedHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"ok":true,"app":"unrelated"}')

            def log_message(self, *_args):
                pass

        unrelated = ThreadingHTTPServer(("127.0.0.1", 0), UnrelatedHandler)
        unrelated_thread = Thread(target=unrelated.serve_forever, daemon=True)
        unrelated_thread.start()
        base_port = unrelated.server_port
        if base_port == 65535:
            raise RuntimeError("Re-run to obtain a local test port below 65535")
        for path in (handle, second_log):
            stream = path.open("wb")
            streams.append(stream)
            processes.append(run_batch(package, f"--no-browser --port {base_port}",
                                       environment=environment, stdout=stream, stderr=subprocess.STDOUT))
        deadline = time.monotonic() + 90
        info = None
        while time.monotonic() < deadline:
            state_path = package / "diagnostics/ui-instance.json"
            try:
                info = json.loads(state_path.read_text(encoding="utf-8"))
                identity = api(info["url"], "api/instance")
                assert identity["app"] == "drama3-studio" and identity["pid"] == info["pid"]
                break
            except (OSError, URLError, ValueError, KeyError, AssertionError):
                if all(process.poll() is not None for process in processes):
                    raise RuntimeError("Both batch launchers exited before a service became ready")
                time.sleep(0.2)
        else:
            raise TimeoutError("Studio did not become ready in 90 seconds")
        url = info["url"]
        owners = []
        for process in processes:
            if process.poll() is None:
                children = psutil.Process(process.pid).children(recursive=True)
                if any(child.pid == identity["pid"] for child in children):
                    owners.append(process)
        checks["server_is_owned_by_this_test"] = len(owners) == 1
        if not checks["server_is_owned_by_this_test"]:
            raise RuntimeError("An existing service was reused; it was left untouched. Stop it before this check.")
        report["url"], report["instance"] = url, identity
        checks["occupied_port_is_preserved"] = api(f"http://127.0.0.1:{base_port}/", "api/instance")["app"] == "unrelated"
        checks["fallback_port_is_used"] = info["port"] != base_port
        checks["portable_interpreter_is_used"] = Path(json.loads(
            (package / "diagnostics/startup.json").read_text())["executable"]).resolve() == (
                package / "runtime/python/python.exe").resolve()
        health = api(url, "api/health")
        checks["real_health_is_ready_without_loading_weights"] = health["ok"] and not health["engine_loaded"]
        checks["advanced_gguf_workflows_available"] = api(url, "api/capabilities")["native_protocol"] >= 4
        for name in ("index.html", "app.js", "studio.js", "audio-editor.js", "references.js", "script-editor.js"):
            path = "" if name == "index.html" else "assets/" + name
            with build_opener(ProxyHandler({})).open(url + path, timeout=5) as response:
                checks["current_asset_" + name] = response.read() == (ROOT / "ui" / name).read_bytes()
        for process in processes:
            if process not in owners:
                checks["simultaneous_double_click_reuses_service"] = process.wait(timeout=20) == 0
        with (work / "reopen.log").open("wb") as stream:
            reopen = run_batch(package, f"--no-browser --port {base_port}",
                               environment=environment, stdout=stream, stderr=subprocess.STDOUT)
            processes.append(reopen)
            checks["later_double_click_returns_success"] = reopen.wait(timeout=20) == 0
        checks["later_double_click_keeps_same_pid"] = api(url, "api/instance")["pid"] == identity["pid"]
        checks["no_gpu_jobs_submitted"] = not api(url, "api/health")["engine_loaded"]
        for process in processes:
            stop_owned_tree(process)

        diagnostic = subprocess.run(
            f'cmd.exe /d /s /c ""{package / "start.bat"}" --diagnostics"',
            cwd=ROOT.parent, env=environment, capture_output=True,
            encoding="utf-8", errors="replace", timeout=90,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        (work / "diagnostics.log").write_text(diagnostic.stdout + diagnostic.stderr, encoding="utf-8")
        checks["same_entry_exports_diagnostics"] = diagnostic.returncode == 0
        checks["native_gpu_diagnostics_pass"] = not json.loads(
            (package / "diagnostics/startup.json").read_text())["errors"]

        missing = work / "缺少运行时 & spaces"
        missing.mkdir()
        shutil.copy2(ROOT / "start.bat", missing / "start.bat")
        # Default double-click policy must pause, while explicit automation can disable it.
        pause_environment = dict(environment)
        pause_environment.pop("DRAMA_NO_PAUSE")
        failed = run_batch(missing, environment=pause_environment, stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        processes.append(failed)
        try:
            failed.wait(timeout=0.75)
            paused = False
        except subprocess.TimeoutExpired:
            paused = True
        output, _ = failed.communicate(b"\n", timeout=15)
        (work / "missing-runtime.log").write_bytes(output)
        checks["missing_runtime_reports_failure"] = failed.returncode == 1 and b"runtime is missing" in output
        checks["error_window_pauses_by_default"] = paused
        checks["missing_runtime_has_local_log"] = (missing / "diagnostics/launcher-error.log").is_file()

        damaged = work / "缺少程序 & spaces"
        (damaged / "tools").mkdir(parents=True)
        shutil.copy2(ROOT / "tools/windows_launcher.py", damaged / "tools/windows_launcher.py")
        failed = subprocess.run(
            [str(package / "runtime/python/python.exe"), "-I", str(damaged / "tools/windows_launcher.py"),
             "--no-browser", "--port", "0"],
            cwd=ROOT.parent, env=environment, capture_output=True, timeout=20,
            encoding="utf-8", errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        (work / "damaged-runtime.log").write_text(failed.stdout + failed.stderr, encoding="utf-8")
        checks["incomplete_app_reports_actionable_error"] = (
            failed.returncode == 1 and "Re-extract the complete Windows package" in failed.stderr)
        checks["python_failure_keeps_traceback"] = "ModuleNotFoundError" in (
            damaged / "diagnostics/launcher-error.log").read_text(encoding="utf-8")
        checks["existing_settings_and_drafts_unchanged"] = all(
            digest(Path(path)) == value for path, value in protected.items())
        if not all(checks.values()):
            raise AssertionError([name for name, passed in checks.items() if not passed])
        report["passed"] = True
    except BaseException as error:
        report.update(passed=False, error=repr(error))
        raise
    finally:
        for process in processes:
            stop_owned_tree(process)
        for stream in streams:
            stream.close()
        if unrelated is not None:
            unrelated.shutdown()
            unrelated.server_close()
        if unrelated_thread is not None:
            unrelated_thread.join(timeout=5)
        target = ROOT / "research/2026-10-20-single-launcher-acceptance.json"
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
