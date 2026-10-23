"""Normal Studio activity stays quiet; useful status and diagnostic detail survive."""

import io
import sys
from threading import Thread
from urllib.error import HTTPError
from urllib.request import ProxyHandler, build_opener

import pytest

from drama_low_vram.studio_logging import brief, console_session, detail, status


def test_console_shows_status_but_not_third_party_output(tmp_path):
    console = io.StringIO()
    original = sys.stdout, sys.stderr
    with console_session(tmp_path, console=console) as log:
        status("Ready: %s", "http://127.0.0.1:8879/")
        print("A verbose model setup detail.")
        print("A library warning.", file=sys.stderr)
        detail("HTTP GET /api/jobs 200")
        status("Job #%s completed. Generation: %.3f s.", "123abc", 7.7181)
    assert (sys.stdout, sys.stderr) == original
    shown = console.getvalue()
    assert "Ready: http://127.0.0.1:8879/" in shown
    assert "Generation: 7.718 s." in shown
    assert "HTTP" not in shown and "model setup" not in shown and "library warning" not in shown
    saved = log.path.read_text(encoding="utf-8")
    for text in ("Ready:", "verbose model setup", "library warning", "HTTP GET /api/jobs 200"):
        assert text in saved


def test_partial_lines_threads_and_ansi_are_written_to_the_detailed_log(tmp_path):
    console = io.StringIO()
    with console_session(tmp_path, console=console) as log:
        print("\x1b[32mPart one", end="")
        print(" and two\x1b[0m.")
        worker = Thread(target=lambda: print("Worker output."))
        worker.start()
        worker.join(timeout=5)
        print("Progress 1\rProgress 2", end="", flush=True)
    saved = log.path.read_text(encoding="utf-8")
    assert "Part one and two." in saved and "\x1b" not in saved
    assert "Worker output." in saved and "Progress 1" in saved and "Progress 2" in saved
    assert console.getvalue() == ""


def test_nested_sessions_do_not_duplicate_or_redirect_status_into_the_log(tmp_path):
    console = io.StringIO()
    with console_session(tmp_path, console=console) as first:
        with console_session(tmp_path / "nested") as second:
            assert first is second
            status("One ready message.")
    assert console.getvalue().count("One ready message.") == 1
    assert first.path.read_text(encoding="utf-8").count("One ready message.") == 1
    assert not (tmp_path / "nested").exists()


def test_diagnostics_rotate_instead_of_growing_without_limit(tmp_path):
    with console_session(tmp_path, console=io.StringIO(), max_bytes=512, backup_count=2) as log:
        for index in range(100):
            detail("A detailed event %03d: %s", index, "x" * 90)
    files = sorted(log.path.parent.glob("studio.log*"))
    assert len(files) == 3
    assert all(file.stat().st_size <= 512 for file in files)
    assert "099" in log.path.read_text(encoding="utf-8")


def test_exception_restores_streams_and_keeps_full_traceback(tmp_path):
    console = io.StringIO()
    original = sys.stdout, sys.stderr
    with pytest.raises(RuntimeError, match="Synthetic failure"):
        with console_session(tmp_path, console=console) as log:
            raise RuntimeError("Synthetic failure")
    assert (sys.stdout, sys.stderr) == original
    assert "Traceback" in log.path.read_text(encoding="utf-8")
    assert "RuntimeError: Synthetic failure" in log.path.read_text(encoding="utf-8")


def test_log_write_failure_does_not_recurse_or_hide_console_status(tmp_path, monkeypatch):
    console = io.StringIO()
    with console_session(tmp_path, console=console) as log:
        monkeypatch.setattr(log.handler, "shouldRollover", lambda _record: (_ for _ in ()).throw(OSError("Disk full")))
        status("Still running.")
        print("Detailed output with an unavailable disk.")
        status("Still responding.")
    shown = console.getvalue()
    assert shown.count("Warning: detailed logging is unavailable.") == 1
    assert "Still running." in shown and "Still responding." in shown


def test_console_errors_are_concise_and_single_line():
    text = brief("\x1b[31mFailure\x1b[0m\n" + "long detail " * 100, 80)
    assert text.startswith("Failure ") and text.endswith("...")
    assert len(text) == 80 and "\n" not in text and "\x1b" not in text


def test_many_real_http_requests_do_not_fill_the_console(tmp_path, monkeypatch):
    import ui.server as ui

    monkeypatch.setattr(ui, "ROOT", tmp_path)
    state = ui.AppState(data_root=tmp_path / "ui_data")
    monkeypatch.setattr(ui, "STATE", state)
    server = ui.StudioHTTPServer(("127.0.0.1", 0), ui.Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    console = io.StringIO()
    opener = build_opener(ProxyHandler({}))
    try:
        with console_session(tmp_path, console=console) as log:
            worker.start()
            for _ in range(20):
                with opener.open(f"http://127.0.0.1:{server.server_port}/api/jobs") as response:
                    assert response.status == 200
            for url in ("/", "/assets/app.js", "/api/studio", "/api/voices"):
                with opener.open(f"http://127.0.0.1:{server.server_port}{url}") as response:
                    assert response.status == 200
            with pytest.raises(HTTPError):
                opener.open(f"http://127.0.0.1:{server.server_port}/missing")
            status("Job #123abc completed.")
            server.shutdown()
            worker.join(timeout=5)
        shown = console.getvalue()
        assert shown.count("\n") == 1 and "completed" in shown
        assert "GET " not in shown and "HTTP/" not in shown
        saved = log.path.read_text(encoding="utf-8")
        assert saved.count('"GET /api/jobs HTTP/1.1" 200') == 20
        assert "/assets/app.js" in saved and "/missing" in saved
    finally:
        server.server_close()
        state.shutdown()


def test_job_result_console_reports_generation_time_not_total_time(tmp_path, monkeypatch):
    import ui.server as ui

    monkeypatch.setattr(ui, "ROOT", tmp_path)
    state = ui.AppState(data_root=tmp_path / "ui_data")
    state._jobs["test-job"] = {
        "id": "test-job", "status": "complete", "created_at": 1,
        "timing": {"generation_seconds": 7.718, "model_load_seconds": 88.0, "total_seconds": 95.718},
    }
    console = io.StringIO()
    try:
        with console_session(tmp_path, console=console):
            state._report_result("test-job")
            state._jobs["test-job"]["status"] = "cancelled"
            state._report_result("test-job")
            state._jobs["test-job"]["status"] = "incomplete"
            state._report_result("test-job")
        shown = console.getvalue()
        assert "Generation: 7.718 s." in shown and "88.000" not in shown and "95.718" not in shown
        assert "cancelled" in shown and "without a complete result" in shown
    finally:
        state.shutdown()


def test_server_stop_status_remains_visible_and_resources_are_closed(tmp_path, monkeypatch):
    import ui.server as ui
    from types import SimpleNamespace

    monkeypatch.setattr(ui, "ROOT", tmp_path)
    closed = []
    monkeypatch.setattr(ui, "STATE", SimpleNamespace(
        checkpoint=tmp_path / "example.gguf", shutdown=lambda: closed.append("state")))

    def stop():
        raise KeyboardInterrupt()

    httpd = SimpleNamespace(
        server_address=("127.0.0.1", 12345), server_port=12345,
        serve_forever=stop, server_close=lambda: closed.append("http"))
    console = io.StringIO()
    with console_session(tmp_path, console=console):
        ui.serve(httpd)
    shown = console.getvalue()
    assert "Ready: http://127.0.0.1:12345/" in shown
    assert "Model: example.gguf (loads on first generation)" in shown
    assert "Detailed log: diagnostics/studio.log" in shown and "Studio stopped." in shown
    assert closed == ["http", "state"]
