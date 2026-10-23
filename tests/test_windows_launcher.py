"""Startup identity and locking must work before importing the inference stack."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import socket
from threading import Thread

import pytest


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "tools/windows_launcher.py"
    spec = importlib.util.spec_from_file_location("testable_windows_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    return module


@contextmanager
def identity_server(payload):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == "/api/instance"
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def test_probe_identifies_this_copy_without_importing_or_loading_an_engine(launcher):
    with identity_server({"app": launcher.APP_ID,
                          "instance_key": launcher.instance_key(launcher.ROOT)}) as port:
        assert launcher.probe_instance("127.0.0.1", port)
        assert launcher.existing_instance("127.0.0.1", port).endswith(f":{port}/")


@pytest.mark.parametrize("payload", [
    {"ok": True}, {"app": "another-app"}, [], None,
    {"app": "drama3-studio", "instance_key": "another-checkout"},
])
def test_probe_does_not_reuse_an_unrelated_server(launcher, payload):
    with identity_server(payload) as port:
        assert not launcher.probe_instance("127.0.0.1", port)


def test_stale_instance_file_cannot_open_an_arbitrary_url(launcher, monkeypatch):
    directory = launcher.ROOT / "diagnostics"
    directory.mkdir()
    (directory / "ui-instance.json").write_text(json.dumps({
        "host": "127.0.0.1", "port": 8898, "url": "https://example.invalid/private",
    }))
    calls = []
    monkeypatch.setattr(launcher, "probe_instance",
                        lambda host, port: calls.append((host, port)) or False)
    assert launcher.existing_instance("127.0.0.1", 8879) is None
    assert calls[0] == ("127.0.0.1", 8898)
    assert len(calls) == launcher.PORT_ATTEMPTS


@pytest.mark.parametrize("content", ["{", "[]", "null", '{"port": true}', '{"port": 99999}'])
def test_corrupt_instance_record_does_not_prevent_startup(launcher, monkeypatch, content):
    directory = launcher.ROOT / "diagnostics"
    directory.mkdir()
    (directory / "ui-instance.json").write_text(content)
    monkeypatch.setattr(launcher, "probe_instance", lambda *_args: False)
    assert launcher.existing_instance("127.0.0.1", 8879) is None


def test_saved_fallback_port_is_reopened_first(launcher, monkeypatch):
    directory = launcher.ROOT / "diagnostics"
    directory.mkdir()
    (directory / "ui-instance.json").write_text('{"host":"127.0.0.1","port":8883}')
    calls = []
    monkeypatch.setattr(launcher, "probe_instance",
                        lambda host, port: calls.append((host, port)) or port == 8883)
    assert launcher.existing_instance("127.0.0.1", 8879) == "http://127.0.0.1:8883/"
    assert calls == [("127.0.0.1", 8883)]


def test_startup_lock_excludes_simultaneous_launch_and_releases_on_error(launcher):
    with pytest.raises(RuntimeError, match="startup failed"):
        with launcher.startup_lock() as first:
            assert first()
            with launcher.startup_lock() as second:
                assert not second()
            raise RuntimeError("startup failed")
    with launcher.startup_lock() as later:
        assert later()


def test_local_cache_settings_override_external_locations(launcher, monkeypatch):
    for key in ("TEMP", "TMP", "HF_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR",
                "TORCH_EXTENSIONS_DIR", "CUDA_CACHE_PATH", "PYTHONUTF8", "HF_HUB_CACHE",
                "MODELSCOPE_CACHE", "UV_CACHE_DIR", "PIP_CACHE_DIR"):
        monkeypatch.setenv(key, r"C:\system-cache")
    launcher.configure_environment()
    assert Path(launcher.os.environ["TEMP"]).is_relative_to(launcher.ROOT)
    assert Path(launcher.os.environ["HF_HOME"]).is_relative_to(launcher.ROOT)


def test_no_server_is_not_treated_as_an_existing_instance(launcher):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    assert not launcher.probe_instance("127.0.0.1", port)


def test_bare_entry_starts_ui_and_opens_browser_by_default(launcher, monkeypatch):
    calls = []
    monkeypatch.setattr(launcher, "configure_environment", lambda: None)
    monkeypatch.setattr(launcher, "run_ui", calls.append)
    launcher.main([])
    assert len(calls) == 1 and calls[0].open_browser is True
    assert calls[0].port == 8879


def test_repeat_launch_opens_existing_page_without_initializing_app_state(launcher, monkeypatch):
    from types import SimpleNamespace
    url = "http://127.0.0.1:8880/"
    monkeypatch.setattr(launcher, "existing_instance", lambda *_args: url)
    opened = []
    monkeypatch.setattr(launcher.webbrowser, "open", opened.append)
    monkeypatch.setattr(launcher, "save_diagnostics",
                        lambda: pytest.fail("Repeated launch must not import or initialize the app again"))
    launcher.run_ui(SimpleNamespace(host="127.0.0.1", port=8879, open_browser=True))
    assert opened == [url]


def test_real_startup_keeps_import_and_library_details_out_of_console(launcher, monkeypatch, capsys):
    from types import SimpleNamespace
    import sys
    from drama_low_vram.studio_logging import status

    @contextmanager
    def lock():
        yield lambda: True

    def diagnostics():
        print("Verbose import details.")
        print("Verbose library warning.", file=sys.stderr)
        return {"gpus": [{"name": "Test RTX GPU"}]}

    def serve(_server, **_kwargs):
        status("Ready: http://127.0.0.1:8879/")
        print("Internal server detail.")

    monkeypatch.setattr(launcher, "existing_instance", lambda *_args: None)
    monkeypatch.setattr(launcher, "startup_lock", lock)
    monkeypatch.setattr(launcher, "save_diagnostics", diagnostics)
    monkeypatch.setitem(sys.modules, "ui.server", SimpleNamespace(
        Handler=object(), StudioHTTPServer=lambda *_args: SimpleNamespace(server_port=8879),
        STATE=SimpleNamespace(shutdown=lambda: None), serve=serve))
    launcher.run_ui(SimpleNamespace(host="127.0.0.1", port=8879, open_browser=False))
    shown = capsys.readouterr()
    assert "Starting Drama 3 Studio" in shown.out and "Ready:" in shown.out
    assert "GPU: Test RTX GPU" in shown.out
    assert "Verbose" not in shown.out + shown.err and "Internal server" not in shown.out
    saved = (launcher.ROOT / "diagnostics/studio.log").read_text(encoding="utf-8")
    assert "Verbose import details" in saved and "Verbose library warning" in saved
    assert "Internal server detail" in saved
