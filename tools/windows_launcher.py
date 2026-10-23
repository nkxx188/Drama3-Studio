"""One Windows entry point: open Studio, reuse its server and keep failures visible."""

import argparse
from contextlib import contextmanager
import errno
import hashlib
from http.client import HTTPException
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor/fish-speech"), str(ROOT)]
DEFAULT_PORT = 8879
PORT_ATTEMPTS = 20
APP_ID = "drama3-studio"


def instance_key(root):
    return hashlib.sha256(str(Path(root).resolve()).casefold().encode("utf-8")).hexdigest()


def configure_environment():
    for key, name in (("TEMP", "tmp"), ("TMP", "tmp"), ("HF_HOME", "huggingface"),
                      ("TRITON_CACHE_DIR", "triton"), ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
                      ("TORCH_EXTENSIONS_DIR", "torch_extensions"), ("CUDA_CACHE_PATH", "cuda")):
        folder = ROOT / ".cache" / name
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(folder)
    os.environ["PYTHONUTF8"] = "1"
    os.environ["HF_HUB_CACHE"] = str(ROOT / ".cache/huggingface/hub")
    os.environ["MODELSCOPE_CACHE"] = str(ROOT / ".cache/modelscope")
    os.environ["UV_CACHE_DIR"] = str(ROOT / ".cache/uv")
    os.environ["PIP_CACHE_DIR"] = str(ROOT / ".cache/pip")


def save_diagnostics(*, deep=False):
    from drama_low_vram.diagnostics import snapshot
    report = snapshot(ROOT, native=deep, deep=deep)
    (ROOT / "diagnostics").mkdir(exist_ok=True)
    (ROOT / "diagnostics/startup.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                                encoding="utf-8")
    return report


def candidate_ports(port):
    return [0] if port == 0 else range(port, min(port + PORT_ATTEMPTS, 65536))


def browser_host(host):
    return "127.0.0.1" if host in ("0.0.0.0", "localhost") else host


def ui_url(host, port):
    host = browser_host(host)
    return f"http://{'[' + host + ']' if ':' in host else host}:{port}/"


def probe_instance(host, port):
    """A cheap app identity check; never invoke CUDA or trust another port's health API."""
    request = Request(ui_url(host, port) + "api/instance", headers={"Connection": "close"})
    try:
        # Local startup must not depend on the user's proxy settings.
        with build_opener(ProxyHandler({})).open(request, timeout=0.3) as response:
            value = json.loads(response.read(8192))
        return (isinstance(value, dict) and value.get("app") == APP_ID
                and value.get("instance_key") == instance_key(ROOT))
    except (OSError, URLError, HTTPException, ValueError):
        return False


def existing_instance(host, port):
    ports = []
    try:
        state = json.loads((ROOT / "diagnostics/ui-instance.json").read_text(encoding="utf-8"))
        saved_port = state.get("port")
        if (state.get("host") == browser_host(host) and isinstance(saved_port, int)
                and not isinstance(saved_port, bool) and 1 <= saved_port <= 65535):
            ports.append(saved_port)
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    ports.extend(value for value in candidate_ports(port) if value and value not in ports)
    for candidate in ports:
        if probe_instance(host, candidate):
            return ui_url(host, candidate)
    return None


@contextmanager
def startup_lock():
    """Serialize double clicks, including while the first process imports Torch."""
    lock_path = ROOT / ".cache/ui-launcher.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        locked = False
        try:
            def acquire():
                nonlocal locked
                if locked:
                    return True
                stream.seek(0)
                try:
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as error:
                    if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
                    return False
                locked = True
                return True
            yield acquire
        finally:
            if locked:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream, fcntl.LOCK_UN)


def reopen(url, *, open_browser):
    print(f"Drama 3 Studio is already running: {url}", flush=True)
    if open_browser:
        webbrowser.open(url)


def run_ui(args):
    existing = existing_instance(args.host, args.port)
    if existing:
        reopen(existing, open_browser=args.open_browser)
        return
    with startup_lock() as acquire:
        deadline = time.monotonic() + 90
        while not acquire():
            existing = existing_instance(args.host, args.port)
            if existing:
                reopen(existing, open_browser=args.open_browser)
                return
            if time.monotonic() >= deadline:
                raise RuntimeError("Another Studio process is still starting. Check its window for errors.")
            time.sleep(0.25)
        # A simultaneous launcher may have become ready just before we took the lock.
        existing = existing_instance(args.host, args.port)
        if existing:
            reopen(existing, open_browser=args.open_browser)
            return
        print("Starting Drama 3 Studio...", flush=True)
        from drama_low_vram.studio_logging import console_session, detail, status
        with console_session(ROOT):
            detail("Starting Drama 3 Studio")
            report = save_diagnostics()
            gpu_names = list(dict.fromkeys(gpu["name"] for gpu in report.get("gpus", [])))
            if gpu_names:
                status("GPU: %s", ", ".join(gpu_names))
            from ui.server import Handler, StudioHTTPServer, STATE, serve
            httpd = None
            try:
                for port in candidate_ports(args.port):
                    try:
                        httpd = StudioHTTPServer((args.host, port), Handler)
                        break
                    except OSError as error:
                        if (error.errno not in (errno.EADDRINUSE, errno.EACCES)
                                and getattr(error, "winerror", None) not in (10013, 10048)):
                            raise
                else:
                    raise RuntimeError("No available local port. Close an unused Studio window and try again.")
                port = httpd.server_port
                url = ui_url(args.host, port)
                folder = ROOT / "diagnostics"
                folder.mkdir(exist_ok=True)
                temporary = folder / "ui-instance.tmp"
                temporary.write_text(json.dumps({
                    "app": APP_ID, "instance_key": instance_key(ROOT), "pid": os.getpid(),
                    "host": browser_host(args.host), "port": port, "url": url,
                }, indent=2) + "\n", encoding="utf-8")
                temporary.replace(folder / "ui-instance.json")
                if args.port and port != args.port:
                    status("Port %s is in use; using %s.", args.port, port)
            except BaseException:
                if httpd is not None:
                    httpd.server_close()
                STATE.shutdown()
                raise
            serve(httpd, open_browser=args.open_browser)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    mode = argv.pop(0) if argv and argv[0] in ("ui", "engine", "gguf", "doctor") else "ui"
    # CLI tools remain available to developers without being extra double-click entries.
    if mode in ("engine", "gguf"):
        configure_environment()
        target = ROOT / ("tools/run_drama_engine.py" if mode == "engine" else "tools/run_drama_gguf.py")
        sys.argv = [str(target), *argv]
        runpy.run_path(str(target), run_name="__main__")
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--diagnostics", action="store_true", help="Export support diagnostics without starting Studio")
    browser = parser.add_mutually_exclusive_group()
    browser.add_argument("--no-browser", dest="open_browser", action="store_false",
                         help="Start the service without opening a browser")
    browser.add_argument("--open-browser", dest="open_browser", action="store_true", help=argparse.SUPPRESS)
    parser.set_defaults(open_browser=True)
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    configure_environment()
    if args.diagnostics or mode == "doctor":
        from drama_low_vram.diagnostics import export
        detailed = save_diagnostics(deep=True)
        print(json.dumps(detailed, ensure_ascii=False, indent=2))
        print("Saved:", export(ROOT, ROOT / "diagnostics/support.zip"))
        if detailed["errors"]:
            raise SystemExit(1)
        return
    run_ui(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as error:
        try:
            folder = ROOT / "diagnostics"
            folder.mkdir(exist_ok=True)
            (folder / "launcher-error.log").write_text(traceback.format_exc(), encoding="utf-8")
        except OSError:
            traceback.print_exc()
        print(f"Startup failed: {type(error).__name__}: {error}", file=sys.stderr)
        if isinstance(error, ModuleNotFoundError):
            print("The Python runtime is incomplete. Re-extract the complete Windows package.", file=sys.stderr)
        print(f"Details: {ROOT / 'diagnostics/launcher-error.log'}", file=sys.stderr)
        raise SystemExit(1)
