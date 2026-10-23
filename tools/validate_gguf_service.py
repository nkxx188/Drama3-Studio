"""Exercise the actual HTTP queue in an isolated local data directory."""

import json
from pathlib import Path
import sys
import threading
import time
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor/fish-speech"), str(ROOT)]
import ui.server as ui
from drama_low_vram.gguf_runtime import GGUFEngine


def main():
    fixture = ROOT / ".cache/phase35-service-fixture"
    fixture.mkdir(parents=True, exist_ok=True)
    original_state = ui.STATE
    original_state.shutdown()
    ui.ROOT = fixture
    ui.GGUFEngine = lambda model, **kwargs: GGUFEngine(model, root=ROOT, context=512, cache=8)
    state = ui.AppState()
    ui.STATE = state
    server = ThreadingHTTPServer(("127.0.0.1", 0), ui.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    report = {"url": origin}
    def api(path, payload=None):
        request = Request(origin + path, None if payload is None else json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=20) as response:
            return json.loads(response.read())
    def wait(job, timeout=180):
        until = time.monotonic() + timeout
        while time.monotonic() < until:
            result = api("/api/jobs/" + job)
            if result["status"] not in ("queued", "running"):
                return result
            time.sleep(0.2)
        raise TimeoutError("Service job timed out")
    try:
        model = next((ROOT / "checkpoints/drama-3-gguf-q4").glob("*.gguf"))
        selected = api("/api/model/select", {"directory": str(model)})
        assert selected["runtime"] == "gguf" and selected["inspection"]["ready"]
        assert api("/api/capabilities")["streaming_audio"]
        one = api("/api/jobs", {"text": "你好，这是本地服务测试。", "max_new_tokens": 128})["id"]
        two = api("/api/jobs", {"text": "你好，这是本地服务测试。", "max_new_tokens": 128})["id"]
        first, second = wait(one), wait(two)
        assert first["status"] == second["status"] == "complete", (first.get("error"), second.get("error"))
        assert second["result"]["session_reused"]
        pid = state._engine._process.pid
        report["queue"] = [first, second]
        cancel = api("/api/jobs", {"text": "请测试取消。这是需要运行一段时间的语音任务。", "max_new_tokens": 256})["id"]
        until = time.monotonic() + 10
        while api("/api/jobs/" + cancel)["status"] != "running":
            if time.monotonic() > until:
                raise TimeoutError("Cancelled job did not start")
            time.sleep(0.05)
        api(f"/api/jobs/{cancel}/cancel", {})
        report["cancel"] = wait(cancel)
        assert report["cancel"]["status"] == "cancelled"
        model = next((ROOT / "checkpoints/drama-3-gguf-q8").glob("*.gguf"))
        report["switch"] = api("/api/model/select", {"directory": str(model)})
        assert report["switch"]["runtime"] == "gguf" and state._engine is None
        report.update(status="passed", previous_worker_pid=pid)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        state.shutdown()
        (ROOT / "research/phase35-service-acceptance.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "queue_seconds": [
        j["result"]["request_seconds"] for j in report["queue"]]}, indent=2))


if __name__ == "__main__":
    main()
