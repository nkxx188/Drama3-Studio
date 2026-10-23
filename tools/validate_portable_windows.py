"""Actual portable-folder CUDA, models, HTTP queue and relocation acceptance.

Run with the package's own interpreter outside the development checkout.
Models are external inputs and never copied into the distribution.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path[:0] = [str(root / "src"), str(root / "vendor/fish-speech"), str(root)]
    assert Path(sys.prefix).resolve() == root / "runtime/python", (sys.prefix, root)
    os.environ.update(TEMP=str(root / ".cache/tmp"), TMP=str(root / ".cache/tmp"),
                      TRITON_CACHE_DIR=str(root / ".cache/relocation-triton"),
                      TORCHINDUCTOR_CACHE_DIR=str(root / ".cache/torchinductor"))
    (root / ".cache/tmp").mkdir(parents=True, exist_ok=True)
    from drama_low_vram.diagnostics import snapshot, export
    report = {"root": str(root), "python": sys.executable, "diagnostics": snapshot(root, deep=True)}
    assert not report["diagnostics"]["errors"], report["diagnostics"]["errors"]
    import ui.server as ui
    assert ui.ROOT == root
    state = ui.STATE
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), ui.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    address = f"http://127.0.0.1:{httpd.server_port}"
    def api(path, payload=None):
        request = Request(address + path, None if payload is None else json.dumps(payload).encode("utf-8"),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read())
    def wait(identifier, limit=600):
        deadline = time.monotonic() + limit
        while time.monotonic() < deadline:
            row = api("/api/jobs/" + identifier)
            if row["status"] not in ("queued", "running"):
                return row
            time.sleep(0.3)
        raise TimeoutError(identifier)
    try:
        voice = state.add_voice("portable test", "你好，这是官方 FP8 质量优先测试。",
                                0, "reference.wav", args.reference.read_bytes())
        assert api("/api/health")["ok"]
        variants = []
        for path, kind in ((args.gguf, "gguf"), (args.checkpoint, "python")):
            selected = api("/api/model/select", {"directory": str(path.resolve())})
            assert selected["inspection"]["ready"]
            requests = []
            for references in ([], [voice["id"]]):
                identifier = api("/api/jobs", {
                    "mode": "tts", "text": "你好，这是一次本地语音测试。",
                    "seed": 123, "max_new_tokens": 192, "voice_ids": references,
                })["id"]
                result = wait(identifier)
                assert result["status"] == "complete", result
                assert Path(result["output"]).is_file()
                requests.append(result)
            variants.append({"kind": kind, "requests": requests})
            report["models"] = variants
        api("/api/model/select", {"directory": str(args.gguf.resolve())})
        assert state._engine is None
        # Real reference encoding and context growth, not a fake capacity
        # response. Repeat source audio to twelve seconds without silently
        # changing user input in the production engine.
        import numpy as np
        import soundfile as sf
        from types import SimpleNamespace
        from drama_low_vram.gguf_runtime import GGUFEngine
        samples, rate = sf.read(args.reference, dtype="float32")
        longer = root / "outputs/acceptance-reference-12s.wav"
        longer.parent.mkdir(exist_ok=True)
        sf.write(longer, np.tile(samples, 1 + int(12 * rate / len(samples)))[:12 * rate], rate)
        engine = GGUFEngine(args.gguf, root=root, context=256)
        try:
            reference = SimpleNamespace(
                audio=longer, transcript=lambda: "<|speaker:0|>" +
                "你好，这是官方 FP8 质量优先测试。" * 3)
            result = engine.generate("你好，这是一次本地语音测试。", root / "outputs/long-reference.wav",
                                     references=[reference], max_frames=192)
            assert result["output"] and Path(result["output"]).is_file() and engine.context > 256
            report["long_reference_context_growth"] = result
        finally:
            engine.close()
        report["diagnostic_export"] = str(export(root, root / "diagnostics/acceptance-support.zip"))
        report["status"] = "passed"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
        state.shutdown()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Relocated portable UI/model acceptance passed", root, flush=True)


if __name__ == "__main__":
    main()
