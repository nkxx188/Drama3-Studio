"""Real GGUF task acceptance through the HTTP queue, with exact-code/PCM gates."""

from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor/fish-speech"), str(ROOT)]

import numpy as np
import soundfile as sf
import torch

import ui.server as ui
from drama_low_vram.gguf_takes import load_native_take
from drama_low_vram.gguf_runtime import GGUFEngine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q8", action="store_true", help="Run the same matrix on Q8")
    parser.add_argument("--only", nargs="*", help="Only run selected additional tasks")
    args = parser.parse_args()
    label = "q8" if args.q8 else "q4"
    folder = ROOT / ".cache" / f"gguf-features-acceptance-{label}"
    folder.mkdir(parents=True, exist_ok=True)
    receipt = ROOT / "research" / f"2026-10-18-gguf-features-{label}.json"
    report = {"model": label, "started": time.time(), "checks": {}}
    ui.STATE.shutdown()
    state = ui.AppState(data_root=folder / "ui_data")
    ui.STATE = state
    # Same resource policy as production, with short bounded generation limits.
    state.checkpoint = next((ROOT / f"checkpoints/drama-3-gguf-{label}").glob("*.gguf")).resolve()
    server = ThreadingHTTPServer(("127.0.0.1", 0), ui.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_port}"

    def api(path, payload=None):
        request = Request(origin + path, None if payload is None else
                          json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=60) as response:
            return json.load(response)

    def save():
        receipt.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def wait(identifier, observe=None, timeout=600):
        until = time.monotonic() + timeout
        while time.monotonic() < until:
            result = state.job(identifier)
            if observe:
                observe(result)
            if result["status"] not in ("queued", "running"):
                return result
            time.sleep(0.15)
        state.cancel(identifier)
        raise TimeoutError(f"Job {identifier} timed out")

    def run(name, payload, *, observe=None, require_complete=True):
        print("RUN", label, name, flush=True)
        job = api("/api/jobs", payload)["id"]
        result = wait(job, observe)
        report["checks"][name] = result
        save()
        print("RESULT", label, name, result["status"],
              result.get("result", {}).get("total_seconds"), result.get("error"), flush=True)
        if require_complete:
            assert result["status"] == "complete", (name, result.get("error"), result.get("result"))
        return result

    def want(name):
        return args.only is None or name in args.only

    try:
        report["capabilities"] = api("/api/capabilities")
        assert all(report["capabilities"][key] for key in (
            "director_revision", "infill", "streaming_text", "streaming_audio", "scene_scheduler"))
        first = run("tts", {"text": "你好，今天的天气很好。", "seed": 123, "max_new_tokens": 256})
        take_path = first["take_file"]
        inspected = api("/api/takes/inspect", {"path": take_path})
        assert inspected["frames"] > 0 and inspected["audio"]
        report["inspected"] = inspected
        if want("ordinary_regression"):
            old = GGUFEngine(state.checkpoint, root=ROOT)
            state._engine.close()
            # Compare the same resource plan and ordinary request with the
            # backed-up validated binary; leave its DLL lookup in runtime/native.
            old.build = old.build | {
                "executable": str(ROOT / ".cache/gguf-feature-backup-20261018/runtime/native/drama3_worker.exe"),
                "protocol": 2,
            }
            try:
                before = old.generate("你好，今天的天气很好。", folder / "baseline.wav",
                                      seed=123, max_frames=256, chunk_size=2000)
                a, _, _ = load_native_take(before["native_take_file"])
                b, _, _ = load_native_take(take_path)
                assert torch.equal(a.codes, b.codes)
                report["checks"]["ordinary_regression"] = {
                    "codec_codes_equal": True, "frames": a.codes.shape[1],
                    "before_seconds": before["request_seconds"],
                    "after_seconds": first["result"]["request_seconds"],
                }
            finally:
                old.close()
            save()
        if want("controls"):
            run("controls", {"text": "你好，请慢慢说。", "direction": "Speak softly and calmly.",
                             "language": "zh", "duration_frames": 64, "max_new_tokens": 256})
        if want("revision"):
            revised = run("revision", {"mode": "revision", "text": "你好，今天的天气非常好。",
                                      "previous_take": take_path, "direction": "More cheerful.",
                                      "max_new_tokens": 256})
            assert revised["take_file"] and revised["result"]["chunks"]
        if want("edit"):
            run("edit", {"mode": "edit", "text": "我们一起出去走走吧。",
                        "previous_take": take_path, "max_new_tokens": 256})
        if want("clone"):
            voice = state.add_voice(
                "Acceptance", "你好，今天的天气很好。", 0, "reference.wav",
                Path(first["output"]).read_bytes())
            run("clone", {"text": "你好，我们出发吧。", "voice_ids": [voice["id"]],
                          "max_new_tokens": 256})
        if want("infill"):
            source, _, _ = load_native_take(take_path)
            count = source.codes.shape[1]
            filled = run("infill", {"mode": "infill", "text": "你好，今天的天气很好。",
                                   "source_take": take_path, "start_frame": count // 3,
                                   "end_frame": 2 * count // 3, "max_new_tokens": 256})
            assert filled["result"]["infill"]["prefix_codes_equal"]
            assert filled["result"]["infill"]["suffix_codes_equal"]
        if want("infill_tail"):
            source, _, _ = load_native_take(take_path)
            count = source.codes.shape[1]
            filled = run("infill_tail", {
                "mode": "infill", "text": "你好，今天的天气非常好。", "source_take": take_path,
                "start_frame": 2 * count // 3, "end_frame": count, "max_new_tokens": 256})
            assert filled["result"]["infill"]["prefix_codes_equal"]
        if want("audio_stream"):
            observed = []
            def observe(value):
                if value["status"] == "running" and any(
                    event["stage"] == "audio" for event in value["events"]
                ):
                    observed.append(time.time())
            streamed = run("audio_stream", {"mode": "audio_stream",
                           "text": "你好，这是流式音频测试，我们可以边生成边收听。",
                           "chunk_frames": 8, "max_new_tokens": 256}, observe=observe)
            segments = streamed["result"]["segments"]
            delivered = np.concatenate([sf.read(row["path"], dtype="int16")[0] for row in segments])
            final = sf.read(streamed["output"], dtype="int16")[0]
            assert observed, "No audio was delivered before job completion"
            assert np.array_equal(delivered, final), "Final WAV differs from delivered stream PCM"
            report["checks"]["audio_stream"]["pcm_equal"] = True
            report["checks"]["audio_stream"]["audio_while_running"] = True
            save()
        if want("stream"):
            run("stream", {"mode": "stream", "chunks": ["你好。", "我们出发吧。"],
                           "max_new_tokens": 256})
        if want("live_stream"):
            job = api("/api/jobs", {"mode": "stream", "live_stream": True,
                                   "chunks": ["你好。"], "max_new_tokens": 256})["id"]
            api(f"/api/jobs/{job}/input", {"text": "我们出发吧。"})
            api(f"/api/jobs/{job}/finish", {})
            live = wait(job)
            report["checks"]["live_stream"] = live
            save()
            assert live["status"] == "complete" and len(live["result"]["segments"]) == 2
        if want("scene"):
            run("scene", {
                "mode": "scene", "scene": "Friends meeting outdoors.",
                "characters": [
                    {"name": "小林", "description": "A calm young man.", "speaker": 0},
                    {"name": "小雨", "description": "A cheerful young woman.", "speaker": 1}],
                "lines": [{"character": "小林", "text": "你好。"},
                          {"character": "小雨", "text": "我们出发吧。"},
                          {"character": "小林", "text": "好的。"}], "max_new_tokens": 256})
        if want("cancel"):
            job = api("/api/jobs", {"mode": "stream", "live_stream": True,
                                   "chunks": ["等一下。"], "max_new_tokens": 256})["id"]
            api(f"/api/jobs/{job}/cancel", {})
            result = wait(job)
            report["checks"]["cancel"] = result
            assert result["status"] == "cancelled"
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        state.shutdown()
        report["total_seconds"] = time.time() - report["started"]
        save()
    print("ACCEPTANCE", label, report["status"], flush=True)


if __name__ == "__main__":
    main()
