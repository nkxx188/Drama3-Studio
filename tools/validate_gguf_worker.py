"""Bounded real Q4/Q8 persistent-session and service acceptance."""

import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor/fish-speech"), str(ROOT)]
from drama_low_vram.gguf_runtime import GGUFEngine, GenerationCancelled, NativeFailure


def main():
    destination = ROOT / "research/phase35-worker-acceptance"
    destination.mkdir(parents=True, exist_ok=True)
    summary = {"models": [], "scope": "Local automatic resources, serial actual requests; no cross-device performance claim"}
    for variant in ("q4", "q8"):
        model = next((ROOT / f"checkpoints/drama-3-gguf-{variant}").glob("*.gguf"))
        engine = GGUFEngine(model)
        references = [SimpleNamespace(
            audio=str(ROOT / "outputs/quality-official-fp8.wav"),
            transcript=lambda: "<|speaker:0|>你好，这是官方 FP8 质量优先测试。")]
        row = {"variant": variant, "requests": []}
        started = time.perf_counter()
        try:
            engine.load()
            pid = engine._process.pid
            row.update(load_wall_seconds=time.perf_counter() - started, pid=pid,
                       resource_plan=engine.plan.report())
            texts = (
                ("tts", "你好，这是一次本地语音测试。", (), 192),
                ("clone1", "今天我们测试本地语音生成。希望声音清楚自然，停顿流畅，方便大家比较不同模型的速度和效果。 [audio tokens: 215]", references, 215),
                ("clone2", "今天我们测试本地语音生成。希望声音清楚自然，停顿流畅，方便大家比较不同模型的速度和效果。 [audio tokens: 215]", references, 215),
            )
            for name, text, refs, limit in texts:
                result = engine.generate(
                    text, destination / f"{variant}-{name}.wav", references=refs,
                    max_frames=limit, temperature=0.7, top_p=0.7, chunk_size=1000)
                assert engine._process.pid == pid
                row["requests"].append({"name": name, **result})
                print(variant, name, result["status"], result["request_seconds"],
                      result["generation"], flush=True)
            # Validate error isolation through the actual native protocol.
            engine._process.stdin.write(json.dumps({
                "id": "invalid", "output": str(destination / "invalid.wav"),
                "request": {"text": "", "max_tokens": 16},
            }) + "\n")
            engine._process.stdin.flush()
            try:
                engine._receive()
                raise AssertionError("Native empty text was accepted")
            except NativeFailure as error:
                assert error.recoverable and engine.loaded
                row["recoverable_error_survived"] = True
            started = time.monotonic()
            try:
                engine.generate("测试取消", destination / f"{variant}-cancel.wav", max_frames=64,
                                cancelled=lambda: time.monotonic() - started > 0.25)
                raise AssertionError("Expected cancellation")
            except GenerationCancelled:
                assert not engine.loaded
                row["cancel_reaped"] = True
            result = engine.generate("你好。", destination / f"{variant}-restart.wav", max_frames=64)
            assert engine._process.pid != pid
            row["restart"] = result
        finally:
            engine.close()
            summary["models"].append(row)
            (destination / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
