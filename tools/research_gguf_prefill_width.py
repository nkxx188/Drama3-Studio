"""Separate resident capacity from prefill work width on the real Q4 model."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drama_low_vram.gguf_runtime import GGUFEngine


def main():
    output = ROOT / "research/phase35-prefill-width"
    output.mkdir(parents=True, exist_ok=True)
    model = next((ROOT / "checkpoints/drama-3-gguf-q4").glob("*.gguf"))
    references = [SimpleNamespace(audio=str(ROOT / "outputs/quality-official-fp8.wav"),
                                  transcript=lambda: "<|speaker:0|>你好，这是官方 FP8 质量优先测试。")]
    text = "今天我们测试本地语音生成。希望声音清楚自然，停顿流畅，方便大家比较不同模型的速度和效果。 [audio tokens: 215]"
    summary = {"scope": "Q4 cache16 unchanged; prefill work width16 vs8; full top-8 and native cache slot mapping preserved", "variants": []}
    for width in (16, 8):
        engine = GGUFEngine(model, context=512, storage="pin", cache=16, prefill_width=width)
        row = {"width": width, "requests": []}
        try:
            for name, limit in (("warmup", 8), ("clone1", 215), ("clone2", 215)):
                result = engine.generate(text, output / f"width{width}-{name}.wav",
                                         references=references, max_frames=limit,
                                         temperature=0.7, top_p=0.7, chunk_size=1000)
                row["requests"].append(result)
                print(width, name, result["status"], result["request_seconds"], result["generation"], flush=True)
        finally:
            engine.close()
            summary["variants"].append(row)
            (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
