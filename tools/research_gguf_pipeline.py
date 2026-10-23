"""Bounded full-request GPU-only expert pipeline comparison."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drama_low_vram.gguf_runtime import GGUFEngine


def main():
    output = ROOT / "research/phase35-full-pipeline"
    output.mkdir(parents=True, exist_ok=True)
    model = next((ROOT / "checkpoints/drama-3-gguf-q8").glob("*.gguf"))
    references = [SimpleNamespace(audio=str(ROOT / "outputs/quality-official-fp8.wav"),
                                  transcript=lambda: "<|speaker:0|>你好，这是官方 FP8 质量优先测试。")]
    text = "今天我们测试本地语音生成。希望声音清楚自然，停顿流畅，方便大家比较不同模型的速度和效果。 [audio tokens: 215]"
    summary = {"scope": "Q8 top-8 GPU compute, context512/cache8/mapped source; serial processes; 8-frame warmup then two natural-EOS clones capped at215. No CPU expert math.", "variants": []}
    for name, width in (("control-before", 0), ("pipeline2", 2), ("pipeline4", 4), ("control-after", 0)):
        engine = GGUFEngine(model, context=512, storage="mapped", cache=8,
                            prefill_width=8, pipeline_width=width)
        row = {"name": name, "requests": []}
        try:
            for request, frames in (("warmup", 8), ("clone1", 215), ("clone2", 215)):
                result = engine.generate(text, output / f"{name}-{request}.wav", references=references,
                                         max_frames=frames, temperature=0.7, top_p=0.7, chunk_size=1000)
                row["requests"].append(result)
                print(name, request, result["status"], result["request_seconds"], result["generation"], flush=True)
        finally:
            engine.close()
            summary["variants"].append(row)
            (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
