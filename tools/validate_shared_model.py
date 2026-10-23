"""Complete official-FP8 generation from the shared Comfy safetensors file."""

from pathlib import Path
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor/fish-speech"), str(ROOT / "tools")]
from drama_low_vram.engine import DramaEngine, EngineOptions
from benchmark_quantized_engine import SHORT, LONG
from urllib.request import urlopen


def check_ui_idle(url):
    with urlopen(url + "/api/health", timeout=3) as response:
        if json.load(response).get("engine_loaded"):
            raise RuntimeError("UI engine already loaded")


def main():
    check_ui_idle("http://127.0.0.1:8877")
    results = []
    with DramaEngine(EngineOptions(
        checkpoint=ROOT / "checkpoints/drama-3-shared", expert_storage="mmap",
        cache_experts=1, staged_experts=8, pair_rows=64, prefill_chunk=32,
        backbone_graph=True, combined_decode_bank=True, host_gather="auto",
        cpu_experts=2, native_threads=4, prefix_cache_mib=256,
    )) as engine:
        for index, text in enumerate((SHORT, LONG, LONG)):
            path = ROOT / "outputs" / f"shared-loader-{index}.wav"
            last = time.perf_counter()

            def progress(frames, limit):
                nonlocal last
                if time.perf_counter() - last >= 20:
                    print(json.dumps({"phase": "shared-generation", "index": index,
                                      "frames": frames}), flush=True)
                    last = time.perf_counter()

            result = engine.synthesize(
                text, path, seed=123, max_new_tokens=768,
                reference_audio=ROOT / "outputs/quality-official-fp8.wav",
                reference_text=SHORT, progress=progress)
            hashes = [chunk["semantic_codes_sha256"] for chunk in result["chunks"]]
            if result["status"] != "complete":
                raise RuntimeError("Shared model did not finish EOS")
            expected = "0d788577902cb135ec936c3bc1f276c0836a0b0f79713c471e819065914b4dfa"
            if index and hashes != [expected]:
                raise RuntimeError("Shared FP8 clone differs from official baseline codes")
            row = {
                "index": index, "file": str(path), "status": result["status"],
                "audio": result["audio"], "total_seconds": result["total_seconds"],
                "decode_seconds": sum(chunk["decode_seconds"] for chunk in result["chunks"]),
                "code_hashes": hashes, "wav_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "lifecycle": result["lifecycle_delta"], "memory": result["memory"],
                "offload": result["offload_delta"],
            }
            results.append(row)
            (ROOT / "research/shared-model-generation.json").write_text(
                json.dumps({"scope": "Complete shared-file compatibility. Timings overlap "
                            "model download and are not a claimed speed comparison.",
                            "results": results}, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"phase": "shared-result", **row}), flush=True)


if __name__ == "__main__":
    main()
