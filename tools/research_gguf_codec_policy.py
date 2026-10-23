"""Bounded Q4 codec/lookup acceptance and quantization-independent capability audit."""

from dataclasses import replace
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import time

import numpy as np
import psutil
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drama_low_vram.gguf_resources import Layout, plan_resources
from drama_low_vram.gguf_runtime import GGUFEngine, native_memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="Reuse completed per-variant evidence from this build")
    args = parser.parse_args()
    output = ROOT / "research/phase36-q4-codec"
    output.mkdir(parents=True, exist_ok=True)
    build = json.loads((ROOT / "research/phase35-worker-build.json").read_text())
    model = next((ROOT / "checkpoints/drama-3-gguf-q4").glob("*.gguf"))
    layout = Layout.read(model)
    summary = {
        "scope": "Local Q4, complete top-8, CPU reference encoding, identical context1024/cache8/"
                 "pin/seed123/sampling; short warmup then clones capped at215 frames. Compare lookup and final "
                 "codec separately. Automatic resource request is a functionality check, not fixed-work timing.",
        "native_build_sha256": build["sha256"], "layout": layout.__dict__, "variants": [],
    }
    if args.resume and (output / "summary.json").is_file():
        previous = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        if previous.get("native_build_sha256") != build["sha256"]:
            raise ValueError("Cannot reuse evidence from a different native build")
        summary["variants"] = previous["variants"]
    reference = [SimpleNamespace(audio=str(ROOT / "outputs/quality-official-fp8.wav"),
                                transcript=lambda: "<|speaker:0|>你好，这是官方 FP8 质量优先测试。")]
    text = "今天我们测试本地语音生成。希望声音清楚自然，停顿流畅，方便大家比较不同模型的速度和效果。 [audio tokens: 215]"

    def save():
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")

    for name, lookup, decoder in (
        ("cpu-original-lookup", False, "cpu"),
        ("cpu-host-lookup", True, "cpu"),
        ("gpu-host-lookup", True, "cuda"),
        ("cpu-control-after", True, "cpu"),
    ):
        if args.resume and any(row["name"] == name and len(row["requests"]) == 3
                               for row in summary["variants"]):
            continue
        memory = native_memory(build)
        selected = replace(layout, host_embedding_supported=lookup)
        plan = plan_resources(selected, free_vram=memory["free_bytes"],
                              available_ram=psutil.virtual_memory().available,
                              cores=psutil.cpu_count(logical=False), context=1024,
                              storage="pin", cache=8, decoder=decoder)
        engine = GGUFEngine(model, plan=plan, decoder=decoder)
        row = {"name": name, "requests": []}
        started = time.perf_counter()
        try:
            engine.load()
            row["load_wall_seconds"] = time.perf_counter() - started
            pid = engine._process.pid
            for request, frames in (("warmup", 8), ("clone1", 215), ("clone2", 215)):
                result = engine.generate(text, output / f"{name}-{request}.wav",
                                         references=reference, max_frames=frames,
                                         temperature=0.7, top_p=0.7, chunk_size=1000)
                assert engine._process.pid == pid
                row["requests"].append(result)
                print(name, request, result["status"], round(result["request_seconds"], 3),
                      result["generation"], flush=True)
        finally:
            engine.close()
            summary["variants"].append(row)
            save()

    control = json.loads((output / "cpu-original-lookup-clone2.take.json").read_text(encoding="utf-8"))
    for row in summary["variants"]:
        if row["name"] == "automatic-resource-policy":
            continue
        codes = json.loads(Path(row["requests"][-1]["native_take_file"]).read_text(encoding="utf-8"))
        row["same_codes_as_original_lookup"] = codes["vq_codes"] == control["vq_codes"]
        assert row["same_codes_as_original_lookup"]
        # The 215-frame cap fixes work when EOS is not emitted; truncated
        # benchmark audio is not counted as ordinary-generation acceptance.
    cpu, cpu_rate = sf.read(output / "cpu-host-lookup-clone2.wav", dtype="float64")
    gpu, gpu_rate = sf.read(output / "gpu-host-lookup-clone2.wav", dtype="float64")
    assert cpu_rate == gpu_rate and cpu.shape == gpu.shape
    noise = np.sum((cpu - gpu) ** 2)
    signal = np.sum(cpu**2)
    summary["waveform_comparison"] = {
        "sample_rate": cpu_rate,
        "relative_l2": float(np.sqrt(noise / max(signal, 1e-30))),
        "snr_db": float(10 * np.log10(signal / max(noise, 1e-30))),
        "max_abs": float(np.max(np.abs(cpu - gpu))),
        "scope": "CPU full vs GPU bounded PCM16; numerical comparison is not a listening verdict",
    }
    assert np.isfinite(gpu).all() and signal > 0 and summary["waveform_comparison"]["snr_db"] > 30

    engine = GGUFEngine(model)
    row = {"name": "automatic-resource-policy", "requests": []}
    try:
        result = engine.generate("你好，这是一次本地语音测试。", output / "auto-tts.wav",
                                 max_frames=192, temperature=0.7, top_p=0.7)
        row["requests"].append(result)
        assert result["complete"] and result["resource_plan"]["decoder_backend"] == "cuda"
        assert result["resource_plan"]["host_embedding"]
        result = engine.generate(
            "今天我们测试本地语音生成，希望声音清楚自然，停顿流畅。", output / "auto-clone.wav",
            references=reference, max_frames=384, temperature=0.7, top_p=0.7, chunk_size=1000)
        row["requests"].append(result)
        assert result["complete"] and result["session_reused"]
    finally:
        engine.close()
        summary["variants"].append(row)
        save()
    for variant in ("q4", "q8"):
        path = next((ROOT / f"checkpoints/drama-3-gguf-{variant}").glob("*.gguf"))
        current = Layout.read(path)
        summary.setdefault("compatible_codec_layouts", {})[variant] = {
            "supported": current.codec_cuda_supported, "signature": current.codec_layout_signature,
        }
    assert len({value["signature"] for value in summary["compatible_codec_layouts"].values()}) == 1
    summary.update(status="passed", complete=True)
    save()
    print(json.dumps(summary["waveform_comparison"], indent=2), flush=True)


if __name__ == "__main__":
    main()
