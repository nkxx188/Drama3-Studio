"""Compare real GGUF reference codes against the official Python codec."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor/fish-speech"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--baseline-probe", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python-device", default="cuda:0")
    args = parser.parse_args()
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    from safetensors.numpy import load_file, save_file

    from drama_low_vram.codec import load_codec
    from drama_low_vram.gguf_archive import Archive
    from drama_low_vram.gguf_runtime import reference_wav

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    prepared = []
    for index, reference in enumerate(args.reference):
        wav_path = reference_wav(reference, ROOT)
        samples, rate = sf.read(wav_path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(samples.mean(axis=1))[None]
        if rate != 44100:
            waveform = torchaudio.functional.resample(waveform, rate, 44100)
        case = output / f"reference-{index + 1}"
        case.mkdir(exist_ok=True)
        # Both implementations consume these exact float32 mono samples.
        sf.write(case / "input.wav", waveform[0].numpy(), 44100, subtype="FLOAT")
        frames = (waveform.shape[-1] + 2047) // 2048
        prepared.append((case, waveform, frames, str(reference.resolve())))

    codec = load_codec(args.codec, device=args.python_device, dtype=torch.float32,
                       max_frames=max(row[2] for row in prepared))
    archive = Archive(args.model)
    # Isolate implementation parity from checkpoint conversion/weight precision.
    with args.model.open("rb") as file, torch.no_grad():
        for name, parameter in codec.named_parameters():
            tensor = archive.by_name["codec_weights/" + name]
            if tensor.kind != 1 or parameter.numel() != int(np.prod(tensor.shape)):
                raise ValueError(f"Codec weight mismatch: {name}: {parameter.shape} vs {tensor.shape}/{tensor.kind}")
            file.seek(tensor.offset)
            values = np.frombuffer(file.read(tensor.nbytes), dtype="<f2").astype(np.float32)
            parameter.copy_(torch.from_numpy(values.reshape(tuple(parameter.shape))))
    for case, waveform, _, _ in prepared:
        started = time.perf_counter()
        with torch.inference_mode():
            audio = waveform[None].to(args.python_device)
            lengths = torch.tensor([waveform.shape[-1]], device=args.python_device)
            indices, valid = codec.encode(audio, lengths)
            codes = indices[0, :, :int(valid[0])].cpu().numpy().astype(np.int32)
        save_file({"codes": codes}, str(case / "python-f32.safetensors"))
        print(json.dumps({"case": case.name, "python_seconds": time.perf_counter() - started,
                          "shape": list(codes.shape)}), flush=True)
    del codec, audio, indices
    gc.collect()
    torch.cuda.empty_cache()

    env = os.environ.copy()
    env["PATH"] = str(args.probe.resolve().parent) + os.pathsep + env.get("PATH", "")
    results = []
    for case, _, _, reference in prepared:
        expected = load_file(str(case / "python-f32.safetensors"))["codes"]
        variants = [("native", args.probe, "--encode"), ("native-f32", args.probe, "--encode-f32")]
        if args.baseline_probe:
            variants.append(("without-cpu-layout-fix", args.baseline_probe, "--encode"))
        comparisons = {}
        for label, probe, operation in variants:
            started = time.perf_counter()
            encoded = case / (label + ".safetensors")
            with (case / (label + ".log")).open("w", encoding="utf-8") as log:
                subprocess.run([str(probe.resolve()), operation, str(args.model.resolve()),
                                str(case / "input.wav"), str(encoded), "cpu", "8"],
                               env=env, cwd=ROOT / ".cache/audio-cpp-phase19",
                               stdout=log, stderr=subprocess.STDOUT, timeout=180, check=True,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            actual = load_file(str(encoded))["codes"]
            if actual.shape != expected.shape:
                raise AssertionError(f"Reference shape differs: {actual.shape} vs {expected.shape}")
            agreement = (actual == expected).mean(axis=1)
            comparisons[label] = {"seconds": time.perf_counter() - started,
                                  "agreement": float(agreement.mean()),
                                  "per_codebook": agreement.tolist()}
        result = {"reference": reference, "shape": list(expected.shape),
                  "comparisons": comparisons}
        # FP32 isolates layout correctness. Native FP16 kernels introduce extra
        # activation rounding, amplified by successive residual quantizers.
        parity = comparisons["native-f32"]
        result["passed"] = (parity["agreement"] >= 0.95
                            and min(parity["per_codebook"]) >= 0.90
                            and comparisons["native"]["per_codebook"][0] >= 0.90)
        if args.baseline_probe:
            result["passed"] &= comparisons["without-cpu-layout-fix"]["agreement"] < 0.01
        results.append(result)
        print(json.dumps(result), flush=True)
    report = {"scope": "CPU encoding vs official Python FP32 graph, identical GGUF FP16 weights and waveform",
              "model": str(args.model.resolve()), "probe": str(args.probe.resolve()),
              "baseline_probe": str(args.baseline_probe.resolve()) if args.baseline_probe else None,
              "passed": all(row["passed"] for row in results), "results": results}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise AssertionError("Native reference codes differ materially from official Python FP32")


if __name__ == "__main__":
    main()
