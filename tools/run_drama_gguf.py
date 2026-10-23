"""Experimental native Windows GGUF entry for verified local Q4/Q8 packages."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid
import sys

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
GIB = 1024**3


def basic_settings(model_bytes, available_ram, physical_cores, *, storage="auto", threads=None):
    """Conservative defaults for the verified Q4 backend, without timing sweeps."""
    pin_budget = model_bytes + 6 * GIB
    return {
        "threads": threads if threads is not None else max(1, min(8, physical_cores or 1)),
        "host_storage": (
            "pin" if storage == "pin" or (storage == "auto" and available_ram >= pin_budget)
            else "mapped"
        ),
        "estimated_pin_budget_bytes": pin_budget,
    }


def allocation_failure(log):
    text = log.lower()
    return any(marker in text for marker in (
        "out of memory", "out-of-memory", "bad_alloc", "cudamemoryallocation",
        "failed to allocate", "allocation failed", "cudahostalloc failed",
        "cannot allocate pinned host expert banks",
        "cannot allocate pinned expert upload staging buffer",
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--runtime", choices=("auto", "worker", "legacy"), default="auto",
                        help="Automatic resource plan/native worker when built; legacy retains experimental knobs")
    parser.add_argument("--pipeline-width", type=int, choices=(0, 2, 4), default=0,
                        help="Optional Q8 GPU-only known-route transfer pipeline; default off")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/gguf.wav")
    parser.add_argument("--prompt-audio", type=Path)
    parser.add_argument("--prompt-text")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-frames", type=int, default=768)
    parser.add_argument("--threads", type=int, help="Default: up to eight physical CPU cores")
    parser.add_argument("--host-storage", choices=("auto", "pin", "mapped"), default="auto")
    parser.add_argument("--variant", choices=("q4", "q8"), default="q4")
    parser.add_argument("--model", type=Path,
                        help="Use a compatible local Drama3 GGUF file with the automatic worker policy")
    parser.add_argument("--device", type=int, help="CUDA device index; automatic free-VRAM selection when omitted")
    parser.add_argument("--codec", choices=("same", "cpu"), default="same",
                        help="CPU codec can leave more VRAM for Q8 generation")
    parser.add_argument("--cache-experts", type=int, default=8,
                        help="Resident expert slots per layer (8..256); intermediate sizes are allowed")
    parser.add_argument("--host-embedding", action="store_true",
                        help="Experimental Q8 mapped BF16 lookup on CPU to free about 986MiB VRAM")
    parser.add_argument("--codec-decode", choices=("full", "stream"), default="full",
                        help="Optional bounded causal decoding of the final waveform")
    parser.add_argument("--decoder", choices=("same", "cpu", "cuda"), default="same",
                        help="Waveform decoder: same uses automatic placement in worker mode")
    parser.add_argument("--decode-cpu-experts", choices=("off", "auto", "0", "1", "2", "3", "4", "5", "6", "7", "8"),
                        default="off", help="Experimental Q8 cooperation; changes arithmetic/codes; auto measures CPU budgets 0/2/4")
    parser.add_argument("--decode-profile", type=Path, default=ROOT / ".cache/gguf-hybrid-profiles",
                        help="Directory for model/hardware-specific cooperative calibration results")
    args = parser.parse_args()
    if args.device is not None and args.device < 0:
        parser.error("CUDA device index must be nonnegative")
    if not args.text.strip() or not 0 <= args.seed <= 2**32 - 1:
        parser.error("Provide nonempty text and a 32-bit unsigned seed")
    if not 1 <= args.max_frames <= 768 or (args.threads is not None and not 1 <= args.threads <= 64):
        parser.error("max-frames must be 1..768 and threads 1..64")
    if not 8 <= args.cache_experts <= 256:
        parser.error("cache-experts must be 8..256 (all top-8 experts still participate)")
    if args.pipeline_width and ((args.model is None and args.variant != "q8") or args.runtime == "legacy"):
        parser.error("The experimental GPU pipeline requires Q8 and the worker runtime")
    if args.host_embedding and args.variant != "q8":
        parser.error("host-embedding is currently validated only for --variant q8")
    codec_probe = args.codec_decode != "full" or args.decoder != "same"
    hybrid = args.decode_cpu_experts != "off"
    if hybrid and args.variant != "q8":
        parser.error("Cooperative decoding is currently supported only for --variant q8")
    if bool(args.prompt_audio) != bool(args.prompt_text):
        parser.error("Voice cloning requires both --prompt-audio and --prompt-text")
    build_file = ROOT / "research" / (
        "phase33-hybrid-build.json" if hybrid
        else "phase30-codec-build.json" if codec_probe
        else "phase29-host-embedding-build.json" if args.host_embedding
        else "phase19-audio-cpp-build.json")
    if args.model:
        from drama_low_vram.gguf_resources import resolve_gguf
        model = resolve_gguf(args.model)
    else:
        manifest = ROOT / "research" / (
            "phase28-q8-model.json" if args.variant == "q8"
            else "phase19-gguf-q4-quality-download-pinned.json")
        if manifest.is_file():
            pin = json.loads(manifest.read_text(encoding="utf-8"))
            model = Path(pin["destination"]) / pin["files"][0]["name"]
            if not model.is_file() or model.stat().st_size != pin["files"][0]["size"]:
                parser.error("The selected verified GGUF model file is missing or changed")
        else:
            from drama_low_vram.gguf_resources import resolve_gguf
            model = resolve_gguf(ROOT / f"checkpoints/drama-3-gguf-{args.variant}")
    manual = (hybrid or args.host_embedding or args.codec != "same"
              or args.codec_decode != "full" or args.cache_experts != 8 or args.threads is not None)
    worker = args.runtime == "worker" or (
        args.runtime == "auto" and not manual and (
            (ROOT / "research/phase35-worker-build.json").is_file() or
            (ROOT / "runtime/native/manifest.json").is_file()))
    if args.model and not worker:
        parser.error("--model uses the worker runtime; build it and use automatic resource options")
    if args.pipeline_width and not worker:
        parser.error("Build the persistent worker before using the optional GPU pipeline")
    if worker:
        if manual:
            parser.error("Worker uses the automatic baseline; use --runtime legacy for experimental knobs")
        from drama_low_vram.gguf_runtime import GGUFEngine
        from types import SimpleNamespace
        references = []
        if args.prompt_audio:
            references.append(SimpleNamespace(audio=str(args.prompt_audio), transcript=lambda: args.prompt_text))
        engine = GGUFEngine(model, root=ROOT, storage=args.host_storage, pipeline_width=args.pipeline_width,
                            decoder="auto" if args.decoder == "same" else args.decoder, device=args.device)
        started = time.perf_counter()
        try:
            report = engine.generate(args.text, args.output, references=references,
                                     seed=args.seed, max_frames=args.max_frames)
            report["seconds_including_cold_load"] = time.perf_counter() - started
            args.output.with_suffix(".json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
            if not report["complete"]:
                raise RuntimeError("Generation reached its frame limit; partial audio is preserved")
        finally:
            engine.close()
        return
    if not build_file.is_file():
        parser.error("Build the native comparison backend with tools/build_audio_cpp_comparison.py")
    build = json.loads(build_file.read_text(encoding="utf-8"))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = ROOT / ".cache/tmp" / ("gguf-request-" + uuid.uuid4().hex)
    temporary.mkdir(parents=True)
    request = {"id": "speech", "text": args.text, "seed": args.seed,
               "max_tokens": args.max_frames, "temperature": 1.0, "top_k": 0, "top_p": 0.9,
               "options": {"text_chunk_size": "1000"}}
    if args.prompt_audio:
        audio = args.prompt_audio.resolve()
        if not audio.is_file():
            parser.error("Reference audio does not exist")
        request.update(voice_ref=str(audio))
        request["options"]["reference_text"] = args.prompt_text
    sequence = temporary / "request.json"
    sequence.write_text(json.dumps([request], ensure_ascii=False) + "\n", encoding="utf-8")
    settings = basic_settings(
        model.stat().st_size, psutil.virtual_memory().available,
        psutil.cpu_count(logical=False), storage=args.host_storage, threads=args.threads)
    pinned = settings["host_storage"] == "pin"
    command = [
        build["executable"], "--task", "tts", "--family", "fish_drama3",
        "--backend", "cuda", "--mode", "offline", "--model", str(model),
        "--threads", str(settings["threads"]), "--session-option", "fish_drama3.disk_experts=true",
        "--session-option", f"fish_drama3.expert_cache_size={args.cache_experts}",
        "--session-option", "fish_drama3.cpu_expert_layers=0",
        "--session-option", "fish_drama3.codec_backend=" + args.codec,
        "--session-option", "fish_drama3.context_length=1024",
        "--request-sequence", str(sequence), "--out-dir", str(temporary),
        "--metrics", "--log",
    ]
    if args.device is not None:
        command += ["--device", str(args.device)]
    if codec_probe:
        command += [
            "--session-option", f"fish_drama3.codec_decode_strategy={args.codec_decode}",
            "--session-option", f"fish_drama3.codec_decoder_backend={args.decoder}",
        ]
    if hybrid:
        command += [
            "--session-option", f"fish_drama3.decode_cpu_experts={args.decode_cpu_experts}",
            "--session-option", "fish_drama3.decode_profile=" + str(args.decode_profile.resolve()),
        ]
    environment = dict(os.environ)
    environment.update(DRAMA_COMPACT_GGUF_HEAD="1",
                       TEMP=str(ROOT / ".cache/tmp"), TMP=str(ROOT / ".cache/tmp"))
    environment.pop("DRAMA_HOST_GGUF_EMBEDDING", None)
    for name in ("DRAMA_Q8_ROUTE_TRACE", "DRAMA_Q8_ROUTE_TRACE_REQUEST", "DRAMA_Q8_STEP_RECORD",
                 "DRAMA_Q8_STEP_REPLAY", "DRAMA_Q8_RAW_CPU", "DRAMA_Q8_COMPATIBLE_QUANT"):
        environment.pop(name, None)
    if args.host_embedding:
        environment["DRAMA_HOST_GGUF_EMBEDDING"] = "1"
    environment["PATH"] = str(Path(build["cuda_toolkit"]) / "bin") + os.pathsep + environment["PATH"]
    started = time.perf_counter()
    attempts, logs = [], []
    for attempt in range(2):
        attempt_dir = temporary / f"attempt-{attempt}"
        attempt_dir.mkdir()
        current = list(command)
        current[current.index("--out-dir") + 1] = str(attempt_dir)
        if pinned:
            current += ["--session-option", "fish_drama3.pin_experts=true"]
        result = subprocess.run(current, cwd=build["source"], env=environment,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log = result.stdout.decode("utf-8", errors="replace")
        attempts.append({
            "host_storage": "pin" if pinned else "mapped", "returncode": result.returncode,
        })
        logs.append(f"Attempt {attempt + 1}, {attempts[-1]['host_storage']}\n{log}")
        if not result.returncode:
            break
        if not (attempt == 0 and pinned and args.host_storage == "auto" and allocation_failure(log)):
            break
        # The failed child has exited: its allocations and CUDA state are gone.
        pinned = False
    output.with_suffix(".log").write_text("\n\n".join(logs), encoding="utf-8")
    report = {
        "output": str(output), "seconds_including_cold_load": time.perf_counter() - started,
        "host_storage": "pin" if pinned else "mapped", "compact_bf16_head": True,
        "model": str(model), "threads": settings["threads"], "attempts": attempts,
        "variant": args.variant,
        "codec_backend": args.codec,
        "cache_experts_per_layer": args.cache_experts,
        "host_embedding": args.host_embedding,
        "codec_decode_strategy": args.codec_decode,
        "codec_decoder_backend": args.decoder,
        "decode_cpu_experts": args.decode_cpu_experts,
        "experimental_cooperative_decoder": hybrid,
        "decode_profile_directory": str(args.decode_profile.resolve()) if hybrid else None,
        "estimated_pin_budget_bytes": settings["estimated_pin_budget_bytes"],
        "backend": "experimental audio.cpp CUDA, with local candidate-row storage optimization",
    }
    if result.returncode:
        report.update(complete=False, status="error", output=None)
        output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"Native generation failed; see {output.with_suffix('.log')}")
    generated = attempt_dir / "speech.wav"
    if not generated.is_file():
        raise RuntimeError("Native backend did not return a WAV")
    shutil.copyfile(generated, output)
    report.update(complete="max_tokens reached" not in log)
    report["status"] = "complete" if report["complete"] else "incomplete"
    output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["complete"]:
        raise RuntimeError("Generation reached its frame limit; partial audio is preserved")


if __name__ == "__main__":
    main()
