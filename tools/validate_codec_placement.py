"""Official codec branch placement: exact encode/decode, VRAM and graph reuse."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from benchmark_engine_decode import DramaEngine, EngineOptions, torch
from drama_low_vram.codec import (
    codec_operation_modules, decode_to_audio, encode_audio,
)
from validate_state_prefix import byte_equal


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-model", action="store_true")
    parser.add_argument("--frames", type=int, nargs="+", default=[16, 375, 650])
    parser.add_argument("--cache-experts", type=int, default=1)
    parser.add_argument("--operations", nargs="+", choices=("encode", "decode"), default=["encode", "decode"])
    parser.add_argument("--output", type=Path, default=ROOT / "research/phase7-codec-placement.json")
    args = parser.parse_args()
    if min(args.frames) < 1:
        parser.error("Positive frame counts required")
    torch.set_num_threads(1)
    report = {
        "scope": "Official codec full/operation-only placement equality and memory. Repeated reference codes are diagnostic inputs, not generated speech or generation speed.",
        "with_model": args.with_model, "operations": [], "all_bytes_equal": True,
    }
    artifacts = {}

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    with DramaEngine(EngineOptions(
        checkpoint=ROOT / "checkpoints/drama-3", expert_storage="mmap",
        cache_experts=args.cache_experts, staged_experts=8, pair_rows=64,
        prefill_chunk=32, backbone_graph=True, combined_decode_bank=True,
        cpu_experts=2, host_gather="auto", prefix_cache_mib=256,
    )) as engine:
        audio = ROOT / "outputs/quality-official-fp8.wav"
        transcript = "<|speaker:0|>你好，这是官方 FP8 质量优先测试。"
        codes, _ = engine.prepare_reference(audio, transcript)
        # Build the largest nonpersistent buffers once, before comparisons.
        with engine._codec_on_gpu(max(args.frames), operation="decode"):
            pass
        if args.with_model:
            engine.load()
            engine._prepare(947)
        cpu_pointers = {
            (id(module), kind, name): tensor.data_ptr()
            for module in engine.codec.modules()
            for kind, values in (("parameter", module._parameters), ("buffer", module._buffers))
            for name, tensor in values.items() if tensor is not None
        }
        report["module_bytes"] = {
            operation: sum(
                tensor.numel() * tensor.element_size()
                for module in codec_operation_modules(engine.codec, operation)
                for tensor in (*module._parameters.values(), *module._buffers.values())
                if tensor is not None
            ) for operation in ("all", "encode", "decode")
        }
        cases = ([("encode", codes.shape[1])] if "encode" in args.operations else [])
        cases += [("decode", count) for count in args.frames] if "decode" in args.operations else []
        for direction, frames in cases:
            inputs = codes.repeat(1, (frames + codes.shape[1] - 1) // codes.shape[1])[:, :frames]
            expected = None
            for placement in ("full", "auto"):
                engine.options = replace(engine.options, codec_placement=placement)
                if args.with_model and engine._parked:
                    engine._prepare(947)
                graphs = engine.graph, engine.backbone_graph
                before = dict(engine._lifecycle)
                torch.cuda.synchronize(engine.device)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(engine.device)
                allocated = torch.cuda.memory_allocated(engine.device)
                free, total = torch.cuda.mem_get_info(engine.device)
                reserved = torch.cuda.memory_reserved(engine.device)
                started = time.perf_counter()

                def operation(codec):
                    if direction == "encode":
                        return encode_audio(audio, codec, engine.device)
                    return decode_to_audio(inputs.clone(), codec, engine.device)

                output = (
                    engine._decode_audio_segments([inputs.clone()])
                    if direction == "decode" and placement == "auto"
                    else engine._run_codec(frames, operation, direction=direction)
                )
                elapsed = time.perf_counter() - started
                if expected is None:
                    expected = output
                equal = byte_equal(output, expected)
                row = {
                    "operation": direction, "frames": frames, "placement": placement,
                    "bytes_equal": equal, "seconds": elapsed,
                    "output_sha256": hashlib.sha256(
                        output.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
                    "baseline_allocated_bytes": allocated,
                    "baseline_reserved_bytes": reserved,
                    "driver_free_bytes": free,
                    "driver_total_bytes": total,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(engine.device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(engine.device),
                    "lifecycle_delta": {
                        key: engine._lifecycle[key] - before[key] for key in before
                    },
                    "graphs_preserved": (
                        not args.with_model or
                        (engine.graph is graphs[0] and engine.backbone_graph is graphs[1])
                    ),
                    "cpu_storage_preserved": all(
                        tensor.device.type == "cpu"
                        and tensor.data_ptr() == cpu_pointers[(id(module), kind, name)]
                        for module in engine.codec.modules()
                        for kind, values in (("parameter", module._parameters), ("buffer", module._buffers))
                        for name, tensor in values.items() if tensor is not None
                    ),
                }
                report["operations"].append(row)
                report["all_bytes_equal"] &= equal and row["cpu_storage_preserved"]
                artifacts[f"{direction}_{frames}_{placement}"] = output
                save()
                print(json.dumps(row), flush=True)
                if not report["all_bytes_equal"]:
                    torch.save(artifacts, args.output.with_suffix(".pt"))
                    raise RuntimeError("Codec branch placement changed bytes/storage")
        torch.save(artifacts, args.output.with_suffix(".pt"))
        if args.with_model:
            before = dict(engine._lifecycle)
            engine._prepare(947)
            report["last_prepare_breakdown"] = dict(engine._last_prepare_timing)
            report["final_prepare_delta"] = {
                key: engine._lifecycle[key] - before[key] for key in before
            }
        save()


if __name__ == "__main__":
    main()
