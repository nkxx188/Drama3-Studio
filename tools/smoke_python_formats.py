"""Short multi-format model execution, optionally forcing the SM75 policy.

Forced policy runs on the current GPU; it verifies the FP16/Torch fallback
implementation and does not claim physical RTX20 validation or audio quality.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--force-turing-policy", action="store_true")
    parser.add_argument("--formats", default="fp8,int4,int8,w6a8,w4a8")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path[:0] = [str(root / "src"), str(root / "vendor/fish-speech"), str(root)]
    import torch
    import drama_low_vram.engine as module
    from drama_low_vram.engine import DramaEngine, EngineOptions
    if args.force_turing_policy:
        inspect = module.inspect_hardware
        module.inspect_hardware = lambda *a, **kw: replace(
            inspect(*a, **kw), compute_capability=(7, 5), triton_available=False)
        import drama_low_vram.quantized_kernels as kernels
        kernels.uses_quantized_torch_fallback = lambda *_: True
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "actual_gpu": torch.cuda.get_device_name(),
        "forced_turing_policy": args.force_turing_policy,
        "scope": "32-frame partial TTS exercises loading, full top-k, backbone/tower/codec; no speed/quality claim",
        "python": sys.executable, "formats": [],
    }
    folders = {"fp8": "drama-3", "int4": "drama-3-int4", "int8": "drama-3-int8",
               "w6a8": "drama-3-w6a8", "w4a8": "drama-3-w4a8"}
    for name in args.formats.split(","):
        engine = DramaEngine(EngineOptions(
            checkpoint=args.models_root / folders[name], expert_storage="auto",
            tower_graph=False, backbone_graph=False, compute_policy="manual", cache_experts=0,
            host_gather="torch", staged_experts=2, pair_rows=16, prefill_chunk=4))
        started = time.perf_counter()
        try:
            result = engine.synthesize("你好。", args.output / f"{name}.wav",
                                       seed=123, max_new_tokens=32, allow_partial=True)
            assert result["output"] and Path(result["output"]).is_file(), result
            report["formats"].append({"name": name, "wall_seconds": time.perf_counter() - started,
                                       "result": result})
            print(name, result["status"], "executed", round(time.perf_counter() - started, 2), flush=True)
        finally:
            engine.close()
            torch.cuda.empty_cache()
            (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                                      encoding="utf-8")
    report["status"] = "passed"
    (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")


if __name__ == "__main__":
    main()
