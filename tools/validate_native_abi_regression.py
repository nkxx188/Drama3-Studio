"""Compare preserved and current native libraries on captured real expert inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor/fish-speech")]

import torch

from drama_low_vram.expert_cache_file import ExpertCacheFile
from drama_low_vram.native_backend import NativeExpertExecutor, NativeLibrary
from drama_low_vram.offload import ExpertStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, default=ROOT / ".cache/native-phase3/drama_native.dll")
    parser.add_argument("--output", type=Path, default=ROOT / "research/native-phase4-abi-regression.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    data = torch.load(ROOT / "research/native-expert-cases.pt", weights_only=True)
    old, new = NativeLibrary(args.old), NativeLibrary()
    source = ExpertCacheFile.open(ROOT / ".cache/drama3-fp8-experts")
    store = ExpertStore(**source.layer_buffers(data["layer"]))
    rows = []
    try:
        for arithmetic in ("reference", "sm89"):
            with NativeExpertExecutor(
                store, threads=3, arithmetic=arithmetic, library=old,
            ) as previous, NativeExpertExecutor(
                store, threads=3, arithmetic=arithmetic, library=new,
            ) as current:
                for index, case in enumerate(data["cases"]):
                    ids, weights = case["ids"].reshape(-1), case["weights"].reshape(-1).float()
                    a = previous.forward_pairs(case["x"], ids, weights)
                    b = current.forward_pairs(case["x"], ids, weights)
                    rows.append({
                        "case": index, "arithmetic": arithmetic,
                        "equal": bool(torch.equal(a, b)),
                    })
    finally:
        source.close()
    report = {
        "old_dll_sha256": hashlib.sha256(old.path.read_bytes()).hexdigest(),
        "new_dll_sha256": hashlib.sha256(new.path.read_bytes()).hexdigest(),
        "abi": 1, "optional_guard_default": False,
        "all_existing_arithmetic_unchanged": all(row["equal"] for row in rows),
        "cases": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}), flush=True)
    if not report["all_existing_arithmetic_unchanged"]:
        raise RuntimeError("Native ABI arithmetic regression")


if __name__ == "__main__":
    main()
