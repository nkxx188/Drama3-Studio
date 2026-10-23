"""Bounded local Q8 route capture and CPU/GPU expert replay.

Captured inference is diagnostic and is never a speed benchmark. Production
CLI identity is checked and restored after building the separate trace binary.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time

from build_native import windows_tools

ROOT = Path(__file__).resolve().parents[1]
RECORD = struct.Struct("<5I8i8f2048f")


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment(build):
    env, _, _, _ = windows_tools()
    env = {key.upper(): value for key, value in env.items()}
    env["TEMP"] = env["TMP"] = str(ROOT / ".cache/tmp")
    env["PATH"] = str(Path(build["cuda_toolkit"]) / "bin") + os.pathsep + env["PATH"]
    env.pop("DRAMA_Q8_ROUTE_TRACE", None)
    env.pop("DRAMA_Q8_ROUTE_TRACE_REQUEST", None)
    return env


def capture(args, build, out):
    source = Path(build["source"])
    original = Path(build["executable"]).with_name("audiocpp_cli.exe")
    saved = ROOT / ".cache/phase32-before-route-trace/audiocpp_cli.exe"
    if sha(original) != sha(saved):
        raise RuntimeError("Production CLI identity changed; refusing overwrite")
    env = environment(build)
    exe = original.with_name("audiocpp_cli-route-trace.exe")
    try:
        with (out / "build.log").open("wb") as log:
            subprocess.run([
                build["cmake"], "--build", str(original.parent.parent),
                "--target", "audiocpp_cli", "--parallel", "2",
            ], env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180, check=True)
        shutil.copy2(original, exe)
    finally:
        shutil.copy2(saved, original)
    if sha(original) != sha(saved):
        raise RuntimeError("Production CLI restore mismatch")
    relative = "src/framework/core/disk_expert_weight_store.cpp"
    before = ROOT / ".cache/phase32-before-route-trace/disk_expert_weight_store.cpp"
    patch = "".join(difflib.unified_diff(
        before.read_text(encoding="utf-8").splitlines(True),
        (source / relative).read_text(encoding="utf-8").splitlines(True),
        fromfile="a/" + relative, tofile="b/" + relative,
    ))
    (out / "route-capture.patch").write_text(patch, encoding="utf-8")
    save(out / "build.json", {
        **build, "executable": str(exe), "sha256": sha(exe),
        "scope": "Read-only live Q8 tensors and routes; captured timings are diagnostic only.",
        "patch": str(out / "route-capture.patch"),
        "patch_sha256": sha(out / "route-capture.patch"),
        "production_cli_restored_sha256": sha(original),
    })
    base = ROOT / "research/phase30-q8-final-stream/01-cache8"
    requests = json.loads((base / "requests.json").read_text(encoding="utf-8"))[:args.requests]
    save(out / "requests.json", requests)
    command = json.loads((base / "summary.json").read_text())["command"]
    command[0] = str(exe)
    command[command.index("--request-sequence") + 1] = str(out / "requests.json")
    command[command.index("--out-dir") + 1] = str(out)
    env.update(DRAMA_COMPACT_GGUF_HEAD="1", DRAMA_HOST_GGUF_EMBEDDING="1",
               DRAMA_Q8_ROUTE_TRACE=str(out / "routes"),
               DRAMA_Q8_ROUTE_TRACE_REQUEST=str(args.requests))
    print("capture starting", out, flush=True)
    started = time.perf_counter()
    with (out / "native.log").open("wb") as log:
        result = subprocess.run(
            command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT,
            timeout=240, creationflags=subprocess.CREATE_NO_WINDOW,
        )
    save(out / "run.json", {
        "command": command, "returncode": result.returncode,
        "elapsed_seconds": time.perf_counter() - started, "performance_benchmark": False,
    })
    print("capture exit", result.returncode, flush=True)
    if result.returncode:
        raise RuntimeError((out / "native.log").read_text(errors="replace")[-3000:])


def analyze(out):
    meta = [json.loads(line) for line in (out / "routes.jsonl").read_text().splitlines()]
    by_key = {(r["request"], r["layer"], r["step"]): r for r in meta}
    if len(by_key) != len(meta):
        raise RuntimeError("Duplicate captured route key")
    raw = (out / "routes.bin").read_bytes()
    if len(raw) % RECORD.size:
        raise RuntimeError("Truncated activation capture")
    details = []
    with (out / "replay.bin").open("wb") as replay:
        for offset in range(0, len(raw), RECORD.size):
            row = RECORD.unpack_from(raw, offset)
            request, layer, step, width, k = row[:5]
            record = by_key[(request, layer, step)]
            if width != 2048 or k != 8 or list(row[5:13]) != record["ids"]:
                raise RuntimeError("Tensor/route capture disagrees")
            replay.write(raw[offset:offset + RECORD.size])
            replay.write(struct.pack("<8i", *record["resident"]))
            details.append((request, step, layer))
    if details != sorted(details):
        raise RuntimeError("Captured replay is not in frame/layer order")
    layers = []
    for layer in range(40):
        rows = [r for r in meta if r["request"] == 3 and r["layer"] == layer]
        if not rows:
            continue
        hits = sum(i in r["resident"] for r in rows for i in r["ids"])
        layers.append({
            "layer": layer, "steps": len(rows), "hits": hits, "queries": len(rows) * 8,
            "hit_ratio": hits / (len(rows) * 8),
            "unique_experts": len({i for r in rows for i in r["ids"]}),
        })
    codes = []
    base = ROOT / "research/phase30-q8-final-stream/01-cache8"
    for name in ("warmup", "clone-1", "clone-2"):
        path = out / name / "drama3_take.json"
        if not path.exists():
            continue
        payload = lambda p: json.loads(bytes.fromhex(json.loads(p.read_text())["payload_hex"]))
        current = payload(path)["vq_codes"]
        previous = payload(base / name / "drama3_take.json")["vq_codes"]
        codes.append({
            "case": name, "frames": len(current[0]), "matches_uninstrumented_codes": current == previous,
            "codes_sha256": hashlib.sha256(json.dumps(current, separators=(",", ":")).encode()).hexdigest(),
        })
    report = {
        "meta_records": len(meta), "activation_records": len(details),
        "record_size_bytes": RECORD.size, "replay_record_size_bytes": RECORD.size + 32,
        "windows": sorted({step for _, step, _ in details}),
        "live_q8_trace": True, "captured_timings_not_for_performance": True,
        "routing_scales": "Selected live softmax probabilities normalized on CPU in F32; not exact GPU scale bytes.",
        "layers": layers, "codes": codes,
        "capture_sha256": sha(out / "routes.bin"), "meta_sha256": sha(out / "routes.jsonl"),
        "replay_sha256": sha(out / "replay.bin"),
    }
    save(out / "capture-audit.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "layers"}, ensure_ascii=False), flush=True)
    if layers:
        hits = sum(r["hits"] for r in layers)
        queries = sum(r["queries"] for r in layers)
        print("actual Q8 decode GPU cache hit ratio", hits / queries, flush=True)


def replay(args, build, out):
    env = environment(build)
    directory = Path(build["executable"]).parent.parent
    with (out / "replay-build.log").open("wb") as log:
        for command in ([
            build["cmake"], "-S", build["source"], "-B", str(directory),
            "-DDRAMA_ROUTE_REPLAY_SOURCE:FILEPATH=" + (ROOT / "native/gguf_route_replay.cpp").as_posix(),
        ], [
            build["cmake"], "--build", str(directory), "--target", "drama3_route_replay", "--parallel", "2",
        ]):
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    exe = directory / "bin/drama3_route_replay.exe"
    save(out / "replay-build.json", {
        "executable": str(exe), "sha256": sha(exe),
        "source_sha256": sha(ROOT / "native/gguf_route_replay.cpp"),
        "helper_sha256": sha(ROOT / "native/gguf_hybrid_probe.cpp"),
        "source": build["source"], "cuda_toolkit": build["cuda_toolkit"],
    })
    command = [
        str(exe), str(ROOT / "checkpoints/drama-3-gguf-q8/drama3-experts-Q8_0-bf16.gguf"),
        str(out / "replay.bin"), str(out / "replay-results.json"),
    ]
    print("multilayer replay starting", flush=True)
    started = time.perf_counter()
    with (out / "replay.log").open("wb") as log:
        result = subprocess.run(
            command, cwd=build["source"], env=env, stdout=log, stderr=subprocess.STDOUT,
            timeout=args.timeout, creationflags=subprocess.CREATE_NO_WINDOW,
        )
    save(out / "replay-run.json", {
        "command": command, "returncode": result.returncode, "elapsed_seconds": time.perf_counter() - started,
    })
    print("multilayer replay exit", result.returncode, flush=True)
    if result.returncode:
        raise RuntimeError((out / "replay.log").read_text(errors="replace")[-3000:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("capture", "analyze", "replay"))
    parser.add_argument("--output", type=Path, default=ROOT / "research/phase32-q8-real-routing/capture-v2")
    parser.add_argument("--requests", choices=(1, 3), type=int, default=3)
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    build = json.loads((ROOT / "research/phase30-codec-build.json").read_text())
    if args.stage == "capture":
        capture(args, build, out)
        analyze(out)
    elif args.stage == "analyze":
        analyze(out)
    else:
        replay(args, build, out)


if __name__ == "__main__":
    main()
