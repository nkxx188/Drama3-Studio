"""Real UI frame streaming against its matching ordinary-clone baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def api(base, route, payload=None):
    request = Request(
        base.rstrip("/") + route,
        None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST")
    with urlopen(request, timeout=20) as response:
        return json.load(response)


def wait_job(base, job_id):
    deadline, last = time.monotonic() + 600, None
    while time.monotonic() < deadline:
        job = api(base, f"/api/jobs/{job_id}")
        event = (job.get("events") or [{}])[-1]
        stage = event.get("stage")
        if stage != last:
            print(json.dumps({"id": job_id, "status": job["status"], "stage": stage}), flush=True)
            last = stage
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(1)
    raise TimeoutError(f"Job {job_id} is still live; inspect it before any retry")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8877")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "research/ui-resident-audio-stream-validation.json")
    args = parser.parse_args()
    voice = next(value for value in api(args.base, "/api/voices")["voices"]
                 if value["speaker"] == 0)
    common = {
        "text": "你好，这是新的帧流测试。",
        "voice_ids": [voice["id"]], "seed": 123, "max_new_tokens": 192,
    }
    jobs = []
    for mode in ("tts", "audio_stream"):
        created = api(args.base, "/api/jobs", {**common, "mode": mode, "chunk_frames": 16})
        job = wait_job(args.base, created["id"])
        jobs.append(job)
        if job["status"] != "complete":
            raise RuntimeError(f"Real UI {mode} task failed: {job.get('error')}")
    baseline, streamed = jobs
    result = streamed["result"]
    with np.load(baseline["take_file"], allow_pickle=False) as expected, np.load(
        streamed["take_file"], allow_pickle=False
    ) as actual:
        codes_equal = bool(np.array_equal(expected["codes"], actual["codes"]))
        frames_equal = bool(np.array_equal(expected["frames"], actual["frames"]))
    parts = np.concatenate([sf.read(row["path"], dtype="int16")[0] for row in result["segments"]])
    final, rate = sf.read(streamed["output"], dtype="int16")
    report = {
        "scope": "Real localhost jobs on the automatically configured UI engine, ordinary reference clone versus frame streaming. Exact VQ/frames and delivered PCM export; listening remains separate.",
        "base_url": args.base, "baseline_job_id": baseline["id"],
        "stream_job_id": streamed["id"], "voice_id": voice["id"],
        "output": streamed["output"], "take_file": streamed["take_file"],
        "audio": result["audio"], "segments": result["segments"],
        "vq_codes_equal": codes_equal, "all_frames_equal": frames_equal,
        "exported_pcm_matches_delivered": bool(np.array_equal(parts, final)),
        "sample_rate": rate,
        "delivered_before_eos": any(row["before_eos"] for row in result["segments"]),
        "resident_segments": sum(row.get("sequence_preservation") == "resident"
                                 for row in result["segments"]),
        "deferred_audio_for_host_budget": result["deferred_audio_for_host_budget"],
        "compute_policy": result["compute_policy"],
        "lifecycle": result["lifecycle"], "total_seconds": result["total_seconds"],
        "baseline_seconds": baseline["result"]["total_seconds"],
    }
    report["passed"] = all(report[key] for key in (
        "vq_codes_equal", "all_frames_equal", "exported_pcm_matches_delivered",
        "delivered_before_eos"))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["passed"]:
        raise RuntimeError("UI frame-stream output gate failed; evidence preserved")


if __name__ == "__main__":
    main()
