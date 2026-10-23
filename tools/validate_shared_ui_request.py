"""One complete cloned request through the restarted local UI job queue."""

import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8877"


def api(path, payload=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=20) as response:
        return json.load(response)


def main():
    if any(job["status"] in ("queued", "running") for job in api("/api/jobs")["jobs"]):
        raise RuntimeError("Validation requires an idle UI job queue")
    model = api("/api/model")
    if not model["inspection"]["ready"] or not model["uses_checkpoint_directly"]:
        raise RuntimeError("The shared default checkpoint is not ready")
    voice = next(voice for voice in api("/api/voices")["voices"]
                 if voice["speaker"] == 0 and "官方 FP8" in voice["transcript"])
    payload = {"mode": "tts", "text": "<|speaker:0|>你好，这是官方 FP8 质量优先测试。",
               "voice_ids": [voice["id"]], "seed": 123, "max_new_tokens": 256}
    job_id = api("/api/jobs", payload)["id"]
    started, last = time.perf_counter(), 0
    while True:
        job = api("/api/jobs/" + job_id)
        now = time.perf_counter()
        if now - last >= 20:
            print(json.dumps({"id": job_id, "status": job["status"],
                              "last_event": (job.get("events") or [{}])[-1]}), flush=True)
            last = now
        if job["status"] not in ("queued", "running"):
            break
        if now - started > 300:
            raise TimeoutError(f"UI request still active: {job_id}")
        time.sleep(1)
    report = {"model": model["inspection"], "payload": payload,
              "wall_seconds": time.perf_counter() - started, "job": job}
    (ROOT / "research/ui-shared-model-generation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"id": job_id, "status": job["status"], "output": job["output"],
                      "error": job["error"]}), flush=True)
    if job["status"] != "complete" or not job["output"]:
        raise RuntimeError("Shared model did not complete the UI clone request")


if __name__ == "__main__":
    main()
