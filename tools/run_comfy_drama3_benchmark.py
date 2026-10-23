"""Queue a local ComfyUI workflow and record node times and sampled resources."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path

import aiohttp
import psutil
import pynvml


async def run(args):
    prompt = json.loads(args.prompt.read_text(encoding="utf-8"))
    workflow = json.loads(args.workflow.read_text(encoding="utf-8")) if args.workflow else None
    client_id = str(uuid.uuid4())
    process = psutil.Process(args.pid)
    started = time.perf_counter()
    samples = []
    events = []
    node_times = []
    active = None
    last_progress_print = 0
    log_offset = args.server_log.stat().st_size
    report = {
        "prompt_file": str(args.prompt.resolve()),
        "base_url": args.url,
        "server_pid": args.pid,
        "status": "pending",
        "resource_sampling_seconds": 1,
        "timing_scope": "Client-observed node transitions; not CUDA event timings.",
    }
    try:
        pynvml.nvmlInit()
        device = pynvml.nvmlDeviceGetHandleByIndex(0)
    except pynvml.NVMLError:
        device = None

    def elapsed():
        return time.perf_counter() - started

    async def observe():
        while True:
            memory = process.memory_info()
            sample = {
                "seconds": elapsed(),
                "process_rss_bytes": memory.rss,
                "process_private_bytes": memory.private,
                "system_available_bytes": psutil.virtual_memory().available,
            }
            if device is not None:
                vram = pynvml.nvmlDeviceGetMemoryInfo(device)
                util = pynvml.nvmlDeviceGetUtilizationRates(device)
                sample.update(
                    gpu_used_bytes=vram.used,
                    gpu_free_bytes=vram.free,
                    gpu_utilization_percent=util.gpu,
                )
            samples.append(sample)
            await asyncio.sleep(1)

    observer = asyncio.create_task(observe())
    prompt_id = None
    try:
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(args.url + "/system_stats") as response:
                report["environment"] = await response.json()
            ws_url = args.url.replace("http://", "ws://").replace("https://", "wss://")
            async with session.ws_connect(ws_url + "/ws?clientId=" + client_id, heartbeat=30) as ws:
                payload = {"prompt": prompt, "client_id": client_id}
                if workflow is not None:
                    payload["extra_data"] = {"extra_pnginfo": {"workflow": workflow}}
                async with session.post(args.url + "/prompt", json=payload) as response:
                    queued = await response.json()
                    if response.status != 200:
                        raise RuntimeError(json.dumps(queued, ensure_ascii=False))
                prompt_id = queued["prompt_id"]
                report["prompt_id"] = prompt_id
                print(f"Queued {prompt_id}", flush=True)
                async for message in ws:
                    if message.type != aiohttp.WSMsgType.TEXT:
                        continue
                    event = json.loads(message.data)
                    kind, data = event["type"], event.get("data", {})
                    if data.get("prompt_id") not in (None, prompt_id):
                        continue
                    if kind not in (
                        "execution_start", "execution_cached", "executing",
                        "executed", "execution_error", "execution_success", "progress",
                    ):
                        continue
                    now = elapsed()
                    events.append({"seconds": now, **event})
                    if kind == "executing":
                        if active is not None:
                            active["seconds"] = now - active["start_seconds"]
                            node_times.append(active)
                            print(
                                f"Finished {active['class_type']}: {active['seconds']:.3f}s",
                                flush=True,
                            )
                        node = data.get("node")
                        active = None
                        if node is not None:
                            active = {
                                "node": node,
                                "class_type": prompt[node]["class_type"],
                                "start_seconds": now,
                            }
                            print(f"Running {active['class_type']}", flush=True)
                        else:
                            break
                    elif kind == "progress":
                        value = data.get("value", 0)
                        if value == 1 or now - last_progress_print >= 15:
                            print(f"Progress {value}/{data.get('max')} at {now:.1f}s", flush=True)
                            last_progress_print = now
                    elif kind == "execution_error":
                        report["execution_error"] = data
                        raise RuntimeError(data.get("exception_message", "ComfyUI execution failed"))
                    elif kind == "execution_success":
                        if active is not None:
                            active["seconds"] = now - active["start_seconds"]
                            node_times.append(active)
                            active = None
                        break
            for attempt in range(30):
                async with session.get(args.url + "/history/" + prompt_id) as response:
                    report["history"] = (await response.json()).get(prompt_id, {})
                if report["history"]:
                    break
                await asyncio.sleep(0.1)
        report["status"] = report["history"].get("status", {}).get("status_str", "unknown")
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        observer.cancel()
        await asyncio.gather(observer, return_exceptions=True)
        report["wall_seconds"] = elapsed()
        report["node_times"] = node_times
        report["events"] = events
        report["resource_samples"] = samples
        if samples:
            report["resource_summary"] = {
                "peak_process_rss_bytes": max(s["process_rss_bytes"] for s in samples),
                "peak_process_private_bytes": max(s["process_private_bytes"] for s in samples),
                "minimum_system_available_bytes": min(s["system_available_bytes"] for s in samples),
                "sampled_peak_whole_gpu_used_bytes": max(s.get("gpu_used_bytes", 0) for s in samples),
            }
        with args.server_log.open("rb") as handle:
            handle.seek(log_offset)
            report["server_log"] = handle.read().decode("utf-8", errors="replace")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Report: {args.output}", flush=True)
        if device is not None:
            pynvml.nvmlShutdown()
    if report["status"] != "success":
        raise RuntimeError(f"Workflow status: {report['status']}")
    print(
        json.dumps(
            {key: report[key] for key in ("status", "wall_seconds", "node_times", "resource_summary")},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8192")
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--workflow", type=Path)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
