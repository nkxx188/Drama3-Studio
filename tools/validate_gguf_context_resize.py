"""Replay local edit -> revision -> TTS on one real native model process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "vendor/fish-speech")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--source-job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from drama_low_vram.gguf_runtime import GGUFEngine
    from drama_low_vram.gguf_takes import load_conditioning_take
    from drama_low_vram.gguf_workflows import GGUFWorkflows
    from drama_low_vram.protocols import InfillSpan, VoiceReference

    job = json.loads(args.source_job.read_text(encoding="utf-8"))
    recipe = job["recipe"]
    source, _, _ = load_conditioning_take(job["take_file"])
    text = recipe["text"].strip()
    references = [VoiceReference(**row) for row in recipe["reference_snapshot"]]
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    engine = GGUFEngine(recipe["model"], context=1024, root=ROOT)
    runtime = args.runtime.resolve()
    manifest = json.loads((runtime / "manifest.json").read_text(encoding="utf-8"))
    engine.build = manifest | {
        "executable": str(runtime / manifest["executable"]), "source": str(ROOT),
        "cuda_toolkit": str(runtime), "dll_directory": str(runtime),
    }
    events = []
    engine.status_callback = lambda event: events.append({"time": time.perf_counter(), **event})
    workflows = GGUFWorkflows(engine)
    report = {"scope": "Real Q4 local-edit, revision and TTS, retaining native weights during context growth",
              "source_job": str(args.source_job.resolve()), "results": [], "status": "running"}

    def save():
        (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")

    try:
        local = workflows.run("infill", text, folder / "local.wav",
                              infill=InfillSpan(source.codes, 8, 20),
                              seed=123, max_new_tokens=768)
        process, first_directory = engine._process, engine._directory
        report["results"].append({"mode": "infill", "pid": process.pid, "result": local})
        assert local["complete"] and local["infill"]["prefix_codes_equal"] and local["infill"]["suffix_codes_equal"]
        save()
        after_load = len(events)
        for mode, extra in [
            ("revision", {"previous_takes": [source], "direction": "\u58f0\u97f3\u5174\u594b\u4e00\u70b9"}),
            ("tts", {}),
        ]:
            result = workflows.run(mode, text, folder / (mode + ".wav"), references=references,
                                   seed=123, max_new_tokens=768, **extra)
            assert result["complete"] and engine._process is process and process.poll() is None
            assert engine._directory == first_directory and result["session_reused"]
            report["results"].append({"mode": mode, "pid": process.pid, "result": result})
            save()
        assert report["results"][1]["result"]["context_resizes"]
        assert not report["results"][2]["result"]["context_resizes"]
        assert not any(event.get("timing_phase") == "model_load" for event in events[after_load:])
        report.update(status="complete", passed=True, events=events)
        save()
        print(json.dumps({
            "passed": True, "pid": process.pid, "context": engine.context,
            "resize": report["results"][1]["result"]["context_resizes"],
        }), flush=True)
    except BaseException as error:
        report.update(status="error", passed=False, error=f"{type(error).__name__}: {error}", events=events)
        save()
        raise
    finally:
        engine.close()


if __name__ == "__main__":
    main()
