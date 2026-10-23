"""Bounded real Q4 acceptance for local infill, continuation and longer scripts."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "vendor/fish-speech")]

import numpy as np
import soundfile as sf
import torch
import ui.server as ui
from drama_low_vram.gguf_takes import load_conditioning_take


def main():
    ui.STATE.shutdown()
    source = json.loads((ROOT / "research/2026-10-19-studio-generation-q4.json").read_text(encoding="utf-8"))
    folder = ROOT / ".cache/studio-edit-acceptance" / uuid.uuid4().hex[:10]
    state = ui.AppState(data_root=folder / "ui_data")
    state.checkpoint = Path(source["model"])
    report = {"started_at": time.time(), "folder": str(folder), "model": str(state.checkpoint),
              "scope": "Real Q4 generation, exact codec/PCM checks, ASR content. Human listening remains separate.",
              "checks": {}, "jobs": {}, "transcripts": {}}
    receipt = ROOT / "research/2026-10-19-studio-edit-generation-q4.json"

    def save():
        receipt.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def run(label, payload):
        print("RUN", label, flush=True)
        identifier = state.create_job(payload)
        deadline = time.monotonic() + 600
        while True:
            job = state.job(identifier)
            if job["status"] not in ("queued", "running"):
                break
            if time.monotonic() >= deadline:
                state.cancel(identifier)
                raise TimeoutError(label)
            time.sleep(.3)
        report["jobs"][label] = job
        save()
        print("RESULT", label, job["status"], job.get("error"), flush=True)
        assert job["status"] == "complete", job
        return job

    try:
        reference = Path(source["jobs"]["design"]["output"])
        upload = state.add_reference_upload(reference.name, reference.read_bytes())
        binding = {"id": upload["id"], "text": "别着急，我们还有时间。坐下来慢慢说。", "speaker": 0}
        original_text = "大家好。我们周三下午开会。记得带上资料。"
        original = run("edit_source", {
            "text": original_text, "voice_bindings": [binding], "max_chunk_chars": 9,
            "seed": 123, "max_new_tokens": 384,
            "studio": {"work_id": "local-edit-acceptance", "segment_id": "first", "layout": "single"},
        })
        take, _, chunks = load_conditioning_take(original["take_file"])
        assert len(chunks) == 3, [chunk.text for chunk in chunks]
        first = chunks[0].codes.shape[1]
        end = first + chunks[1].codes.shape[1]
        report["selection"] = {"start_frame": first, "end_frame": end,
                               "source": "Recorded middle generation chunk, not inferred word timestamps."}
        changed = run("local_word_edit", {
            "source_version": {"job_id": original["id"]}, "operation": "infill",
            "text": original_text.replace("周三", "周四"), "start_frame": first, "end_frame": end,
            "seed": 123, "max_new_tokens": 384,
        })
        edited, _, _ = load_conditioning_take(changed["take_file"])
        suffix = take.codes.shape[1] - end
        report["checks"]["local_edit_preserves_real_prefix_codes"] = torch.equal(
            edited.codes[:, :first], take.codes[:, :first])
        report["checks"]["local_edit_preserves_real_suffix_codes"] = suffix > 0 and torch.equal(
            edited.codes[:, -suffix:], take.codes[:, end:])
        report["checks"]["local_edit_has_new_words_and_source"] = (
            changed["recipe"]["text"] == "大家好。我们周四下午开会。记得带上资料。"
            and changed["recipe"]["source_take"] == original["take_file"])
        state.adopt_version({
            "work_id": "local-edit-acceptance", "segment_id": "first", "version": {"job_id": changed["id"]}})
        long_text = (
            "今天我们一起准备明天的活动。首先，请大家检查需要的资料，把重要的文件放进包里。"
            "接着确认出发时间，给同行的朋友发一条消息。到达以后，我们先在门口集合，再一起去会议室。"
            "如果路上遇到堵车，也不用着急，提前告诉大家就好。所有事情安排妥当以后，记得早点休息。"
        )
        long_job = run("long_text", {
            "text": long_text, "voice_bindings": [binding], "max_chunk_chars": 40,
            "seed": 123, "max_new_tokens": 512,
            "studio": {"work_id": "long-continuation-acceptance", "segment_id": "first", "layout": "single"},
        })
        long_take, _, long_chunks = load_conditioning_take(long_job["take_file"])
        report["checks"]["long_text_uses_real_automatic_chunks"] = (
            len(long_chunks) >= 3 and len(long_job["result"]["chunks"]) == len(long_chunks)
            and long_job["result"]["audio"]["duration_seconds"] >= 15)
        next_text = "第二天早上，我们准时出发，路上聊得很开心。"
        continued = run("continue_long_text", {
            "source_version": {"job_id": long_job["id"]}, "operation": "continue", "text": next_text,
            "seed": 123, "max_new_tokens": 384,
            "studio": {"work_id": "long-continuation-acceptance", "segment_id": "next", "layout": "single"},
        })
        recipe = continued["recipe"]
        actual = state._previous(recipe, state._engine)
        report["checks"]["continuation_uses_only_exact_first_and_last_chunks"] = (
            len(actual) == 2 and torch.equal(actual[0].codes, long_chunks[0].codes)
            and torch.equal(actual[1].codes, long_chunks[-1].codes)
            and actual[0].text == long_chunks[0].text and actual[1].text == long_chunks[-1].text
            and sum(part.codes.shape[1] for part in actual) < long_take.codes.shape[1])
        report["checks"]["continuation_payload_contains_only_new_target_text"] = recipe["text"] == next_text
        report["composition"] = state.export_work({"work_id": "long-continuation-acceptance"})
        output, rate = sf.read(report["composition"]["output"], dtype="int16")
        expected = np.concatenate([sf.read(job["output"], dtype="int16")[0] for job in (long_job, continued)])
        report["checks"]["continuation_export_matches_adopted_pcm"] = rate == 44100 and np.array_equal(output, expected)
        state._engine.close()
        for label in ("edit_source", "local_word_edit", "long_text", "continue_long_text"):
            task = state.asr.start(Path(report["jobs"][label]["output"]), language="auto")
            deadline = time.monotonic() + 300
            while task["status"] in ("queued", "running"):
                if time.monotonic() >= deadline:
                    raise TimeoutError("ASR " + label)
                time.sleep(.4)
                task = state.asr.get(task["id"])
            report["transcripts"][label] = task
            save()
            assert task["status"] == "complete", task
        normalize = lambda text: "".join(character for character in text if character.isalnum())
        report["checks"]["source_asr_matches_target"] = (
            normalize(report["transcripts"]["edit_source"]["result"]["text"]) == normalize(original_text))
        report["checks"]["local_edit_asr_says_new_weekday"] = (
            normalize(report["transcripts"]["local_word_edit"]["result"]["text"])
            == normalize(original_text.replace("周三", "周四")))
        report["checks"]["continuation_asr_contains_only_new_text"] = (
            normalize(report["transcripts"]["continue_long_text"]["result"]["text"]) == normalize(next_text))
        report["checks"]["long_text_asr_matches_target"] = (
            normalize(report["transcripts"]["long_text"]["result"]["text"]) == normalize(long_text))
        report["status"] = "passed" if all(report["checks"].values()) else "needs_review"
        save()
        listening = ROOT / "docs/webui-listening-acceptance-2026-10-19.md"
        if listening.is_file():
            content = listening.read_text(encoding="utf-8")
            marker = "## 局部改词与长文本续接"
            content = content.split(marker, 1)[0].rstrip()
            lines = [content, "", marker, "",
                     "这一组为真实 Q4 输出，结构/内容检查通过不代表听感验收完成。", ""]
            for label, heading in (
                ("edit_source", "修改前：大家好。我们周三下午开会。记得带上资料。"),
                ("local_word_edit", "局部改词后：周三改为周四"),
                ("long_text", "约 25 秒长文本，自动分为 5 个生成块"),
                ("continue_long_text", "只生成后文，使用准确首尾块作历史"),
            ):
                job = report["jobs"][label]
                lines.extend([heading, f"![{heading}]({job['output'].replace(chr(92), '/')})", "",
                              "ASR：" + report["transcripts"][label]["result"]["text"], ""])
            lines.extend(["采用的长文本与后文组合：",
                          f"![长文本续接组合]({report['composition']['output'].replace(chr(92), '/')})", ""])
            listening.write_text("\n".join(lines), encoding="utf-8")
        print("ACCEPTANCE", report["status"], json.dumps(report["checks"], ensure_ascii=False), flush=True)
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        state.shutdown()
        report["total_seconds"] = time.time() - report["started_at"]
        save()


if __name__ == "__main__":
    main()
