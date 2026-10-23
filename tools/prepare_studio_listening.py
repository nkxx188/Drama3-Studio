"""Generate a matched joint-dialogue take and index real outputs for listening."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "vendor/fish-speech")]

import ui.server as ui


def main():
    ui.STATE.shutdown()
    context = json.loads((ROOT / "research/2026-10-19-studio-context-generation-q4.json").read_text(encoding="utf-8"))
    assert context["status"] == "passed"
    state = ui.AppState(data_root=Path(context["folder"]) / "ui_data")
    state.checkpoint = Path(context["model"])
    baseline = context["jobs"]["baseline_scene"]
    receipt = ROOT / "research/2026-10-19-studio-listening-q4.json"
    report = {"scope": "Matched input and references; human listening pending. No naturalness score inferred from ASR.",
              "model": str(state.checkpoint), "per_line": baseline, "started_at": time.time()}

    def save():
        receipt.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    try:
        payload = {key: baseline["recipe"][key] for key in (
            "mode", "scene", "characters", "lines", "line_ids", "voice_bindings",
            "seed", "temperature", "top_p", "top_k", "max_new_tokens",
        )}
        payload.update(scene_strategy="joint", studio={
            "work_id": "context-listening", "segment_id": "joint-comparison", "layout": "scene",
        })
        identifier = state.create_job(payload)
        print("RUN matched_joint", identifier, flush=True)
        deadline = time.monotonic() + 600
        while True:
            job = state.job(identifier)
            if job["status"] not in ("queued", "running"):
                break
            if time.monotonic() >= deadline:
                state.cancel(identifier)
                raise TimeoutError("Matched joint dialogue")
            time.sleep(.4)
        report["joint"] = job
        save()
        assert job["status"] == "complete", job
        report["checks"] = {
            "same_script": job["recipe"]["text"] == baseline["recipe"]["text"],
            "same_references": job["recipe"]["reference_snapshot"] == baseline["recipe"]["reference_snapshot"],
            "same_sampling": all(job["recipe"][key] == baseline["recipe"][key]
                                 for key in ("seed", "temperature", "top_p", "top_k", "max_new_tokens")),
            "joint_one_take": bool(job["take_file"] and job["result"]["strategy"] == "joint"
                                   and not job["result"]["segments"]),
        }
        state._engine.close()
        task = state.asr.start(Path(job["output"]), language="auto")
        deadline = time.monotonic() + 300
        while task["status"] in ("queued", "running"):
            if time.monotonic() >= deadline:
                raise TimeoutError("Joint content transcription")
            time.sleep(.4)
            task = state.asr.get(task["id"])
        report["transcript"] = task
        report["checks"]["asr_available_for_review"] = task["status"] == "complete"
        report["status"] = "ready_for_listening" if all(report["checks"].values()) else "failed"
        lines = [
            "# Drama 3 WebUI 真实音频验收", "",
            "日期：2026-10-19。以下均为真实 Q4 输出，未加入用户作品。", "",
            "试听未完成，不能凭 complete 或 ASR 判定自然度、音色相似度与角色区分。",
            "多人两种方式使用相同剧本、参考及采样条件；逐句 seed 随行号变化，联合使用一次 seed，随机过程不等价。",
            "冷启动与不同输出长度不适合据此比较运行速度。", "",
            "## 多人对话", "",
            "林：明天几点出发？", "",
            "周：早上九点，我们在门口见。", "",
            "林：好，那我们明天见。", "",
            "逐句生成后拼接：", f"![逐句对话]({baseline['output'].replace(chr(92), '/')})", "",
            "整段联合生成：", f"![联合对话]({job['output'].replace(chr(92), '/')})", "",
            "联合输出 ASR（仅供核对内容）：", task.get("result", {}).get("text", task.get("error", "")), "",
            "## 采用前文后更新后句", "",
            "原第三句：",
            f"![原第三句]({baseline['result']['segments'][2]['path'].replace(chr(92), '/')})", "",
            "改用新的首句后，使用当前前文生成的新候选：",
            f"![更新后第三句]({context['jobs']['updated_third_line']['output'].replace(chr(92), '/')})", "",
            "采用后的完整组合：", f"![采用后对话]({context['composition']['output'].replace(chr(92), '/')})", "",
            "## 回听记录", "",
            "| 检查 | 听什么 | 结论 |", "| --- | --- | --- |",
            "| 内容 | 漏字、重复、旧词、是否完整读完 | 待回听 |",
            "| 角色 | 声音归属与换人是否清楚 | 待回听 |",
            "| 参考复用 | 与上传参考的音色是否相似 | 待回听 |",
            "| 对话 | 接话、停顿、语气与整段连贯性 | 待回听 |",
            "| 前文更新 | 新后句是否保持声音、是否自然衔接 | 待回听 |",
        ]
        for filename, heading, entries in (
            ("2026-10-19-studio-generation-q4.json", "声音设计与复用",
             [("design", "设计声音"), ("reuse_designed_voice", "复用设计声音")]),
            ("2026-10-19-studio-joint-derivatives-q4-seed123.json", "联合改词",
             [("reword", "改词后的完整对话")]),
        ):
            file = ROOT / "research" / filename
            if not file.is_file():
                continue
            data = json.loads(file.read_text(encoding="utf-8"))
            lines += ["", f"## {heading}", ""]
            for key, label in entries:
                entry = data.get("jobs", {}).get(key)
                if entry and entry.get("output"):
                    lines += [label, f"![{label}]({entry['output'].replace(chr(92), '/')})", ""]
        listening = ROOT / "docs/webui-listening-acceptance-2026-10-19.md"
        marker = "## 局部改词与长文本续接"
        if listening.is_file():
            existing = listening.read_text(encoding="utf-8")
            if marker in existing:
                lines.extend(["", marker + existing.split(marker, 1)[1]])
        listening.write_text("\n".join(lines) + "\n", encoding="utf-8")
        save()
        assert report["status"] == "ready_for_listening"
        print("LISTENING", report["status"], task.get("result", {}).get("text", ""), flush=True)
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
