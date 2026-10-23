"""Verify comparison WAVs and write a compact report with optional local ASR."""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import html
import json
from pathlib import Path
import wave

LABELS = {
    "fp8": "官方 FP8", "int4": "官方 INT4（W4A16）", "int8": "INT8 ConvRot",
    "w6a8": "W6A8", "w4a8": "W4A8", "gguf-q4": "GGUF Q4_K_S",
}


def normalized(text):
    return "".join(c for c in text if c.isalnum()).lower()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--asr", type=Path)
    args = parser.parse_args()
    folder = args.directory.resolve()
    report = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if report["status"] != "complete":
        raise ValueError("The model comparison has not finished")
    asr = {}
    if args.asr:
        asr = {
            str(Path(row["audio"]).resolve()): row
            for row in json.loads(args.asr.read_text(encoding="utf-8"))["rows"]
        }
    rows = []
    for model in report["models"]:
        measured = json.loads((Path(model["directory"]) / "summary.json").read_text(encoding="utf-8"))
        for request in measured["requests"]:
            if request["warmup"]:
                continue
            audio = request.get("audio")
            if audio is None:
                raise ValueError(f"Missing audio: {model['label']}/{request['case']}")
            path = Path(audio["path"])
            with wave.open(str(path), "rb") as wav:
                duration = wav.getnframes() / wav.getframerate()
                digest = hashlib.sha256(wav.readframes(wav.getnframes())).hexdigest()
                assert wav.getnchannels() == 1 and wav.getsampwidth() == 2
                assert wav.getframerate() == 44100
                assert wav.getnframes() == request["frames"] * 2048
                assert duration <= report["duration_cap_seconds"] + 1e-6
                assert digest == audio["pcm_sha256"]
            transcript = asr.get(str(path.resolve()), {}).get("text")
            text_match = (
                normalized(transcript) == normalized(report["text"]) if transcript is not None else None
            )
            similarity = (
                difflib.SequenceMatcher(
                    None, normalized(report["text"]), normalized(transcript)).ratio()
                if transcript is not None else None
            )
            row = {
                "model": model["label"], "model_label": LABELS[model["label"]],
                "task": request["case"], "seconds": request["total_seconds"],
                "audio_seconds": duration, "rtf": request["total_seconds"] / duration,
                "frames": request["frames"], "finish_reason": request["finish_reason"][0],
                "prefill_seconds": request.get("prefill_seconds"),
                "decode_seconds": request.get("decode_seconds"),
                "codec_seconds": request.get("codec_seconds"),
                "reference_seconds": request.get("reference_seconds"),
                "reference_cache_hit": request.get("reference_cache_hit"),
                "model_load_seconds": measured.get("model_load_seconds"),
                "initial_prepare_seconds": measured.get("initial_prepare_seconds"),
                "compute_policy": measured.get("compute_policy"),
                "native_outside_request_seconds": measured.get("outside_request_seconds"),
                "audio": str(path), "pcm_sha256": digest,
                "asr_text": transcript, "asr_normalized_match": text_match,
                "asr_sequence_similarity": similarity, "resources": model.get("resources"),
            }
            rows.append(row)
    result = {
        "scope": report["scope"], "text": report["text"],
        "duration_cap_seconds": report["duration_cap_seconds"],
        "sampling": report["sampling"], "rows": rows,
        "asr_note": "ASR checks words, not audio naturalness or speaker similarity.",
        "hardware": "Windows 11 / RTX 4070 Laptop 8GB / Ryzen 9 7940H / 48GB RAM",
    }
    (folder / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    columns = (
        "model_label", "task", "seconds", "audio_seconds", "rtf", "frames", "finish_reason",
        "model_load_seconds", "initial_prepare_seconds", "native_outside_request_seconds",
        "asr_text", "asr_normalized_match", "audio",
    )
    with (folder / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# 六模型约10秒文生／克隆对比（2026-10-17）", "",
        result["hardware"], "",
        f"目标文本：{report['text']}", "",
        "每项215帧上限（9.98458秒）＋时长软提示；允许模型提前EOS。"
        "length表示达到上限截断，不能当作自然完整结束。"
        "生成采用相同seed=123、temperature=0.7、top_k=30、top_p=0.7；"
        "不同量化/后端的采样实现和质量不等价。", "",
        "每种任务前有一条明确记录的8帧预热；每模型独立进程，文生始终先于克隆，"
        "避免克隆参考被继承到文生。每项正式请求仅测一次，不代表稳定中位数。"
        "完整请求含prefill、逐帧生成、codec和必要运行准备；"
        "模型加载、首次校准及预热另列。主引擎正式克隆复用预热时编码的参考音频；"
        "第一次换入参考的耗时保留在warmup-clone记录内。", "",
        "| 模型 | 文生耗时／音频时长 | 结束 | 克隆耗时／音频时长 | 结束 |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for key, label in LABELS.items():
        by_case = {row["task"]: row for row in rows if row["model"] == key}
        cells = [label]
        for case in ("tts", "clone"):
            row = by_case.get(case)
            cells.extend(
                [f"{row['seconds']:.2f}s ／ {row['audio_seconds']:.3f}s", row["finish_reason"]]
                if row else ["无结果", "错误"])
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## 加载、校准与资源", "",
              "这些开销不在上表正式生成时间内，不能把它们与热请求速度混为一谈。", ""]
    for model in report["models"]:
        measured = model["report"]
        resources = model.get("resources", {})
        gpu = resources.get("sampled_whole_gpu_peak_bytes", 0) / 1024**3
        ram = resources.get("peak_rss_bytes", 0) / 1024**3
        if model["label"] == "gguf-q4":
            setup = (
                f"原生进程除请求外耗时{measured.get('outside_request_seconds', 0):.2f}s"
                "（含启动／加载／收尾，不能精确拆成模型加载）")
        else:
            policy = measured.get("compute_policy", {})
            setup = (
                f"模型加载{measured.get('model_load_seconds', 0):.2f}s，"
                f"初始准备{measured.get('initial_prepare_seconds', 0):.2f}s"
                f"（其中计算策略{policy.get('seconds', 0):.2f}s，状态{policy.get('status')}）")
        lines.append(
            f"- {LABELS[model['label']]}：{setup}；加载/预热/正式请求全过程"
            f"采样峰值RSS约{ram:.2f}GiB、整卡显存约{gpu:.2f}GiB。")
    lines += ["", "RSS包括文件页；整卡显存包括桌面／驱动。以上是采样近似峰值。", "",
              "## 音频与内容检查", "",
              "ASR只检查读词，不代替听感或音色相似度验收。"
              "截断结尾或标点差异也可能导致不完全匹配。", ""]
    for row in rows:
        task = "文生" if row["task"] == "tts" else "克隆"
        lines.append(f"- [{row['model_label']} {task}]({Path(row['audio']).as_posix()})："
                     + (row["asr_text"] if row["asr_text"] is not None else "待ASR检查"))
    (folder / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    body = []
    for row in rows:
        rel = Path(row["audio"]).relative_to(folder).as_posix()
        body.append(
            f"<tr><td>{html.escape(row['model_label'])}</td>"
            f"<td>{'文生' if row['task']=='tts' else '克隆'}</td>"
            f"<td>{row['seconds']:.2f}s</td><td>{row['audio_seconds']:.3f}s</td>"
            f"<td>{html.escape(row['finish_reason'])}</td>"
            f"<td><audio controls preload='none' src='{html.escape(rel)}'></audio></td>"
            f"<td>{html.escape(row['asr_text'] or '')}</td></tr>")
    document = (
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Drama3 六模型音频对比</title><style>"
        "body{font:16px/1.6 system-ui;max-width:1440px;margin:32px auto;padding:0 20px;color:#222}"
        "table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}"
        "audio{width:240px}p{max-width:1000px}.scroll{overflow:auto}</style>"
        "<h1>Drama3 六模型约10秒对比</h1>"
        f"<p>{html.escape(report['text'])}</p>"
        "<p>215帧上限约9.985秒；eos为自然结束，length为到上限截断。"
        "耗时为预热后的单次完整请求，加载和首次校准另计；主引擎克隆参考已预编码。"
        "不同模型精度、后端和输出质量不同；ASR只核对文字。</p>"
        "<div class='scroll'><table><thead><tr><th>模型</th><th>任务</th><th>耗时</th>"
        "<th>音频</th><th>结束</th><th>试听</th><th>转写</th></tr></thead><tbody>"
        + "".join(body) + "</tbody></table></div></html>")
    (folder / "listen.html").write_text(document, encoding="utf-8")
    print(json.dumps(
        [{"model": row["model"], "task": row["task"], "seconds": row["seconds"],
          "audio_seconds": row["audio_seconds"], "finish": row["finish_reason"],
          "asr_match": row["asr_normalized_match"]} for row in rows],
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
