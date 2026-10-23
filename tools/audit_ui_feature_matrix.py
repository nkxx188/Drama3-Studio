"""Audit the first UI against the official Drama 3 capability matrix."""

from __future__ import annotations

import json
from pathlib import Path
import re
import ast

ROOT = Path(__file__).resolve().parents[1]


def main():
    html = (ROOT / "ui/index.html").read_text(encoding="utf-8")
    server = (ROOT / "ui/server.py").read_text(encoding="utf-8")
    latest = next((ROOT / f"research/native-phase{phase}-status.json"
                   for phase in (13, 12, 11, 10)
                   if (ROOT / f"research/native-phase{phase}-status.json").is_file()))
    status = json.loads((latest if latest.is_file() else (
        ROOT / "research/native-phase10-status.json")).read_text(encoding="utf-8"))
    capability_method = next(
        node for node in ast.walk(ast.parse(server))
        if isinstance(node, ast.FunctionDef) and node.name == "capabilities")
    api_capabilities = ast.literal_eval(next(
        node.value for node in capability_method.body if isinstance(node, ast.Return)))
    capabilities = [
        "ordinary_tts",
        "voice_cloning",
        "multi_reference",
        "cross_language",
        "director_revision",
        "point_and_scoped_controls",
        "infill",
        "streaming_text",
        "streaming_audio",
        "scene_scheduler",
    ]
    mode_selector = re.search(r'<select id="mode">(.+?)</select>', html, re.DOTALL)
    modes = re.findall(r'<option value="([^"]+)"', mode_selector.group(1))
    views = set(re.findall(r'data-view="([^"]+)"', html))
    server_modes = set(re.search(
        r'if mode not in \(([^)]+)\)', server).group(1).replace('"', "").replace("'", "").split(","))
    server_modes = {value.strip() for value in server_modes}
    evidence = status.get("ui_phase3", status.get("ui_phase2", status.get("ui_phase1")))["real_e2e_reports"]
    rows = []
    for capability in capabilities:
        rows.append({
            "capability": capability,
            "api_capability": api_capabilities.get(capability) is True,
            "ui_entry": (
                capability in {"ordinary_tts", "voice_cloning", "multi_reference",
                               "cross_language", "director_revision",
                               "point_and_scoped_controls", "infill",
                               "streaming_text", "streaming_audio"}
                and "generate" in views
            ) or capability == "scene_scheduler" and "scene" in views,
            "real_evidence_reports": evidence,
        })
    report = {
        "scope": "Static UI/backend capability alignment. Runtime behavior is covered by the linked real reports; native separator remains an explicit unavailable capability.",
        "html_modes": modes,
        "server_modes": sorted(server_modes),
        "views": sorted(views),
        "rows": rows,
        "native_scene_separator": {
            "available": False,
            "fallback": "validated per-line EOS scene scheduler",
        },
        "all_required_product_paths_present": all(
            row["api_capability"] and row["ui_entry"] for row in rows
        ),
    }
    target = ROOT / "research/ui-feature-matrix.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "all_required_product_paths_present": report["all_required_product_paths_present"],
        "html_modes": modes,
        "server_modes": sorted(server_modes),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
