"""Read-only evidence audit; does not infer listening quality or load a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.request import urlopen
import xml.etree.ElementTree as ET

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8879")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    report = {
        "checked_at": time.time(), "service": base,
        "scope": "Saved tests, HTTP asset identity, generation receipts and playable audio metadata.",
        "checks": {}, "browser_reports": {}, "assets": {}, "audio": [], "generation": {},
        "manual_acceptance": {
            "status": "pending", "items": [
                "Voice similarity and naturalness", "Speaker distinction and dialogue flow",
                "Scoped delivery controls", "Infill joins", "Hands-on workflow feedback",
            ],
        },
    }
    for filename, minimum in (
        ("2026-10-20-dialogue-ui-acceptance.json", 100),
        ("2026-10-20-voice-design-ui-acceptance.json", 90),
        ("2026-10-20-library-layout-ui-acceptance.json", 100),
        ("2026-10-20-tag-palette-ui-acceptance.json", 50),
        ("2026-10-20-studio-simplification-ui-acceptance.json", 65),
        ("2026-10-19-waveform-creation-regression.json", 58),
        ("2026-10-19-studio-recovery-ui-acceptance.json", 17),
        ("2026-10-19-audio-editor-ui-acceptance.json", 30),
    ):
        file = ROOT / "research" / filename
        data = json.loads(file.read_text(encoding="utf-8"))
        report["browser_reports"][filename] = {
            "passed": data["passed"], "checks": len(data["checks"]), "scope": data["scope"],
            "modified_at": file.stat().st_mtime,
        }
        report["checks"][filename] = (
            data["passed"] and not data["errors"] and len(data["checks"]) >= minimum
            and all(data["checks"].values()))

    suites = ET.parse(ROOT / "research/2026-10-20-ui-simplification-pytest.xml").findall(".//testsuite")
    report["integration_tests"] = {
        name: sum(int(suite.attrib.get(name, 0)) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    report["checks"]["integration_tests"] = (
        report["integration_tests"]["tests"] >= 129
        and report["integration_tests"]["failures"] == report["integration_tests"]["errors"] == 0)

    layout_suites = ET.parse(ROOT / "research/2026-10-20-library-layout-pytest.xml").findall(".//testsuite")
    report["layout_integration_tests"] = {
        name: sum(int(suite.attrib.get(name, 0)) for suite in layout_suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    report["checks"]["layout_integration_tests"] = (
        report["layout_integration_tests"]["tests"] >= 75
        and report["layout_integration_tests"]["failures"] == report["layout_integration_tests"]["errors"] == 0)

    design_suites = ET.parse(ROOT / "research/2026-10-20-voice-design-pytest.xml").findall(".//testsuite")
    report["design_integration_tests"] = {
        name: sum(int(suite.attrib.get(name, 0)) for suite in design_suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    report["checks"]["design_integration_tests"] = (
        report["design_integration_tests"]["tests"] >= 77
        and report["design_integration_tests"]["failures"] == report["design_integration_tests"]["errors"] == 0)

    writing_suites = ET.parse(ROOT / "research/2026-10-20-separate-writing-pytest.xml").findall(".//testsuite")
    report["writing_integration_tests"] = {
        name: sum(int(suite.attrib.get(name, 0)) for suite in writing_suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    report["checks"]["writing_integration_tests"] = (
        report["writing_integration_tests"]["tests"] >= 106
        and report["writing_integration_tests"]["failures"] == report["writing_integration_tests"]["errors"] == 0)

    tag_suites = ET.parse(ROOT / "research/2026-10-20-continuous-tags-pytest.xml").findall(".//testsuite")
    report["continuous_tag_integration_tests"] = {
        name: sum(int(suite.attrib.get(name, 0)) for suite in tag_suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    report["checks"]["continuous_tag_integration_tests"] = (
        report["continuous_tag_integration_tests"]["tests"] >= 108
        and report["continuous_tag_integration_tests"]["failures"] == report["continuous_tag_integration_tests"]["errors"] == 0)

    dialogue_suites = ET.parse(ROOT / "research/2026-10-20-dialogue-workspace-pytest.xml").findall(".//testsuite")
    report["dialogue_integration_tests"] = {
        name: sum(int(suite.attrib.get(name, 0)) for suite in dialogue_suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    report["checks"]["dialogue_integration_tests"] = (
        report["dialogue_integration_tests"]["tests"] >= 210
        and report["dialogue_integration_tests"]["failures"] == report["dialogue_integration_tests"]["errors"] == 0)
    parser = json.loads((ROOT / "research/2026-10-20-dialogue-parser-acceptance.json").read_text(encoding="utf-8"))
    report["checks"]["dialogue_parser"] = parser["passed"] and len(parser["checks"]) >= 26 and all(parser["checks"].values())

    catalog = json.loads((ROOT / "research/2026-10-20-portable-model-catalog-acceptance.json").read_text(encoding="utf-8"))
    report["model_catalog"] = {"passed": catalog["passed"], "models": len(catalog["models"]),
                               "checks": catalog["checks"]}
    report["checks"]["portable_shared_model_catalog"] = catalog["passed"] and all(catalog["checks"].values())

    for relative in ("index.html", "app.css", "app.js", "studio.js", "references.js", "audio-editor.js", "player.js",
                     "i18n.js", "controls.js", "studio.css", "script-editor.js"):
        file = ROOT / "ui" / relative
        route = "/" if relative == "index.html" else "/assets/" + relative
        with urlopen(base + route, timeout=15) as response:
            served = response.read()
        same = served == file.read_bytes()
        report["assets"][relative] = {"same_as_local": same, "sha256": hashlib.sha256(served).hexdigest(),
                                      "modified_at": file.stat().st_mtime}
    report["checks"]["served_assets_match_saved_sources"] = all(
        asset["same_as_local"] for asset in report["assets"].values())
    newest_asset = max(asset["modified_at"] for asset in report["assets"].values())
    report["checks"]["browser_checks_follow_current_asset_changes"] = all(
        data["modified_at"] >= newest_asset for data in report["browser_reports"].values())

    with urlopen(base + "/api/capabilities", timeout=15) as response:
        report["capabilities"] = json.load(response)
    report["checks"]["current_workflows_advertised"] = (
        report["capabilities"].get("native_protocol") == 4
        and all(report["capabilities"].get(name) for name in (
            "ordinary_tts", "voice_cloning", "voice_design", "joint_dialogue",
            "director_revision", "infill", "scene_scheduler", "streaming_text", "streaming_audio")))

    for filename in ("2026-10-19-studio-generation-q4.json", "2026-10-19-studio-generation-q8.json",
                     "2026-10-19-studio-context-generation-q4.json",
                     "2026-10-19-studio-edit-generation-q4.json"):
        data = json.loads((ROOT / "research" / filename).read_text(encoding="utf-8"))
        report["generation"][filename] = {"status": data["status"], "scope": data["scope"],
                                          "checks": data["checks"]}
    report["checks"]["saved_generation_checks"] = all(
        data["status"] == "passed" and all(data["checks"].values()) for data in report["generation"].values())

    document = (ROOT / "docs/webui-listening-acceptance-2026-10-19.md").read_text(encoding="utf-8")
    for value in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", document):
        file = Path(value).resolve()
        if not file.is_relative_to(ROOT):
            raise ValueError("Listening material is outside this project")
        info = sf.info(file)
        decoded, finite = 0, True
        with sf.SoundFile(file) as stream:
            for block in stream.blocks(blocksize=44100, dtype="float32", always_2d=True):
                decoded += len(block)
                finite = finite and bool(np.isfinite(block).all())
        report["audio"].append({"path": str(file), "duration_seconds": info.duration,
                                "sample_rate": info.samplerate, "channels": info.channels,
                                "decoded_frames": decoded, "all_samples_finite": finite,
                                "decoded_complete": decoded == info.frames})
    report["checks"]["all_thirteen_listening_files_readable"] = (
        len(report["audio"]) == 13 and all(
            audio["duration_seconds"] > 0 and audio["decoded_complete"] and audio["all_samples_finite"]
            for audio in report["audio"]))
    report["status"] = "ready_for_manual_acceptance" if all(report["checks"].values()) else "needs_review"
    target = ROOT / "research/2026-10-19-studio-delivery-audit.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": report["checks"],
                      "audio_count": len(report["audio"]), "manual_acceptance": "pending"},
                     ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ready_for_manual_acceptance" else 1


if __name__ == "__main__":
    raise SystemExit(main())
