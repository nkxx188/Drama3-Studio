"""Check the localhost UI contract without loading the model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def fetch(base, path):
    request = Request(base.rstrip("/") + path, headers={"Accept": "application/json,text/html"})
    with urlopen(request, timeout=10) as response:
        return response.status, response.headers, response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8765")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checks = []
    for path in ("/", "/assets/app.css", "/assets/app.js"):
        status, headers, body = fetch(args.base, path)
        checks.append({
            "path": path, "status": status,
            "content_type": headers.get_content_type(), "bytes": len(body),
            "nonempty": bool(body),
        })
    status, _, body = fetch(args.base, "/")
    html = body.decode("utf-8")
    checks.append({
        "path": "html-contract", "status": status,
        "workflow_modes": re.findall(
            r'<option value="([^"]+)"',
            re.search(r'<select id="mode">(.+?)</select>', html, re.DOTALL).group(1)),
        "views": sorted(set(re.findall(r'data-view="([^"]+)"', html))),
        "scene_fallback_visible": "Native separators: experimental / unavailable" in html,
    })
    for path in ("/api/health", "/api/capabilities", "/api/voices", "/api/jobs",
                 "/api/model", "/api/model/task"):
        status, headers, body = fetch(args.base, path)
        data = json.loads(body.decode("utf-8"))
        checks.append({
            "path": path, "status": status,
            "content_type": headers.get_content_type(),
            "json_object": isinstance(data, dict),
            "keys": sorted(data) if isinstance(data, dict) else [],
        })
    audio_candidates = sorted((ROOT / "ui_data/jobs").glob("*/result.wav"))
    if audio_candidates:
        relative = audio_candidates[-1].relative_to(ROOT).as_posix()
        status, headers, body = fetch(args.base, f"/files/{relative}")
        checks.append({
            "path": f"/files/{relative}", "status": status,
            "content_type": headers.get_content_type(), "bytes": len(body),
            "audio_nonempty": bool(body),
        })
    report = {
        "scope": "Local HTTP UI contract without model loading or browser automation.",
        "base": args.base,
        "checks": checks,
        "passed": all(
            item.get("status") == 200
            and item.get(
                "nonempty",
                item.get("json_object", item.get("scene_fallback_visible",
                item.get("audio_nonempty", True))),
            )
            for item in checks
        ),
    }
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
