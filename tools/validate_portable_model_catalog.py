"""Validate the running portable launcher and its shared model catalog, without inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import ProxyHandler, Request, build_opener
import uuid


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8879")
    parser.add_argument("--package-root", type=Path, default=ROOT / "dist/Drama3 Windows NVIDIA")
    args = parser.parse_args()
    package = args.package_root.resolve()
    opener = build_opener(ProxyHandler({}))

    def api(path, value=None):
        request = Request(args.url.rstrip("/") + path,
                          data=json.dumps(value).encode("utf-8") if value is not None else None,
                          headers={"Content-Type": "application/json"})
        with opener.open(request, timeout=30) as response:
            return json.load(response)

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

    protected = {path: digest(path) for path in (
        package / "ui_data/settings.json", package / "ui_data/studio.json",
    )}
    before = api("/api/model")
    instance = api("/api/instance")
    expected = set()
    for folder in (ROOT / "checkpoints").iterdir():
        if folder.is_file() and folder.suffix.lower() == ".gguf":
            expected.add(folder.resolve())
        elif folder.is_dir():
            config_file = folder / "config.json"
            if config_file.is_file():
                config = json.loads(config_file.read_text(encoding="utf-8"))
                if isinstance(config, dict) and config.get("model_type") == "fish_omni":
                    expected.add(folder.resolve())
            expected.update(path.resolve() for path in folder.iterdir()
                            if path.is_file() and path.suffix.lower() == ".gguf")
    models = before["available_models"]
    found = {Path(row["directory"]).resolve() for row in models}
    # Probe fresh packaged code with only the original external selection,
    # independent of the running service's remembered-model registry.
    sys.path.insert(0, str(package))
    from tools.windows_launcher import configure_environment
    configure_environment()
    import ui.server as server
    probe = server.AppState(data_root=ROOT / ".cache/model-catalog-probes" / uuid.uuid4().hex[:10])
    try:
        probe.checkpoint = Path(before["selected_directory"])
        probe.model_paths = []
        fresh_models = probe.available_models()
        fresh_roots = probe.model_search_roots()
        probe_unloaded = probe._engine is None
    finally:
        probe.shutdown()
        server.STATE.shutdown()
    startup = json.loads((package / "diagnostics/startup.json").read_text(encoding="utf-8"))
    report = {
        "scope": "Actual portable start.bat/HTTP catalog and fresh packaged discovery with an empty model registry; no inference.",
        "checked_at": time.time(), "package": str(package), "instance": instance,
        "selected_directory": before["selected_directory"], "models": models,
        "fresh_models": fresh_models, "fresh_search_roots": [str(path) for path in fresh_roots],
        "checks": {
            "portable_instance_identity": instance["instance_key"] == hashlib.sha256(
                str(package).casefold().encode("utf-8")).hexdigest(),
            "bundled_interpreter_used": Path(startup["executable"]).resolve() == package / "runtime/python/python.exe",
            "source_pool_has_multiple_models": len(expected) > 1,
            "all_source_pool_models_visible": expected <= found,
            "fresh_packaged_code_uses_portable_root": server.ROOT == package,
            "fresh_discovery_without_registry_finds_all_models": expected <= {
                Path(row["directory"]).resolve() for row in fresh_models},
            "fresh_discovery_does_not_load_weights": probe_unloaded,
            "model_paths_are_unique": len(found) == len(models),
            "python_and_gguf_formats_visible": {row["runtime"] for row in models} == {"python", "gguf"},
            "auxiliary_asr_excluded": not any("qwen3-asr" in row["directory"].casefold() for row in models),
            "no_weights_copied_into_portable_pool": not any((package / "checkpoints").iterdir()),
            "no_model_loaded": not api("/api/health")["engine_loaded"],
        },
    }
    report["inspections"] = []
    for row in models:
        status = api("/api/model/inspect", {"directory": row["directory"]})
        report["inspections"].append({"name": row["name"], "ready": status["inspection"]["ready"]})
    reopened = subprocess.run(
        ["cmd.exe", "/d", "/c", "start.bat", "--no-browser"], cwd=package,
        env=dict(os.environ, DRAMA_NO_PAUSE="1"), capture_output=True, timeout=60,
        encoding="utf-8", errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    report["reopen"] = {"returncode": reopened.returncode, "output": reopened.stdout + reopened.stderr}
    after = api("/api/model")
    report["checks"].update({
        "start_script_reopens_same_instance": reopened.returncode == 0 and api("/api/instance")["pid"] == instance["pid"],
        "selected_model_unchanged": after["selected_directory"] == before["selected_directory"],
        "catalog_stable_after_reopen": after["available_models"] == models,
        "user_settings_and_drafts_unchanged": all(digest(path) == value for path, value in protected.items()),
        "no_inference_after_inspection": not api("/api/health")["engine_loaded"],
    })
    report["passed"] = all(report["checks"].values())
    target = ROOT / "research/2026-10-20-portable-model-catalog-acceptance.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
