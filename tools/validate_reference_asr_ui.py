"""Isolated WebUI ASR acceptance with real Qwen GPU transcription.

User voices/history/settings are read only. Actual recognition runs against a
copied reference; browser race checks explicitly stub transcription tasks.
"""

from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    import ui.server as ui

    ui.STATE.shutdown()
    folder = ROOT / ".cache" / f"asr-ui-{uuid.uuid4().hex[:10]}"
    folder.mkdir(parents=True)
    state = ui.STATE = ui.AppState(data_root=folder)
    sources = sorted((ROOT / "ui_data/voices").glob("*/voice.json"))
    if not sources:
        raise RuntimeError("A local reference clip is required for the real ASR check")
    metadata = json.loads(sources[0].read_text(encoding="utf-8"))
    original = sources[0].parent / metadata["audio"]
    reference = folder / ("reference" + original.suffix)
    shutil.copyfile(original, reference)
    gpu_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else None
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), ui.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    env = dict(
        os.environ, DRAMA_UI_URL=f"http://127.0.0.1:{httpd.server_port}",
        DRAMA_ASR_TEST_AUDIO=str(reference), DRAMA_ASR_TEST_FOLDER=str(folder),
    )
    try:
        result = subprocess.run(
            [shutil.which("node") or "node", str(ROOT / "tools/validate_reference_asr_ui.cjs")],
            cwd=ROOT, env=env, timeout=240,
        )
        report_path = ROOT / "research/qwen3-asr-ui-validation.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        state._executor.submit(lambda: None).result(timeout=30)
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        remaining_tensors = []
        for obj in gc.get_objects():
            if type(obj) in (torch.Tensor, torch.nn.Parameter) and obj.is_cuda:
                remaining_tensors.append({"shape": list(obj.shape), "bytes": obj.numel() * obj.element_size()})
        after = torch.cuda.memory_allocated() if torch.cuda.is_available() else None
        # cuBLAS keeps an 8,519,680-byte workspace in this runtime; it is not an
        # ASR weight tensor. Validate released tensors and bound that residual.
        report["checks"]["gpu_model_memory_released"] = (
            not remaining_tensors and (after is None or after <= gpu_before + 16 * 1024**2))
        report["gpu_allocated_after"] = after
        report["remaining_cuda_tensors"] = remaining_tensors
        report["memory_note"] = "Model/input/output tensors released; a bounded CUDA library workspace may remain."
        # New transcriber instance proves the cache is read from disk.
        from drama_low_vram.reference_asr import ReferenceTranscriber
        with ThreadPoolExecutor(max_workers=1) as pool:
            def no_inference(*_args):
                raise RuntimeError("A persistent cache hit must not invoke the model")
            restored = ReferenceTranscriber(ROOT, folder, pool, no_inference)
            cached = restored.start(reference)
            report["checks"]["cache_survives_service_restart"] = (
                cached["status"] == "complete" and cached["cache_hit"])
        report["passed"] = not report["errors"] and all(report["checks"].values())
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return result.returncode or (0 if report["passed"] else 1)
    finally:
        httpd.shutdown()
        httpd.server_close()
        state.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
