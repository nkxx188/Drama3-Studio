"""Local startup diagnostics and a bounded, user-exportable support archive."""

import json
from pathlib import Path
import platform
import sys
import zipfile

import psutil

from .nvidia_policy import policy_for
from .runtime_files import load_native_runtime


def snapshot(root, *, native=True, deep=False):
    root = Path(root).resolve()
    report = {
        "platform": platform.platform(), "machine": platform.machine(),
        "python": platform.python_version(), "executable": sys.executable,
        "ram_total_bytes": psutil.virtual_memory().total,
        "ram_available_bytes": psutil.virtual_memory().available,
        "drive_free_bytes": psutil.disk_usage(str(root)).free, "errors": [],
    }
    try:
        import torch
        report["torch"] = torch.__version__
        report["torch_cuda"] = torch.version.cuda
        report["torch_architectures"] = torch.cuda.get_arch_list()
        report["cpu_capability"] = torch.backends.cpu.get_cpu_capability()
        report["gpus"] = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            free, total = torch.cuda.mem_get_info(index)
            report["gpus"].append({"index": index, "name": properties.name, "free_bytes": free,
                                   "total_bytes": total,
                                   "policy": policy_for((properties.major, properties.minor)).report()})
    except Exception as error:
        report["errors"].append("Python CUDA: " + str(error))
    if "gpus" in report and not any(gpu["policy"]["supported"] for gpu in report["gpus"]):
        report["errors"].append("No RTX20/SM75 or newer CUDA GPU was found for the Python engine.")
    if native:
        try:
            from .gguf_runtime import native_memory
            build = load_native_runtime(root)
            report["native_build"] = {key: build.get(key) for key in
                                      ("sha256", "architectures", "protocol", "portable")}
            report["native_probe"] = native_memory(build, root)
            if deep:
                from .gguf_runtime import build_environment, PREFIX
                import subprocess
                result = subprocess.run(
                    [build["executable"], "--self-test",
                     str(report["native_probe"].get("selected_device", 0))],
                    cwd=str(root), env=build_environment(build, root), capture_output=True,
                    text=True, encoding="utf-8", errors="replace", timeout=60,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                report["native_self_test"] = [
                    json.loads(line[len(PREFIX):]) for line in result.stdout.splitlines()
                    if line.startswith(PREFIX)]
                if result.returncode:
                    report["errors"].append("Native GPU self-test failed: " + result.stderr[-2000:])
                elif not report["native_self_test"] or not report["native_self_test"][-1].get("passed"):
                    report["errors"].append("Native GPU self-test returned no valid success result.")
        except Exception as error:
            report["errors"].append("Native CUDA: " + str(error))
    return report


def export(root, destination):
    """Exclude user audio, texts, model weights, settings and raw request files."""
    root, destination = Path(root).resolve(), Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("diagnostics.json", json.dumps(snapshot(root), ensure_ascii=False, indent=2))
        manifest = root / "runtime/native/manifest.json"
        if manifest.is_file():
            archive.write(manifest, "native-manifest.json")
        launch_error = root / "diagnostics/launcher-error.log"
        if launch_error.is_file():
            with launch_error.open("rb") as stream:
                stream.seek(max(0, launch_error.stat().st_size - 64 * 1024))
                archive.writestr("launcher-error.log", stream.read())
        # Native timing records have no input text/audio content. Bound the
        # number and size of exported logs rather than exporting the cache.
        folders = sorted((root / ".cache/gguf-sessions").glob("*"), key=lambda p: p.stat().st_mtime,
                         reverse=True)[:3]
        for folder in folders:
            for name in ("native.log", "stderr.log"):
                file = folder / name
                if file.is_file():
                    with file.open("rb") as stream:
                        stream.seek(max(0, file.stat().st_size - 256 * 1024))
                        archive.writestr(f"sessions/{folder.name}/{name}", stream.read())
    return destination


def select_python_device():
    import torch
    choices = []
    for index in range(torch.cuda.device_count()):
        cc = torch.cuda.get_device_capability(index)
        if policy_for(cc).supported:
            free, _ = torch.cuda.mem_get_info(index)
            choices.append((free, index))
    if not choices:
        raise RuntimeError("No supported NVIDIA CUDA GPU found. RTX20/SM75 or newer and a compatible driver are required.")
    return "cuda:" + str(max(choices)[1])
