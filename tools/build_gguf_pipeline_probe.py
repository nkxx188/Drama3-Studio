"""Build the opt-in pipeline experiment separately from the GGUF service."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

from build_native import windows_tools

ROOT = Path(__file__).resolve().parents[1]


def main():
    build = json.loads((ROOT / "research/phase35-worker-build.json").read_text())
    env, *_ = windows_tools()
    env = {k.upper(): v for k, v in env.items()}
    kit = Path(build["cuda_toolkit"])
    env.update(TEMP=str(ROOT / ".cache/tmp"), TMP=str(ROOT / ".cache/tmp"), CUDA_PATH=str(kit),
               CUDAToolkit_ROOT=str(kit))
    env["PATH"] = str(kit / "bin") + os.pathsep + env["PATH"]
    directory = Path(build["executable"]).parent.parent
    with (ROOT / "research/phase35-pipeline-build.log").open("w", encoding="utf-8") as out:
        subprocess.run([
            build["cmake"], "-S", build["source"], "-B", str(directory),
            "-DDRAMA_PIPELINE_PROBE_SOURCE:FILEPATH=" + (ROOT / "native/gguf_pipeline_probe.cpp").as_posix(),
        ], env=env, stdout=out, stderr=subprocess.STDOUT, check=True)
        subprocess.run([build["cmake"], "--build", str(directory), "--target", "drama3_pipeline_probe", "--parallel", "2"],
                       env=env, stdout=out, stderr=subprocess.STDOUT, check=True)
    executable = directory / "bin/drama3_pipeline_probe.exe"
    receipt = build | {"executable": str(executable),
                       "sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
                       "source_sha256": hashlib.sha256((ROOT / "native/gguf_pipeline_probe.cpp").read_bytes()).hexdigest()}
    (ROOT / "research/phase35-pipeline-build.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
