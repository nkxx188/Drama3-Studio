"""Build the separate persistent worker without relinking existing CLIs."""

import hashlib
import difflib
import json
import os
from pathlib import Path
import subprocess

from build_native import windows_tools

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    baseline = json.loads((ROOT / "research/phase19-audio-cpp-build.json").read_text())
    cli = Path(baseline["executable"])
    identities = {str(p): sha(p) for p in cli.parent.glob("audiocpp_cli*.exe")}
    env, _, _, _ = windows_tools()
    env = {k.upper(): v for k, v in env.items()}
    kit = Path(baseline["cuda_toolkit"])
    env.update(TEMP=str(ROOT / ".cache/tmp"), TMP=str(ROOT / ".cache/tmp"), CUDA_PATH=str(kit),
               CUDAToolkit_ROOT=str(kit), VSLANG="1033")
    env["PATH"] = str(kit / "bin") + os.pathsep + env["PATH"]
    build = cli.parent.parent
    log = ROOT / "research/phase35-worker-build.log"
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run([
            baseline["cmake"], "-S", baseline["source"], "-B", str(build),
            "-DDRAMA_WORKER_SOURCE:FILEPATH=" + (ROOT / "native/gguf_worker.cpp").as_posix(),
        ], env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        subprocess.run([baseline["cmake"], "--build", str(build), "--target", "drama3_worker", "--parallel", "2"],
                       env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    for name, before in identities.items():
        if sha(Path(name)) != before:
            raise RuntimeError("Existing CLI changed: " + name)
    executable = cli.with_name("drama3_worker.exe")
    source = Path(baseline["source"])
    patch = subprocess.check_output(["git", "diff", "--", "CMakeLists.txt"], cwd=source)
    (ROOT / "research/phase35-worker-cmake.patch").write_bytes(patch)
    modified = [
        "CMakeLists.txt", "src/community_models/fish_drama3/backbone.cpp",
        "src/community_models/fish_drama3/generator.cpp", "src/community_models/fish_drama3/hybrid_decode.inc",
        "include/engine/framework/core/disk_expert_weight_store.h",
        "src/framework/core/disk_expert_weight_store.cpp",
    ]
    # Full current vendor delta from pinned upstream, including prerequisites.
    portable = ROOT / "research/phase35-native-runtime.patch"
    delta = subprocess.check_output(["git", "diff", "--binary"], cwd=source)
    for name in subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], cwd=source).decode().splitlines():
        file = source / name
        if file.is_file():
            patch = f"diff --git a/{name} b/{name}\nnew file mode 100644\n" + "".join(
                difflib.unified_diff([], file.read_text(encoding="utf-8").splitlines(keepends=True),
                                     fromfile="/dev/null", tofile="b/" + name))
            delta += patch.encode("utf-8")
    portable.write_bytes(delta)
    snapshot = ROOT / ".cache/phase35-native-snapshots" / sha(executable)[:16]
    snapshot.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(executable, snapshot / executable.name)
    receipt = baseline | {
        "executable": str(executable), "sha256": sha(executable),
        "worker_source": str(ROOT / "native/gguf_worker.cpp"),
        "worker_source_sha256": sha(ROOT / "native/gguf_worker.cpp"),
        "existing_cli_hashes": identities,
        "snapshot_executable": str(snapshot / executable.name),
        "full_upstream_delta": str(portable),
        "full_upstream_delta_sha256": sha(portable),
        "modified_sources": {name: sha(source / name) for name in modified},
        "scope": "Persistent serial worker; CPU expert computation disabled by service planner",
        "prerequisites": ["phase29-audio-cpp-host-embedding.patch", "phase30-audio-cpp-codec.patch",
                          "phase33-native-hybrid/native-hybrid.patch"],
    }
    (ROOT / "research/phase35-worker-build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
