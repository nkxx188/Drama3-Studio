"""Check archive contents, CRC and clean extraction of the Windows runtime."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile

from package_windows import archive_files

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=ROOT / "dist/Drama3 Windows NVIDIA")
    parser.add_argument("--extract-to", type=Path, default=ROOT / "dist/archive-check")
    args = parser.parse_args()
    folder, extracted = args.folder.resolve(), args.extract_to.resolve()
    if not extracted.is_relative_to(ROOT / "dist") or extracted.exists():
        raise ValueError("Use a fresh extraction folder inside this project's dist directory")
    archive_path = folder.with_suffix(".zip")
    report = {}
    with archive_path.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    assert archive_path.with_suffix(".zip.sha256").read_text().strip() == digest
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        expected = {file.relative_to(folder.parent).as_posix() for file in archive_files(folder)}
        assert names == expected, (sorted(expected - names)[:10], sorted(names - expected)[:10])
        extracted.mkdir(parents=True)
        for member in archive.infolist():
            if not (extracted / member.filename).resolve().is_relative_to(extracted):
                raise ValueError("Archive member escaped extraction directory")
        # zipfile validates each entry's CRC while reading it to EOF during
        # extraction; a separate full decompression pass is unnecessary.
        archive.extractall(extracted)
        report.update(files=len(names), uncompressed_bytes=sum(i.file_size for i in archive.infolist()),
                      compressed_bytes=archive_path.stat().st_size, sha256=digest, crc_checked_via_extract=True)
    clean = extracted / folder.name
    # The batch path is generated inside the checked workspace. Use cmd's
    # documented outer-quote form so spaces survive without shell interpolation.
    command = f'cmd.exe /d /s /c ""{clean / "start.bat"}" --diagnostics"'
    result = subprocess.run(command, cwd=clean, capture_output=True,
                            env=dict(os.environ, DRAMA_NO_PAUSE="1"),
                            text=True, encoding="utf-8", errors="replace", timeout=120)
    report["clean_doctor_returncode"] = result.returncode
    report["doctor_output"] = result.stdout + result.stderr
    assert result.returncode == 0, report["doctor_output"]
    report["clean_diagnostics"] = json.loads((clean / "diagnostics/startup.json").read_text(encoding="utf-8"))
    assert not report["clean_diagnostics"]["errors"]
    assert report["clean_diagnostics"]["native_self_test"][-1]["passed"]
    assert Path(report["clean_diagnostics"]["executable"]).is_relative_to(clean)
    assert not (clean / "ui_data").exists()
    env = {key: value for key, value in os.environ.items()
           if key.upper() not in ("CC", "CXX", "CUDA_PATH", "CUDA_HOME", "INCLUDE", "LIB", "LIBPATH")
           and not key.upper().startswith("TRITON_")}
    system = Path(os.environ["SystemRoot"])
    env["PATH"] = os.pathsep.join(str(p) for p in
                                 (system / "System32", system, clean / "runtime/native", clean / "runtime/python"))
    env["TRITON_CACHE_DIR"] = str(clean / ".cache/clean-jit")
    env["TEMP"] = env["TMP"] = str(clean / ".cache/tmp")
    (clean / ".cache/tmp").mkdir(parents=True, exist_ok=True)
    smoke = subprocess.run(
        [str(clean / "runtime/python/python.exe"), "-I", str(clean / "tools/verify_python_runtime.py")],
        cwd=clean, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    report["clean_jit_returncode"] = smoke.returncode
    report["clean_jit_output"] = smoke.stdout + smoke.stderr
    assert smoke.returncode == 0, report["clean_jit_output"]
    report.update(status="passed", extracted_root=str(clean), archive=str(archive_path))
    (ROOT / "research/phase37-archive-audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in
                      ("doctor_output", "clean_diagnostics")}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
