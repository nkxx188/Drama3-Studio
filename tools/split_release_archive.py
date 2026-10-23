"""Split a completed ZIP into numbered volumes without recompressing it."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


PART_BYTES = 1900 * 1024**2


def split_archive(archive, destination, *, part_bytes=PART_BYTES):
    archive, destination = Path(archive).resolve(), Path(destination).resolve()
    if (isinstance(part_bytes, bool) or not isinstance(part_bytes, int)
            or not 1 <= part_bytes <= PART_BYTES):
        raise ValueError("Release parts must be between 1 byte and 1900 MiB")
    if not archive.is_file() or not zipfile.is_zipfile(archive):
        raise ValueError("Select a completed ZIP archive")
    # A fresh directory prevents older volumes being mixed into this release.
    destination.mkdir(parents=True, exist_ok=False)
    whole = hashlib.sha256()
    parts = []
    with archive.open("rb") as source:
        index = 1
        while block := source.read(min(8 * 1024**2, part_bytes)):
            target = destination / f"{archive.name}.{index:03d}"
            digest, written = hashlib.sha256(), 0
            with target.open("xb") as output:
                while block:
                    output.write(block)
                    digest.update(block)
                    whole.update(block)
                    written += len(block)
                    if written == part_bytes:
                        break
                    block = source.read(min(8 * 1024**2, part_bytes - written))
            parts.append({"name": target.name, "bytes": written, "sha256": digest.hexdigest()})
            print(f"Created {target.name}: {written / 1024**2:.2f} MiB", flush=True)
            index += 1
    report = {
        "schema_version": 1, "archive": archive.name, "bytes": archive.stat().st_size,
        "sha256": whole.hexdigest(), "parts": parts,
        "extract": "Keep every numbered volume in one folder and extract .zip.001 with 7-Zip.",
    }
    (destination / "release-parts.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (destination / "SHA256SUMS.txt").write_text(
        "".join(f"{part['sha256']}  {part['name']}\n" for part in parts), encoding="ascii")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True,
                        help="A new folder for this release's volumes and checksums")
    args = parser.parse_args()
    report = split_archive(args.archive, args.output)
    print(json.dumps({"archive_sha256": report["sha256"], "parts": len(report["parts"]),
                      "output": str(args.output.resolve())}, indent=2), flush=True)


if __name__ == "__main__":
    main()
