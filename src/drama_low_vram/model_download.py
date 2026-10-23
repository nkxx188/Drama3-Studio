"""Download the pinned Drama 3 checkpoint with bounded, resumable HTTP ranges."""

from __future__ import annotations

import argparse
import concurrent.futures
from contextlib import contextmanager
import hashlib
import json
import os
import re
import ssl
import threading
import time
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path


MODEL = "fishaudio/drama-3"
REVISION = "f693e8e10a346fc9114c46624f3735bcb3a7bc1b"
HF_REVISION = "e0a9218aa48888270f06ef1ed9ea67622c0e17d2"
API = f"https://modelscope.cn/api/v1/models/{MODEL}/repo"
ROOT = Path(__file__).resolve().parents[2]
CONTEXT = ssl.create_default_context()
RANGE_PATTERN = re.compile(r"bytes (\d+)-(\d+)/(\d+)")


def open_url(url: str, headers: dict[str, str] | None = None):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Drama-Low-VRAM/0.1", **(headers or {})},
    )
    return urllib.request.urlopen(request, timeout=30, context=CONTEXT)


def manifest() -> list[dict]:
    document = json.loads(
        Path(__file__).with_name("data").joinpath("drama3-manifest.json").read_text(encoding="utf-8"))
    if (document["model"] != MODEL or document["revision"] != REVISION
            or document["equivalent_huggingface_weight_revision"] != HF_REVISION):
        raise RuntimeError("Bundled checkpoint manifest identity mismatch")
    files = document["files"]
    if len([f for f in files if f["name"].endswith(".safetensors")]) != 8:
        raise RuntimeError("Expected all eight Drama 3 safetensors shards")
    return files


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def download_error_details(error):
    if isinstance(error, urllib.error.HTTPError):
        return {"error_kind": "http", "http_status": error.code,
                "error": f"HTTP {error.code}: {error.reason}"}
    reason = getattr(error, "reason", error)
    return {"error_kind": "network" if isinstance(error, (urllib.error.URLError, TimeoutError, OSError))
            else "response", "error": f"{type(error).__name__}: {reason}"}


class DownloadCancelled(RuntimeError):
    """A resumable download was explicitly cancelled."""


class Downloader:
    def __init__(self, destination: Path, files: list[dict], workers: int, chunk: int,
                 *, source="auto", progress=None, cancel_event=None,
                 model=MODEL, revision=REVISION, hf_revision=HF_REVISION):
        if (source not in ("auto", "modelscope", "huggingface")
                or isinstance(workers, bool) or not isinstance(workers, int)
                or not 1 <= workers <= 64
                or isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1
                or not files):
            raise ValueError("Invalid download source, workers or chunk size")
        self.destination = Path(destination).resolve()
        if (not isinstance(model, str) or not re.fullmatch(r"[\w.-]+/[\w.-]+", model)
                or not isinstance(revision, str) or not re.fullmatch(r"[\w.-]+", revision)
                or not isinstance(hf_revision, str) or not re.fullmatch(r"[\w.-]+", hf_revision)):
            raise ValueError("Invalid model repository or revision")
        self.model, self.revision, self.hf_revision = model, revision, hf_revision
        names = set()
        for entry in files:
            name = entry["name"]
            target = (self.destination / name).resolve()
            if (not isinstance(name, str) or not name or target == self.destination
                    or not target.is_relative_to(self.destination) or name in names
                    or ":" in name or "\\" in name
                    or isinstance(entry["size"], bool) or not isinstance(entry["size"], int)
                    or entry["size"] < 1 or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])):
                raise ValueError("Invalid pinned model file")
            names.add(name)
        self.destination.mkdir(parents=True, exist_ok=True)
        self.files = files
        self.workers = workers
        self.chunk = chunk
        self.source = source
        self.selected_source = "modelscope" if source == "auto" else source
        self.progress = progress
        self.cancel_event = cancel_event if cancel_event is not None else threading.Event()
        self.lock = threading.Lock()
        self.urls: dict[str, str] = {}
        self.done_bytes = 0
        # Live bytes are display-only until their complete range can be resumed.
        self.inflight_bytes: dict[tuple[str, int], int] = {}
        self.network_bytes = 0
        self.started = time.monotonic()
        self.last_report = self.started
        self.state_path = self.destination / ".download-state.json"
        self.state = {
            "model": self.model,
            "revision": self.revision,
            "chunk_bytes": chunk,
            "files": {},
        }
        if self.state_path.exists():
            try:
                previous = json.loads(self.state_path.read_text(encoding="utf-8"))
                if (isinstance(previous, dict) and previous.get("revision") == self.revision
                        and previous.get("model") == self.model
                        and previous.get("chunk_bytes") == chunk
                        and isinstance(previous.get("files"), dict)):
                    self.state = previous
            except (OSError, ValueError, TypeError):
                # Partial bytes are never trusted without matching state and
                # the final SHA256 check. A torn state file can be rebuilt.
                pass

    def check_cancelled(self):
        if self.cancel_event.is_set():
            raise DownloadCancelled("Download cancelled; completed ranges are preserved")

    def emit(self, stage, **values):
        if self.progress is not None:
            with self.lock:
                event = {
                    "stage": stage,
                    "done_bytes": self.done_bytes + sum(self.inflight_bytes.values()),
                    "total_bytes": sum(entry["size"] for entry in self.files),
                    "network_bytes": self.network_bytes, "source": self.selected_source,
                    **values,
                }
            self.progress(event)

    def verify_hash(self, path):
        digest, counted, last = hashlib.sha256(), 0, 0.0
        with path.open("rb") as handle:
            while True:
                self.check_cancelled()
                block = handle.read(4 * 1024**2)
                if not block:
                    break
                digest.update(block)
                counted += len(block)
                now = time.monotonic()
                if now - last >= 0.4:
                    self.emit("verifying", file=path.name, file_bytes=counted,
                              file_total_bytes=path.stat().st_size)
                    last = now
        return digest.hexdigest()

    def save_state(self) -> None:
        temporary = self.state_path.with_suffix(".json.tmp")
        self.local_path(temporary.name)
        temporary.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        os.replace(temporary, self.state_path)

    def local_path(self, name):
        target = self.destination / name
        if not target.resolve().is_relative_to(self.destination):
            raise ValueError("Download workspace file resolves outside its chosen directory")
        return target

    @contextmanager
    def exclusive_destination(self):
        # OS locks are released on process exit, including an interrupted
        # download. The small file is not evidence of a running downloader.
        with self.local_path(".download.lock").open("a+b") as handle:
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError:
                    raise RuntimeError("Another downloader is using this model directory") from None
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    raise RuntimeError("Another downloader is using this model directory") from None
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def direct_url(self, entry, source):
        model = entry.get("repository", self.model)
        revision = entry.get("revision", self.revision)
        if source == "huggingface" and entry.get("huggingface_compatible"):
            return (
                f"https://huggingface.co/{model}/resolve/{entry.get('hf_revision', self.hf_revision)}/"
                f"{urllib.parse.quote(entry['name'], safe='/')}?download=true")
        return f"https://modelscope.cn/api/v1/models/{model}/repo?" + urllib.parse.urlencode(
            {"Revision": revision, "FilePath": entry["name"]})

    def source_url(self, entry, attempt=0) -> str:
        name = entry["name"]
        with self.lock:
            if name in self.urls:
                return self.urls[name]
        source = self.selected_source
        if self.source == "auto" and attempt >= 3:
            source = "huggingface" if source == "modelscope" else "modelscope"
        return self.direct_url(entry, source)

    def select_source(self, entry):
        automatic = self.source == "auto" and entry.get("huggingface_compatible")
        count = min(4 * 1024**2, entry["size"]) if automatic else 1

        def probe(source):
            self.check_cancelled()
            started = time.monotonic()
            try:
                with open_url(self.direct_url(entry, source), {
                    "Range": f"bytes=0-{count - 1}", "Accept-Encoding": "identity",
                }) as response:
                    expected = f"bytes 0-{count - 1}/{entry['size']}"
                    partial = response.status == 206 and response.headers.get("Content-Range") == expected
                    complete = response.status == 200 and count == entry["size"]
                    if not (partial or complete):
                        raise RuntimeError("Source probe did not honor the bounded byte range")
                    data = response.read(count + 1)
                    self.check_cancelled()
                    if len(data) != count:
                        raise RuntimeError("Source probe returned an incorrect byte count")
                    elapsed = time.monotonic() - started
                    return {"source": source, "seconds": elapsed, "bytes": len(data),
                            "sample_sha256": hashlib.sha256(data).hexdigest()}
            except DownloadCancelled:
                raise
            except Exception as exc:
                return {"source": source, **download_error_details(exc)}

        self.emit("source_probe" if automatic else "connecting")
        if automatic:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                probes = list(pool.map(probe, ("modelscope", "huggingface")))
        else:
            probes = [probe(self.selected_source)]
        good = [row for row in probes if "seconds" in row]
        if not good:
            failure = probes[0]
            self.emit("source_error", file=entry["name"], probes=probes,
                      **{key: failure[key] for key in ("error_kind", "http_status", "error")
                         if key in failure})
            raise RuntimeError(
                f"Cannot access download source for {entry['name']}: "
                + "; ".join(f"{row['source']}: {row['error']}" for row in probes))
        if len(good) == 2 and good[0]["sample_sha256"] != good[1]["sample_sha256"]:
            raise RuntimeError("Pinned sources returned different model bytes")
        if good:
            self.selected_source = min(good, key=lambda row: row["seconds"])["source"]
        self.emit("source_selected", probes=probes)

    def partial_path(self, name: str) -> Path:
        return self.local_path(name + ".partial")

    def download_chunk(self, entry: dict, index: int) -> None:
        name, size = entry["name"], entry["size"]
        transfer = (name, index)
        start = index * self.chunk
        end = min(size, start + self.chunk) - 1
        expected = end - start + 1
        for attempt in range(6):
            self.check_cancelled()
            received = 0
            try:
                with open_url(
                    self.source_url(entry, attempt),
                    {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                ) as response:
                    if response.status == 206:
                        match = RANGE_PATTERN.fullmatch(
                            response.headers.get("Content-Range", "")
                        )
                        if not match or tuple(map(int, match.groups())) != (start, end, size):
                            raise RuntimeError(f"Incorrect Content-Range for {name}")
                    elif not (response.status == 200 and start == 0 and expected == size):
                        raise RuntimeError(f"Server ignored bounded Range for {name}")
                    length = response.headers.get("Content-Length")
                    if length is not None and int(length) != expected:
                        raise RuntimeError(f"Incorrect response length for {name}")
                    # CDN URLs contain short-lived signatures. Keep them in memory only.
                    with self.lock:
                        self.urls[name] = response.geturl()
                    read = getattr(response, "read1", response.read)
                    with self.partial_path(name).open("r+b", buffering=0) as output:
                        output.seek(start)
                        while received < expected:
                            self.check_cancelled()
                            block = read(min(1024 * 1024, expected - received))
                            if not block:
                                raise OSError(f"Incomplete range for {name}")
                            output.write(block)
                            received += len(block)
                            with self.lock:
                                first_data = self.network_bytes == 0
                                self.network_bytes += len(block)
                                self.inflight_bytes[transfer] = received
                            self.report(force=first_data)
                        if response.read(1):
                            raise RuntimeError(f"Response exceeded requested range for {name}")
                with self.lock:
                    completed = self.state["files"][name]["completed_chunks"]
                    self.inflight_bytes.pop(transfer, None)
                    if index not in completed:
                        completed.append(index)
                        self.done_bytes += expected
                    self.save_state()
                return
            except DownloadCancelled:
                with self.lock:
                    self.inflight_bytes.pop(transfer, None)
                self.report(force=True)
                raise
            except Exception as error:
                with self.lock:
                    self.inflight_bytes.pop(transfer, None)
                    self.urls.pop(name, None)
                self.report(force=True)
                if attempt == 5:
                    self.emit("source_error", file=name, **download_error_details(error))
                    raise RuntimeError(
                        f"Failed {name} chunk {index}: {download_error_details(error)['error']}"
                    ) from None
                print(
                    f"Retry {name} chunk {index}, attempt {attempt + 2}/6 "
                    f"({type(error).__name__})",
                    flush=True,
                )
                if self.cancel_event.wait(min(8, 2**attempt)):
                    self.check_cancelled()

    def report(self, force: bool = False) -> None:
        now = time.monotonic()
        with self.lock:
            if not force and now - self.last_report < (0.4 if self.progress else 10):
                return
            self.last_report = now
            done = self.done_bytes + sum(self.inflight_bytes.values())
            network = self.network_bytes
        total = sum(entry["size"] for entry in self.files)
        elapsed = max(0.001, now - self.started)
        speed = network / elapsed / (1024**2)
        self.emit("downloading" if network or done else "connecting",
                  speed_mib_s=speed, elapsed_seconds=elapsed)
        if self.progress is not None:
            return
        print(
            f"Download {done / 1024**3:.2f}/{total / 1024**3:.2f} GiB "
            f"({done / total:.1%}); average network {speed:.1f} MiB/s; "
            f"elapsed {elapsed / 60:.1f} min",
            flush=True,
        )

    def prepare(self) -> list[tuple[dict, int]]:
        tasks = []
        for entry in self.files:
            self.check_cancelled()
            name, size, digest = entry["name"], entry["size"], entry["sha256"]
            target = self.destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                print(f"Verifying existing {name}", flush=True)
                if target.stat().st_size == size and self.verify_hash(target) == digest:
                    self.done_bytes += size
                    continue
                raise RuntimeError(f"Existing {name} does not match pinned checkpoint")
            partial = self.partial_path(name)
            previous = self.state["files"].get(name)
            matching = (
                isinstance(previous, dict)
                and previous.get("size") == size
                and previous.get("sha256") == digest
                and partial.exists()
                and partial.stat().st_size == size
                and isinstance(previous.get("completed_chunks"), list)
                and all(isinstance(index, int) and not isinstance(index, bool)
                        and 0 <= index < (size + self.chunk - 1) // self.chunk
                        for index in previous["completed_chunks"])
            )
            if not matching:
                with partial.open("wb") as output:
                    output.truncate(size)
                previous = {"size": size, "sha256": digest, "completed_chunks": []}
                self.state["files"][name] = previous
            completed = set(previous["completed_chunks"])
            count = (size + self.chunk - 1) // self.chunk
            for index in range(count):
                if index in completed:
                    self.done_bytes += min(self.chunk, size - index * self.chunk)
                else:
                    tasks.append((entry, index))
        with self.lock:
            self.save_state()
        return tasks

    def run(self) -> None:
        with self.exclusive_destination():
            self._run()

    def _run(self) -> None:
        tasks = self.prepare()
        if tasks:
            self.select_source(max((entry for entry, _ in tasks), key=lambda entry: entry["size"]))
        self.report(force=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [
                pool.submit(self.download_chunk, entry, index)
                for entry, index in tasks
            ]
            try:
                for future in concurrent.futures.as_completed(futures):
                    future.result()
                    self.report()
            except BaseException:
                self.cancel_event.set()
                for future in futures:
                    future.cancel()
                raise
        results = []
        for entry in self.files:
            self.check_cancelled()
            target = self.destination / entry["name"]
            partial = self.partial_path(entry["name"])
            if not target.exists():
                print(f"SHA256 verification: {entry['name']}", flush=True)
                actual = self.verify_hash(partial)
                if actual != entry["sha256"]:
                    with self.lock:
                        self.state["files"][entry["name"]]["completed_chunks"] = []
                        self.save_state()
                    raise RuntimeError(
                        f"SHA256 mismatch for {entry['name']}; rerun to download it again"
                    )
                os.replace(partial, target)
            results.append({**entry, "verified": True})
        verification = {
            "model": self.model,
            "source": self.selected_source,
            "source_requested": self.source,
            "revision": self.revision,
            "equivalent_huggingface_weight_revision": self.hf_revision,
            "total_bytes": sum(entry["size"] for entry in self.files),
            "files": results,
        }
        target = self.local_path("download-verification.json")
        temporary = self.local_path("download-verification.json.tmp")
        temporary.write_text(json.dumps(verification, indent=2), encoding="utf-8")
        os.replace(temporary, target)
        self.report(force=True)
        self.emit("complete")
        print(f"Complete: all {len(results)} files passed SHA256 verification.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "checkpoints" / "drama-3")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-mib", type=int, default=64)
    parser.add_argument("--source", choices=("auto", "modelscope", "huggingface"), default="auto")
    args = parser.parse_args()
    if not 1 <= args.workers <= 64 or not 1 <= args.chunk_mib <= 512:
        parser.error("workers must be 1..64 and chunk-mib 1..512")
    destination = args.directory.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    files = manifest()
    print(
        f"Source: {args.source}; revision: {REVISION}; "
        f"{len(files)} files, {sum(f['size'] for f in files) / 1e9:.2f} GB; "
        f"{args.workers} workers x {args.chunk_mib} MiB",
        flush=True,
    )
    Downloader(destination, files, args.workers, args.chunk_mib * 1024**2,
               source=args.source).run()


if __name__ == "__main__":
    main()
