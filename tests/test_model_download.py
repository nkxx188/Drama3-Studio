"""Real bounded HTTP ranges, interrupted resumes and immutable target files."""

from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from threading import Event, Thread
import time
from urllib.error import HTTPError, URLError

import pytest

from drama_low_vram.model_download import Downloader, DownloadCancelled, manifest
from drama_low_vram.model_installation import ModelInstallation, inspect_checkpoint


def entry(name, data, *, compatible=True):
    return {
        "name": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
        "huggingface_compatible": compatible,
    }


@contextmanager
def http_ranges(data, *, wrong_range=False, stream_gate=None, truncate_first=False):
    requests = []
    truncated = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            match = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
            if not match:
                self.send_error(400)
                return
            start, end = map(int, match.groups())
            requests.append((self.path, start, end))
            block = data[start:end + 1]
            self.send_response(206)
            actual_start = start + 1 if wrong_range else start
            self.send_header("Content-Range", f"bytes {actual_start}-{end}/{len(data)}")
            self.send_header("Content-Length", str(len(block)))
            self.end_headers()
            if truncate_first and len(block) > 1 and not truncated:
                truncated.append(True)
                self.wfile.write(block[:len(block) // 2])
                return
            if stream_gate is not None and len(block) > 1:
                split = min(256 * 1024, len(block) // 2)
                self.wfile.write(block[:split])
                self.wfile.flush()
                if not stream_gate.wait(timeout=10):
                    return
                self.wfile.write(block[split:])
                return
            self.wfile.write(block)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def local_downloader(tmp_path, data, base, **kwargs):
    downloader = Downloader(tmp_path, [entry("weight.bin", data)], 1, 1024, **kwargs)
    downloader.direct_url = lambda _entry, source: f"{base}/{source}"
    return downloader


def test_actual_http_ranges_and_complete_sha256_cover_every_byte(tmp_path):
    data = bytes(range(256)) * 32
    events = []
    with http_ranges(data) as (base, requests):
        downloader = local_downloader(tmp_path, data, base, source="auto", progress=events.append)
        downloader.run()
    assert (tmp_path / "weight.bin").read_bytes() == data
    assert not (tmp_path / "weight.bin.partial").exists()
    assert {path for path, _start, _end in requests} == {"/modelscope", "/huggingface"}
    assert all(0 <= start <= end < len(data) for _path, start, end in requests)
    verified = json.loads((tmp_path / "download-verification.json").read_text())
    assert verified["files"][0]["sha256"] == hashlib.sha256(data).hexdigest()
    assert events[-1]["stage"] == "complete"
    assert any(event["stage"] == "source_selected" for event in events)
    assert downloader.done_bytes == len(data)


def test_progress_updates_before_the_first_range_finishes_without_committing_it(tmp_path):
    data = bytes(range(256)) * 8192
    release, partial_received = Event(), Event()
    events, errors = [], []

    def progress(event):
        events.append(event)
        if event["stage"] == "downloading" and 0 < event["done_bytes"] < len(data):
            partial_received.set()

    with http_ranges(data, stream_gate=release) as (base, _requests):
        downloader = Downloader(
            tmp_path, [entry("weight.bin", data)], 1, len(data),
            source="modelscope", progress=progress)
        downloader.direct_url = lambda _entry, source: f"{base}/{source}"

        def run():
            try:
                downloader.run()
            except Exception as error:
                errors.append(error)

        thread = Thread(target=run, daemon=True)
        thread.start()
        try:
            assert partial_received.wait(timeout=5)
            assert not (tmp_path / "weight.bin").exists()
            assert downloader.done_bytes == 0
            saved = json.loads((tmp_path / ".download-state.json").read_text())
            assert saved["files"]["weight.bin"]["completed_chunks"] == []
        finally:
            release.set()
            thread.join(timeout=5)
        assert not thread.is_alive()
        assert not errors
    assert (tmp_path / "weight.bin").read_bytes() == data
    assert events[-1]["stage"] == "complete"
    assert events[-1]["done_bytes"] == len(data)
    assert not downloader.inflight_bytes


def test_retry_does_not_count_incomplete_range_bytes_twice(tmp_path):
    data = bytes(range(256)) * 8192
    events = []
    with http_ranges(data, truncate_first=True) as (base, requests):
        downloader = Downloader(
            tmp_path, [entry("weight.bin", data)], 1, len(data),
            source="modelscope", progress=events.append)
        downloader.direct_url = lambda _entry, source: f"{base}/{source}"
        downloader.run()
    assert len(requests) == 3
    assert downloader.network_bytes > len(data)
    assert all(0 <= event["done_bytes"] <= len(data) for event in events)
    assert events[-1]["done_bytes"] == downloader.done_bytes == len(data)
    assert not downloader.inflight_bytes
    assert (tmp_path / "weight.bin").read_bytes() == data


def test_cancelled_transfer_resumes_only_completed_chunks_then_verifies(tmp_path):
    data = bytes(range(256)) * 16
    cancel = Event()
    with http_ranges(data) as (base, _requests):
        downloader = local_downloader(tmp_path, data, base, source="modelscope", cancel_event=cancel)
        downloader.prepare()
        downloader.download_chunk(downloader.files[0], 0)
        cancel.set()
        with pytest.raises(DownloadCancelled):
            downloader.download_chunk(downloader.files[0], 1)
        state = json.loads((tmp_path / ".download-state.json").read_text())
        assert state["files"]["weight.bin"]["completed_chunks"] == [0]
    with http_ranges(data) as (base, requests):
        resumed = local_downloader(tmp_path, data, base, source="modelscope")
        resumed.run()
    assert {start for _path, start, end in requests if end > start} == {1024, 2048, 3072}
    assert (tmp_path / "weight.bin").read_bytes() == data


def test_full_hash_mismatch_never_publishes_a_model_file(tmp_path):
    expected = b"a" * 4096
    with http_ranges(b"b" * 4096) as (base, _requests):
        downloader = local_downloader(tmp_path, expected, base, source="modelscope")
        with pytest.raises(RuntimeError, match="SHA256 mismatch"):
            downloader.run()
    assert not (tmp_path / "weight.bin").exists()
    state = json.loads((tmp_path / ".download-state.json").read_text())
    assert state["files"]["weight.bin"]["completed_chunks"] == []


def test_existing_different_target_is_preserved_and_never_redownloaded(tmp_path):
    existing = tmp_path / "weight.bin"
    existing.write_bytes(b"keep me")
    downloader = Downloader(tmp_path, [entry("weight.bin", b"expected")], 1, 4)
    downloader.select_source = lambda _entry: pytest.fail("Invalid existing targets must stop before networking")
    with pytest.raises(RuntimeError, match="does not match"):
        downloader.run()
    assert existing.read_bytes() == b"keep me"


def test_invalid_range_fails_a_probe_without_accepting_bytes(tmp_path):
    data = b"x" * 1024
    events = []
    with http_ranges(data, wrong_range=True) as (base, _requests):
        downloader = local_downloader(tmp_path, data, base, source="auto", progress=events.append)
        with pytest.raises(RuntimeError, match="Cannot access download source"):
            downloader.select_source(downloader.files[0])
    probes = events[-1]["probes"]
    assert all("error" in probe for probe in probes)
    assert not (tmp_path / "weight.bin").exists()


@pytest.mark.parametrize("error,kind,status", [
    (URLError(TimeoutError("connection timed out")), "network", None),
    (HTTPError("https://example.com/model", 404, "Not Found", None, None), "http", 404),
])
def test_source_probe_distinguishes_network_failures_from_missing_files(
        tmp_path, monkeypatch, error, kind, status):
    events = []

    def fail_request(*_args, **_kwargs):
        raise error

    monkeypatch.setattr("drama_low_vram.model_download.open_url", fail_request)
    downloader = Downloader(tmp_path, [entry("weights.bin", b"x")], 16, 64 * 1024**2,
                            source="huggingface", progress=events.append)
    with pytest.raises(RuntimeError, match="Cannot access download source"):
        downloader.run()
    assert events[-1]["stage"] == "source_error"
    assert events[-1]["error_kind"] == kind
    assert events[-1].get("http_status") == status
    assert events[-1]["network_bytes"] == 0
    assert events[-1]["done_bytes"] == 0
    assert str(getattr(error, "reason", error)) in events[-1]["error"]
    assert not (tmp_path / "weights.bin").exists()


@pytest.mark.parametrize("name", ["../outside", "/outside", "C:/outside", "nested/../../outside"])
def test_pinned_paths_cannot_escape_the_download_directory(tmp_path, name):
    with pytest.raises(ValueError):
        Downloader(tmp_path, [entry(name, b"x")], 1, 1)


def test_packaged_manifest_has_all_shards_and_correct_source_compatibility():
    files = manifest()
    assert len(files) == 19
    assert sum(row["name"].endswith(".safetensors") for row in files) == 8
    assert sum(row["size"] for row in files) == 41_502_594_526
    indexed = {row["name"]: row for row in files}
    assert indexed["codec.pth"]["huggingface_compatible"]
    assert not indexed["README.md"]["huggingface_compatible"]
    assert not indexed["configuration.json"]["huggingface_compatible"]


def test_folder_inspection_is_metadata_only_and_missing_paths_are_not_created(tmp_path):
    data = b"metadata"
    model = tmp_path / "model"
    files = [entry("config.json", data)]
    assert not inspect_checkpoint(model, files=files)["ready"]
    assert not model.exists()
    model.mkdir()
    (model / "config.json").write_bytes(data)
    assert inspect_checkpoint(model, files=files)["ready"]
    (model / "config.json").write_bytes(b"mismatch")
    inspected = inspect_checkpoint(model, files=files)
    assert not inspected["ready"] and inspected["invalid"]


def wait_installation(installation):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = installation.snapshot()
        if result["status"] not in ("queued", "running", "cancelling"):
            return result
        time.sleep(0.01)
    pytest.fail("The small local installation job did not terminate")


def test_background_verification_persists_completion_and_rejects_corrupt_bytes(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    data = b"verified weights"
    (model / "weight.bin").write_bytes(data)
    installation = ModelInstallation(tmp_path / "ui_data", files=[entry("weight.bin", data)])
    try:
        installation.start("verify", model)
        result = wait_installation(installation)
        assert result["status"] == "complete"
        assert result["progress"]["stage"] == "complete"
        assert installation.state_file.is_file()
        (model / "weight.bin").write_bytes(b"invalid weights")
        installation.start("verify", model)
        assert wait_installation(installation)["status"] == "error"
        assert (model / "weight.bin").read_bytes() == b"invalid weights"
    finally:
        installation.close()


def test_interrupted_model_job_is_recovered_without_automatic_network_access(tmp_path):
    (tmp_path / "model-task.json").write_text(
        '{"id":"old","status":"running","operation":"download"}', encoding="utf-8")
    installation = ModelInstallation(tmp_path, files=[entry("file.bin", b"x")])
    try:
        assert installation.snapshot()["status"] == "interrupted"
    finally:
        installation.close()


def test_process_lock_refuses_a_second_downloader_and_recovers_when_released(tmp_path):
    one = Downloader(tmp_path, [entry("weight.bin", b"x")], 1, 1)
    two = Downloader(tmp_path, [entry("weight.bin", b"x")], 1, 1)
    with one.exclusive_destination():
        with pytest.raises(RuntimeError, match="Another downloader"):
            with two.exclusive_destination():
                pytest.fail("The second writer acquired an occupied destination")
    with two.exclusive_destination():
        assert (tmp_path / ".download.lock").is_file()


def test_cancellation_during_full_hash_cannot_publish_completion(tmp_path):
    data = b"weights"
    target = tmp_path / "weight.bin"
    target.write_bytes(data)
    cancel = Event()
    events = []

    def progress(value):
        events.append(value["stage"])
        if value["stage"] == "verifying":
            cancel.set()

    downloader = Downloader(
        tmp_path, [entry("weight.bin", data)], 1, 4, progress=progress, cancel_event=cancel)
    with pytest.raises(DownloadCancelled):
        downloader.run()
    assert "complete" not in events and target.read_bytes() == data
    assert not (tmp_path / "download-verification.json").exists()


def test_cancel_a_queued_installation_does_not_create_its_destination(tmp_path):
    installation = ModelInstallation(tmp_path / "state", files=[entry("weight.bin", b"x")])
    release, entered = Event(), Event()
    def busy():
        entered.set()
        release.wait(timeout=5)
    installation._executor.submit(busy)
    assert entered.wait(timeout=5)
    try:
        installation.start("download", tmp_path / "new-model")
        result = installation.cancel()
        assert result["status"] == "cancelled"
        assert not (tmp_path / "new-model").exists()
    finally:
        release.set()
        installation.close()


def test_background_installation_downloads_and_verifies_through_the_shared_range_engine(tmp_path, monkeypatch):
    data = b"local-range-test" * 2048
    with http_ranges(data) as (base, _requests):
        class LocalDownloader(Downloader):
            def direct_url(self, _entry, source):
                return f"{base}/{source}"

        monkeypatch.setattr("drama_low_vram.model_installation.Downloader", LocalDownloader)
        installation = ModelInstallation(tmp_path / "state", files=[entry("weight.bin", data)])
        try:
            installation.start("download", tmp_path / "download", chunk_mib=1)
            result = wait_installation(installation)
            assert result["status"] == "complete"
            assert result["inspection"]["ready"]
            assert (tmp_path / "download/weight.bin").read_bytes() == data
        finally:
            installation.close()


def test_shared_model_verification_reads_external_payload_without_copying_it(tmp_path):
    import json

    model = tmp_path / "descriptor"
    model.mkdir()
    shared = tmp_path / "comfy-model.safetensors"
    payload = b"shared verified tensor payload"
    shared.write_bytes(payload)
    for name in ("config.json", "tokenizer.json", "codec.pth", "control-head.safetensors"):
        (model / name).write_bytes(b"metadata")
    document = {
        "format": "drama3-packed-shared-v1", "model": str(shared),
        "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
    }
    (model / "shared-model.json").write_text(json.dumps(document), encoding="utf-8")
    installation = ModelInstallation(tmp_path / "state")
    try:
        assert inspect_checkpoint(model)["ready"]
        installation.start("verify", model)
        result = wait_installation(installation)
        assert result["status"] == "complete"
        assert result["progress"]["done_bytes"] == len(payload)
        assert result["progress"]["total_bytes"] == len(payload)
        assert not (model / shared.name).exists()
        shared.write_bytes(b"X" * len(payload))
        installation.start("verify", model)
        assert wait_installation(installation)["status"] == "error"
    finally:
        installation.close()
