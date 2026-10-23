"""Native-entry resource defaults and bounded allocation failure recovery."""

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def cli(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "tools/run_drama_gguf.py"
    spec = spec_from_file_location("testable_drama_gguf_cli", source)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    research = tmp_path / "research"
    research.mkdir()
    (research / "phase19-audio-cpp-build.json").write_text(json.dumps({
        "executable": "native.exe", "source": str(tmp_path), "cuda_toolkit": str(tmp_path),
    }))
    (tmp_path / "model.gguf").write_bytes(b"verified-placeholder")
    (research / "phase19-gguf-q4-quality-download-pinned.json").write_text(json.dumps({
        "destination": str(tmp_path),
        "files": [{"name": "model.gguf", "size": (tmp_path / "model.gguf").stat().st_size}],
    }))
    monkeypatch.setattr(module.psutil, "virtual_memory", lambda: SimpleNamespace(available=32 * module.GIB))
    monkeypatch.setattr(module.psutil, "cpu_count", lambda **_: 4)
    return module


def test_storage_uses_selected_model_size_and_threads_follow_available_cpu(cli):
    assert cli.basic_settings(24 * cli.GIB, 32 * cli.GIB, 4)["host_storage"] == "pin"
    assert cli.basic_settings(24 * cli.GIB, 24 * cli.GIB, 4)["host_storage"] == "mapped"
    assert cli.basic_settings(40 * cli.GIB, 32 * cli.GIB, 4)["host_storage"] == "mapped"
    assert cli.basic_settings(24 * cli.GIB, 32 * cli.GIB, 2)["threads"] == 2
    assert cli.basic_settings(24 * cli.GIB, 32 * cli.GIB, 32)["threads"] == 8
    assert cli.basic_settings(24 * cli.GIB, 32 * cli.GIB, None)["threads"] == 1
    assert cli.basic_settings(24 * cli.GIB, 32 * cli.GIB, 2, threads=12)["threads"] == 12


def test_native_backend_actual_pinned_allocation_messages_allow_fallback(cli):
    assert cli.allocation_failure("cannot allocate pinned host expert banks")
    assert cli.allocation_failure("cannot allocate pinned expert upload staging buffer")
    assert not cli.allocation_failure("GGUF expert bank not found: model_weights")


@pytest.mark.parametrize("decoder", ["same", "cpu", "cuda"])
def test_worker_entry_uses_auto_resources_and_closes_session(cli, tmp_path, monkeypatch, decoder):
    import drama_low_vram.gguf_runtime as runtime
    calls = []
    class FakeWorker:
        def __init__(self, model, **kwargs):
            calls.append((model, kwargs))
        def generate(self, text, output, **kwargs):
            Path(output).write_bytes(b"worker-wave")
            return {"complete": True, "status": "complete", "request_seconds": 1.0}
        def close(self):
            calls.append("closed")
    monkeypatch.setattr(runtime, "GGUFEngine", FakeWorker)
    output = tmp_path / "worker.wav"
    monkeypatch.setattr("sys.argv", ["gguf", "--runtime", "worker", "--text", "hello",
                                     "--decoder", decoder, "--output", str(output)])
    cli.main()
    assert calls[0][1]["pipeline_width"] == 0
    assert calls[0][1]["storage"] == "auto"
    assert calls[0][1]["decoder"] == ("auto" if decoder == "same" else decoder)
    assert calls[0][1]["device"] is None
    assert calls[-1] == "closed"
    assert json.loads(output.with_suffix(".json").read_text())["complete"]


def test_custom_gguf_file_uses_shared_worker_policy_without_a_variant_manifest(cli, tmp_path, monkeypatch):
    import drama_low_vram.gguf_runtime as runtime
    model = tmp_path / "custom-compatible.gguf"
    model.write_bytes(b"custom")
    (tmp_path / "research/phase19-gguf-q4-quality-download-pinned.json").unlink()
    calls = []
    class FakeWorker:
        def __init__(self, selected, **kwargs):
            calls.append((selected, kwargs))
        def generate(self, text, output, **kwargs):
            Path(output).write_bytes(b"wave")
            return {"complete": True}
        def close(self):
            pass
    monkeypatch.setattr(runtime, "GGUFEngine", FakeWorker)
    monkeypatch.setattr("sys.argv", ["gguf", "--runtime", "worker", "--model", str(model),
                                     "--text", "hello", "--output", str(tmp_path / "custom.wav")])
    cli.main()
    assert calls[0][0] == model
    assert calls[0][1]["decoder"] == "auto"


def test_q8_selects_its_own_validated_model_without_changing_the_q4_default(cli, tmp_path, monkeypatch):
    model = tmp_path / "local-q8.gguf"
    model.write_bytes(b"local-q8-placeholder")
    (tmp_path / "research/phase28-q8-model.json").write_text(json.dumps({
        "destination": str(tmp_path),
        "files": [{"name": model.name, "size": model.stat().st_size}],
    }))
    (tmp_path / "research/phase29-host-embedding-build.json").write_text(json.dumps({
        "executable": "host-native.exe", "source": str(tmp_path), "cuda_toolkit": str(tmp_path),
    }))
    output = tmp_path / "q8.wav"
    calls = []
    environments = []

    def run(command, **kwargs):
        calls.append(command)
        environments.append(kwargs["env"])
        (Path(command[command.index("--out-dir") + 1]) / "speech.wav").write_bytes(b"complete")
        return SimpleNamespace(returncode=0, stdout=b"EOS complete")

    monkeypatch.setattr(cli.subprocess, "run", run)
    monkeypatch.setattr("sys.argv", ["gguf", "--text", "hello", "--variant", "q8", "--codec", "cpu",
                                   "--cache-experts", "12", "--host-embedding",
                                   "--output", str(output)])
    cli.main()
    assert calls[0][calls[0].index("--model") + 1] == str(model)
    assert calls[0][0] == "host-native.exe"
    assert environments[0]["DRAMA_HOST_GGUF_EMBEDDING"] == "1"
    assert "fish_drama3.codec_backend=cpu" in calls[0]
    assert "fish_drama3.expert_cache_size=12" in calls[0]
    assert json.loads(output.with_suffix(".json").read_text())["variant"] == "q8"
    assert json.loads(output.with_suffix(".json").read_text())["cache_experts_per_layer"] == 12
    assert (tmp_path / "model.gguf").read_bytes() == b"verified-placeholder"


@pytest.mark.parametrize("slots", ["7", "257"])
def test_out_of_range_cache_does_not_launch_native_process(cli, monkeypatch, slots):
    run = Mock()
    monkeypatch.setattr(cli.subprocess, "run", run)
    monkeypatch.setattr("sys.argv", ["gguf", "--text", "hello", "--cache-experts", slots])
    with pytest.raises(SystemExit):
        cli.main()
    run.assert_not_called()


def test_split_codec_selects_verified_probe_and_preserves_cpu_reference(cli, tmp_path, monkeypatch):
    model = tmp_path / "local-q8.gguf"
    model.write_bytes(b"local-q8-placeholder")
    (tmp_path / "research/phase28-q8-model.json").write_text(json.dumps({
        "destination": str(tmp_path),
        "files": [{"name": model.name, "size": model.stat().st_size}],
    }))
    (tmp_path / "research/phase30-codec-build.json").write_text(json.dumps({
        "executable": "codec-native.exe", "source": str(tmp_path), "cuda_toolkit": str(tmp_path),
    }))
    output = tmp_path / "split.wav"
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        (Path(command[command.index("--out-dir") + 1]) / "speech.wav").write_bytes(b"complete")
        return SimpleNamespace(returncode=0, stdout=b"EOS complete")

    monkeypatch.setattr(cli.subprocess, "run", run)
    monkeypatch.setattr("sys.argv", [
        "gguf", "--text", "hello", "--variant", "q8", "--codec", "cpu",
        "--decoder", "cuda", "--codec-decode", "stream", "--host-embedding", "--output", str(output),
    ])
    cli.main()
    assert calls[0][0] == "codec-native.exe"
    assert "fish_drama3.codec_backend=cpu" in calls[0]
    assert "fish_drama3.codec_decoder_backend=cuda" in calls[0]
    assert "fish_drama3.codec_decode_strategy=stream" in calls[0]
    assert json.loads(output.with_suffix(".json").read_text())["codec_decode_strategy"] == "stream"


def test_cooperation_uses_own_binary_and_sanitizes_diagnostic_environment(cli, tmp_path, monkeypatch):
    model = tmp_path / "q8.gguf"
    model.write_bytes(b"q8-placeholder")
    (tmp_path / "research/phase28-q8-model.json").write_text(json.dumps({
        "destination": str(tmp_path),
        "files": [{"name": model.name, "size": model.stat().st_size}],
    }))
    (tmp_path / "research/phase33-hybrid-build.json").write_text(json.dumps({
        "executable": "hybrid.exe", "source": str(tmp_path), "cuda_toolkit": str(tmp_path),
    }))
    monkeypatch.setenv("DRAMA_Q8_STEP_REPLAY", "unrelated-previous-trace.bin")
    output = tmp_path / "hybrid.wav"
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs["env"]))
        (Path(command[command.index("--out-dir") + 1]) / "speech.wav").write_bytes(b"complete")
        return SimpleNamespace(returncode=0, stdout=b"EOS complete")

    monkeypatch.setattr(cli.subprocess, "run", run)
    monkeypatch.setattr("sys.argv", [
        "gguf", "--text", "hello", "--variant", "q8", "--decode-cpu-experts", "auto",
        "--codec", "cpu", "--decoder", "cuda", "--codec-decode", "stream",
        "--decode-profile", str(tmp_path / "profiles"), "--output", str(output),
    ])
    cli.main()
    command, env = calls[0]
    assert command[0] == "hybrid.exe"
    assert "fish_drama3.decode_cpu_experts=auto" in command
    assert "fish_drama3.decode_profile=" + str(tmp_path / "profiles") in command
    assert "fish_drama3.codec_decoder_backend=cuda" in command
    assert "DRAMA_Q8_STEP_REPLAY" not in env
    assert json.loads(output.with_suffix(".json").read_text())["decode_cpu_experts"] == "auto"


def test_q4_cooperation_is_rejected_before_loading_model(cli, monkeypatch):
    run = Mock()
    monkeypatch.setattr(cli.subprocess, "run", run)
    monkeypatch.setattr("sys.argv", ["gguf", "--text", "hello", "--decode-cpu-experts", "auto"])
    with pytest.raises(SystemExit):
        cli.main()
    run.assert_not_called()


def test_auto_pin_failure_retries_mapped_and_never_uses_the_failed_attempts_audio(cli, tmp_path, monkeypatch):
    output = tmp_path / "voice.wav"
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        target = Path(command[command.index("--out-dir") + 1]) / "speech.wav"
        target.write_bytes(b"failed-audio" if len(calls) == 1 else b"complete-audio")
        return SimpleNamespace(
            returncode=1 if len(calls) == 1 else 0,
            stdout=b"cudaHostAlloc failed: out of memory" if len(calls) == 1 else b"EOS complete",
        )

    monkeypatch.setattr(cli.subprocess, "run", run)
    monkeypatch.setattr("sys.argv", ["gguf", "--text", "hello", "--output", str(output)])
    cli.main()
    assert len(calls) == 2
    assert "fish_drama3.pin_experts=true" in calls[0]
    assert "fish_drama3.pin_experts=true" not in calls[1]
    assert calls[0][calls[0].index("--out-dir") + 1] != calls[1][calls[1].index("--out-dir") + 1]
    assert output.read_bytes() == b"complete-audio"
    report = json.loads(output.with_suffix(".json").read_text())
    assert report["complete"] and report["host_storage"] == "mapped"
    assert report["threads"] == 4
    assert len(report["attempts"]) == 2


@pytest.mark.parametrize("message,extra", [
    (b"unsupported model tensor", []),
    (b"CUDA device-side assert", []),
    (b"cudaHostAlloc failed: out of memory", ["--host-storage", "pin"]),
])
def test_model_errors_and_explicit_pin_failures_do_not_start_a_second_attempt(
        cli, tmp_path, monkeypatch, message, extra):
    output = tmp_path / "failed.wav"
    run = Mock(return_value=SimpleNamespace(returncode=1, stdout=message))
    monkeypatch.setattr(cli.subprocess, "run", run)
    monkeypatch.setattr("sys.argv", ["gguf", "--text", "hello", "--output", str(output), *extra])
    with pytest.raises(RuntimeError, match="Native generation failed"):
        cli.main()
    run.assert_called_once()
    assert not output.exists()
    report = json.loads(output.with_suffix(".json").read_text())
    assert not report["complete"] and report["status"] == "error"
