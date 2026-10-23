"""The CLI must reuse one engine and surface incomplete request outcomes."""

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def load_cli():
    source = Path(__file__).resolve().parents[1] / "tools" / "run_drama_engine.py"
    spec = spec_from_file_location("testable_drama_engine_cli", source)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cli(monkeypatch):
    module = load_cli()
    engine = SimpleNamespace(synthesize=Mock(return_value={"status": "complete"}))
    lifecycle = []

    class FakeEngine:
        def __init__(self, options):
            self.options = options
            lifecycle.append(("construct", options))

        def __enter__(self):
            lifecycle.append(("enter", None))
            return engine

        def __exit__(self, *_args):
            lifecycle.append(("close", None))

    monkeypatch.setattr(module, "DramaEngine", FakeEngine)
    return SimpleNamespace(module=module, engine=engine, lifecycle=lifecycle, monkeypatch=monkeypatch)


def test_voice_clone_cli_forwards_both_reference_fields(cli, tmp_path):
    output = tmp_path / "voice.wav"
    audio = tmp_path / "reference.wav"
    cli.monkeypatch.setattr(
        "sys.argv",
        [
            "run_drama_engine.py", "--text", "Hello there.",
            "--output", str(output), "--prompt-audio", str(audio),
            "--prompt-text", "Reference speech.", "--seed", "567",
        ],
    )
    assert cli.module.main() == 0
    call = cli.engine.synthesize.call_args
    assert call.args == ("Hello there.", output)
    assert call.kwargs["reference_audio"] == audio
    assert call.kwargs["reference_text"] == "Reference speech."
    assert call.kwargs["seed"] == 567
    assert [event for event, _ in cli.lifecycle] == ["construct", "enter", "close"]


def test_json_request_accepts_zero_top_k(cli, tmp_path):
    request = cli.module._validate_request(
        {"text": "Valid.", "top_k": 0},
        index=1,
        default_output=tmp_path / "default.wav",
        default_seed=123,
        default_max_new_tokens=768,
        default_max_chunk_chars=160,
        default_allow_partial=False,
        batched=False,
    )
    assert request["top_k"] == 0


def test_jsonl_batch_uses_a_single_engine_for_all_requests(cli, tmp_path):
    requests = tmp_path / "batch.jsonl"
    first, second = tmp_path / "first.wav", tmp_path / "second.wav"
    requests.write_text(
        (
            f'{{"text": "First.", "output": "{first.as_posix()}"}}\n'
            f'{{"text": "Second.", "output": "{second.as_posix()}", "seed": 987}}\n'
        ),
        encoding="utf-8",
    )
    cli.monkeypatch.setattr(
        "sys.argv", ["run_drama_engine.py", "--requests", str(requests)]
    )
    assert cli.module.main() == 0
    assert cli.engine.synthesize.call_count == 2
    assert [call.args[1] for call in cli.engine.synthesize.call_args_list] == [first, second]
    assert cli.engine.synthesize.call_args_list[1].kwargs["seed"] == 987
    assert sum(event == "construct" for event, _ in cli.lifecycle) == 1


def test_packed_staging_is_forwarded_only_when_requested(cli):
    cli.monkeypatch.setattr(
        "sys.argv", ["run_drama_engine.py", "--text", "Hello.", "--packed-staging"]
    )
    assert cli.module.main() == 0
    assert cli.lifecycle[0][1].packed_staging is True


def test_official_expert_decode_override_is_forwarded(cli):
    cli.monkeypatch.setattr(
        "sys.argv", ["run_drama_engine.py", "--text", "Hello.",
                     "--expert-decode", "official"])
    assert cli.module.main() == 0
    assert cli.lifecycle[0][1].expert_decode == "official"


def test_automatic_compute_profile_and_exact_context_override_are_forwarded(cli, tmp_path):
    directory = tmp_path / "profiles"
    cli.monkeypatch.setattr("sys.argv", [
        "run_drama_engine.py", "--text", "Hello.", "--compute-policy", "auto",
        "--compute-profile-dir", str(directory), "--context-bucket-size", "0",
    ])
    assert cli.module.main() == 0
    options = cli.lifecycle[0][1]
    assert options.compute_policy == "auto"
    assert options.compute_profile_dir == directory
    assert options.context_bucket_size == 0


@pytest.mark.parametrize("mode", ["auto", "torch", "native"])
def test_host_gather_override_is_forwarded(cli, mode):
    cli.monkeypatch.setattr(
        "sys.argv", ["run_drama_engine.py", "--text", "Hello.", "--host-gather", mode])
    assert cli.module.main() == 0
    assert cli.lifecycle[0][1].host_gather == mode


@pytest.mark.parametrize("mode", ["auto", "full"])
def test_codec_placement_override_is_forwarded(cli, mode):
    cli.monkeypatch.setattr(
        "sys.argv", ["run_drama_engine.py", "--text", "Hello.", "--codec-placement", mode])
    assert cli.module.main() == 0
    assert cli.lifecycle[0][1].codec_placement == mode


def test_experimental_cpu_decode_configuration_is_forwarded(cli):
    cli.monkeypatch.setattr(
        "sys.argv", ["run_drama_engine.py", "--text", "Hello.",
                     "--cpu-experts", "4", "--native-threads", "3"])
    assert cli.module.main() == 0
    assert cli.lifecycle[0][1].cpu_experts == 4
    assert cli.lifecycle[0][1].native_threads == 3


def test_host_thread_override_is_explicit(cli):
    configure = Mock()
    cli.monkeypatch.setattr(cli.module.torch, "set_num_threads", configure)
    cli.monkeypatch.setattr(
        "sys.argv", ["run_drama_engine.py", "--text", "Hello."])
    assert cli.module.main() == 0
    configure.assert_not_called()
    cli.monkeypatch.setattr(
        "sys.argv", ["run_drama_engine.py", "--text", "Hello.", "--host-threads", "1"])
    assert cli.module.main() == 0
    configure.assert_called_once_with(1)


def test_prefix_state_cache_budget_is_forwarded(cli):
    cli.monkeypatch.setattr(
        "sys.argv", ["run_drama_engine.py", "--text", "Hello.", "--prefix-cache-mib", "128"])
    assert cli.module.main() == 0
    assert cli.lifecycle[0][1].prefix_cache_mib == 128


def test_incomplete_request_produces_nonzero_exit_code(cli, tmp_path):
    cli.engine.synthesize.return_value = {"status": "incomplete"}
    cli.monkeypatch.setattr(
        "sys.argv", ["run_drama_engine.py", "--text", "Speech text."]
    )
    assert cli.module.main() == 2


def test_explicit_partial_output_mode_can_accept_incomplete_request(cli):
    cli.engine.synthesize.return_value = {"status": "incomplete"}
    cli.monkeypatch.setattr(
        "sys.argv", ["run_drama_engine.py", "--text", "Speech text.", "--allow-partial"]
    )
    assert cli.module.main() == 0
    assert cli.engine.synthesize.call_args.kwargs["allow_partial"] is True


def test_engine_is_closed_when_request_raises(cli):
    cli.engine.synthesize.side_effect = RuntimeError("Generation failed")
    cli.monkeypatch.setattr(
        "sys.argv", ["run_drama_engine.py", "--text", "Speech text."]
    )
    assert cli.module.main() == 1
    assert cli.lifecycle[-1][0] == "close"


@pytest.mark.parametrize("arguments", [[], ["--text", "Hello", "--requests", "batch.jsonl"]])
def test_invalid_cli_request_source_is_rejected_before_engine_construction(cli, arguments):
    cli.monkeypatch.setattr("sys.argv", ["run_drama_engine.py", *arguments])
    with pytest.raises(SystemExit) as result:
        cli.module.main()
    assert result.value.code == 2
    assert not cli.lifecycle


@pytest.mark.parametrize(
    "value",
    [
        {"text": "Valid.", "seed": True},
        {"text": "Valid.", "seed": 1.5},
        {"text": "Valid.", "seed": "123"},
        {"text": "Valid.", "seed": -1},
        {"text": "Valid.", "seed": 2**64},
        {"text": "Valid.", "max_new_tokens": False},
        {"text": "Valid.", "max_new_tokens": 1.5},
        {"text": "Valid.", "max_chunk_chars": "160"},
        {"text": "Valid.", "top_k": True},
        {"text": "Valid.", "temperature": float("nan")},
        {"text": "Valid.", "temperature": 2},
        {"text": "Valid.", "top_p": float("inf")},
        {"text": "Valid.", "allow_partial": "true"},
        {"text": ""},
        {"text": " \n"},
        {"text": 123},
        {"other": "Missing text."},
        {"text": "Valid.", "unsupported": 1},
        {"text": "Valid.", "reference_audio": "voice.wav"},
        {"text": "Valid.", "reference_text": "reference words"},
        {"text": "Valid.", "output": 123},
        ["Not an object"],
    ],
)
def test_invalid_json_request_is_rejected_before_engine_loading(cli, tmp_path, value):
    source = tmp_path / "invalid.jsonl"
    source.write_text(json.dumps(value) + "\n", encoding="utf-8")
    cli.monkeypatch.setattr(
        "sys.argv", ["run_drama_engine.py", "--requests", str(source)]
    )
    with pytest.raises(SystemExit) as result:
        cli.module.main()
    assert result.value.code == 2
    assert not cli.lifecycle


@pytest.mark.parametrize("text", ["", " \n \n", "{invalid json}\n"])
def test_empty_or_malformed_batch_fails_before_engine_loading(cli, tmp_path, text):
    source = tmp_path / "empty.jsonl"
    source.write_text(text, encoding="utf-8")
    cli.monkeypatch.setattr(
        "sys.argv", ["run_drama_engine.py", "--requests", str(source)]
    )
    with pytest.raises(SystemExit) as result:
        cli.module.main()
    assert result.value.code == 2
    assert not cli.lifecycle


def test_default_batch_outputs_are_unique_and_keep_cli_sampling_defaults(cli, tmp_path):
    source = tmp_path / "batch.jsonl"
    source.write_text(
        '{"text": "First."}\n{"text": "Second."}\n', encoding="utf-8"
    )
    default = tmp_path / "custom.wav"
    cli.monkeypatch.setattr(
        "sys.argv",
        ["run_drama_engine.py", "--requests", str(source), "--output", str(default)],
    )
    assert cli.module.main() == 0
    first, second = cli.engine.synthesize.call_args_list
    assert first.args[1] == tmp_path / "custom-001.wav"
    assert second.args[1] == tmp_path / "custom-002.wav"
    assert first.kwargs["temperature"] == cli.module.DEFAULT_TEMPERATURE
    assert first.kwargs["top_p"] == cli.module.DEFAULT_TOP_P
    assert cli.lifecycle[0][1].staged_experts is None
    assert cli.lifecycle[0][1].pair_rows is None
    assert cli.lifecycle[0][1].prefill_chunk is None


def test_duplicate_outputs_in_batch_are_rejected_before_loading(cli, tmp_path):
    source = tmp_path / "batch.jsonl"
    output = (tmp_path / "same.wav").as_posix()
    source.write_text(
        json.dumps({"text": "First.", "output": output}) + "\n"
        + json.dumps({"text": "Second.", "output": output}) + "\n",
        encoding="utf-8",
    )
    cli.monkeypatch.setattr(
        "sys.argv", ["run_drama_engine.py", "--requests", str(source)]
    )
    with pytest.raises(SystemExit):
        cli.module.main()
    assert not cli.lifecycle


def test_batch_continues_after_a_request_error_and_returns_error_status(cli, tmp_path, capsys):
    source = tmp_path / "batch.jsonl"
    source.write_text(
        '{"text": "First."}\n{"text": "Second."}\n', encoding="utf-8"
    )
    cli.engine.synthesize.side_effect = [
        ValueError("Invalid first reference"),
        {"status": "complete"},
    ]
    cli.monkeypatch.setattr(
        "sys.argv", ["run_drama_engine.py", "--requests", str(source)]
    )
    assert cli.module.main() == 1
    assert cli.engine.synthesize.call_count == 2
    error = json.loads(capsys.readouterr().err)
    assert error["status"] == "error"
    assert error["request_index"] == 1
    assert "Invalid first reference" in error["error"]
    assert cli.lifecycle[-1][0] == "close"


def test_batch_stops_after_cuda_failure(cli, tmp_path, capsys):
    source = tmp_path / "batch.jsonl"
    source.write_text(
        '{"text": "First."}\n{"text": "Second."}\n', encoding="utf-8"
    )
    cli.engine.synthesize.side_effect = [
        RuntimeError("CUDA error: illegal memory access"),
        {"status": "complete"},
    ]
    cli.monkeypatch.setattr(
        "sys.argv", ["run_drama_engine.py", "--requests", str(source)]
    )
    assert cli.module.main() == 1
    assert cli.engine.synthesize.call_count == 1
    assert json.loads(capsys.readouterr().err)["status"] == "error"
    assert cli.lifecycle[-1][0] == "close"


@pytest.mark.parametrize(
    "option,value",
    [
        ("--staged-experts", "0"),
        ("--pair-rows", "-1"),
        ("--prefill-chunk", "0"),
        ("--cache-experts", "-1"),
        ("--cpu-experts", "-1"),
        ("--cpu-experts", "9"),
        ("--native-threads", "0"),
        ("--native-threads", "65"),
        ("--host-threads", "0"),
        ("--host-threads", "65"),
        ("--prefix-cache-mib", "-1"),
    ],
)
def test_invalid_resource_limits_are_rejected_before_engine_loading(cli, option, value):
    cli.monkeypatch.setattr(
        "sys.argv", ["run_drama_engine.py", "--text", "Speech.", option, value]
    )
    with pytest.raises(SystemExit) as result:
        cli.module.main()
    assert result.value.code == 2
    assert not cli.lifecycle


def test_json_error_reports_actual_file_line_number(cli, tmp_path, capsys):
    source = tmp_path / "batch.jsonl"
    source.write_text('\n{"text":"First."}\n\n{broken json}\n', encoding="utf-8")
    cli.monkeypatch.setattr(
        "sys.argv", ["run_drama_engine.py", "--requests", str(source)]
    )
    with pytest.raises(SystemExit):
        cli.module.main()
    assert "line 4" in capsys.readouterr().err
