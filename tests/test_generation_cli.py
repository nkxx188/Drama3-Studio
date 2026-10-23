"""CLI contract tests without loading the Drama 3 checkpoint."""

from __future__ import annotations

import contextlib
import gc
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "drama_generation_cli", ROOT / "tools" / "run_drama3_low_vram.py"
)
assert SPEC is not None and SPEC.loader is not None
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


def runtime_config(**overrides):
    values = {
        "status": "ready",
        "can_run": True,
        "device": "cuda:1",
        "backend": "triton",
        "lm_head_device": "cpu",
        "max_staged_experts": 4,
        "max_pair_rows": 32,
        "cache_experts": 6,
        "prefill_chunk_size": 8,
        "codec_max_frames": 2048,
        "pin_memory": False,
        "warnings": (),
    }
    values.update(overrides)
    values["as_dict"] = lambda **kwargs: {
        key: value
        for key, value in values.items()
        if key != "as_dict"
    }
    return SimpleNamespace(**values)


class GenerationCLIArgumentsTests(unittest.TestCase):
    def test_defaults_follow_official_sampling(self):
        args = cli.build_parser().parse_args(["--text", "hello"])
        self.assertEqual(args.max_new_tokens, cli.DEFAULT_MAX_NEW_TOKENS)
        self.assertEqual(args.max_new_tokens, 1024)
        self.assertEqual(args.temperature, cli.DEFAULT_TEMPERATURE)
        self.assertEqual(args.top_p, cli.DEFAULT_TOP_P)
        self.assertEqual(args.top_k, cli.DEFAULT_TOP_K)
        self.assertEqual(args.seed, 123)
        self.assertFalse(args.allow_partial)

    def test_auto_honors_selected_device_and_disabled_pinning(self):
        args = cli.build_parser().parse_args(["--text", "hello", "--device", "cuda:1"])
        options = cli.resolve_runtime_options(args, runtime_config())
        self.assertEqual(options["device"], "cuda:1")
        self.assertFalse(options["pin_memory"])
        self.assertEqual(options["cache_experts"], 6)
        self.assertEqual(options["prefill_chunk_size"], 8)

    def test_explicit_zero_cache_and_manual_values_win(self):
        args = cli.build_parser().parse_args(
            [
                "--text", "hello",
                "--backend", "torch",
                "--lm-head-device", "cuda",
                "--cache-experts", "0",
                "--max-staged-experts", "2",
                "--max-pair-rows", "10",
                "--prefill-chunk-size", "3",
                "--no-pin-memory",
            ]
        )
        options = cli.resolve_runtime_options(args, runtime_config(pin_memory=True))
        self.assertEqual(options["backend"], "torch")
        self.assertEqual(options["lm_head_device"], "cuda")
        self.assertEqual(options["cache_experts"], 0)
        self.assertEqual(options["max_staged_experts"], 2)
        self.assertEqual(options["max_pair_rows"], 10)
        self.assertEqual(options["prefill_chunk_size"], 3)
        self.assertFalse(options["pin_memory"])

    def test_zero_generation_or_staging_budgets_are_rejected(self):
        for option in ("--max-new-tokens", "--max-staged-experts", "--max-pair-rows"):
            with self.subTest(option=option):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        cli.build_parser().parse_args(["--text", "hello", option, "0"])
                self.assertEqual(raised.exception.code, 2)

    def test_seed_resets_cpu_sampling_randomness(self):
        with patch.object(cli.torch.cuda, "is_available", return_value=False):
            cli._seed_generation(123)
            first = (torch.rand(5), np.random.random(5))
            cli._seed_generation(123)
            second = (torch.rand(5), np.random.random(5))
        torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
        np.testing.assert_array_equal(first[1], second[1])

    def test_report_cannot_overwrite_audio(self):
        with tempfile.TemporaryDirectory() as name:
            output = Path(name) / "sample.wav"
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    cli.main(
                        [
                            "--checkpoint", name, "--text", "hello",
                            "--output", str(output), "--report", str(output),
                        ]
                    )
        self.assertEqual(raised.exception.code, 2)

    def test_invalid_device_gets_readable_argument_error(self):
        with tempfile.TemporaryDirectory() as name:
            with patch.object(cli, "autotune", side_effect=ValueError("cuda:7 is unavailable")):
                with contextlib.redirect_stderr(io.StringIO()) as error:
                    with self.assertRaises(SystemExit) as raised:
                        cli.main(
                            ["--checkpoint", name, "--text", "hello", "--device", "cuda:7"]
                        )
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("cuda:7 is unavailable", error.getvalue())

    def test_busy_resources_stop_before_model_loading(self):
        config = runtime_config(
            status="resource_busy", can_run=False, warnings=("Free VRAM is too low",)
        )
        with tempfile.TemporaryDirectory() as name:
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(cli, "autotune", return_value=config))
                load = stack.enter_context(patch.object(cli, "load_offloaded_drama3"))
                error = stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                with self.assertRaises(SystemExit) as raised:
                    cli.main(["--checkpoint", name, "--text", "hello"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("Free VRAM is too low", error.getvalue())
        load.assert_not_called()


class GenerationCLIRunTests(unittest.TestCase):
    def run_cli(self, directory, *, reason="eos", frames=3, extra=(), decode_error=None):
        output = directory / "sample.wav"
        model = SimpleNamespace(tokenizer=object(), config=object(), cache_capacity=1030)
        manager = SimpleNamespace(
            stats=SimpleNamespace(as_dict=lambda: {"h2d_bytes": 1234, "hot_cache_hits": 2}),
            pool=SimpleNamespace(pin_memory=False, pin_memory_error=None),
        )
        plan = SimpleNamespace(
            cache_experts=6,
            max_staged_experts=4,
            max_pair_rows=32,
            prefill_chunk=8,
            as_dict=lambda: {"cache_experts": 6, "safe_gpu_budget_bytes": 7 * 1024**3}
        )
        pcm = {
            "sample_rate": 44100,
            "samples": frames * 2048,
            "duration_seconds": frames * 2048 / 44100,
        }
        with contextlib.ExitStack() as stack:
            mocks = {}
            replacements = {
                "autotune": {"return_value": runtime_config()},
                "load_offloaded_drama3": {"return_value": (model, manager)},
                "build_drama3_prompt": {"return_value": torch.zeros(6, 11, dtype=torch.long)},
                "_configure_loaded_runtime": {"return_value": plan},
                "generate_drama3": {
                    "return_value": (torch.zeros(frames, 11, dtype=torch.long), reason)
                },
                "write_audio": (
                    {"side_effect": decode_error}
                    if decode_error is not None else {"return_value": pcm}
                ),
                "release_drama3": {},
                "_seed_generation": {},
                "_synchronize": {},
                "_reset_gpu_peak": {},
                "_memory_snapshot": {
                    "return_value": {
                        "process": {"resident_bytes": 100},
                        "cuda": {"peak_allocated_bytes": 200},
                    }
                },
            }
            for name, kwargs in replacements.items():
                mocks[name] = stack.enter_context(patch.object(cli, name, **kwargs))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            code = cli.main(
                [
                    "--checkpoint", str(directory),
                    "--text", "hello",
                    "--output", str(output),
                    "--device", "cuda:1",
                    *extra,
                ]
            )
        report = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
        return code, report, mocks

    def test_eos_is_complete_and_reports_actual_runtime(self):
        with tempfile.TemporaryDirectory() as name:
            code, report, mocks = self.run_cli(Path(name))
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["finish_reason"], "eos")
        self.assertEqual(report["frames"], 3)
        self.assertIsNotNone(report["output"])
        self.assertEqual(report["runtime_plan"]["cache_experts"], 6)
        self.assertEqual(report["effective_runtime"]["cache_experts"], 6)
        self.assertFalse(report["effective_runtime"]["pin_memory"])
        self.assertEqual(report["precision"]["routed_experts"], "torch.float8_e4m3fn")
        self.assertTrue(report["precision"]["all_routed_experts_preserved"])
        self.assertGreaterEqual(report["end_to_end_rtf"], 0)
        self.assertGreaterEqual(report["generation_rtf_including_prefill"], 0)
        self.assertIn("generation", report["memory"])
        self.assertIn("codec_decode_and_write", report["phase_seconds"])
        mocks["autotune"].assert_called_once_with(device="cuda:1")
        kwargs = mocks["load_offloaded_drama3"].call_args.kwargs
        self.assertEqual(kwargs["device"], "cuda:1")
        self.assertFalse(kwargs["pin_memory"])
        self.assertEqual(kwargs["cache_experts"], 0)
        configure_kwargs = mocks["_configure_loaded_runtime"].call_args.kwargs
        self.assertEqual(configure_kwargs["sequence_capacity"], 1030)
        self.assertIsNone(configure_kwargs["cache_experts"])
        self.assertEqual(configure_kwargs["prefill_chunk_size"], 8)
        self.assertTrue(configure_kwargs["allow_staging_reduction"])
        mocks["_seed_generation"].assert_called_once_with(123)
        mocks["release_drama3"].assert_called_once()

    def test_token_limit_does_not_write_complete_audio(self):
        with tempfile.TemporaryDirectory() as name:
            code, report, mocks = self.run_cli(Path(name), reason="length")
        self.assertEqual(code, 2)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["finish_reason"], "length")
        self.assertIsNone(report["output"])
        self.assertIsNone(report["audio"])
        self.assertIn("before EOS", report["error"])
        mocks["write_audio"].assert_not_called()
        mocks["release_drama3"].assert_called_once()

    def test_explicit_partial_output_remains_incomplete(self):
        with tempfile.TemporaryDirectory() as name:
            code, report, mocks = self.run_cli(
                Path(name), reason="length", extra=("--allow-partial",)
            )
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "incomplete")
        self.assertTrue(report["allow_partial"])
        self.assertIsNotNone(report["output"])
        mocks["write_audio"].assert_called_once()

    def test_empty_eos_is_an_error_and_releases_model(self):
        with tempfile.TemporaryDirectory() as name:
            code, report, mocks = self.run_cli(Path(name), frames=0)
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "error")
        self.assertIn("no audio frames", report["error"])
        mocks["write_audio"].assert_not_called()
        mocks["release_drama3"].assert_called_once()

    def test_decoder_failure_does_not_report_success(self):
        with tempfile.TemporaryDirectory() as name:
            code, report, mocks = self.run_cli(
                Path(name), decode_error=RuntimeError("invalid waveform")
            )
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["error"], "RuntimeError: invalid waveform")
        self.assertIsNone(report["output"])
        mocks["release_drama3"].assert_called_once()

    def test_manual_explicit_cache_is_passed_to_postload_planner(self):
        with tempfile.TemporaryDirectory() as name:
            code, report, mocks = self.run_cli(
                Path(name),
                extra=("--manual", "--cache-experts", "3", "--seed", "410"),
            )
        self.assertEqual(code, 0)
        self.assertIsNone(report["auto_runtime"])
        self.assertEqual(report["seed"], 410)
        mocks["autotune"].assert_not_called()
        self.assertEqual(
            mocks["_configure_loaded_runtime"].call_args.kwargs["cache_experts"], 3
        )
        self.assertFalse(
            mocks["_configure_loaded_runtime"].call_args.kwargs[
                "allow_staging_reduction"
            ]
        )


class GenerationCLIWaveformTests(unittest.TestCase):
    def test_decode_uses_exact_frame_budget_and_releases_codec(self):
        fake_codec = SimpleNamespace(sample_rate=44100)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(cli, "resolve_codec_path", return_value=Path("codec.pth")))
            load = stack.enter_context(patch.object(cli, "load_codec", return_value=fake_codec))
            stack.enter_context(patch.object(cli, "decode_to_audio", return_value=torch.ones(4096)))
            write = stack.enter_context(patch.object(cli, "write_wav"))
            release = stack.enter_context(patch.object(cli, "release_codec"))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            metadata = cli.write_audio(
                Path("checkpoint"), torch.zeros(10, 2, dtype=torch.long),
                Path("sample.wav"), "cuda:0", torch.bfloat16
            )
        self.assertEqual(load.call_args.kwargs["max_frames"], 2)
        self.assertEqual(metadata["samples"], 4096)
        self.assertAlmostEqual(metadata["duration_seconds"], 4096 / 44100)
        write.assert_called_once()
        release.assert_called_once_with(fake_codec)

    def test_nonfinite_waveform_is_rejected_before_write(self):
        fake_codec = SimpleNamespace(sample_rate=44100)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(cli, "resolve_codec_path", return_value=Path("codec.pth")))
            stack.enter_context(patch.object(cli, "load_codec", return_value=fake_codec))
            stack.enter_context(patch.object(cli, "decode_to_audio", return_value=torch.tensor([float("nan")])))
            write = stack.enter_context(patch.object(cli, "write_wav"))
            release = stack.enter_context(patch.object(cli, "release_codec"))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            with self.assertRaisesRegex(RuntimeError, "NaN or infinity"):
                cli.write_audio(
                    Path("checkpoint"), torch.zeros(10, 2, dtype=torch.long),
                    Path("sample.wav"), "cuda:0", torch.bfloat16
                )
        write.assert_not_called()
        release.assert_called_once_with(fake_codec)


class GenerationCLIReleaseTests(unittest.TestCase):
    def test_release_diagnostics_do_not_retain_cpu_expert_buffers(self):
        model = torch.nn.Module()
        model.backbone = torch.nn.Module()
        model.audio_decoder = torch.nn.Identity()
        model.backbone.experts = torch.nn.Module()
        model.backbone.experts.fp8 = True
        model.backbone.experts.register_buffer(
            "w13_weight", torch.ones((2, 256, 128), dtype=torch.float8_e4m3fn)
        )
        model.backbone.experts.register_buffer(
            "w2_weight", torch.ones((2, 128, 128), dtype=torch.float8_e4m3fn)
        )
        model.backbone.experts.register_buffer(
            "w13_scale_inv", torch.ones((2, 2, 1))
        )
        model.backbone.experts.register_buffer(
            "w2_scale_inv", torch.ones((2, 1, 1))
        )
        from drama_low_vram.offload import install_expert_offload

        manager = install_expert_offload(
            model.backbone, backend="torch", pin_memory=False
        )
        references = cli._expert_release_references(manager)
        before = cli._expert_release_summary(references)
        self.assertEqual(before["live_expert_buffers_after_release"], 4)
        with patch.object(torch.cuda, "is_available", return_value=False):
            cli.release_drama3(model, manager)
        model, manager = None, None
        gc.collect()
        after = cli._expert_release_summary(references)
        self.assertEqual(after["tracked_expert_buffers"], 4)
        self.assertEqual(after["live_expert_buffers_after_release"], 0)
        self.assertEqual(after["live_expert_bytes_after_release"], 0)


if __name__ == "__main__":
    unittest.main()
