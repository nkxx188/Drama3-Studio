"""Measured compute policy selection, including a valid GPU-only result.

Calibration is explicit and uses the production cursor, routing, transfers,
native work and merge. It never claims a short-window gain is a full TTS gain.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import platform
from pathlib import Path
import statistics
import time
import os
import math
import importlib.metadata

import torch

from .native_backend import default_library_path
from .verification import DecodeCursor
from .engine import seed_sampling
from .sequence_state import RandomState


def _distribution_version(*names):
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


@dataclass(frozen=True)
class ComputeCandidate:
    cpu_experts: int = 0
    native_threads: int = 4
    host_gather: str = "torch"

    def __post_init__(self):
        if (isinstance(self.cpu_experts, bool) or not isinstance(self.cpu_experts, int)
                or not 0 <= self.cpu_experts <= 8
                or isinstance(self.native_threads, bool) or not isinstance(self.native_threads, int)
                or not 1 <= self.native_threads <= 64
                or self.host_gather not in ("torch", "auto", "native")):
            raise ValueError("Invalid compute candidate")


def _timing_windows(row):
    windows = row.get("windows", [])
    return (
        {window["offset"]: window["samples_seconds"] for window in windows}
        if windows else {None: row["samples_seconds"]}
    )


def _balanced_seconds(row):
    """Give equal-length decode windows equal weight, including late windows."""
    windows = _timing_windows(row)
    if any(not samples for samples in windows.values()):
        return math.inf
    return statistics.mean(statistics.median(samples) for samples in windows.values())


def _repeatable_improvement(candidate, reference):
    """Compare observed repeat ranges without claiming a confidence interval.

    The aggregate worst observed candidate must beat the aggregate best
    observed reference. A window with disjoint ranges in the wrong direction
    vetoes the candidate; overlapping ranges in an early window can be a tie.
    """
    windows, reference_windows = _timing_windows(candidate), _timing_windows(reference)
    if windows.keys() != reference_windows.keys():
        return {"accepted": False, "reason": "incomparable_windows"}
    if any(len(samples) < 2 for samples in (*windows.values(), *reference_windows.values())):
        return {"accepted": False, "reason": "insufficient_repeats"}
    slower = [
        offset for offset, samples in windows.items()
        if min(samples) > max(reference_windows[offset])
    ]
    gain_bound = statistics.mean(
        min(reference_windows[offset]) - max(samples)
        for offset, samples in windows.items()
    )
    return {
        "accepted": not slower and gain_bound > 0,
        "reason": (
            "slower_window" if slower else
            "repeat_ranges_overlap" if gain_bound <= 0 else "repeatable_gain"
        ),
        "gain_lower_bound_seconds": gain_bound,
        "slower_windows": slower,
    }


def select_candidate(rows):
    """Preserve exact output and select repeatable gains of any measured size."""
    valid = []
    for row in rows:
        ComputeCandidate(**row["candidate"])
        samples = row["samples_seconds"]
        if (not isinstance(samples, (list, tuple)) or any(
            isinstance(value, bool) or not isinstance(value, (float, int))
            or not math.isfinite(value) or value <= 0 for value in samples
        )):
            raise ValueError("Invalid compute timing samples")
        windows = row.get("windows", [])
        if windows:
            offsets = [window["offset"] for window in windows]
            if (any(isinstance(offset, bool) or not isinstance(offset, int) or offset < 0
                    for offset in offsets) or len(set(offsets)) != len(offsets)
                    or [value for window in windows for value in window["samples_seconds"]] != list(samples)
                    or any(window["bytes_equal"] is not True for window in windows)
                    != (row["bytes_equal"] is not True)):
                raise ValueError("Inconsistent compute timing windows")
        if row["bytes_equal"] is True and samples:
            valid.append(row)
    baseline = next(
        (row for row in valid if row["candidate"]["cpu_experts"] == 0
         and row["candidate"]["host_gather"] == "torch"), None)
    if baseline is None:
        raise ValueError("A valid GPU-only/Torch calibration baseline is required")
    gpu_choices = [baseline]
    comparisons = []
    for row in valid:
        if row is baseline or row["candidate"]["cpu_experts"]:
            continue
        comparison = _repeatable_improvement(row, baseline)
        comparisons.append({
            "candidate": row["candidate"], "reference": baseline["candidate"], **comparison,
        })
        if comparison["accepted"]:
            gpu_choices.append(row)
    gpu = min(gpu_choices, key=_balanced_seconds)
    eligible = []
    for row in valid:
        if not row["candidate"]["cpu_experts"]:
            continue
        comparison = _repeatable_improvement(row, gpu)
        comparisons.append({
            "candidate": row["candidate"], "reference": gpu["candidate"], **comparison,
        })
        if comparison["accepted"]:
            eligible.append(row)
    winner = min(eligible, key=_balanced_seconds) if eligible else gpu
    base = statistics.median(baseline["samples_seconds"])
    return {
        "candidate": winner["candidate"],
        "median_seconds": statistics.median(winner["samples_seconds"]),
        "baseline_median_seconds": base,
        "measured_time_reduction_fraction": 1 - statistics.median(winner["samples_seconds"]) / base,
        "selection_policy": "repeatable_window_ranges",
        "window_balanced_seconds": _balanced_seconds(winner),
        "baseline_window_balanced_seconds": _balanced_seconds(baseline),
        "best_gpu_median_seconds": statistics.median(gpu["samples_seconds"]),
        "comparisons": comparisons,
    }


def calibration_key(engine):
    dll = default_library_path()
    payload = {
        "device": torch.cuda.get_device_name(engine.device),
        "capability": list(torch.cuda.get_device_capability(engine.device)),
        "cpu": platform.processor(), "torch": torch.__version__,
        "logical_cpus": os.cpu_count(), "python": platform.python_version(),
        "triton": _distribution_version("triton", "triton-windows"),
        "cuda": torch.version.cuda, "native_sha256": (
            hashlib.sha256(dll.read_bytes()).hexdigest() if dll.is_file() else None),
        "model_config": hashlib.sha256(
            (engine.checkpoint / "config.json").read_bytes()).hexdigest(),
        "model_fingerprint": getattr(
            getattr(engine.model, "_drama_low_vram_expert_cache", None), "model_fingerprint", None),
        "checkpoint_path": str(engine.checkpoint),
        "staging": engine.manager.max_staged_experts,
        "pair_rows": engine.manager.max_pair_rows,
        "cache_experts": engine.manager.cache_experts,
        "prefill_chunk": getattr(getattr(engine, "_ready_plan", None), "prefill_chunk", None),
        "expert_decode": engine.options.expert_decode,
        "host_threads": torch.get_num_threads(),
        "math": {
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "allow_bf16_reduced_precision_reduction": (
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
        "schema": 3,
        "runtime_sources": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (
                "calibration.py", "hybrid_decode.py", "fp8_decode.py", "offload.py",
                "engine.py", "generation.py", "backbone_graph.py", "tower_graph.py",
                "verification.py", "prefix_cache.py", "sequence_state.py", "model.py",
                "weight_sources.py",
                "codec.py", "codec_decoder.py", "autotune.py",
            )
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload


@torch.inference_mode()
def calibrate_compute(engine, prompt, *, frames=8, repeats=3, candidates=None,
                      progress=None, output=None, offset=0):
    if frames < 2 or repeats < 2:
        raise ValueError("Compute calibration needs at least two frames and repetitions")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("Compute calibration offset must be nonnegative")
    if candidates is None:
        candidates = [
            ComputeCandidate(0, 4, "torch"), ComputeCandidate(0, 4, "auto"),
            ComputeCandidate(2, 2, "auto"), ComputeCandidate(2, 4, "auto"),
            ComputeCandidate(4, 4, "auto"),
        ]
    if candidates[0].cpu_experts != 0 or candidates[0].host_gather != "torch":
        raise ValueError("First calibration candidate must be GPU-only/Torch")
    with engine._lock:
        engine.load()
        engine._prepare(len(prompt) + offset + frames + 2)
        if engine.graph is None or engine.backbone_graph is None or not engine.backbone_graph.enabled:
            raise ValueError("Compute calibration requires the validated decode graphs")
        if engine._active_output_controls:
            raise ValueError("Compute calibration uses ordinary audio candidates")
        original_gather = engine.manager.host_gather_mode
        original_hybrid = engine.manager.hybrid_decode
        original = (
            None if original_hybrid is None else ComputeCandidate(
                original_hybrid.cpu_jobs, original_hybrid.executor.threads, original_gather)
        )
        # Preserve the caller's RNG before constructing the calibration prefix.
        random_state = RandomState.capture([engine.device.index])
        report = {
            "scope": "Fixed short decode continuation, same snapshot, real gather/PCIe/native/merge costs. No codec or completed-speech speed claim.",
            "frames": frames, "offset": offset, "repeats": repeats, "rows": [],
        }
        try:
            seed_sampling(1984)
            cursor = DecodeCursor.prefill(
                engine.model, prompt, chunk_size=engine._ready_plan.prefill_chunk,
                tower_graph=engine.graph, backbone_graph=engine.backbone_graph,
                allow_prefill_fallback=engine.options.prefill_chunk is None)
            engine._remember_prefill_chunk(cursor.prefix_report)
            report["prefill"] = cursor.prefix_report
            for _ in range(offset):
                cursor.step()
                if cursor.finished:
                    raise ValueError("Calibration text ended before its diagnostic window")
            base = cursor.capture()
            reference = None
            for candidate in candidates:
                row = {"candidate": candidate.__dict__, "bytes_equal": True,
                       "samples_seconds": [], "fallback": None}
                report["rows"].append(row)
                try:
                    current = engine.manager.hybrid_decode
                    if candidate.cpu_experts:
                        if (current is None or current.executor.threads != candidate.native_threads):
                            engine.manager.configure_hybrid_decode(
                                candidate.cpu_experts, threads=candidate.native_threads, device=engine.device)
                        else:
                            current.cpu_jobs = candidate.cpu_experts
                    engine.manager.configure_host_gather(candidate.host_gather)
                    for repetition in range(repeats + 1):
                        cursor.restore(base)
                        # Warm pages are shared, but per-layer hot cache starts
                        # the same for every measurement.
                        engine.manager.clear_hot_cache()
                        hybrid = engine.manager.hybrid_decode
                        context = (
                            hybrid.suspend() if hybrid is not None and not candidate.cpu_experts
                            else nullcontext()
                        )
                        with context:
                            torch.cuda.synchronize(engine.device)
                            started = time.perf_counter()
                            records = [cursor.step() for _ in range(frames)]
                            torch.cuda.synchronize(engine.device)
                            elapsed = time.perf_counter() - started
                        if cursor.finished:
                            raise ValueError("Calibration text ended inside its diagnostic window")
                        # Keep comparison copies outside the measured interval.
                        values = torch.cat([record["frame"] for record in records]).cpu()
                        logits = torch.cat([record["semantic_logits"] for record in records]).cpu()
                        if reference is None:
                            reference = values.clone(), logits.clone()
                        equal = (
                            torch.equal(values, reference[0]) and
                            torch.equal(logits.view(torch.uint8), reference[1].view(torch.uint8))
                        )
                        row["bytes_equal"] &= equal
                        if not equal:
                            break
                        if repetition:
                            row["samples_seconds"].append(elapsed)
                    row["median_seconds"] = (
                        statistics.median(row["samples_seconds"]) if row["samples_seconds"] else None)
                except (ValueError, OSError) as exc:
                    row["fallback"] = f"{type(exc).__name__}: {exc}"
                    row["bytes_equal"] = False
                if progress:
                    progress(row)
                if output:
                    Path(output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            report["selection"] = select_candidate(report["rows"])
            key, identity = calibration_key(engine)
            report["key"], report["identity"] = key, identity
            if output:
                Path(output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            return report
        finally:
            # Caller explicitly applies a selection; research calibration never
            # silently leaves the last candidate active.
            if original is None:
                engine.manager.configure_hybrid_decode(0)
            else:
                current = engine.manager.hybrid_decode
                if current is None or current.executor.threads != original.native_threads:
                    engine.manager.configure_hybrid_decode(
                        original.cpu_experts, threads=original.native_threads, device=engine.device)
                else:
                    current.cpu_jobs = original.cpu_experts
            engine.manager.configure_host_gather(original_gather)
            random_state.restore()


def calibrate_compute_windows(engine, prompt, *, frames=8, repeats=2,
                             offsets=(0, 16, 48), candidates=None, progress=None, output=None):
    """Measure more than the first acoustic window before selecting CPU work."""
    if (not offsets or len(set(offsets)) != len(offsets)
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                   for value in offsets)):
        raise ValueError("Compute windows require distinct nonnegative offsets")
    with engine._lock:
        previous = engine._computing_policy
        engine._computing_policy = True
        try:
            return _measure_compute_windows(
                engine, prompt, frames=frames, repeats=repeats, offsets=offsets,
                candidates=candidates, progress=progress, output=output)
        finally:
            engine._computing_policy = previous


def _measure_compute_windows(engine, prompt, *, frames, repeats, offsets,
                             candidates, progress, output):
    report = {
        "scope": "Several short production decode windows, real gather/transfer/compute/merge. Completed-speech validation remains separate.",
        "frames": frames, "repeats": repeats, "offsets": list(offsets), "rows": [],
    }
    combined = {}
    for offset in offsets:
        measured = calibrate_compute(
            engine, prompt, frames=frames, repeats=repeats, offset=offset,
            candidates=candidates, progress=(
                (lambda row: progress({"offset": offset, **row})) if progress else None))
        for row in measured["rows"]:
            key = json.dumps(row["candidate"], sort_keys=True)
            merged = combined.setdefault(key, {
                "candidate": row["candidate"], "bytes_equal": True,
                "samples_seconds": [], "windows": [],
            })
            merged["bytes_equal"] &= row["bytes_equal"]
            merged["samples_seconds"].extend(row["samples_seconds"])
            merged["windows"].append({
                "offset": offset, "bytes_equal": row["bytes_equal"],
                "samples_seconds": row["samples_seconds"], "fallback": row["fallback"],
            })
        report.update(rows=list(combined.values()), key=measured["key"], identity=measured["identity"])
        if output:
            Path(output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["selection"] = select_candidate(report["rows"])
    if output:
        Path(output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def apply_compute_selection(engine, selection):
    candidate = ComputeCandidate(**selection["candidate"])
    with engine._lock:
        engine.manager.configure_hybrid_decode(
            candidate.cpu_experts, threads=candidate.native_threads, device=engine.device)
        engine.manager.configure_host_gather(candidate.host_gather)
    return candidate


def save_compute_profile(report, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{report['key']}.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return target


def load_compute_profile(engine, directory):
    key, identity = calibration_key(engine)
    path = Path(directory) / f"{key}.json"
    try:
        if not path.is_file() or path.stat().st_size > 1024**2:
            return None
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["key"] != key or report["identity"] != identity:
            return None
        candidate = ComputeCandidate(**report["selection"]["candidate"])
        # Recompute the decision from measured rows rather than trusting a
        # stale selection field after schema/environment changes.
        selected = select_candidate(report["rows"])
        if selected["candidate"] != report["selection"]["candidate"]:
            return None
        report["selection"] = selected
        return report
    except (OSError, ValueError, TypeError, KeyError):
        return None


def ensure_compute_policy(engine):
    """Run once after runtime preparation; a cache miss measures this machine.

    Ordinary generation always prefills again afterwards. Fatal CUDA errors
    propagate; only unsupported optional paths and profile I/O use a fallback.
    """
    started = time.perf_counter()
    baseline = {"candidate": ComputeCandidate().__dict__}
    directory = (
        Path(engine.options.compute_profile_dir) if engine.options.compute_profile_dir
        else engine.checkpoint.parent.parent / ".cache/compute-profiles"
    )
    report = {"mode": "auto", "status": "gpu_fallback", "profile_file": None}
    # A user choosing auto permits selecting GPU-only, including when an
    # optional native module or graph backend is unavailable.
    apply_compute_selection(engine, baseline)
    if (engine.graph is None or engine.backbone_graph is None
            or not engine.backbone_graph.enabled):
        return {
            **report, "reason": "Measured compute policy requires decode graphs",
            "selection": baseline, "seconds": time.perf_counter() - started,
        }
    cached = load_compute_profile(engine, directory)
    if cached is not None:
        try:
            apply_compute_selection(engine, cached["selection"])
            return {
                **report, "status": "cached", "selection": cached["selection"],
                "profile_file": str(directory / f"{cached['key']}.json"),
                "key": cached["key"], "seconds": time.perf_counter() - started,
            }
        except (ValueError, OSError) as exc:
            report["cached_fallback"] = f"{type(exc).__name__}: {exc}"
            apply_compute_selection(engine, baseline)
    controls = tuple(engine._active_output_controls)
    capacity = engine.model.cache_capacity
    try:
        from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt

        prompt = build_drama3_prompt(
            "<|speaker:0|>你好，这是一次完整的语音推理性能校准，请保持清楚自然的声音。",
            engine.model.tokenizer, engine.model.config)
        emit = getattr(engine, "_emit_status", None)
        measured = calibrate_compute_windows(
            engine, prompt, progress=(
                (lambda row: emit({"stage": "compute_calibration", **row})) if emit else None))
        apply_compute_selection(engine, measured["selection"])
        report.update(status="calibrated", selection=measured["selection"], key=measured["key"])
        try:
            report["profile_file"] = str(save_compute_profile(measured, directory))
        except OSError as exc:
            report["profile_save_warning"] = f"{type(exc).__name__}: {exc}"
    except (ValueError, OSError) as exc:
        apply_compute_selection(engine, baseline)
        report.update(reason=f"{type(exc).__name__}: {exc}", selection=baseline)
    finally:
        # A first scene may have temporarily switched to ordinary candidates.
        # Rebuild only the tower sampler, retaining valid backbone addresses.
        if engine._active_output_controls != controls:
            engine._prepare(max(capacity, engine.model.cache_capacity), output_controls=controls)
    report["seconds"] = time.perf_counter() - started
    return report
