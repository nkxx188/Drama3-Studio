"""Protocol/lifetime tests with a real lightweight child, no GPU required."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

import drama_low_vram.gguf_runtime as runtime
from drama_low_vram.gguf_resources import Layout, GIB, MIB, plan_resources


@pytest.fixture
def engine(tmp_path, monkeypatch):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"test-model-identity")
    research = tmp_path / "research"
    research.mkdir()
    (research / "phase35-worker-build.json").write_text(json.dumps({
        "executable": sys.executable, "source": str(tmp_path), "cuda_toolkit": str(tmp_path),
    }))
    fake = tmp_path / "worker.py"
    fake.write_text('''
import sys,json,time,os,wave
from pathlib import Path
config=json.loads(Path(sys.argv[1]).read_text())
def emit(value): print("@drama3 "+json.dumps(value),flush=True)
emit({"event":"ready","load_seconds":0.001,"protocol":int(os.environ.get("DRAMA_TEST_PROTOCOL","1")),
      "context_resize":os.environ.get("DRAMA_TEST_CONTEXT_RESIZE")=="1"})
for line in sys.stdin:
 r=json.loads(line)
 if r.get("op")=="close":break
 if r.get("op")=="reserve_context":
  if os.environ.get("DRAMA_TEST_RESIZE_DELAY"):time.sleep(30)
  if os.environ.get("DRAMA_TEST_RESIZE_OOM") and int(config["options"]["fish_drama3.context_length"])<1536:
   emit({"event":"error","id":r["id"],"message":"CUDA out of memory","recoverable":False});sys.exit(1)
  config["options"]["fish_drama3.context_length"]=str(r["capacity"])
  emit({"event":"context_ready","id":r["id"],"capacity":r["capacity"],"seconds":0.002});continue
 if r.get("op")=="prewarm_reference":
  options=r["request"]["options"]
  if options.get("encode_reference_only")!="true":
   emit({"event":"error","id":r["id"],"message":"speech generation is not allowed during prewarm","recoverable":True});continue
  refs=json.loads(options["references"])
  prepared=[json.dumps({"text":ref["text"],"vq_codes":[[book]*4 for book in range(10)]})
            for ref in refs]
  emit({"event":"prewarmed","id":r["id"],"references":prepared});continue
 text=r["request"]["text"]
 if "invalid" in text:
  emit({"event":"error","id":r["id"],"message":"invalid request","recoverable":True});continue
 if "slow" in text: time.sleep(30)
 if "fatal" in text:
  emit({"event":"error","id":r["id"],"message":"execution failed","recoverable":False});sys.exit(1)
 if "oom" in text and config["options"]["fish_drama3.codec_decoder_backend"]=="cuda":
  emit({"event":"error","id":r["id"],"message":"CUDA out of memory","recoverable":False});sys.exit(1)
 if "needscontext" in text and int(config["options"]["fish_drama3.context_length"])<1536:
  emit({"event":"error","id":r["id"],"message":"capacity exceeded; required_context=1300","recoverable":True});continue
 output=Path(r["output"])
 def wav(path):
  with wave.open(str(path),"wb") as f:
   f.setparams((1,2,44100,0,"NONE","not compressed"));f.writeframes(b"\\0\\0"*65536)
 wav(output)
 take=output.with_suffix(".take.json")
 take.write_text(json.dumps({"text":text,"vq_codes":[[i]*32 for i in range(10)]}))
 if r.get("streaming"):
  part=output.with_name(output.stem+".part-1.wav");wav(part)
  emit({"event":"audio","id":r["id"],"path":str(part),"index":0,"start_sample":0,
        "samples":65536,"sample_rate":44100})
  if "lateoom" in text:
   emit({"event":"error","id":r["id"],"message":"CUDA out of memory","recoverable":False});sys.exit(1)
 emit({"event":"result","id":r["id"],"output":r["output"],"pid":os.getpid(),
       "native_take_file":str(take),"request_echo":r["request"],
       "generation":{"reached_limit":False,"frames":32},"request_seconds":0.001})
''', encoding="utf-8")
    real_popen = subprocess.Popen

    def popen(command, **kwargs):
        return real_popen([sys.executable, str(fake), command[1]], **kwargs)

    monkeypatch.setattr(runtime.subprocess, "Popen", popen)
    layout = Layout(str(model), 40 * GIB, 40, 256, 32640 * MIB, int(127.5 * MIB),
                    7074 * MIB, 1326 * MIB, 793 * MIB, 954 * MIB, 986 * MIB, True, (8,),
                    codec_cuda_supported=True, host_embedding_supported=True)
    plan = plan_resources(layout, free_vram=int(7.5 * GIB), available_ram=24 * GIB, cores=4)
    result = runtime.GGUFEngine(model, root=tmp_path, plan=plan)
    yield result
    result.close()


def test_independent_requests_share_process_and_model_change_invalidates(engine, tmp_path, monkeypatch):
    one = engine.generate("one", tmp_path / "one.wav")
    two = engine.generate("two", tmp_path / "two.wav")
    assert one["pid"] == two["pid"]
    assert not one["session_reused"] and two["session_reused"]
    monkeypatch.setattr(runtime, "native_memory", lambda *_: {"free_bytes": int(7.5 * GIB)})
    monkeypatch.setattr(runtime.Layout, "read", lambda _: engine.plan.layout)
    engine.model_path.write_bytes(b"changed-model-identity-longer")
    three = engine.generate("three", tmp_path / "three.wav")
    assert three["pid"] != one["pid"] and not three["session_reused"]


def test_zero_top_k_is_accepted_and_forwarded_to_native_worker(engine, tmp_path):
    result = engine.generate("unbounded top-k", tmp_path / "top-k-zero.wav", top_k=0)
    assert result["request_echo"]["top_k"] == 0


def test_validation_error_keeps_session_and_fatal_error_reaps_it(engine, tmp_path):
    first = engine.generate("one", tmp_path / "one.wav")
    with pytest.raises(runtime.NativeFailure, match="invalid request"):
        engine.generate("invalid", tmp_path / "invalid.wav")
    assert engine.loaded
    assert engine.generate("two", tmp_path / "two.wav")["pid"] == first["pid"]
    with pytest.raises(runtime.NativeFailure, match="execution failed"):
        engine.generate("fatal", tmp_path / "fatal.wav")
    assert not engine.loaded
    assert engine.generate("three", tmp_path / "three.wav")["pid"] != first["pid"]


def test_cancellation_reaps_child_and_next_request_starts_cleanly(engine, tmp_path):
    engine.load()
    process = engine._process
    started = time.monotonic()
    with pytest.raises(runtime.GenerationCancelled):
        engine.generate("slow", tmp_path / "cancel.wav",
                        cancelled=lambda: time.monotonic() - started > 0.2)
    assert process.poll() is not None
    assert not engine.loaded and not (tmp_path / "cancel.wav").exists()
    assert engine.generate("again", tmp_path / "again.wav")["pid"] != process.pid


def test_allocation_fallback_reaps_failed_child_before_restart(engine, tmp_path, monkeypatch):
    original = engine._receive
    processes = []

    def receive(**kwargs):
        if not processes:
            processes.append(engine._process)
            raise runtime.NativeFailure("cannot allocate CUDA buffer")
        assert processes[0].poll() is not None
        return original(**kwargs)

    monkeypatch.setattr(engine, "_receive", receive)
    # More than eight slots guarantees a smaller-cache fallback.
    engine.plan = replace(engine.plan, cache_experts=12)
    engine.load()
    assert engine.plan.cache_experts == 8
    assert len(engine.attempts) == 2


def test_reference_conversion_uses_content_identity_and_reuses_cached_wav(tmp_path):
    import numpy as np
    import soundfile as sf
    source = tmp_path / "reference.flac"
    sf.write(source, np.zeros(800, dtype=np.float32), 8000)
    one = runtime.reference_wav(source, tmp_path)
    stat = one.stat().st_mtime_ns
    assert runtime.reference_wav(source, tmp_path) == one
    assert one.stat().st_mtime_ns == stat
    sf.write(source, np.ones(800, dtype=np.float32) * 0.1, 8000)
    assert runtime.reference_wav(source, tmp_path) != one


def test_native_reference_preparation_caches_codes_without_generating_audio(
        engine, tmp_path, monkeypatch):
    import numpy as np
    import soundfile as sf
    from types import SimpleNamespace

    monkeypatch.setenv("DRAMA_TEST_PROTOCOL", "5")
    engine.build["protocol"] = 5
    sources = [tmp_path / "voice-a.wav", tmp_path / "voice-b.wav"]
    sf.write(sources[0], np.zeros(8000, dtype=np.float32), 8000)
    sf.write(sources[1], np.ones(8000, dtype=np.float32) * .1, 8000)
    references = [
        SimpleNamespace(audio=source, transcript=lambda: f"Reference {index}.")
        for index, source in enumerate(sources)
    ]

    prepared = engine.prewarm_references(references)
    assert prepared["status"] == "ready" and prepared["references"] == 2
    process = engine._process
    assert list(tmp_path.glob("*.wav")) == sources
    assert engine.references_prepared(references)

    result = engine.generate(
        "Generated words.", tmp_path / "generated.wav", references=references)
    descriptors = json.loads(result["request_echo"]["options"]["references"])
    assert all("audio_path" not in descriptor for descriptor in descriptors)
    assert all(descriptor["vq_codes"] == [[book] * 4 for book in range(10)]
               for descriptor in descriptors)
    assert engine._process is process and result["session_reused"]


def test_native_reference_preparation_requires_protocol_five(engine, tmp_path):
    from types import SimpleNamespace

    reference = SimpleNamespace(audio=tmp_path / "missing.wav", transcript=lambda: "Words.")
    with pytest.raises(RuntimeError, match="protocol 5"):
        engine.prewarm_references([reference])
    assert not engine.loaded


def test_inflight_oom_reaches_cpu_codec_fallback_after_host_and_cache_retries(engine, tmp_path, monkeypatch):
    engine.plan = replace(engine.plan, host_storage="pin", cache_experts=12, decoder_backend="cuda")
    original_receive = engine._receive
    processes = []
    def receive(**kwargs):
        if not processes or engine._process.pid != processes[-1].pid:
            if processes:
                assert processes[-1].poll() is not None
            processes.append(engine._process)
        return original_receive(**kwargs)
    monkeypatch.setattr(engine, "_receive", receive)
    result = engine.generate("oom", tmp_path / "recovered.wav")
    assert result["complete"] and len(processes) == 4
    assert engine.plan.host_storage == "mapped" and engine.plan.cache_experts == 8
    assert engine.plan.decoder_backend == "cpu"
    assert all(process.poll() is not None for process in processes[:-1])


def test_capacity_error_grows_context_in_a_fresh_worker_without_changing_request(engine, tmp_path, monkeypatch):
    engine.context = 512
    engine.plan = replace(engine.plan, context_length=512)
    engine.load()
    first = engine._process
    engine._provided_plan = False
    monkeypatch.setattr(runtime, "native_memory", lambda *_: {"free_bytes": int(7.5 * GIB)})
    layout = engine.plan.layout
    monkeypatch.setattr(runtime.Layout, "read", lambda _: layout)
    result = engine.generate("needscontext", tmp_path / "long-reference.wav")
    assert result["complete"] and engine.context == 1536
    assert engine._process.pid != first.pid and first.poll() is not None
    assert not result["session_reused"]

@pytest.mark.parametrize("task", ["", "infill", "stream", "joint"])
def test_capacity_growth_preserves_model_process_across_workflows(engine, tmp_path, monkeypatch, task):
    import torch
    from drama_low_vram.protocols import InfillSpan
    monkeypatch.setenv("DRAMA_TEST_PROTOCOL", "4")
    monkeypatch.setenv("DRAMA_TEST_CONTEXT_RESIZE", "1")
    engine.build["protocol"] = 4
    engine.context = 512
    engine.plan = replace(engine.plan, context_length=512)
    engine.generate("initial", tmp_path / "initial.wav")
    process, directory = engine._process, engine._directory
    engine._provided_plan = False
    events = []
    engine.status_callback = events.append
    extra = {"infill": InfillSpan(torch.zeros(10, 20, dtype=torch.int64), 2, 8)} if task == "infill" else {}
    result = engine.generate("needscontext", tmp_path / "resized.wav", task_mode=task, **extra)
    assert engine._process is process and process.poll() is None
    assert engine._directory == directory and engine.context == 1536
    assert engine.plan.context_length == 1536 and engine._signature() == engine._identity
    assert result["session_reused"] and result["context_resizes"] == [
        {"from": 512, "to": 1536, "seconds": 0.002, "weights_reloaded": False}]
    assert result["request_echo"]["seed"] == 123
    assert not any(event.get("timing_phase") == "model_load" for event in events)
    again = engine.generate("again", tmp_path / "again.wav")
    assert engine._process is process and again["context_resizes"] == [] and again["session_reused"]


def test_resize_allocation_failure_falls_back_to_fresh_plan_and_reaps_worker(engine, tmp_path, monkeypatch):
    monkeypatch.setenv("DRAMA_TEST_CONTEXT_RESIZE", "1")
    monkeypatch.setenv("DRAMA_TEST_RESIZE_OOM", "1")
    engine.context = 512
    engine.plan = replace(engine.plan, context_length=512)
    engine.generate("initial", tmp_path / "initial.wav")
    process = engine._process
    engine._provided_plan = False
    layout = engine.plan.layout
    monkeypatch.setattr(runtime, "native_memory", lambda *_: {"free_bytes": int(7.5 * GIB)})
    monkeypatch.setattr(runtime.Layout, "read", lambda _: layout)
    result = engine.generate("needscontext", tmp_path / "resized.wav")
    assert result["complete"] and engine.context == 1536
    assert process.poll() is not None and engine._process is not process
    assert not result["session_reused"]


def test_cancellation_during_cache_resize_reaps_worker(engine, tmp_path, monkeypatch):
    monkeypatch.setenv("DRAMA_TEST_CONTEXT_RESIZE", "1")
    engine.context = 512
    engine.plan = replace(engine.plan, context_length=512)
    engine.load()
    process = engine._process
    engine._provided_plan = False
    received = engine._receive
    def receive(**kwargs):
        if kwargs.get("request_id") and not hasattr(receive, "first"):
            receive.first = True
            raise runtime.NativeFailure("capacity exceeded; required_context=1300", recoverable=True)
        return received(**kwargs)
    monkeypatch.setattr(engine, "_receive", receive)
    started = time.monotonic()
    with pytest.raises(runtime.GenerationCancelled):
        engine.generate("slow", tmp_path / "cancel-resize.wav",
                        cancelled=lambda: time.monotonic() - started > 0.1)
    assert process.poll() is not None and not engine.loaded


def test_excessive_reference_is_rejected_without_loading_or_silently_trimming(engine, tmp_path):
    from types import SimpleNamespace
    import numpy as np
    import soundfile as sf
    path = tmp_path / "long.wav"
    sf.write(path, np.zeros(8000 * 121, dtype=np.float32), 8000)
    with pytest.raises(ValueError, match="120 seconds"):
        engine.generate("hello", tmp_path / "out.wav", references=[
            SimpleNamespace(audio=path, transcript=lambda: "Long reference")])
    assert not engine.loaded


def test_native_probe_selects_device_and_disables_unvalidated_pipeline(engine, monkeypatch):
    from types import SimpleNamespace
    layout = engine.plan.layout
    engine.plan = None
    engine._provided_plan = False
    monkeypatch.setattr(runtime.Layout, "read", lambda _: layout)
    monkeypatch.setattr(runtime, "native_memory", lambda *_: {
        "free_bytes": 12 * GIB, "selected_device": 1,
        "devices": [{"index": 1, "major": 7, "minor": 5, "supported": True,
                     "free_bytes": 12 * GIB, "total_bytes": 12 * GIB}]})
    engine.load()
    assert engine.selected_device == 1
    assert engine.hardware_policy.compute_dtype == "float16"
    config = json.loads((engine._directory / "config.json").read_text())
    assert config["device"] == 1 and config["options"]["fish_drama3.cpu_expert_layers"] == "0"
    engine.close()
    engine.pipeline_width = 2
    with pytest.raises(ValueError, match="only for SM89"):
        engine.load()
    assert not engine.loaded


def test_advanced_request_sends_real_history_and_delivers_audio_before_result(engine, tmp_path, monkeypatch):
    import torch
    from drama_low_vram.protocols import PreviousTake
    monkeypatch.setenv("DRAMA_TEST_PROTOCOL", "3")
    engine.build["protocol"] = 3
    history = PreviousTake("Prior speech.", torch.arange(10).reshape(10, 1))
    delivered = []
    result = engine.generate(
        "Hello.", tmp_path / "advanced.wav", previous_takes=[history], direction="Softer.",
        language="zh", duration_frames=32, streaming_audio=True, segment_ready=delivered.append)
    assert delivered and result["audio"]["samples"] == delivered[0]["samples"]
    options = result["request_echo"]["options"]
    descriptor = json.loads(options["previous_takes"])[0]
    assert descriptor["vq_codes"] == [[i] for i in range(10)]
    assert "Direction: Softer." in result["request_echo"]["text"]
    assert "[audio tokens: 32]" in result["request_echo"]["text"]
    assert Path(result["take_file"]).is_file()


def test_oom_after_audio_delivery_never_retries_or_duplicates_segments(engine, tmp_path, monkeypatch):
    monkeypatch.setenv("DRAMA_TEST_PROTOCOL", "3")
    engine.build["protocol"] = 3
    delivered = []
    with pytest.raises(runtime.NativeFailure, match="out of memory"):
        engine.generate("lateoom", tmp_path / "late.wav",
                        streaming_audio=True, segment_ready=delivered.append)
    assert len(delivered) == 1 and Path(delivered[0]["path"]).is_file()
    assert not engine.loaded


def test_task_markers_and_unsupported_worker_are_rejected_before_loading(engine, tmp_path):
    with pytest.raises(ValueError, match="Structural task"):
        engine.generate("Hello <|fim_prefix|>", tmp_path / "bad.wav")
    with pytest.raises(RuntimeError, match="protocol 3"):
        engine.generate("Hello.", tmp_path / "bad.wav", task_mode="stream", lookahead="Next.")
    assert not engine.loaded
