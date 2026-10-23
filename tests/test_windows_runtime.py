"""Capability selection and portable resolution without relying on other GPUs."""

import hashlib
import json
from pathlib import Path

import pytest

from drama_low_vram.nvidia_policy import policy_for, uses_quantized_torch_fallback
from drama_low_vram.runtime_files import load_native_runtime


@pytest.mark.parametrize("cc,dtype,fp8", [
    ((7, 5), "float16", False), ((8, 6), "bfloat16", False),
    ((8, 9), "bfloat16", True), ((12, 0), "bfloat16", True),
])
def test_supported_generations_have_explicit_generic_and_experimental_policy(cc, dtype, fp8):
    policy = policy_for(cc)
    assert policy.supported and policy.compute_dtype == dtype and policy.fp8_tensor_cores == fp8
    assert policy.experimental_hybrid == (cc == (8, 9))
    assert policy.experimental_pipeline == (cc == (8, 9))


def test_older_devices_are_reported_unsupported():
    assert not policy_for((6, 1)).supported


@pytest.mark.parametrize("mode", [0, 1, 2, 3])
def test_turing_quantized_formats_have_a_torch_fallback(mode):
    assert uses_quantized_torch_fallback((7, 5), mode)
    assert not uses_quantized_torch_fallback((8, 9), mode)
    assert not uses_quantized_torch_fallback((12, 0), mode)


def test_ampere_only_falls_back_for_fp8_relative_scale_formats():
    assert uses_quantized_torch_fallback((8, 6), 2)
    assert uses_quantized_torch_fallback((8, 6), 3)
    assert not uses_quantized_torch_fallback((8, 6), 0)
    assert not uses_quantized_torch_fallback((8, 6), 1)


def test_packaged_runtime_resolves_from_relocated_folder_and_rejects_escape(tmp_path):
    root = tmp_path / "moved app with spaces"
    native = root / "runtime/native"
    native.mkdir(parents=True)
    (native / "drama3_worker.exe").write_bytes(b"worker")
    manifest = native / "manifest.json"
    manifest.write_text(json.dumps({"executable": "drama3_worker.exe", "protocol": 2}))
    result = load_native_runtime(root)
    assert result["executable"] == str(native / "drama3_worker.exe")
    assert result["source"] == str(root)
    assert result["dll_directory"] == str(native)
    manifest.write_text(json.dumps({"executable": "../../../outside.exe"}))
    with pytest.raises(RuntimeError, match="outside"):
        load_native_runtime(root)


def test_reproducible_native_patch_keeps_reference_encoder_fixes():
    root = Path(__file__).resolve().parents[1]
    patch = root / "native/audio-cpp-runtime.patch"
    payload = patch.read_bytes()
    metadata = json.loads((root / "native/audio-cpp-runtime.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(payload).hexdigest() == metadata["sha256"]
    assert len(payload) == metadata["bytes"]
    source = payload.decode("utf-8")
    assert ("+    const bool materialize_input = ctx.backend_type == core::BackendType::Metal"
            " || ctx.backend_type == core::BackendType::Cpu;") in source
    assert ("+    if (ctx.backend_type == core::BackendType::Cpu) {\n"
            "+        x = core::ensure_backend_addressable_layout(ctx, x);") in source
    assert "+    bool encoder_bf16_compute = false;" in source
    assert "fish_drama3.codec_encoder_bf16_compute" in source
