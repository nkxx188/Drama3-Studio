"""Optional synchronous CPU C ABI, independent of the PyTorch binary ABI."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import threading

import torch

from .offload import EXPERT_BUFFER_NAMES, ExpertStore

_IMPLEMENTATIONS = {"auto": 0, "scalar": 1, "avx2": 2, "avx512": 3}


class NativeQuantizationBoundary(RuntimeError):
    """Native work completed, but hidden FP8 rounding requires GPU recovery."""


class _ExpertTrace(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in (
        "gate_up", "gate_up_bf16", "activation", "hidden_q", "hidden_scales",
        "down", "shared_gate_up", "shared_hidden_q", "shared_hidden_scales",
    )]


class _GatherRows(ctypes.Structure):
    _fields_ = [
        ("source", ctypes.c_void_p), ("target", ctypes.c_void_p),
        ("row_bytes", ctypes.c_uint64),
    ]


def default_library_path():
    filename = "drama_native.dll" if os.name == "nt" else "libdrama_native.so"
    root = Path(__file__).resolve().parents[2]
    packaged = root / "runtime/native" / filename
    return packaged if packaged.is_file() else root / ".cache/native" / filename


def _pointer(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def _reject_output_alias(output, inputs):
    first = output.data_ptr()
    last = first + output.numel() * output.element_size()
    for tensor in inputs:
        source_first = tensor.data_ptr()
        source_last = source_first + tensor.numel() * tensor.element_size()
        if first < source_last and source_first < last:
            raise ValueError("Native output must not overlap input or expert weights")


class NativeLibrary:
    def __init__(self, path=None):
        self.path = Path(path or default_library_path()).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Optional native library missing: {self.path}. Run tools/build_native.py."
            )
        self.dll = ctypes.CDLL(str(self.path))
        for name in ("drama_native_abi", "drama_native_features"):
            function = getattr(self.dll, name)
            function.argtypes = []
            function.restype = ctypes.c_int
        self.dll.drama_native_error.argtypes = []
        self.dll.drama_native_error.restype = ctypes.c_char_p
        self.gather_rows = getattr(self.dll, "drama_gather_rows", None)
        if self.gather_rows is not None:
            self.gather_rows.argtypes = [
                ctypes.POINTER(_GatherRows), ctypes.c_int, ctypes.c_int,
                ctypes.c_void_p, ctypes.c_int,
            ]
            self.gather_rows.restype = ctypes.c_int
        self.dll.drama_decode_fp8.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        self.dll.drama_decode_fp8.restype = ctypes.c_int
        self.dll.drama_quantize_bf16.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int]
        self.dll.drama_quantize_bf16.restype = ctypes.c_int
        self.dll.drama_executor_create.argtypes = [ctypes.c_int] * 5
        self.dll.drama_executor_create.restype = ctypes.c_void_p
        self.dll.drama_executor_destroy.argtypes = [ctypes.c_void_p]
        self.dll.drama_executor_destroy.restype = None
        self.dll.drama_executor_implementation.argtypes = [ctypes.c_void_p]
        self.dll.drama_executor_implementation.restype = ctypes.c_int
        self.set_arithmetic = getattr(self.dll, "drama_executor_set_arithmetic", None)
        if self.set_arithmetic is not None:
            self.set_arithmetic.argtypes = [ctypes.c_void_p, ctypes.c_int]
            self.set_arithmetic.restype = ctypes.c_int
        self.set_silu_table = getattr(self.dll, "drama_executor_set_silu_table", None)
        if self.set_silu_table is not None:
            self.set_silu_table.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            self.set_silu_table.restype = ctypes.c_int
        self.set_quantization_guard = getattr(self.dll, "drama_executor_set_quantization_guard", None)
        if self.set_quantization_guard is not None:
            self.set_quantization_guard.argtypes = [ctypes.c_void_p, ctypes.c_int]
            self.set_quantization_guard.restype = ctypes.c_int
        self.dll.drama_executor_forward.argtypes = (
            [ctypes.c_void_p] * 5 + [ctypes.c_int] +
            [ctypes.c_void_p] * 3 + [ctypes.c_int, ctypes.c_void_p]
        )
        self.dll.drama_executor_forward.restype = ctypes.c_int
        self.dll.drama_executor_forward_quantized.argtypes = (
            [ctypes.c_void_p] * 5 + [ctypes.c_int] +
            [ctypes.c_void_p] * 4 + [ctypes.c_int, ctypes.c_void_p]
        )
        self.dll.drama_executor_forward_quantized.restype = ctypes.c_int
        self.trace_quantized = getattr(self.dll, "drama_executor_trace_quantized", None)
        if self.trace_quantized is not None:
            self.trace_quantized.argtypes = (
                self.dll.drama_executor_forward_quantized.argtypes +
                [ctypes.POINTER(_ExpertTrace)]
            )
            self.trace_quantized.restype = ctypes.c_int
        if self.dll.drama_native_abi() != 1:
            raise RuntimeError("Unsupported drama_native ABI")
        self.features = self.dll.drama_native_features()

    def gather_experts(self, store, ids, targets):
        """Gather original bytes into bounded contiguous CPU staging buffers."""
        if self.gather_rows is None:
            raise RuntimeError("Native gather requires rebuilding tools/build_native.py")
        store.validate()
        if (ids.device.type != "cpu" or ids.dtype != torch.int64 or ids.ndim != 1
                or not ids.is_contiguous() or not 1 <= ids.numel() <= 256):
            raise ValueError("Gather IDs must be a bounded contiguous CPU int64 vector")
        sources = [getattr(store, name) for name in EXPERT_BUFFER_NAMES]
        outputs = []
        for name, source in zip(EXPERT_BUFFER_NAMES, sources):
            target = targets[name]
            if (target.device.type != "cpu" or target.dtype != source.dtype
                    or target.shape != (ids.numel(), *source.shape[1:])
                    or not target.is_contiguous()):
                raise ValueError("Native gather requires matching contiguous CPU targets")
            _reject_output_alias(target, sources + [ids] + outputs)
            outputs.append(target)
        descriptors = (_GatherRows * len(sources))(*[
            _GatherRows(source.data_ptr(), target.data_ptr(),
                        source.numel() // store.num_experts * source.element_size())
            for source, target in zip(sources, outputs)
        ])
        self.check(self.gather_rows(
            descriptors, len(sources), store.num_experts, _pointer(ids), ids.numel()))

    def check(self, status):
        if status:
            message = self.dll.drama_native_error()
            text = message.decode("utf-8", errors="replace") if message else "Native error"
            error = (
                NativeQuantizationBoundary
                if text == "GPU FP8 rounding boundary requires fallback" else RuntimeError
            )
            raise error(text)

    def decode(self, raw):
        if raw.device.type != "cpu" or raw.dtype != torch.uint8 or not raw.is_contiguous():
            raise ValueError("Decode requires contiguous CPU uint8 bytes")
        output = torch.empty(raw.shape, dtype=torch.float32)
        self.check(self.dll.drama_decode_fp8(_pointer(raw), _pointer(output), raw.numel()))
        return output

    def quantize(self, values):
        if (values.device.type != "cpu" or values.dtype != torch.bfloat16
                or not values.is_contiguous() or values.numel() < 128
                or values.numel() % 128):
            raise ValueError("Quantization requires contiguous BF16 CPU 128-value blocks")
        raw = torch.empty_like(values, dtype=torch.uint8)
        scales = torch.empty(values.numel() // 128, dtype=torch.float32)
        self.check(self.dll.drama_quantize_bf16(
            _pointer(values), _pointer(raw), _pointer(scales), values.numel(),
        ))
        return raw, scales


class NativeExpertExecutor:
    """One BF16 row -> ordered FP32 expert partials; opt-in research backend.

    Calls hold references to the original immutable buffers and synchronously
    wait for all native workers. CDLL releases the GIL during native execution.
    A lock serializes calls/close on one executor; distinct executors are separate.
    This class is not wired into the default CUDA engine.
    """
    def __init__(self, store: ExpertStore, *, capacity=8, threads=1,
                 implementation="auto", arithmetic="reference", silu_table=None,
                 guard_fp8_boundaries=False, library=None):
        store.validate()
        if implementation not in _IMPLEMENTATIONS:
            raise ValueError("Unknown native implementation")
        if arithmetic not in ("reference", "sm89"):
            raise ValueError("Unknown native arithmetic")
        if not isinstance(guard_fp8_boundaries, bool):
            raise ValueError("guard_fp8_boundaries must be a boolean")
        if (isinstance(capacity, bool) or not isinstance(capacity, int)
                or not 1 <= capacity <= 256 or isinstance(threads, bool)
                or not isinstance(threads, int) or not 1 <= threads <= 64):
            raise ValueError("Invalid native capacity or thread count")
        self.library = library or NativeLibrary()
        if arithmetic == "sm89" and self.library.set_arithmetic is None:
            raise RuntimeError("SM89 arithmetic requires rebuilding tools/build_native.py")
        if guard_fp8_boundaries and self.library.set_quantization_guard is None:
            raise RuntimeError("FP8 boundary guard requires rebuilding tools/build_native.py")
        if silu_table is not None and (
            silu_table.device.type != "cpu" or silu_table.dtype != torch.float32
            or silu_table.shape != (65536,) or not silu_table.is_contiguous()
        ):
            raise ValueError("GPU SiLU table must be contiguous CPU FP32 [65536]")
        if silu_table is not None and self.library.set_silu_table is None:
            raise RuntimeError("GPU SiLU table requires rebuilding tools/build_native.py")
        # Keep a private immutable-by-convention copy for the borrowed native pointer.
        self._silu_table = None if silu_table is None else silu_table.clone()
        self.store = store
        self.capacity = capacity
        self.threads = threads
        self.arithmetic = arithmetic
        self._lock = threading.Lock()
        self._handle = self.library.dll.drama_executor_create(
            store.hidden, store.intermediate, capacity, threads,
            _IMPLEMENTATIONS[implementation],
        )
        if not self._handle:
            self.library.check(-1)
        self.implementation = {
            1: "scalar", 2: "avx2", 3: "avx512",
        }[self.library.dll.drama_executor_implementation(self._handle)]
        if arithmetic == "sm89":
            self.library.check(self.library.set_arithmetic(self._handle, 1))
        if self._silu_table is not None:
            self.library.check(self.library.set_silu_table(
                self._handle, _pointer(self._silu_table)))
        if guard_fp8_boundaries:
            self.library.check(self.library.set_quantization_guard(self._handle, 1))

    def _bank(self, store):
        """Bind a bank for one synchronous call without changing executor state."""
        if store is None or store is self.store:
            return self.store
        store.validate()
        if (store.hidden, store.intermediate) != (
            self.store.hidden, self.store.intermediate
        ):
            raise ValueError("Native expert bank must match the executor geometry")
        return store

    def forward_pairs(self, x, ids, weights, *, out=None, store=None):
        store = self._bank(store)
        if (x.device.type != "cpu" or x.dtype != torch.bfloat16
                or x.shape not in ((self.store.hidden,), (1, self.store.hidden))
                or not x.is_contiguous()):
            raise ValueError("Native decode expects one contiguous BF16 CPU hidden row")
        if (ids.device.type != "cpu" or ids.dtype != torch.int64
                or ids.ndim != 1 or not ids.is_contiguous()
                or not 1 <= ids.numel() <= self.capacity):
            raise ValueError("Native expert IDs must be a bounded int64 CPU vector")
        if (weights.device.type != "cpu" or weights.dtype != torch.float32
                or weights.shape != ids.shape or not weights.is_contiguous()):
            raise ValueError("Native route weights must be contiguous FP32 CPU values")
        expected_shape = (ids.numel(), self.store.hidden)
        if out is None:
            out = torch.empty(expected_shape, dtype=torch.float32)
        if (out.device.type != "cpu" or out.dtype != torch.float32
                or out.shape != expected_shape or not out.is_contiguous()):
            raise ValueError("Invalid native FP32 output buffer")
        _reject_output_alias(out, [x, ids, weights] + [
            getattr(bank, name) for bank in (self.store, store)
            for name in EXPERT_BUFFER_NAMES
        ] + ([self._silu_table] if self._silu_table is not None else []))
        with self._lock:
            if not self._handle:
                raise RuntimeError("Native executor is closed")
            buffers = [_pointer(getattr(store, name)) for name in EXPERT_BUFFER_NAMES]
            self.library.check(self.library.dll.drama_executor_forward(
                self._handle, *buffers, store.num_experts,
                _pointer(x), _pointer(ids), _pointer(weights), ids.numel(), _pointer(out),
            ))
        return out

    def forward_quantized_pairs(self, xq, scales, ids, weights, *, out=None, store=None):
        store = self._bank(store)
        if (xq.device.type != "cpu" or xq.dtype not in (torch.uint8, torch.float8_e4m3fn)
                or xq.numel() != self.store.hidden or not xq.is_contiguous()
                or scales.device.type != "cpu" or scales.dtype != torch.float32
                or scales.numel() != self.store.hidden // 128 or not scales.is_contiguous()):
            raise ValueError("Expected contiguous CPU FP8 activation bytes and FP32 scales")
        if (ids.device.type != "cpu" or ids.dtype != torch.int64 or ids.ndim != 1
                or not ids.is_contiguous() or not 1 <= ids.numel() <= self.capacity
                or weights.device.type != "cpu" or weights.dtype != torch.float32
                or weights.shape != ids.shape or not weights.is_contiguous()):
            raise ValueError("Invalid native route IDs/weights")
        shape = (ids.numel(), self.store.hidden)
        if out is None:
            out = torch.empty(shape, dtype=torch.float32)
        if (out.device.type != "cpu" or out.dtype != torch.float32
                or out.shape != shape or not out.is_contiguous()):
            raise ValueError("Invalid native FP32 output buffer")
        _reject_output_alias(out, [xq, scales, ids, weights] + [
            getattr(bank, name) for bank in (self.store, store)
            for name in EXPERT_BUFFER_NAMES
        ] + ([self._silu_table] if self._silu_table is not None else []))
        with self._lock:
            if not self._handle:
                raise RuntimeError("Native executor is closed")
            buffers = [_pointer(getattr(store, name)) for name in EXPERT_BUFFER_NAMES]
            self.library.check(self.library.dll.drama_executor_forward_quantized(
                self._handle, *buffers, store.num_experts, _pointer(xq),
                _pointer(scales), _pointer(ids), _pointer(weights), ids.numel(), _pointer(out),
            ))
        return out

    def trace_quantized_pairs(
        self, xq, scales, ids, weights, *, shared_gate_up=None,
        shared_hidden_q=None, shared_hidden_scales=None,
    ):
        """Capture native stages; optionally inject GPU intermediates for isolation."""
        if self.library.trace_quantized is None:
            raise RuntimeError("Native diagnostics require rebuilding tools/build_native.py")
        # Reuse the ordinary call's input contract before passing borrowed pointers.
        if (xq.device.type != "cpu" or xq.dtype not in (torch.uint8, torch.float8_e4m3fn)
                or xq.numel() != self.store.hidden or not xq.is_contiguous()
                or scales.device.type != "cpu" or scales.dtype != torch.float32
                or scales.numel() != self.store.hidden // 128 or not scales.is_contiguous()):
            raise ValueError("Expected contiguous CPU FP8 activation bytes and FP32 scales")
        if (ids.device.type != "cpu" or ids.dtype != torch.int64 or ids.ndim != 1
                or not ids.is_contiguous() or not 1 <= ids.numel() <= self.capacity
                or weights.device.type != "cpu" or weights.dtype != torch.float32
                or weights.shape != ids.shape or not weights.is_contiguous()):
            raise ValueError("Invalid native route IDs/weights")
        count, hidden, intermediate = ids.numel(), self.store.hidden, self.store.intermediate
        for value, shape, dtypes in (
            (shared_gate_up, (count, 2 * intermediate), (torch.bfloat16,)),
            (shared_hidden_q, (count, intermediate), (torch.uint8, torch.float8_e4m3fn)),
            (shared_hidden_scales, (count, intermediate // 128), (torch.float32,)),
        ):
            if value is not None and (
                value.device.type != "cpu" or value.dtype not in dtypes
                or value.shape != shape or not value.is_contiguous()
            ):
                raise ValueError("Invalid shared diagnostic intermediate")
        if (shared_hidden_q is None) != (shared_hidden_scales is None):
            raise ValueError("Shared hidden bytes and scales must be supplied together")
        outputs = {
            name: torch.empty(shape, dtype=dtype)
            for name, shape, dtype in (
                ("gate_up", (count, 2 * intermediate), torch.float32),
                ("gate_up_bf16", (count, 2 * intermediate), torch.bfloat16),
                ("activation", (count, intermediate), torch.float32),
                ("hidden_q", (count, intermediate), torch.uint8),
                ("hidden_scales", (count, intermediate // 128), torch.float32),
                ("down", (count, hidden), torch.float32),
                ("partials", (count, hidden), torch.float32),
            )
        }
        trace = _ExpertTrace(
            *[_pointer(outputs[name]) for name, _ in _ExpertTrace._fields_[:6]],
            *[None if value is None else _pointer(value) for value in (
                shared_gate_up, shared_hidden_q, shared_hidden_scales,
            )],
        )
        with self._lock:
            if not self._handle:
                raise RuntimeError("Native executor is closed")
            buffers = [_pointer(getattr(self.store, name)) for name in EXPERT_BUFFER_NAMES]
            self.library.check(self.library.trace_quantized(
                self._handle, *buffers, self.store.num_experts, _pointer(xq),
                _pointer(scales), _pointer(ids), _pointer(weights), count,
                _pointer(outputs["partials"]), ctypes.byref(trace),
            ))
        return outputs

    def close(self):
        with self._lock:
            if self._handle:
                self.library.dll.drama_executor_destroy(self._handle)
                self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def __del__(self):
        if getattr(self, "_handle", None):
            try:
                self.close()
            except Exception:
                pass
