# Drama CPU core, ABI v1

This is an optional research module. DramaEngine can opt in on SM89 with
`--cpu-experts 2 --native-threads 4`; the default CPU share remains zero.
Original FP8/BF16 GPU inference is the quality reference.

Build on the current Windows developer machine:

```powershell
.venv\Scripts\python.exe tools\build_native.py
```

The build uses the installed MSVC/CMake/Ninja tools and Windows SDK. Source,
objects, DLL and temporary files stay on the project drive. It does not install
or upgrade PyTorch. Output is `.cache/native/drama_native.dll`; its build
receipt records the toolchain and source/artifact hashes.

The library does not link PyTorch. `drama_native.h` exposes a small synchronous
C ABI, and `src/drama_low_vram/native_backend.py` validates and borrows the
existing contiguous CPU tensors. Owners must keep expert tensors immutable.
Output cannot alias inputs. Each executor serializes calls and close; separate
executors have separate scratch and native workers.

Implemented:

- E4M3 finite/NaN/signed-zero decoding and nearest-even encoding.
- Dynamic 128-value activation grouping and FP32 scales.
- Blockwise gate/up, BF16 intermediate rounding, SiLU/requantization, down and
  weighted FP32 partials in the supplied expert order.
- Scalar reference, AVX2/FMA and AVX-512 kernels in separate translation units.
- Runtime CPU/OS feature checks and explicit implementation selection.
- Bounded persistent activation/intermediate scratch; no full-bank BF16 copy.
- Persistent native workers across experts, with checked exception delivery.
- Optional already-quantized GPU activation handoff for hybrid research.
- Optional diagnostic intermediates and GPU gate/up or hidden-quant injections;
  existing ABI v1 entry points remain usable with older libraries.
- Explicit `arithmetic="sm89"` for empirical sixteen-product/truncated-dot
  emulation, with scalar/AVX2/AVX-512 implementations. Shared original GPU
  quantized input and a GPU-generated 256KiB BF16 SiLU table align the measured
  real expert cases and selected-layer teacher-forcing windows.
- Per-call expert-bank binding lets all layers share one executor, worker pool
  and native scratch. Bank owners survive synchronous completion; geometry,
  dtype and output aliasing are checked without copying the weight bank.
- Optional hidden-quantization boundary guard. It reports a distinct status
  after all native workers finish; the engine recomputes those route slots with
  the original GPU arithmetic. Existing ABI v1 arithmetic is unchanged unless
  the new guard is requested.
- Integrated fixed-frame tests cover all 40 layers; completed speech comparisons
  retain identical codes/WAV. The tested CPU share two gives about 4% complete
  request improvement locally. Default selection remains GPU-only.
- Optional ABI-v1 `drama_gather_rows` extension gathers original FP8/scale bytes
  into bounded CPU staging buffers in one C call. It does not quantize, expand
  or copy the full expert bank. `--host-gather auto` uses this extension for
  contiguous groups of at least four experts when available; smaller groups,
  packed arenas and resident host-cache copies retain their existing paths.
  `--host-gather torch` is the default. Missing/older DLLs fall back in auto
  mode; explicit `native` requires the extension.
- Hybrid return can be ordered through a CUDA event rather than a host fence.
  This is a research setting (`synchronize_return=False`), tested across CUDA
  streams, failure and release. The default host fence remains enabled because
  the measured short-window end-to-end benefit was not consistent.

Native `auto` chooses the widest supported ISA, not a measured fastest
implementation/thread count. The future hardware calibration must select by
actual completed workload cost and can select CPU share zero.

Current limitations:

- Batch-one activation only; CPU partials are returned before route reduction.
- `hybrid_decode.py` integrates batch-one cooperation with DramaEngine, using
  one shared worker/workspace and the existing manager-owned staging/caches.
  Automatic CPU-share calibration remains unimplemented; CPU share is explicit.
- Raw CPU and GPU quantization may differ at FP8 half-way boundaries because
  division implementations differ.
- Ordinary `reference` arithmetic still differs from the official GPU kernel.
  The explicit guarded SM89 scheme matches the measured multi-layer windows,
  but is hardware-specific research, not a portable ISA guarantee.
  Raw CPU input quantization remains a separate, unresolved rounding boundary;
  the cooperative path uses shared GPU-produced FP8 input.
- Only this x64 Windows development machine has been built/tested. The source
  includes a GCC/Clang x86 path, which has not been validated.

Validation:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_native_backend.py -q
.venv\Scripts\python.exe tools\profile_engine_runtime.py --frames 12
.venv\Scripts\python.exe tools\benchmark_native_experts.py --cases-count 11 --repeats 7
.venv\Scripts\python.exe tools\diagnose_native_stages.py --cases-count 11
.venv\Scripts\python.exe tools\benchmark_hybrid_layer.py --cases-count 11 --repeats 7
.venv\Scripts\python.exe tools\diagnose_native_stages.py --arithmetic sm89 --gpu-silu-table `
  --output research\native-stage-sm89-table.json --tensors research\native-stage-sm89-table.pt
```

The 12-frame profiling request deliberately stops at the frame budget. It is a
diagnostic/capture task and does not demonstrate complete speech generation.
Captures come from actual layer-zero MoE decode calls, not final backbone
hidden states.

Numerical and component results are in
`research/2026-10-15-native-core-phase1.md`. Stage isolation and the single-layer
experiment are in `research/2026-10-15-native-stage-and-hybrid-layer.md`. The main
remaining difference was isolated to raw tensor-core dot results; small FP32
differences can change BF16 rounding and subsequent FP8 bytes. A shared GPU
input avoids the initial quantization discrepancy. A subsequent hidden FP8
rounding boundary is handled by the new guard and GPU recovery. The alignment
and direct GPU decode work are in
`research/2026-10-15-sm89-alignment-and-decode.md`; multi-layer numerical and
completed speech results are in `research/2026-10-15-multilayer-hybrid.md`.
