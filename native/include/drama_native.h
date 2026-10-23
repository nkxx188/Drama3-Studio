#pragma once
#include <stdint.h>
#ifdef _WIN32
#  ifdef DRAMA_NATIVE_BUILD
#    define DRAMA_API __declspec(dllexport)
#  else
#    define DRAMA_API __declspec(dllimport)
#  endif
#else
#  define DRAMA_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
/* ABI v1, borrowed immutable CPU buffers, synchronous calls. No tensor ABI. */
DRAMA_API int drama_native_abi(void);
/* Bits: scalar=1, AVX2+FMA=2, AVX512F/BW=4. Includes OS extended-state support. */
DRAMA_API int drama_native_features(void);
DRAMA_API const char* drama_native_error(void);
/* Optional ABI-v1 extension: synchronous byte-preserving gather of contiguous
 * expert rows. Sources and targets are borrowed, disjoint and remain alive
 * through return. All descriptors/IDs are checked before any target write. */
typedef struct DramaGatherRows {
    const uint8_t* source;
    uint8_t* target;
    uint64_t row_bytes;
} DramaGatherRows;
DRAMA_API int drama_gather_rows(const DramaGatherRows* buffers, int buffer_count,
                               int experts, const int64_t* ids, int count);
/* Optional route-planning probe. Prefer low-ranked nonresident slots for CPU;
 * outputs retain original route order. All validation precedes any write.
 * CPU output has cpu_jobs entries, GPU output count-cpu_jobs entries. */
DRAMA_API int drama_partition_routes(
    const int64_t* ids, int count, const int64_t* resident, int resident_count,
    int experts, int cpu_jobs, int64_t* cpu_slots, int64_t* gpu_slots);
DRAMA_API int drama_decode_fp8(const uint8_t* values, float* output, int count);
DRAMA_API int drama_quantize_bf16(const uint16_t* values, uint8_t* output,
                                float* scales, int count);
/* implementation: 0=auto, 1=scalar reference, 2=AVX2, 3=AVX512. */
DRAMA_API void* drama_executor_create(int hidden, int intermediate, int capacity,
                                     int threads, int implementation);
DRAMA_API void drama_executor_destroy(void* executor);
DRAMA_API int drama_executor_implementation(void* executor);
/* Arithmetic: 0=ordinary FP32 CPU dot, 1=empirical SM89 FP8 dot emulation.
 * Mode 1 is hardware-specific research, not a portable GPU contract. */
DRAMA_API int drama_executor_set_arithmetic(void* executor, int arithmetic);
/* Borrow an optional [65536] FP32 table of GPU SiLU(BF16 bits). Caller owns it
 * until executor destruction. Null restores the ordinary C++ activation. */
DRAMA_API int drama_executor_set_silu_table(void* executor, const float* table);
/* Optional strict hybrid guard: reject hidden FP8 halfway boundaries that can
 * depend on GPU division/software conversion. Caller recomputes on the GPU. */
DRAMA_API int drama_executor_set_quantization_guard(void* executor, int enabled);
/* Original contiguous [E,2I,H], [E,2I/128,H/128], [E,H,I], [E,H/128,I/128].
 * x is one BF16 hidden row; out is [jobs,H] FP32 in supplied route order.
 * Duplicate expert IDs retain separate weighted partials. */
DRAMA_API int drama_executor_forward(
    void* executor, const uint8_t* w13, const float* s13,
    const uint8_t* w2, const float* s2, int experts,
    const uint16_t* x, const int64_t* ids, const float* weights,
    int jobs, float* out);
/* Hybrid handoff: borrow the exact GPU-produced FP8 activation and scales,
 * avoiding CPU/GPU division differences at FP8 rounding boundaries. */
DRAMA_API int drama_executor_forward_quantized(
    void* executor, const uint8_t* w13, const float* s13,
    const uint8_t* w2, const float* s2, int experts,
    const uint8_t* xq, const float* x_scales, const int64_t* ids,
    const float* weights, int jobs, float* out);
/* Diagnostic-only ordered intermediates. All output buffers are required.
 * Shared inputs, when non-null, replace the BF16 gate/up or hidden FP8 stage.
 * Sizes: gate_up [jobs,2I], activation/hidden_q [jobs,I],
 * hidden_scales [jobs,I/128], down [jobs,H]. */
typedef struct DramaExpertTrace {
    float* gate_up;
    uint16_t* gate_up_bf16;
    float* activation;
    uint8_t* hidden_q;
    float* hidden_scales;
    float* down;
    const uint16_t* shared_gate_up;
    const uint8_t* shared_hidden_q;
    const float* shared_hidden_scales;
} DramaExpertTrace;
DRAMA_API int drama_executor_trace_quantized(
    void* executor, const uint8_t* w13, const float* s13,
    const uint8_t* w2, const float* s2, int experts,
    const uint8_t* xq, const float* x_scales, const int64_t* ids,
    const float* weights, int jobs, float* out, DramaExpertTrace* trace);
#ifdef __cplusplus
}
#endif
