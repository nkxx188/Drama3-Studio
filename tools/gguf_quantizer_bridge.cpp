#include "ggml.h"
#include "ggml-quants.h"
#include <cstdint>
#include <omp.h>

extern "C" __declspec(dllexport) int drama_quantize_rows(
    int type, const float * source, void * destination, int64_t rows,
    int64_t columns, int threads) {
    if ((type != GGML_TYPE_Q4_K && type != GGML_TYPE_Q8_0) ||
        columns % ggml_blck_size(static_cast<ggml_type>(type)) ||
        rows < 1 || threads < 1) return -1;
    const auto kind = static_cast<ggml_type>(type);
    ggml_quantize_init(kind);
    const auto bytes = ggml_row_size(kind, columns);
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t row = 0; row < rows; ++row) {
        auto * target = static_cast<char *>(destination) + row * bytes;
        if (type == GGML_TYPE_Q4_K)
            quantize_row_q4_K_ref(source + row * columns,
                reinterpret_cast<block_q4_K *>(target), columns);
        else
            quantize_row_q8_0_ref(source + row * columns,
                reinterpret_cast<block_q8_0 *>(target), columns);
    }
    return 0;
}
