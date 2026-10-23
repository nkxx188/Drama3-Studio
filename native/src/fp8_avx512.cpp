#include "internal.hpp"
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace drama {
float dot128_avx512(const uint8_t* weights, const float* values) {
    __m512 sum = _mm512_setzero_ps();
    for (int k = 0; k < 128; k += 16) {
        const __m512i raw = _mm512_cvtepu8_epi32(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights + k)));
        const __m512i mag = _mm512_and_si512(raw, _mm512_set1_epi32(127));
        const __m512i normal = _mm512_add_epi32(_mm512_slli_epi32(mag, 20),
                                               _mm512_set1_epi32(120 << 23));
        __m512 value = _mm512_castsi512_ps(normal);
        value = _mm512_mask_mov_ps(value, _mm512_cmp_epi32_mask(mag, _mm512_set1_epi32(8), _MM_CMPINT_LT),
            _mm512_mul_ps(_mm512_cvtepi32_ps(mag), _mm512_set1_ps(1.0f / 512)));
        value = _mm512_mask_mov_ps(value, _mm512_cmpeq_epi32_mask(mag, _mm512_set1_epi32(127)),
            _mm512_castsi512_ps(_mm512_set1_epi32(0x7fc00000)));
        const __m512i sign = _mm512_slli_epi32(_mm512_and_si512(raw, _mm512_set1_epi32(128)), 24);
        value = _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(value), sign));
        sum = _mm512_fmadd_ps(value, _mm512_loadu_ps(values + k), sum);
    }
    return _mm512_reduce_add_ps(sum);
}

float dot128_sm89_avx512(const uint8_t* weights, const float* values) {
    float acc = 0;
    for (int k = 0; k < 128; k += 16) {
        const __m512i raw = _mm512_cvtepu8_epi32(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights + k)));
        const __m512i mag = _mm512_and_si512(raw, _mm512_set1_epi32(127));
        if (_mm512_cmpeq_epi32_mask(mag, _mm512_set1_epi32(127)))
            throw std::invalid_argument("Non-finite FP8 weight");
        __m512 w = _mm512_castsi512_ps(_mm512_add_epi32(
            _mm512_slli_epi32(mag, 20), _mm512_set1_epi32(120 << 23)));
        w = _mm512_mask_mov_ps(w,
            _mm512_cmp_epi32_mask(mag, _mm512_set1_epi32(8), _MM_CMPINT_LT),
            _mm512_mul_ps(_mm512_cvtepi32_ps(mag), _mm512_set1_ps(1.0f / 512)));
        w = _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(w),
            _mm512_slli_epi32(_mm512_and_si512(raw, _mm512_set1_epi32(128)), 24)));
        const __m512 x = _mm512_loadu_ps(values + k);
        const __m512i abs_x = _mm512_and_si512(_mm512_castps_si512(x), _mm512_set1_epi32(0x7fffffff));
        const __m512i abs_w = _mm512_and_si512(_mm512_castps_si512(w), _mm512_set1_epi32(0x7fffffff));
        const __m512i exponent_sum = _mm512_sub_epi32(
            _mm512_add_epi32(_mm512_srli_epi32(abs_x, 23), _mm512_srli_epi32(abs_w, 23)),
            _mm512_set1_epi32(254));
        const auto nonzero = _mm512_cmpneq_epi32_mask(abs_x, _mm512_setzero_si512()) &
                             _mm512_cmpneq_epi32_mask(abs_w, _mm512_setzero_si512());
        int maximum = _mm512_reduce_max_epi32(_mm512_mask_mov_epi32(
            _mm512_set1_epi32(-1024), nonzero, exponent_sum));
        maximum = std::max(maximum, finite_binary_exponent(acc));
        if (maximum == -1024) continue;
        const float inverse = power_of_two(13 - maximum);
        const __m512 product = _mm512_mul_ps(w, x);
        const int sum = static_cast<int>(acc * inverse) + _mm512_reduce_add_epi32(
            _mm512_cvttps_epi32(_mm512_mul_ps(product, _mm512_set1_ps(inverse))));
        acc = truncate_sm89(static_cast<float>(sum) * power_of_two(maximum - 13));
    }
    return acc;
}
}  // namespace drama
