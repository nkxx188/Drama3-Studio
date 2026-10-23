#include "internal.hpp"
#include <immintrin.h>
#include <algorithm>
#include <cmath>

namespace drama {
static __m256 decode8(const uint8_t* source) {
    const __m128i packed = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(source));
    const __m256i raw = _mm256_cvtepu8_epi32(packed);
    const __m256i mag = _mm256_and_si256(raw, _mm256_set1_epi32(127));
    const __m256i normal = _mm256_add_epi32(_mm256_slli_epi32(mag, 20),
                                          _mm256_set1_epi32(120 << 23));
    const __m256 sub = _mm256_mul_ps(_mm256_cvtepi32_ps(mag), _mm256_set1_ps(1.0f / 512));
    __m256 value = _mm256_blendv_ps(
        _mm256_castsi256_ps(normal), sub,
        _mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_set1_epi32(8), mag)));
    value = _mm256_blendv_ps(value, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fc00000)),
        _mm256_castsi256_ps(_mm256_cmpeq_epi32(mag, _mm256_set1_epi32(127))));
    return _mm256_xor_ps(value, _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_and_si256(raw, _mm256_set1_epi32(128)), 24)));
}
float dot128_avx2(const uint8_t* weights, const float* values) {
    __m256 sum = _mm256_setzero_ps();
    for (int k = 0; k < 128; k += 8)
        sum = _mm256_fmadd_ps(decode8(weights + k), _mm256_loadu_ps(values + k), sum);
    __m128 reduced = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
    reduced = _mm_hadd_ps(reduced, reduced);
    reduced = _mm_hadd_ps(reduced, reduced);
    return _mm_cvtss_f32(reduced);
}

float dot128_sm89_avx2(const uint8_t* weights, const float* values) {
    // Preserve the sixteen-product grouping across both eight-lane vectors.
    // Scalar exponent/scale control is shared by the two AVX2 conversions.
    float acc = 0;
    for (int k = 0; k < 128; k += 16) {
        alignas(32) float decoded[16], inputs[16];
        _mm256_store_ps(decoded, decode8(weights + k));
        _mm256_store_ps(decoded + 8, decode8(weights + k + 8));
        _mm256_store_ps(inputs, _mm256_loadu_ps(values + k));
        _mm256_store_ps(inputs + 8, _mm256_loadu_ps(values + k + 8));
        int maximum = finite_binary_exponent(acc);
        for (int n = 0; n < 16; ++n) {
            if (!std::isfinite(decoded[n]))
                return dot128_sm89_scalar(weights, values);
            if (decoded[n] != 0 && inputs[n] != 0)
                maximum = std::max(maximum,
                    finite_binary_exponent(decoded[n]) + finite_binary_exponent(inputs[n]));
        }
        if (maximum == -1024) continue;
        const float inverse = power_of_two(13 - maximum);
        __m256i integers = _mm256_setzero_si256();
        for (int n = 0; n < 16; n += 8) {
            const __m256 products = _mm256_mul_ps(
                _mm256_load_ps(decoded + n), _mm256_load_ps(inputs + n));
            integers = _mm256_add_epi32(integers, _mm256_cvttps_epi32(
                _mm256_mul_ps(products, _mm256_set1_ps(inverse))));
        }
        alignas(32) int lanes[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), integers);
        int sum = static_cast<int>(acc * inverse);
        for (int value : lanes) sum += value;
        acc = truncate_sm89(static_cast<float>(sum) * power_of_two(maximum - 13));
    }
    return acc;
}
}  // namespace drama
