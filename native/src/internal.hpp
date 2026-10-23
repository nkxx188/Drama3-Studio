#pragma once
#include <cstdint>
#include <cstring>

namespace drama {
float decode_fp8(uint8_t value);
uint8_t encode_fp8(float value);
float from_bf16(uint16_t value);
uint16_t to_bf16(float value);
void quantize(const float* values, uint8_t* bytes, float* decoded,
              float* scales, int count, bool guard_boundaries = false);
using Dot128 = float (*)(const uint8_t*, const float*);
float dot128_scalar(const uint8_t*, const float*);
float dot128_sm89_scalar(const uint8_t*, const float*);
float dot128_sm89_avx2(const uint8_t*, const float*);
float dot128_sm89_avx512(const uint8_t*, const float*);
inline int finite_binary_exponent(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7fffffff) ? static_cast<int>((bits >> 23) & 255) - 127 : -1024;
}
inline float power_of_two(int exponent) {
    const uint32_t bits = static_cast<uint32_t>(exponent + 127) << 23;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}
inline float truncate_sm89(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits &= ~uint32_t{1023};
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}
float dot128_avx2(const uint8_t*, const float*);
float dot128_avx512(const uint8_t*, const float*);
int features();
}  // namespace drama
