#include "internal.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace drama {
float decode_fp8(uint8_t value) {
    const int mag = value & 127, exp = mag >> 3, mantissa = mag & 7;
    float result;
    if (mag == 127) result = std::numeric_limits<float>::quiet_NaN();
    else if (exp == 0) result = std::ldexp(static_cast<float>(mantissa), -9);
    else result = std::ldexp(1.0f + mantissa * 0.125f, exp - 7);
    return (value & 128) ? -result : result;
}

uint8_t encode_fp8(float value) {
    if (!std::isfinite(value)) throw std::invalid_argument("Non-finite FP8 input");
    const uint8_t sign = std::signbit(value) ? 128 : 0;
    value = std::min(std::abs(value), 448.0f);
    uint32_t result;
    if (value < 0.015625f) {
        const float scaled = value * 512.0f;
        result = static_cast<uint32_t>(scaled);
        const float fraction = scaled - result;
        if (fraction > 0.5f || (fraction == 0.5f && (result & 1))) ++result;
    } else {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        bits += (1u << 19) - 1 + ((bits >> 20) & 1);
        result = (bits >> 20) - (120u << 3);
    }
    return sign | static_cast<uint8_t>(result);
}

float from_bf16(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t to_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fff + ((bits >> 16) & 1);
    return static_cast<uint16_t>(bits >> 16);
}

void quantize(const float* values, uint8_t* bytes, float* decoded,
              float* scales, int count, bool guard_boundaries) {
    for (int start = 0; start < count; start += 128) {
        float maximum = 1e-10f;
        for (int k = 0; k < 128; ++k) {
            if (!std::isfinite(values[start + k]))
                throw std::invalid_argument("Non-finite activation");
            maximum = std::max(maximum, std::abs(values[start + k]));
        }
        // Triton's constant division lowers to multiplication by the rounded
        // reciprocal on this supported GPU path. A hardware CPU divide differs
        // by one ULP and can change FP8 half-way rounding.
        const float scale = maximum * (1.0f / 448.0f);
        scales[start / 128] = scale;
        for (int k = 0; k < 128; ++k) {
            const float normalized = values[start + k] / scale;
            if (guard_boundaries) {
                const float magnitude = std::abs(normalized);
                bool ambiguous = false;
                if (magnitude >= 0.015625f && magnitude < 448.0f) {
                    uint32_t bits;
                    std::memcpy(&bits, &magnitude, sizeof(bits));
                    // Normal E4M3 has 20 discarded FP32 fraction bits. Allow
                    // four ULPs around the midpoint for division and software
                    // conversion ties. The supported GPU is the authority.
                    const int fraction = static_cast<int>(bits & 0xfffff);
                    ambiguous = std::abs(fraction - 0x80000) <= 4;
                } else if (magnitude > 0 && magnitude < 0.015625f) {
                    const float scaled = magnitude * 512.0f;
                    const float fraction = scaled - std::floor(scaled);
                    ambiguous = std::abs(fraction - 0.5f) <=
                        std::max(1e-12f, scaled * (4.0f / 8388608.0f));
                }
                if (ambiguous)
                    throw std::runtime_error("GPU FP8 rounding boundary requires fallback");
            }
            const uint8_t q = encode_fp8(normalized);
            if (bytes) bytes[start + k] = q;
            if (decoded) decoded[start + k] = decode_fp8(q);
        }
    }
}

float dot128_scalar(const uint8_t* weights, const float* values) {
    static const auto table = [] {
        std::array<float, 256> result{};
        for (int n = 0; n < 256; ++n) result[n] = decode_fp8(static_cast<uint8_t>(n));
        return result;
    }();
    float sum = 0;
    for (int k = 0; k < 128; ++k) sum += table[weights[k]] * values[k];
    return sum;
}

float dot128_sm89_scalar(const uint8_t* weights, const float* values) {
    // Empirically validated on SM89: align groups of 16 by the operands'
    // exponent sum, truncate at 13 fractional bits, then truncate the sum.
    // Do not use the exponent of the product: its normalization changes ties.
    static const auto table = [] {
        std::array<float, 256> result{};
        for (int n = 0; n < 256; ++n) result[n] = decode_fp8(static_cast<uint8_t>(n));
        return result;
    }();
    float acc = 0;
    for (int start = 0; start < 128; start += 16) {
        float products[16];
        int maximum = finite_binary_exponent(acc);
        for (int k = 0; k < 16; ++k) {
            const float weight = table[weights[start + k]];
            if (!std::isfinite(weight)) throw std::invalid_argument("Non-finite FP8 weight");
            products[k] = weight * values[start + k];
            if (weight != 0 && values[start + k] != 0)
                maximum = std::max(maximum,
                    finite_binary_exponent(weight) + finite_binary_exponent(values[start + k]));
        }
        if (maximum == -1024) continue;
        const float inverse = power_of_two(13 - maximum);
        int sum = static_cast<int>(acc * inverse);
        for (float product : products) sum += static_cast<int>(product * inverse);
        acc = truncate_sm89(static_cast<float>(sum) * power_of_two(maximum - 13));
    }
    return acc;
}
}  // namespace drama
