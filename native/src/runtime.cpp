#include "drama_native.h"
#include "internal.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#include <immintrin.h>
#endif

namespace drama {
int features() {
    int result = 1;
    int regs[4]{};
#if defined(_MSC_VER)
    __cpuid(regs, 0);
    if (regs[0] < 7) return result;
    __cpuidex(regs, 1, 0);
    if (!(regs[2] & (1 << 27)) || !(regs[2] & (1 << 28))) return result;
    const bool fma = (regs[2] & (1 << 12)) != 0;
    const uint64_t state = _xgetbv(0);
    __cpuidex(regs, 7, 0);
#else
    unsigned a, b, c, d;
    if (__get_cpuid_max(0, nullptr) < 7) return result;
    __cpuid_count(1, 0, a, b, c, d);
    if (!(c & (1u << 27)) || !(c & (1u << 28))) return result;
    const bool fma = (c & (1u << 12)) != 0;
    uint32_t lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    const uint64_t state = (static_cast<uint64_t>(hi) << 32) | lo;
    __cpuid_count(7, 0, a, b, c, d);
    regs[1] = static_cast<int>(b);
#endif
    if (fma && (state & 6) == 6 && (regs[1] & (1 << 5))) result |= 2;
    if ((result & 2) && (state & 0xe6) == 0xe6 &&
        (regs[1] & (1 << 16)) && (regs[1] & (1 << 30))) result |= 4;
    return result;
}
}  // namespace drama

namespace {
thread_local std::string last_error;
struct Job {
    const uint8_t* w13 = nullptr;
    const float* s13 = nullptr;
    const uint8_t* w2 = nullptr;
    const float* s2 = nullptr;
    const int64_t* ids = nullptr;
    const float* weights = nullptr;
    float* output = nullptr;
    int jobs = 0;
    DramaExpertTrace* trace = nullptr;
};

class Executor {
public:
    Executor(int h, int i, int capacity, int threads, int requested)
        : h_(h), i_(i), capacity_(capacity), input_(h), xq_(h), xs_(h / 128),
          intermediate_(static_cast<size_t>(capacity) * i) {
        const int available = drama::features();
        implementation_ = requested == 0 ? ((available & 4) ? 3 : (available & 2) ? 2 : 1) : requested;
        if (implementation_ < 1 || implementation_ > 3 ||
            !(available & (1 << (implementation_ - 1))))
            throw std::invalid_argument("Requested CPU ISA is unavailable");
        dot_ = implementation_ == 3 ? drama::dot128_avx512 :
               implementation_ == 2 ? drama::dot128_avx2 : drama::dot128_scalar;
        try {
            for (int n = 1; n < threads; ++n)
                workers_.emplace_back([this] { worker(); });
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(work_mutex_);
                stop_ = true;
                ++epoch_;
            }
            start_.notify_all();
            for (auto& worker : workers_) worker.join();
            throw;
        }
    }
    ~Executor() {
        {
            std::lock_guard<std::mutex> lock(work_mutex_);
            stop_ = true;
            ++epoch_;
        }
        start_.notify_all();
        for (auto& worker : workers_) worker.join();
    }
    int implementation() const { return implementation_; }
    void arithmetic(int mode) {
        std::lock_guard<std::mutex> single_call(call_mutex_);
        if (mode != 0 && mode != 1) throw std::invalid_argument("Unsupported native arithmetic mode");
        dot_ = mode == 1 ? (
                   implementation_ == 3 ? drama::dot128_sm89_avx512 :
                   implementation_ == 2 ? drama::dot128_sm89_avx2 : drama::dot128_sm89_scalar) :
               implementation_ == 3 ? drama::dot128_avx512 :
               implementation_ == 2 ? drama::dot128_avx2 : drama::dot128_scalar;
    }
    void silu_table(const float* table) {
        std::lock_guard<std::mutex> single_call(call_mutex_);
        silu_table_ = table;
    }
    void quantization_guard(bool enabled) {
        std::lock_guard<std::mutex> single_call(call_mutex_);
        quantization_guard_ = enabled;
    }

    void forward(const Job& job, const uint16_t* x,
                 const uint8_t* quantized = nullptr, const float* scales = nullptr) {
        std::lock_guard<std::mutex> single_call(call_mutex_);
        if (job.jobs < 1 || job.jobs > capacity_)
            throw std::invalid_argument("Job count is outside executor capacity");
        if (quantized) {
            for (int n = 0; n < h_; ++n) {
                xq_[n] = drama::decode_fp8(quantized[n]);
                if (!std::isfinite(xq_[n])) throw std::invalid_argument("Non-finite FP8 activation");
            }
            for (int n = 0; n < h_ / 128; ++n) {
                if (!(scales[n] > 0) || !std::isfinite(scales[n]))
                    throw std::invalid_argument("Invalid FP8 activation scale");
                xs_[n] = scales[n];
            }
        } else {
            for (int n = 0; n < h_; ++n) {
                input_[n] = drama::from_bf16(x[n]);
                if (!std::isfinite(input_[n])) throw std::invalid_argument("Non-finite activation");
            }
            drama::quantize(input_.data(), nullptr, xq_.data(), xs_.data(), h_);
        }
        for (int n = 0; n < job.jobs; ++n)
            if (!std::isfinite(job.weights[n])) throw std::invalid_argument("Non-finite routing weight");
        {
            std::lock_guard<std::mutex> lock(work_mutex_);
            job_ = job;
            next_.store(0);
            remaining_ = static_cast<int>(workers_.size());
            failure_ = nullptr;
            ++epoch_;
        }
        start_.notify_all();
        try { consume(); } catch (...) { fail(std::current_exception()); }
        std::unique_lock<std::mutex> lock(work_mutex_);
        done_.wait(lock, [this] { return remaining_ == 0; });
        if (failure_) std::rethrow_exception(failure_);
    }

private:
    float row(const uint8_t* weights, const float* scales, int columns,
              const float* input, const float* input_scales) const {
        float acc = 0;
        for (int block = 0; block < columns / 128; ++block) {
            const float product = dot_(weights + block * 128, input + block * 128);
            // Match the current Triton lowering: multiply activation scale,
            // then fuse weight scaling with the outer accumulation.
            acc = std::fma(product * input_scales[block], scales[block], acc);
        }
        return acc;
    }
    void expert(int index) {
        const int64_t id = job_.ids[index];
        const auto w13 = job_.w13 + static_cast<size_t>(id) * 2 * i_ * h_;
        const auto s13 = job_.s13 + static_cast<size_t>(id) * (2 * i_ / 128) * (h_ / 128);
        const auto w2 = job_.w2 + static_cast<size_t>(id) * h_ * i_;
        const auto s2 = job_.s2 + static_cast<size_t>(id) * (h_ / 128) * (i_ / 128);
        float* local = intermediate_.data() + static_cast<size_t>(index) * i_;
        auto trace = job_.trace;
        const size_t gu_base = static_cast<size_t>(index) * 2 * i_;
        const size_t hidden_base = static_cast<size_t>(index) * i_;
        const size_t scale_base = static_cast<size_t>(index) * (i_ / 128);
        for (int n = 0; n < i_; ++n) {
            float gate = row(w13 + static_cast<size_t>(n) * h_,
                s13 + (n / 128) * (h_ / 128), h_, xq_.data(), xs_.data());
            float up = row(w13 + static_cast<size_t>(n + i_) * h_,
                s13 + ((n + i_) / 128) * (h_ / 128), h_, xq_.data(), xs_.data());
            if (trace) {
                trace->gate_up[gu_base + n] = gate;
                trace->gate_up[gu_base + n + i_] = up;
            }
            // The official first GEMM stores BF16 before fused SiLU quantization.
            const uint16_t gate_bits = trace && trace->shared_gate_up ?
                trace->shared_gate_up[gu_base + n] : drama::to_bf16(gate);
            const uint16_t up_bits = trace && trace->shared_gate_up ?
                trace->shared_gate_up[gu_base + n + i_] : drama::to_bf16(up);
            if (trace) {
                trace->gate_up_bf16[gu_base + n] = gate_bits;
                trace->gate_up_bf16[gu_base + n + i_] = up_bits;
            }
            gate = drama::from_bf16(gate_bits);
            up = drama::from_bf16(up_bits);
            local[n] = (silu_table_ ? silu_table_[gate_bits] :
                        gate / (1.0f + std::exp(-gate))) * up;
            if (trace) trace->activation[hidden_base + n] = local[n];
        }
        float scales[128]; // max intermediate=16384; checked at construction.
        if (trace && trace->shared_hidden_q) {
            for (int n = 0; n < i_; ++n) {
                const auto value = trace->shared_hidden_q[hidden_base + n];
                trace->hidden_q[hidden_base + n] = value;
                local[n] = drama::decode_fp8(value);
                if (!std::isfinite(local[n])) throw std::invalid_argument("Non-finite hidden FP8");
            }
            for (int n = 0; n < i_ / 128; ++n) {
                const float scale = trace->shared_hidden_scales[scale_base + n];
                if (!(scale > 0) || !std::isfinite(scale))
                    throw std::invalid_argument("Invalid hidden FP8 scale");
                scales[n] = scale;
            }
        } else {
            drama::quantize(local, trace ? trace->hidden_q + hidden_base : nullptr,
                            local, scales, i_, quantization_guard_);
        }
        if (trace)
            std::copy_n(scales, i_ / 128, trace->hidden_scales + scale_base);
        float* output = job_.output + static_cast<size_t>(index) * h_;
        for (int n = 0; n < h_; ++n) {
            const float value = row(w2 + static_cast<size_t>(n) * i_,
                s2 + (n / 128) * (i_ / 128), i_, local, scales);
            if (trace) trace->down[static_cast<size_t>(index) * h_ + n] = value;
            output[n] = value * job_.weights[index];
            if (!std::isfinite(output[n]))
                throw std::invalid_argument("Non-finite expert output");
        }
    }
    void consume() {
        for (;;) {
            const int index = next_.fetch_add(1);
            if (index >= job_.jobs) return;
            expert(index);
        }
    }
    void fail(std::exception_ptr error) {
        std::lock_guard<std::mutex> lock(work_mutex_);
        if (!failure_) failure_ = error;
    }
    void worker() {
        uint64_t seen = 0;
        for (;;) {
            std::unique_lock<std::mutex> lock(work_mutex_);
            start_.wait(lock, [this, seen] { return stop_ || epoch_ != seen; });
            if (stop_) return;
            seen = epoch_;
            lock.unlock();
            try { consume(); } catch (...) { fail(std::current_exception()); }
            lock.lock();
            if (--remaining_ == 0) done_.notify_one();
        }
    }
    int h_, i_, capacity_, implementation_;
    drama::Dot128 dot_ = nullptr;
    const float* silu_table_ = nullptr;
    bool quantization_guard_ = false;
    std::vector<float> input_, xq_, xs_, intermediate_;
    std::vector<std::thread> workers_;
    std::mutex call_mutex_, work_mutex_;
    std::condition_variable start_, done_;
    std::atomic<int> next_{0};
    bool stop_ = false;
    uint64_t epoch_ = 0;
    int remaining_ = 0;
    Job job_;
    std::exception_ptr failure_;
};

template<class F> int checked(F operation) {
    last_error.clear();
    try { operation(); return 0; }
    catch (const std::exception& error) { last_error = error.what(); }
    catch (...) { last_error = "Unknown native exception"; }
    return -1;
}
}

extern "C" {
int drama_native_abi() { return 1; }
int drama_native_features() { return drama::features(); }
const char* drama_native_error() { return last_error.c_str(); }
int drama_partition_routes(
    const int64_t* ids, int count, const int64_t* resident, int resident_count,
    int experts, int cpu_jobs, int64_t* cpu_slots, int64_t* gpu_slots) {
    return checked([&] {
        if (!ids || !cpu_slots || !gpu_slots || count < 1 || count > 8 ||
            experts < 1 || experts > 65536 || resident_count < 0 ||
            resident_count > experts || (resident_count && !resident) ||
            cpu_jobs < 0 || cpu_jobs > count)
            throw std::invalid_argument("Invalid route planning geometry/buffers");
        for (int n = 0; n < count; ++n)
            if (ids[n] < 0 || ids[n] >= experts)
                throw std::invalid_argument("Route expert ID outside bank");
        for (int n = 0; n < resident_count; ++n)
            if (resident[n] < 0 || resident[n] >= experts)
                throw std::invalid_argument("Resident expert ID outside bank");
        auto overlaps = [](const int64_t* a, int na, const int64_t* b, int nb) {
            if (!na || !nb) return false;
            const auto aa = reinterpret_cast<uintptr_t>(a);
            const auto bb = reinterpret_cast<uintptr_t>(b);
            return aa < bb + static_cast<uintptr_t>(nb) * sizeof(int64_t) &&
                   bb < aa + static_cast<uintptr_t>(na) * sizeof(int64_t);
        };
        if (overlaps(cpu_slots, cpu_jobs, gpu_slots, count - cpu_jobs) ||
            overlaps(cpu_slots, cpu_jobs, ids, count) ||
            overlaps(gpu_slots, count - cpu_jobs, ids, count) ||
            overlaps(cpu_slots, cpu_jobs, resident, resident_count) ||
            overlaps(gpu_slots, count - cpu_jobs, resident, resident_count))
            throw std::invalid_argument("Route planning outputs overlap borrowed inputs");
        bool hot[8]{}, selected[8]{};
        for (int slot = 0; slot < count; ++slot)
            for (int n = 0; n < resident_count; ++n)
                hot[slot] = hot[slot] || ids[slot] == resident[n];
        int remaining = cpu_jobs;
        for (int pass = 0; pass < 2 && remaining; ++pass)
            for (int slot = count - 1; slot >= 0 && remaining; --slot)
                if (hot[slot] == (pass != 0)) {
                    selected[slot] = true;
                    --remaining;
                }
        int cpu = 0, gpu = 0;
        for (int slot = 0; slot < count; ++slot)
            if (selected[slot]) cpu_slots[cpu++] = slot;
            else gpu_slots[gpu++] = slot;
    });
}
int drama_gather_rows(const DramaGatherRows* buffers, int buffer_count,
                      int experts, const int64_t* ids, int count) {
    return checked([&] {
        if (!buffers || !ids || buffer_count < 1 || buffer_count > 4 ||
            experts < 1 || count < 1 || count > 256)
            throw std::invalid_argument("Invalid gather geometry/buffers");
        for (int row = 0; row < count; ++row)
            if (ids[row] < 0 || ids[row] >= experts)
                throw std::invalid_argument("Gather expert ID outside bank");
        // Validate every field before copying; a bad final descriptor or ID
        // must leave all targets untouched.
        for (int field = 0; field < buffer_count; ++field) {
            const auto& buffer = buffers[field];
            if (!buffer.source || !buffer.target || !buffer.row_bytes ||
                buffer.row_bytes > std::numeric_limits<size_t>::max() /
                    static_cast<size_t>(std::max(experts, count)))
                throw std::invalid_argument("Invalid gather row size/pointer");
        }
        for (int field = 0; field < buffer_count; ++field) {
            const auto& buffer = buffers[field];
            const auto bytes = static_cast<size_t>(buffer.row_bytes);
            for (int row = 0; row < count; ++row)
                std::memcpy(buffer.target + static_cast<size_t>(row) * bytes,
                            buffer.source + static_cast<size_t>(ids[row]) * bytes,
                            bytes);
        }
    });
}
int drama_decode_fp8(const uint8_t* values, float* output, int count) {
    return checked([&] {
        if (!values || !output || count < 0) throw std::invalid_argument("Invalid FP8 decode buffers");
        for (int n = 0; n < count; ++n) output[n] = drama::decode_fp8(values[n]);
    });
}
int drama_quantize_bf16(const uint16_t* values, uint8_t* output, float* scales, int count) {
    return checked([&] {
        if (!values || !output || !scales || count < 128 || count % 128)
            throw std::invalid_argument("Quantization needs whole 128-value blocks");
        std::vector<float> decoded(count);
        for (int n = 0; n < count; ++n) decoded[n] = drama::from_bf16(values[n]);
        drama::quantize(decoded.data(), output, nullptr, scales, count);
    });
}
void* drama_executor_create(int hidden, int intermediate, int capacity, int threads, int implementation) {
    Executor* result = nullptr;
    checked([&] {
        if (hidden < 128 || hidden > 32768 || hidden % 128 ||
            intermediate < 128 || intermediate > 16384 || intermediate % 128 ||
            capacity < 1 || capacity > 256 || threads < 1 || threads > 64)
            throw std::invalid_argument("Unsupported executor geometry/capacity/threads");
        result = new Executor(hidden, intermediate, capacity, threads, implementation);
    });
    return result;
}
void drama_executor_destroy(void* executor) { delete static_cast<Executor*>(executor); }
int drama_executor_implementation(void* executor) {
    return executor ? static_cast<Executor*>(executor)->implementation() : 0;
}
int drama_executor_set_arithmetic(void* executor, int arithmetic) {
    return checked([&] {
        if (!executor) throw std::invalid_argument("Missing native executor");
        static_cast<Executor*>(executor)->arithmetic(arithmetic);
    });
}
int drama_executor_set_silu_table(void* executor, const float* table) {
    return checked([&] {
        if (!executor) throw std::invalid_argument("Missing native executor");
        static_cast<Executor*>(executor)->silu_table(table);
    });
}
int drama_executor_set_quantization_guard(void* executor, int enabled) {
    return checked([&] {
        if (!executor || (enabled != 0 && enabled != 1))
            throw std::invalid_argument("Invalid native quantization guard");
        static_cast<Executor*>(executor)->quantization_guard(enabled != 0);
    });
}
int drama_executor_forward(void* executor, const uint8_t* w13, const float* s13,
    const uint8_t* w2, const float* s2, int experts, const uint16_t* x,
    const int64_t* ids, const float* weights, int jobs, float* out) {
    return checked([&] {
        if (!executor || !w13 || !s13 || !w2 || !s2 || !x || !ids || !weights || !out ||
            experts < 1 || jobs < 1 || jobs > 256)
            throw std::invalid_argument("Invalid expert buffers");
        for (int n = 0; n < jobs; ++n)
            if (ids[n] < 0 || ids[n] >= experts) throw std::invalid_argument("Expert ID outside bank");
        static_cast<Executor*>(executor)->forward({w13, s13, w2, s2, ids, weights, out, jobs}, x);
    });
}
int drama_executor_forward_quantized(void* executor, const uint8_t* w13, const float* s13,
    const uint8_t* w2, const float* s2, int experts, const uint8_t* xq,
    const float* scales, const int64_t* ids, const float* weights, int jobs, float* out) {
    return checked([&] {
        if (!executor || !w13 || !s13 || !w2 || !s2 || !xq || !scales || !ids ||
            !weights || !out || experts < 1 || jobs < 1 || jobs > 256)
            throw std::invalid_argument("Invalid quantized expert buffers");
        for (int n = 0; n < jobs; ++n)
            if (ids[n] < 0 || ids[n] >= experts) throw std::invalid_argument("Expert ID outside bank");
        static_cast<Executor*>(executor)->forward(
            {w13, s13, w2, s2, ids, weights, out, jobs}, nullptr, xq, scales);
    });
}
int drama_executor_trace_quantized(void* executor, const uint8_t* w13, const float* s13,
    const uint8_t* w2, const float* s2, int experts, const uint8_t* xq,
    const float* scales, const int64_t* ids, const float* weights, int jobs,
    float* out, DramaExpertTrace* trace) {
    return checked([&] {
        if (!executor || !w13 || !s13 || !w2 || !s2 || !xq || !scales || !ids ||
            !weights || !out || experts < 1 || jobs < 1 || jobs > 256 ||
            !trace || !trace->gate_up || !trace->gate_up_bf16 || !trace->activation ||
            !trace->hidden_q || !trace->hidden_scales || !trace->down)
            throw std::invalid_argument("Invalid diagnostic expert buffers");
        if (!!trace->shared_hidden_q != !!trace->shared_hidden_scales)
            throw std::invalid_argument("Shared hidden bytes and scales must be supplied together");
        for (int n = 0; n < jobs; ++n)
            if (ids[n] < 0 || ids[n] >= experts) throw std::invalid_argument("Expert ID outside bank");
        static_cast<Executor*>(executor)->forward(
            {w13, s13, w2, s2, ids, weights, out, jobs, trace}, nullptr, xq, scales);
    });
}
}
