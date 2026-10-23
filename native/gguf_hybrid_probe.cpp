// Bounded one-layer Q8 cooperation probe. No production inference changes.
#include "engine/framework/core/backend.h"
#include "engine/framework/core/disk_expert_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/io/binary.h"
#include "engine/framework/io/json.h"

#include <ggml-alloc.h>
#include <gguf.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace core = engine::core;
namespace json = engine::io::json;
using Clock = std::chrono::steady_clock;
constexpr int H = 2048, I = 512, K = 8;
struct CtxDelete { void operator()(ggml_context * p) const { ggml_free(p); } };
using Context = std::unique_ptr<ggml_context, CtxDelete>;
Context context(size_t bytes = 4 * 1024 * 1024) {
    Context result(ggml_init({bytes, nullptr, true}));
    if (!result) throw std::runtime_error("cannot allocate graph header context");
    return result;
}
double elapsed(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}
struct Buffer {
    ggml_backend_buffer_t p = nullptr;
    ~Buffer() { if (p) ggml_backend_buffer_free(p); }
};
struct Allocator {
    ggml_gallocr_t p = nullptr;
    ~Allocator() { if (p) ggml_gallocr_free(p); }
};
struct Case { std::vector<float> x, weights; std::vector<int32_t> ids; };

struct CpuBank {
    engine::io::BinaryBlob mapping;
    Context ctx = context();
    Buffer buffer;
    std::array<ggml_tensor *, 3> weights{};
    CpuBank(const std::string & path, const std::vector<std::string> & names)
        : mapping(engine::io::read_binary_blob(path, true)) {
        ggml_context * headers = nullptr;
        auto * header = gguf_init_from_file(path.c_str(), {true, &headers});
        if (!header || !headers) throw std::runtime_error("invalid expert GGUF");
        const auto key = gguf_find_key(header, "audiocpp.tensor_names");
        buffer.p = ggml_backend_cpu_buffer_from_ptr(
            const_cast<std::byte *>(mapping.data()), mapping.size());
        ggml_backend_buffer_set_usage(buffer.p, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (int64_t n = 0; n < gguf_get_n_tensors(header); ++n) {
            const std::string logical = key >= 0 ? gguf_get_arr_str(header, key, n)
                                                : gguf_get_tensor_name(header, n);
            auto found = std::find(names.begin(), names.end(), logical);
            if (found == names.end()) continue;
            const auto * meta = ggml_get_tensor(headers, gguf_get_tensor_name(header, n));
            if (meta->type != GGML_TYPE_Q8_0 || meta->ne[2] != 256 || meta->ne[3] != 1)
                throw std::runtime_error("probe expects native Q8_0 expert banks");
            auto * t = ggml_new_tensor_3d(ctx.get(), meta->type, meta->ne[0], meta->ne[1], meta->ne[2]);
            const auto offset = gguf_get_data_offset(header) + gguf_get_tensor_offset(header, n);
            if (offset + ggml_nbytes(t) > mapping.size()) throw std::runtime_error("truncated expert bank");
            if (ggml_backend_tensor_alloc(buffer.p, t,
                const_cast<std::byte *>(mapping.data()) + offset) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("cannot bind mapped CPU expert view");
            weights[found - names.begin()] = t;
        }
        gguf_free(header);
        ggml_free(headers);
        for (auto * t : weights) if (!t) throw std::runtime_error("missing selected expert bank");
    }
};

ggml_tensor * reduce(ggml_context * ctx, ggml_tensor * partial, ggml_tensor * weights) {
    auto * scaled = ggml_mul(ctx, partial, ggml_reshape_3d(ctx, weights, 1, K, 1));
    auto * rows = ggml_cont(ctx, ggml_permute(ctx, scaled, 1, 0, 2, 3));
    return ggml_reshape_2d(ctx, ggml_sum_rows(ctx, rows), H, 1);
}

struct ExpertGraph {
    core::ExecutionContext & execution;
    Context ctx = context();
    Buffer inputs;
    Allocator allocator;
    core::HostGraphPlan plan;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * x = nullptr, * ids = nullptr, * partial = nullptr, * output = nullptr, * weights = nullptr;
    int count;
    ExpertGraph(core::ExecutionContext & e, std::array<ggml_tensor *, 3> banks, int n, bool sum = false)
        : execution(e), count(n) {
        x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, H, 1, 1);
        ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, count, 1);
        ggml_set_input(x); ggml_set_input(ids);
        if (sum) {
            weights = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, K);
            ggml_set_input(weights);
        }
        auto * gate = ggml_mul_mat_id(ctx.get(), banks[0], x, ids);
        auto * up = ggml_mul_mat_id(ctx.get(), banks[1], x, ids);
        auto * active = ggml_mul(ctx.get(), ggml_silu(ctx.get(), gate), up);
        partial = ggml_mul_mat_id(ctx.get(), banks[2], active, ids);
        ggml_set_output(partial);
        output = sum ? reduce(ctx.get(), partial, weights) : partial;
        ggml_set_output(output);
        graph = ggml_new_graph_custom(ctx.get(), 128, false);
        ggml_build_forward_expand(graph, output);
        allocator.p = ggml_gallocr_new(ggml_backend_get_default_buffer_type(e.backend()));
        if (!allocator.p || !ggml_gallocr_alloc_graph(allocator.p, graph))
            throw std::runtime_error("cannot allocate expert graph");
        core::prepare_host_graph_plan(execution, graph, plan);
    }
    ~ExpertGraph() {
        ggml_backend_synchronize(execution.backend());
        core::release_backend_graph_resources(execution.backend(), graph, true);
    }
    void compute() {
        if (core::compute_graph(execution, graph, plan) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("expert graph compute failed");
    }
};

struct MergeGraph {
    core::ExecutionContext & execution;
    Context ctx = context();
    Allocator allocator;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * partial, * weights, * output;
    explicit MergeGraph(core::ExecutionContext & e) : execution(e) {
        partial = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, H, K, 1);
        weights = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, K);
        ggml_set_input(partial); ggml_set_input(weights);
        output = reduce(ctx.get(), partial, weights);
        ggml_set_output(output);
        graph = ggml_new_graph_custom(ctx.get(), 64, false);
        ggml_build_forward_expand(graph, output);
        allocator.p = ggml_gallocr_new(ggml_backend_get_default_buffer_type(e.backend()));
        if (!allocator.p || !ggml_gallocr_alloc_graph(allocator.p, graph))
            throw std::runtime_error("cannot allocate merge graph");
    }
    ~MergeGraph() { core::release_backend_graph_resources(execution.backend(), graph, true); }
    void compute() {
        if (core::compute_backend_graph(execution.backend(), graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("merge compute failed");
    }
};

class Worker {
    std::thread thread;
    std::mutex mutex;
    std::condition_variable cv;
    bool stop = false, pending = false, done = true;
    std::function<void()> task;
    std::exception_ptr error;
public:
    double compute_ms = 0;
    Worker() { thread = std::thread([this] {
        std::unique_lock<std::mutex> lock(mutex);
        for (;;) {
            cv.wait(lock, [&] { return stop || pending; });
            if (stop) return;
            auto f = std::move(task);
            pending = false;
            lock.unlock();
            auto started = Clock::now();
            try { f(); } catch (...) { error = std::current_exception(); }
            compute_ms = elapsed(started);
            lock.lock();
            done = true;
            cv.notify_all();
        }
    }); }
    ~Worker() {
        { std::lock_guard<std::mutex> lock(mutex); stop = true; }
        cv.notify_all(); thread.join();
    }
    void launch(std::function<void()> f) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!done) throw std::runtime_error("CPU worker already active");
        task = std::move(f); done = false; pending = true; error = nullptr;
        cv.notify_all();
    }
    void wait() {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return done; });
        if (error) std::rethrow_exception(error);
    }
};

struct Variant {
    core::ExecutionContext & gpu;
    int jobs, threads;
    std::unique_ptr<core::ExecutionContext> cpu;
    std::unique_ptr<core::DiskExpertWeightStore> store;
    size_t group = 0;
    std::unique_ptr<ExpertGraph> gpu_graph, cpu_graph;
    std::unique_ptr<MergeGraph> merge;
    Context source_ctx = context();
    Buffer source_buffer, handoff;
    ggml_tensor * source_x, * source_ids;
    float * host_x, * cpu_partial;
    int32_t * host_ids;
    std::unique_ptr<Worker> worker;
    std::vector<double> times, handoff_times, cpu_times;
    double max_relative_error = 0, max_partial_relative_error = 0, max_absolute_error = 0;
    uint64_t timed_loads = 0, timed_hits = 0, timed_bytes = 0;
    Variant(core::ExecutionContext & e, const CpuBank & bank, const std::string & path,
            const std::vector<std::string> & names, int n, int nthreads)
        : gpu(e), jobs(n), threads(nthreads) {
        source_x = ggml_new_tensor_3d(source_ctx.get(), GGML_TYPE_F32, H, 1, 1);
        source_ids = ggml_new_tensor_1d(source_ctx.get(), GGML_TYPE_I32, K);
        source_buffer.p = ggml_backend_alloc_ctx_tensors(source_ctx.get(), gpu.backend());
        const auto host_type = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(gpu.backend()));
        if (!source_buffer.p || !host_type) throw std::runtime_error("missing source/pinned support");
        handoff.p = ggml_backend_buft_alloc_buffer(host_type, (H + H * K) * sizeof(float) + K * sizeof(int32_t));
        if (!handoff.p || ggml_backend_buffer_get_type(handoff.p) != host_type)
            throw std::runtime_error("cannot allocate bounded pinned handoff");
        host_x = static_cast<float *>(ggml_backend_buffer_get_base(handoff.p));
        cpu_partial = host_x + H;
        host_ids = reinterpret_cast<int32_t *>(cpu_partial + H * K);
        if (jobs < K) {
            store = std::make_unique<core::DiskExpertWeightStore>(path, false);
            group = store->add_group(names, gpu.backend(), K);
            std::array<ggml_tensor *, 3> buffers{};
            for (size_t p = 0; p < buffers.size(); ++p) buffers[p] = store->tensor(group, p).tensor;
            gpu_graph = std::make_unique<ExpertGraph>(gpu, buffers, K - jobs, jobs == 0);
        }
        if (jobs) {
            core::BackendConfig config; config.threads = threads;
            cpu = std::make_unique<core::ExecutionContext>(config);
            cpu_graph = std::make_unique<ExpertGraph>(*cpu, bank.weights, jobs);
            merge = std::make_unique<MergeGraph>(gpu);
            worker = std::make_unique<Worker>();
        }
    }
    void prepare(const Case & c, const std::string & mode) {
        ggml_backend_tensor_set(source_x, c.x.data(), 0, H * sizeof(float));
        ggml_backend_tensor_set(source_ids, c.ids.data(), 0, K * sizeof(int32_t));
        auto * weights = jobs ? merge->weights : gpu_graph->weights;
        ggml_backend_tensor_set(weights, c.weights.data(), 0, K * sizeof(float));
        if (store && mode == "cold") store->invalidate_cache();
        if (store && mode == "hit")
            store->stage(group, std::vector<int32_t>(c.ids.begin(), c.ids.begin() + K - jobs));
    }
    double run(bool record) {
        const auto before = store ? store->statistics() : core::DiskExpertWeightStore::Statistics{};
        const auto start = Clock::now();
        ggml_backend_tensor_get_async(gpu.backend(), source_ids, host_ids, 0, K * sizeof(int32_t));
        if (jobs) ggml_backend_tensor_get_async(gpu.backend(), source_x, host_x, 0, H * sizeof(float));
        ggml_backend_synchronize(gpu.backend());
        const double handoff_ms = elapsed(start);
        if (jobs) worker->launch([this] {
            ggml_backend_tensor_set(cpu_graph->x, host_x, 0, H * sizeof(float));
            ggml_backend_tensor_set(cpu_graph->ids, host_ids + K - jobs, 0, jobs * sizeof(int32_t));
            cpu_graph->compute();
            ggml_backend_tensor_get(cpu_graph->partial, cpu_partial, 0, H * jobs * sizeof(float));
        });
        if (gpu_graph) {
            const std::vector<int32_t> required(host_ids, host_ids + K - jobs);
            const auto local = store->stage(group, required);
            ggml_backend_tensor_copy_async(gpu.backend(), gpu.backend(), source_x, gpu_graph->x);
            ggml_backend_tensor_set_async(gpu.backend(), gpu_graph->ids, local.data(), 0, local.size() * sizeof(int32_t));
            gpu_graph->compute();
            if (jobs) {
                // Preserve the original top-k lane order; device-to-device
                // partial copy is real cooperation overhead and remains timed.
                auto view_ctx = context(4096);
                auto * destination = ggml_view_1d(view_ctx.get(), merge->partial, H * (K - jobs), 0);
                auto * source = ggml_reshape_1d(view_ctx.get(), gpu_graph->partial, H * (K - jobs));
                ggml_backend_tensor_copy_async(gpu.backend(), gpu.backend(), source, destination);
            }
        }
        if (jobs) {
            worker->wait();
            ggml_backend_tensor_set_async(gpu.backend(), merge->partial, cpu_partial,
                H * (K - jobs) * sizeof(float), H * jobs * sizeof(float));
            merge->compute();
        }
        ggml_backend_synchronize(gpu.backend());
        const double milliseconds = elapsed(start);
        if (record) {
            times.push_back(milliseconds); handoff_times.push_back(handoff_ms);
            if (jobs) cpu_times.push_back(worker->compute_ms);
            if (store) {
                const auto after = store->statistics();
                timed_loads += after.expert_loads - before.expert_loads;
                timed_hits += after.cache_hits - before.cache_hits;
                timed_bytes += after.bytes_read - before.bytes_read;
            }
        }
        return milliseconds;
    }
    std::vector<float> output() {
        return core::read_tensor_f32(jobs ? merge->output : gpu_graph->output);
    }
};

double median(std::vector<double> values) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}
double relative(const std::vector<float> & actual, const std::vector<float> & expected, double & maximum) {
    double squared = 0, reference = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i])) throw std::runtime_error("nonfinite hybrid result");
        const double delta = double(actual[i]) - expected[i];
        squared += delta * delta; reference += double(expected[i]) * expected[i];
        maximum = std::max(maximum, std::abs(delta));
    }
    return std::sqrt(squared / std::max(1e-30, reference));
}

int main(int argc, char ** argv) {
    try {
        if (argc != 4 && argc != 5)
            throw std::runtime_error("model.gguf cases.json output.json [variant-sweeps]");
        const bool variant_sweeps = argc == 5 && std::string(argv[4]) == "variant-sweeps";
        if (argc == 5 && !variant_sweeps) throw std::runtime_error("unknown benchmark schedule");
        const auto data = json::parse_file(argv[2]);
        const auto layer = json::require_i64(data, "layer");
        const auto prefix = "model_weights/model.language_model.layers." + std::to_string(layer) + ".mlp.experts.";
        const std::vector<std::string> names{prefix + "gate_proj.weight", prefix + "up_proj.weight", prefix + "down_proj.weight"};
        std::vector<Case> cases;
        for (const auto & row : data.require("cases").as_array()) {
            Case c;
            for (const auto & x : row.require("x").as_array()) c.x.push_back(x.as_f32());
            for (const auto & x : row.require("ids").as_array()) c.ids.push_back(static_cast<int32_t>(x.as_i64()));
            for (const auto & x : row.require("weights").as_array()) c.weights.push_back(x.as_f32());
            if (c.x.size() != H || c.ids.size() != K || c.weights.size() != K)
                throw std::runtime_error("case geometry mismatch");
            cases.push_back(std::move(c));
        }
        if (cases.empty()) throw std::runtime_error("empty case set");
        core::BackendConfig gpu_config; gpu_config.type = core::BackendType::Cuda;
        core::ExecutionContext gpu(gpu_config);
        CpuBank bank(argv[1], names);
        json::Value::Array reports;
        for (const std::string mode : {"cold", "rolling", "hit"}) {
            std::vector<std::unique_ptr<Variant>> variants;
            variants.push_back(std::make_unique<Variant>(gpu, bank, argv[1], names, 0, 0));
            for (int threads : {2, 4, 8})
                for (int jobs : {1, 2, 4, 8})
                    variants.push_back(std::make_unique<Variant>(gpu, bank, argv[1], names, jobs, threads));
            std::vector<std::vector<float>> expected(cases.size()), expected_partials(cases.size());
            for (size_t n = 0; n < cases.size(); ++n) {
                variants[0]->prepare(cases[n], mode);
                variants[0]->run(false);
                expected[n] = variants[0]->output();
                expected_partials[n] = core::read_tensor_f32(variants[0]->gpu_graph->partial);
            }
            for (int repetition = -2; repetition < 5; ++repetition) {
                // Sweeping all cases before changing variant avoids immediately
                // warming the same expert bytes for the next CPU job count.
                const size_t outer = variant_sweeps ? variants.size() : cases.size();
                const size_t inner = variant_sweeps ? cases.size() : variants.size();
                for (size_t a = 0; a < outer; ++a) {
                    for (size_t b = 0; b < inner; ++b) {
                        const size_t n = variant_sweeps ? b : a;
                        const size_t j = variant_sweeps ? a : b;
                        const size_t index = (j + repetition + 2) % variants.size();
                        auto & v = *variants[index];
                        v.prepare(cases[n], mode);
                        v.run(repetition >= 0);
                        if (repetition == 0) {
                            v.max_relative_error = std::max(v.max_relative_error,
                                relative(v.output(), expected[n], v.max_absolute_error));
                            if (v.jobs) {
                                std::vector<float> a(v.cpu_partial, v.cpu_partial + H * v.jobs);
                                std::vector<float> b(expected_partials[n].begin() + H * (K - v.jobs),
                                                     expected_partials[n].end());
                                double unused = 0;
                                v.max_partial_relative_error = std::max(v.max_partial_relative_error, relative(a, b, unused));
                            }
                        }
                    }
                }
            }
            const double base = median(variants[0]->times);
            for (auto & v : variants) {
                json::Value::Array samples;
                for (const auto x : v->times) samples.push_back(json::Value::make_number(x));
                auto row = json::Value::make_object({
                    {"mode", json::Value::make_string(mode)}, {"cpu_jobs", json::Value::make_number(v->jobs)},
                    {"cpu_threads", json::Value::make_number(v->threads)},
                    {"median_ms", json::Value::make_number(median(v->times))},
                    {"baseline_median_ms", json::Value::make_number(base)},
                    {"speedup", json::Value::make_number(base / median(v->times))},
                    {"handoff_median_ms", json::Value::make_number(median(v->handoff_times))},
                    {"cpu_compute_median_ms", json::Value::make_number(median(v->cpu_times))},
                    {"max_merged_relative_l2", json::Value::make_number(v->max_relative_error)},
                    {"max_cpu_partial_relative_l2", json::Value::make_number(v->max_partial_relative_error)},
                    {"max_merged_absolute_error", json::Value::make_number(v->max_absolute_error)},
                    {"timed_expert_loads", json::Value::make_number(v->timed_loads)},
                    {"timed_cache_hits", json::Value::make_number(v->timed_hits)},
                    {"timed_bytes_read", json::Value::make_number(v->timed_bytes)},
                    {"samples_ms", json::Value::make_array(std::move(samples))},
                });
                std::cout << mode << " cpu=" << v->jobs << " threads=" << v->threads
                          << " median_ms=" << median(v->times) << " speedup=" << base / median(v->times)
                          << " error=" << v->max_relative_error << std::endl;
                reports.push_back(std::move(row));
            }
            std::ofstream(argv[3]) << json::stringify(json::Value::make_object({
                {"layer", json::Value::make_number(layer)},
                {"cases", json::Value::make_number(cases.size())},
                {"schedule", json::Value::make_string(variant_sweeps ? "variant-sweeps" : "case-interleaved")},
                {"rows", json::Value::make_array(reports)}}));
        }
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
