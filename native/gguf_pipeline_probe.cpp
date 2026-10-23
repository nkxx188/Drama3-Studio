// Known-route, full top-8 GPU compute. A bounded overlap experiment; CPU only
// gathers bytes. Does not alter the service or cooperative expert code.
#define main drama3_old_expert_probe_main
#include "gguf_hybrid_probe.cpp"
#undef main
#include <cuda_runtime_api.h>
#include <cstring>
#include <map>
#include "engine/framework/core/host_memory.h"

static void checked(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
// Bounded profile hot set, research only. Unlike a full pinned bank it
// duplicates at most 128MiB. This best-case layer-local profile is not a claim
// that 40 layers fit the same hot coverage into 128MiB.
struct HostHotSet {
    std::map<int32_t, std::array<void *, 3>> rows;
    size_t bytes = 0;
    uint64_t hits = 0, misses = 0;
    HostHotSet(const CpuBank & source, const std::vector<Case> & cases, size_t budget) {
        const auto available = core::available_host_memory_bytes();
        budget = available ? std::min(budget, available / 16) : 0;
        std::map<int32_t, int> frequency;
        for (const auto & c : cases) for (auto id : c.ids) ++frequency[id];
        std::vector<std::pair<int32_t, int>> ranked(frequency.begin(), frequency.end());
        std::stable_sort(ranked.begin(), ranked.end(), [](auto a, auto b) { return a.second > b.second; });
        size_t per_expert = 0;
        for (auto * weight : source.weights) per_expert += weight->nb[2];
        for (const auto & item : ranked) {
            if (bytes + per_expert > budget) break;
            auto & expert = rows[item.first];
            for (size_t p = 0; p < 3; ++p) {
                const auto count = source.weights[p]->nb[2];
                const auto status = cudaHostAlloc(&expert[p], count, cudaHostAllocDefault);
                if (status != cudaSuccess) {
                    for (auto * partial : expert) if (partial) cudaFreeHost(partial);
                    rows.erase(item.first);
                    return; // Allocation pressure shrinks the optional set.
                }
                std::memcpy(expert[p], static_cast<const std::byte *>(source.weights[p]->data) + item.first * count, count);
            }
            bytes += per_expert;
        }
    }
    ~HostHotSet() { for (const auto & item : rows) for (auto * p : item.second) cudaFreeHost(p); }
    const void * find(int32_t id, int p) {
        auto found = rows.find(id);
        if (p == 0) found == rows.end() ? ++misses : ++hits;
        return found == rows.end() ? nullptr : found->second[p];
    }
};
struct SlotBank {
    Context ctx = context();
    Buffer gpu;
    std::array<ggml_tensor *, 3> tensors{};
    void * pinned = nullptr;
    size_t bytes = 0;
    cudaEvent_t uploaded = nullptr, consumed = nullptr;
    std::unique_ptr<ExpertGraph> graph;
    SlotBank(core::ExecutionContext & execution, const CpuBank & source, int count) {
        for (int p = 0; p < 3; ++p) {
            const auto * original = source.weights[p];
            tensors[p] = ggml_new_tensor_3d(ctx.get(), original->type, original->ne[0], original->ne[1], count);
            bytes += ggml_nbytes(tensors[p]);
        }
        gpu.p = ggml_backend_alloc_ctx_tensors(ctx.get(), execution.backend());
        if (!gpu.p) throw std::runtime_error("cannot allocate bounded GPU banks");
        checked(cudaHostAlloc(&pinned, bytes, cudaHostAllocDefault));
        checked(cudaEventCreateWithFlags(&uploaded, cudaEventDisableTiming));
        checked(cudaEventCreateWithFlags(&consumed, cudaEventDisableTiming));
        graph = std::make_unique<ExpertGraph>(execution, tensors, count);
        std::vector<int32_t> local(count);
        std::iota(local.begin(), local.end(), 0);
        ggml_backend_tensor_set(graph->ids, local.data(), 0, count * sizeof(int32_t));
    }
    ~SlotBank() {
        graph.reset();
        if (uploaded) cudaEventDestroy(uploaded);
        if (consumed) cudaEventDestroy(consumed);
        if (pinned) cudaFreeHost(pinned);
    }
};

struct Pipeline {
    core::ExecutionContext & gpu;
    const CpuBank & source;
    int width;
    bool overlap;
    cudaStream_t copy = nullptr, compute = nullptr;
    std::vector<std::unique_ptr<SlotBank>> banks;
    MergeGraph merge;
    HostHotSet * hot = nullptr;
    double gather_ms = 0, wait_ms = 0;
    Pipeline(core::ExecutionContext & execution, const CpuBank & src, int w, bool async, HostHotSet * cache = nullptr)
        : gpu(execution), source(src), width(w), overlap(async), merge(execution), hot(cache) {
        compute = static_cast<cudaStream_t>(core::backend_cuda_stream(gpu.backend()));
        if (!compute) throw std::runtime_error("native CUDA stream unavailable");
        checked(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking));
        for (int i = 0; i < (overlap ? 2 : 1); ++i) {
            banks.push_back(std::make_unique<SlotBank>(gpu, source, width));
            checked(cudaEventRecord(banks.back()->uploaded, copy));
            checked(cudaEventRecord(banks.back()->consumed, compute));
        }
    }
    ~Pipeline() {
        cudaDeviceSynchronize();
        banks.clear();
        cudaStreamDestroy(copy);
    }
    void prepare(const Case & c) {
        for (auto & bank : banks)
            ggml_backend_tensor_set(bank->graph->x, c.x.data(), 0, H * sizeof(float));
        ggml_backend_tensor_set(merge.weights, c.weights.data(), 0, K * sizeof(float));
        checked(cudaDeviceSynchronize());
    }
    std::vector<float> run(const Case & c) {
        for (int group = 0; group < K / width; ++group) {
            auto & bank = *banks[group % banks.size()];
            auto started = Clock::now();
            checked(cudaEventSynchronize(bank.uploaded)); // staging can now be overwritten
            wait_ms += elapsed(started);
            size_t offset = 0;
            checked(cudaStreamWaitEvent(copy, bank.consumed, 0));
            for (int p = 0; p < 3; ++p) {
                started = Clock::now();
                const auto expert_bytes = source.weights[p]->nb[2];
                std::vector<const void *> sources(width);
                for (int lane = 0; lane < width; ++lane) {
                    const auto id = c.ids[group * width + lane];
                    sources[lane] = hot ? hot->find(id, p) : nullptr;
                    if (!sources[lane]) {
                        auto * target = static_cast<std::byte *>(bank.pinned) + offset + lane * expert_bytes;
                        std::memcpy(target, static_cast<const std::byte *>(source.weights[p]->data) +
                                    id * expert_bytes, expert_bytes);
                        sources[lane] = target;
                    }
                }
                gather_ms += elapsed(started);
                if (!hot)
                    checked(cudaMemcpyAsync(bank.tensors[p]->data, static_cast<std::byte *>(bank.pinned) + offset,
                                            width * expert_bytes, cudaMemcpyHostToDevice, copy));
                else
                    for (int lane = 0; lane < width; ++lane)
                        checked(cudaMemcpyAsync(static_cast<std::byte *>(bank.tensors[p]->data) + lane * expert_bytes,
                                                sources[lane], expert_bytes, cudaMemcpyHostToDevice, copy));
                offset += width * expert_bytes;
            }
            // Wait only on this destination bank's prior use; no device-wide
            // barrier. The next copy stream can overlap the current compute.
            checked(cudaEventRecord(bank.uploaded, copy));
            checked(cudaStreamWaitEvent(compute, bank.uploaded, 0));
            if (ggml_backend_graph_compute_async(gpu.backend(), bank.graph->graph) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("asynchronous expert compute failed");
            checked(cudaMemcpyAsync(static_cast<float *>(merge.partial->data) + group * width * H,
                                    bank.graph->partial->data, width * H * sizeof(float),
                                    cudaMemcpyDeviceToDevice, compute));
            checked(cudaEventRecord(bank.consumed, compute));
            if (!overlap) checked(cudaStreamSynchronize(compute));
        }
        if (ggml_backend_graph_compute_async(gpu.backend(), merge.graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("asynchronous merge failed");
        std::vector<float> output(H);
        ggml_backend_tensor_get_async(gpu.backend(), merge.output, output.data(), 0, H * sizeof(float));
        ggml_backend_synchronize(gpu.backend());
        return output;
    }
};

static std::vector<Case> cases_for_layer0(const char * path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot read captured routes");
    std::vector<Case> cases;
    for (;;) {
        uint32_t header[5];
        file.read(reinterpret_cast<char *>(header), sizeof(header));
        if (file.eof() && file.gcount() == 0) break;
        if (!file || header[3] != H || header[4] != K) throw std::runtime_error("Invalid route header");
        Case c; c.ids.resize(K); c.weights.resize(K); c.x.resize(H);
        std::array<int32_t, K> resident{};
        file.read(reinterpret_cast<char *>(c.ids.data()), K * sizeof(int32_t));
        file.read(reinterpret_cast<char *>(c.weights.data()), K * sizeof(float));
        file.read(reinterpret_cast<char *>(c.x.data()), H * sizeof(float));
        file.read(reinterpret_cast<char *>(resident.data()), K * sizeof(int32_t));
        if (!file) throw std::runtime_error("Truncated replay");
        if (header[1] == 0 && cases.size() < 32) cases.push_back(std::move(c));
    }
    if (cases.empty()) throw std::runtime_error("No layer0 routes");
    return cases;
}

int main(int argc, char ** argv) {
    try {
        if (argc != 4) throw std::runtime_error("Usage: pipeline_probe model.gguf replay.bin output.json");
        core::BackendConfig config; config.type = core::BackendType::Cuda; config.threads = 8;
        core::ExecutionContext gpu(config);
        const std::string root = "model_weights/model.language_model.layers.0.mlp.experts.";
        CpuBank source(argv[1], {root + "gate_proj.weight", root + "up_proj.weight", root + "down_proj.weight"});
        const auto cases = cases_for_layer0(argv[2]);
        HostHotSet hot(source, cases, 128 * 1024 * 1024);
        Pipeline baseline(gpu, source, 8, false);
        Pipeline serial4(gpu, source, 4, false);
        Pipeline split4(gpu, source, 4, true);
        Pipeline serial2(gpu, source, 2, false);
        Pipeline split2(gpu, source, 2, true);
        Pipeline hot2(gpu, source, 2, true, &hot);
        std::array<Pipeline *, 6> variants{&baseline, &serial4, &split4, &serial2, &split2, &hot2};
        std::array<std::vector<double>, 6> times;
        std::array<double, 6> max_error{};
        // All variants get graph/kernel and file-page warmup.
        for (auto * v : variants) {
            for (const auto & c : cases) {
                v->prepare(c);
                (void)v->run(c);
            }
            v->gather_ms = v->wait_ms = 0;
        }
        std::vector<std::vector<float>> expected(cases.size());
        for (int repeat = 0; repeat < 4; ++repeat) {
            for (size_t index = 0; index < cases.size(); ++index) {
                // Alternate order to reduce cache/temperature order bias.
                for (int order = 0; order < 6; ++order) {
                    int variant = (order + repeat) % 6;
                    auto * v = variants[variant];
                    v->prepare(cases[index]);
                    const auto start = Clock::now();
                    auto output = v->run(cases[index]);
                    times[variant].push_back(elapsed(start));
                    if (variant == 0) expected[index] = output;
                    else if (!expected[index].empty()) {
                        double difference = 0, norm = 0;
                        for (int i = 0; i < H; ++i) {
                            difference += std::pow(output[i] - expected[index][i], 2);
                            norm += expected[index][i] * expected[index][i];
                        }
                        max_error[variant] = std::max(max_error[variant], std::sqrt(difference / std::max(1e-20, norm)));
                    }
                }
            }
        }
        json::Value::Array rows;
        for (int v = 0; v < 6; ++v) {
            auto values = times[v]; std::sort(values.begin(), values.end());
            rows.push_back(json::Value::make_object({
                {"group_width", json::Value::make_number(variants[v]->width)},
                {"overlap", json::Value::make_bool(variants[v]->overlap)},
                {"hot_cache_bytes", json::Value::make_number(variants[v]->hot ? static_cast<double>(hot.bytes) : 0)},
                {"median_ms", json::Value::make_number(values[values.size()/2])},
                {"mean_ms", json::Value::make_number(std::accumulate(values.begin(), values.end(), 0.0) / values.size())},
                {"gather_mean_ms", json::Value::make_number(variants[v]->gather_ms / values.size())},
                {"staging_wait_mean_ms", json::Value::make_number(variants[v]->wait_ms / values.size())},
                {"max_relative_l2", json::Value::make_number(max_error[v])},
                {"samples", json::Value::make_number(static_cast<double>(values.size()))}}));
        }
        std::ofstream report(argv[3]);
        report << json::stringify(json::Value::make_object({
            {"scope", json::Value::make_string("Real layer0 all-miss full top-8; warmed mapped source; includes gather/transfer/compute/merge/sync. Hot set uses same cases as favorable profiling coverage; no 40-layer or full-request speed claim.")},
            {"variants", json::Value::make_array(std::move(rows))}})) << "\n";
        return report ? 0 : 2;
    } catch (const std::exception & error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
