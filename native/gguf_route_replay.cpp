// Research-only replay of real Q8 routes in frame/layer order.
// Reuse the measured graph and worker helpers without changing the older CLI.
#define main drama3_single_layer_probe_main
#include "gguf_hybrid_probe.cpp"
#undef main
#include <cstring>
#include <limits>
#include <map>
#include <set>

struct TraceCase : Case {
    uint32_t request, layer, step;
    std::vector<int32_t> resident;
};

std::vector<TraceCase> read_cases(const char * path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot read native Q8 trace");
    std::vector<TraceCase> result;
    for (;;) {
        uint32_t header[5];
        file.read(reinterpret_cast<char *>(header), sizeof(header));
        if (file.eof() && file.gcount() == 0) break;
        if (!file || header[3] != H || header[4] != K || header[1] >= 40)
            throw std::runtime_error("invalid native Q8 trace header");
        TraceCase c;
        c.request = header[0]; c.layer = header[1]; c.step = header[2];
        c.x.resize(H); c.ids.resize(K); c.weights.resize(K); c.resident.resize(K);
        file.read(reinterpret_cast<char *>(c.ids.data()), K * sizeof(int32_t));
        file.read(reinterpret_cast<char *>(c.weights.data()), K * sizeof(float));
        file.read(reinterpret_cast<char *>(c.x.data()), H * sizeof(float));
        file.read(reinterpret_cast<char *>(c.resident.data()), K * sizeof(int32_t));
        if (!file || std::set<int32_t>(c.ids.begin(), c.ids.end()).size() != K)
            throw std::runtime_error("truncated/duplicate native Q8 trace");
        for (auto id : c.ids) if (id < 0 || id >= 256) throw std::runtime_error("bad expert ID");
        result.push_back(std::move(c));
    }
    if (result.empty()) throw std::runtime_error("empty Q8 replay");
    return result;
}

std::vector<std::string> names_for(int layer) {
    const auto prefix = "model_weights/model.language_model.layers." +
                        std::to_string(layer) + ".mlp.experts.";
    return {prefix + "gate_proj.weight", prefix + "up_proj.weight", prefix + "down_proj.weight"};
}

struct AllCpuBanks {
    engine::io::BinaryBlob mapping;
    Context ctx = context();
    Buffer buffer;
    std::array<std::array<ggml_tensor *, 3>, 40> banks{};
    explicit AllCpuBanks(const std::string & path) : mapping(engine::io::read_binary_blob(path, true)) {
        ggml_context * headers = nullptr;
        auto * meta = gguf_init_from_file(path.c_str(), {true, &headers});
        if (!meta || !headers) throw std::runtime_error("cannot parse Q8 GGUF");
        const auto key = gguf_find_key(meta, "audiocpp.tensor_names");
        std::map<std::string, std::pair<int, int>> wanted;
        for (int layer = 0; layer < 40; ++layer) {
            const auto names = names_for(layer);
            for (int p = 0; p < 3; ++p) wanted.emplace(names[p], std::make_pair(layer, p));
        }
        buffer.p = ggml_backend_cpu_buffer_from_ptr(
            const_cast<std::byte *>(mapping.data()), mapping.size());
        ggml_backend_buffer_set_usage(buffer.p, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (int64_t i = 0; i < gguf_get_n_tensors(meta); ++i) {
            const std::string logical = key >= 0 ? gguf_get_arr_str(meta, key, i)
                                                : gguf_get_tensor_name(meta, i);
            const auto found = wanted.find(logical);
            if (found == wanted.end()) continue;
            auto * shape = ggml_get_tensor(headers, gguf_get_tensor_name(meta, i));
            if (shape->type != GGML_TYPE_Q8_0 || shape->ne[2] != 256)
                throw std::runtime_error("expected Q8_0 experts");
            auto * tensor = ggml_new_tensor_3d(ctx.get(), shape->type, shape->ne[0], shape->ne[1], shape->ne[2]);
            const auto offset = gguf_get_data_offset(meta) + gguf_get_tensor_offset(meta, i);
            if (offset + ggml_nbytes(tensor) > mapping.size() ||
                ggml_backend_tensor_alloc(buffer.p, tensor,
                    const_cast<std::byte *>(mapping.data()) + offset) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("cannot bind CPU expert view");
            banks[found->second.first][found->second.second] = tensor;
        }
        ggml_free(headers); gguf_free(meta);
        for (const auto & bank : banks) for (auto * p : bank)
            if (!p) throw std::runtime_error("missing expert projection");
    }
};

struct Replay {
    core::ExecutionContext & gpu;
    const AllCpuBanks & banks;
    int cpu_budget, threads;
    bool keep_hits;
    core::ExecutionContext cpu;
    core::DiskExpertWeightStore store;
    Worker worker;
    Context ctx = context();
    Buffer source, handoff;
    ggml_tensor * source_x, * source_ids;
    float * host_x, * cpu_partial;
    int32_t * host_ids;
    std::unique_ptr<MergeGraph> merge;
    std::array<size_t, 40> groups;
    std::array<std::vector<int32_t>, 40> placement;
    std::map<std::pair<int, int>, std::unique_ptr<ExpertGraph>> gpu_graphs, cpu_graphs;
    std::vector<double> times;
    std::array<double, 40> layer_ms{};
    std::array<uint64_t, 40> layer_calls{};
    std::array<double, 40> layer_error{};
    std::vector<double> errors;
    uint64_t cpu_jobs = 0, gpu_jobs = 0, kept_hits = 0, loads = 0, copied_bytes = 0;
    double error = 0, absolute_error = 0;

    Replay(core::ExecutionContext & g, const AllCpuBanks & b, const std::string & path,
           int budget, bool preserve_hits, int nthreads = 4)
        : gpu(g), banks(b), cpu_budget(budget), threads(nthreads), keep_hits(preserve_hits),
          cpu(core::BackendConfig{core::BackendType::Cpu, 0, nthreads}), store(path, false) {
        source_x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, H, 1, 1);
        source_ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, K);
        source.p = ggml_backend_alloc_ctx_tensors(ctx.get(), gpu.backend());
        const auto host_type = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(gpu.backend()));
        handoff.p = ggml_backend_buft_alloc_buffer(host_type, (H + H * K) * sizeof(float) + K * sizeof(int32_t));
        if (!source.p || !handoff.p || ggml_backend_buffer_get_type(handoff.p) != host_type)
            throw std::runtime_error("cannot allocate bounded replay handoff");
        host_x = static_cast<float *>(ggml_backend_buffer_get_base(handoff.p));
        cpu_partial = host_x + H;
        host_ids = reinterpret_cast<int32_t *>(cpu_partial + H * K);
        if (budget) merge = std::make_unique<MergeGraph>(gpu);
        for (int layer = 0; layer < 40; ++layer) {
            groups[layer] = store.add_group(names_for(layer), gpu.backend(), K);
            placement[layer].assign(K, -1);
        }
    }
    ExpertGraph & graph(int layer, int count, bool on_cpu) {
        auto & cache = on_cpu ? cpu_graphs : gpu_graphs;
        auto & ptr = cache[{layer, count}];
        if (!ptr) {
            std::array<ggml_tensor *, 3> weights = banks.banks[layer];
            if (!on_cpu) for (int p = 0; p < 3; ++p) weights[p] = store.tensor(groups[layer], p).tensor;
            ptr = std::make_unique<ExpertGraph>(on_cpu ? cpu : gpu, weights, count, !on_cpu && count == K);
        }
        return *ptr;
    }
    void seed(const std::vector<TraceCase> & cases, size_t begin) {
        // A captured window starts with the real all-GPU resident set. Later
        // steps use each candidate's own placement, not the baseline's state.
        for (size_t i = begin; i < std::min(begin + 40, cases.size()); ++i) {
            const auto & c = cases[i];
            if (c.step != cases[begin].step) break;
            std::vector<int32_t> ids;
            for (auto id : c.resident) if (id >= 0) ids.push_back(id);
            placement[c.layer].assign(K, -1);
            if (!ids.empty()) {
                const auto slots = store.stage(groups[c.layer], ids);
                for (size_t j = 0; j < ids.size(); ++j) placement[c.layer][slots[j]] = ids[j];
            }
        }
    }
    std::vector<float> run(const TraceCase & c, bool record, bool readback) {
        // Inputs already exist on GPU in the real decoder. These writes and
        // the numerical check are outside the measured expert segment.
        ggml_backend_tensor_set(source_x, c.x.data(), 0, H * sizeof(float));
        ggml_backend_tensor_set(source_ids, c.ids.data(), 0, K * sizeof(int32_t));
        const auto before = store.statistics();
        const auto started = Clock::now();
        ggml_backend_tensor_get_async(gpu.backend(), source_ids, host_ids, 0, K * sizeof(int32_t));
        if (cpu_budget) ggml_backend_tensor_get_async(gpu.backend(), source_x, host_x, 0, H * sizeof(float));
        ggml_backend_synchronize(gpu.backend());
        std::vector<int> cpu_lanes, gpu_lanes;
        const auto & old = placement[c.layer];
        for (int lane = K - 1; lane >= 0 && cpu_lanes.size() < static_cast<size_t>(cpu_budget); --lane) {
            if (!keep_hits || std::find(old.begin(), old.end(), host_ids[lane]) == old.end())
                cpu_lanes.push_back(lane);
        }
        std::sort(cpu_lanes.begin(), cpu_lanes.end());
        for (int lane = 0; lane < K; ++lane)
            if (std::find(cpu_lanes.begin(), cpu_lanes.end(), lane) == cpu_lanes.end()) gpu_lanes.push_back(lane);
        auto * cpu_graph = cpu_lanes.empty() ? nullptr : &graph(c.layer, static_cast<int>(cpu_lanes.size()), true);
        auto * gpu_graph = gpu_lanes.empty() ? nullptr : &graph(c.layer, static_cast<int>(gpu_lanes.size()), false);
        const bool mixed = !cpu_lanes.empty();
        ggml_backend_tensor_set_async(gpu.backend(), mixed ? merge->weights : gpu_graph->weights,
                                      c.weights.data(), 0, K * sizeof(float));
        if (cpu_graph) worker.launch([&, cpu_graph] {
            std::vector<int32_t> ids;
            for (auto lane : cpu_lanes) ids.push_back(host_ids[lane]);
            ggml_backend_tensor_set(cpu_graph->x, host_x, 0, H * sizeof(float));
            ggml_backend_tensor_set(cpu_graph->ids, ids.data(), 0, ids.size() * sizeof(int32_t));
            cpu_graph->compute();
            ggml_backend_tensor_get(cpu_graph->partial, cpu_partial, 0, H * ids.size() * sizeof(float));
        });
        if (gpu_graph) {
            std::vector<int32_t> required;
            for (auto lane : gpu_lanes) required.push_back(host_ids[lane]);
            const auto slots = store.stage(groups[c.layer], required);
            for (size_t i = 0; i < slots.size(); ++i) placement[c.layer][slots[i]] = required[i];
            ggml_backend_tensor_copy_async(gpu.backend(), gpu.backend(), source_x, gpu_graph->x);
            ggml_backend_tensor_set_async(gpu.backend(), gpu_graph->ids, slots.data(), 0, slots.size() * sizeof(int32_t));
            gpu_graph->compute();
            if (mixed) {
                auto views = context(16384);
                // Preserve original lane order; transfers are timed.
                for (size_t i = 0; i < gpu_lanes.size(); ++i) {
                    auto * from = ggml_view_1d(views.get(), gpu_graph->partial, H, i * H * sizeof(float));
                    auto * to = ggml_view_1d(views.get(), merge->partial, H, gpu_lanes[i] * H * sizeof(float));
                    ggml_backend_tensor_copy_async(gpu.backend(), gpu.backend(), from, to);
                }
            }
        }
        if (cpu_graph) {
            worker.wait();
            for (size_t i = 0; i < cpu_lanes.size(); ++i)
                ggml_backend_tensor_set_async(gpu.backend(), merge->partial, cpu_partial + i * H,
                                             cpu_lanes[i] * H * sizeof(float), H * sizeof(float));
        }
        if (mixed) merge->compute();
        ggml_backend_synchronize(gpu.backend());
        const auto ms = elapsed(started);
        if (record) {
            const auto after = store.statistics();
            times.push_back(ms); layer_ms[c.layer] += ms; ++layer_calls[c.layer];
            cpu_jobs += cpu_lanes.size(); gpu_jobs += gpu_lanes.size();
            loads += after.expert_loads - before.expert_loads;
            copied_bytes += after.bytes_read - before.bytes_read;
            for (auto lane : gpu_lanes)
                if (std::find(old.begin(), old.end(), host_ids[lane]) != old.end()) ++kept_hits;
        }
        return readback ? core::read_tensor_f32(mixed ? merge->output : gpu_graph->output) : std::vector<float>{};
    }
};

int main(int argc, char ** argv) {
    try {
        if (argc != 4) throw std::runtime_error("model.gguf replay.bin output.json");
        const auto cases = read_cases(argv[2]);
        core::BackendConfig config; config.type = core::BackendType::Cuda;
        core::ExecutionContext gpu(config);
        AllCpuBanks banks(argv[1]);
        std::vector<std::vector<float>> expected(cases.size());
        json::Value::Array reports;
        // Baseline, simple split, then keep-hit split. One shared worker and CPU
        // backend per variant avoid creating a CPU pool for every model layer.
        const std::vector<std::pair<int, bool>> policies{
            {0, false}, {2, false}, {4, false}, {4, true}, {8, false}, {0, false}};
        size_t candidate = 0;
        for (auto policy : policies) {
            Replay replay(gpu, banks, argv[1], policy.first, policy.second);
            const auto overall = Clock::now();
            for (int pass = -1; pass < 3; ++pass) {
                replay.store.invalidate_cache();
                size_t previous_step = std::numeric_limits<size_t>::max();
                for (size_t i = 0; i < cases.size(); ++i) {
                    const bool new_window = i == 0 ||
                        (cases[i].layer == 0 && cases[i].step != previous_step + 1);
                    if (new_window) replay.seed(cases, i);
                    if (cases[i].layer == 39) previous_step = cases[i].step;
                    const auto output = replay.run(cases[i], pass >= 0, pass == 0);
                    if (pass == 0) {
                        if (candidate == 0) expected[i] = output;
                        else {
                            const auto error = relative(output, expected[i], replay.absolute_error);
                            replay.error = std::max(replay.error, error);
                            replay.errors.push_back(error);
                            replay.layer_error[cases[i].layer] = std::max(replay.layer_error[cases[i].layer], error);
                        }
                    }
                }
            }
            const double sum = std::accumulate(replay.times.begin(), replay.times.end(), 0.0);
            json::Value::Array layer_rows;
            for (int layer = 0; layer < 40; ++layer) layer_rows.push_back(json::Value::make_object({
                {"layer", json::Value::make_number(layer)},
                {"calls", json::Value::make_number(replay.layer_calls[layer])},
                {"mean_ms", json::Value::make_number(replay.layer_ms[layer] / replay.layer_calls[layer])},
                {"max_relative_l2", json::Value::make_number(replay.layer_error[layer])}
            }));
            std::sort(replay.errors.begin(), replay.errors.end());
            auto row = json::Value::make_object({
                {"candidate_index", json::Value::make_number(candidate)},
                {"cpu_budget", json::Value::make_number(policy.first)},
                {"keep_hits", json::Value::make_bool(policy.second)},
                {"cpu_threads", json::Value::make_number(replay.threads)},
                {"calls", json::Value::make_number(replay.times.size())},
                {"completed_ms", json::Value::make_number(sum)},
                {"median_layer_ms", json::Value::make_number(median(replay.times))},
                {"mean_frame_expert_ms", json::Value::make_number(sum / (replay.times.size() / 40))},
                {"cpu_expert_jobs", json::Value::make_number(replay.cpu_jobs)},
                {"gpu_expert_jobs", json::Value::make_number(replay.gpu_jobs)},
                {"expert_loads", json::Value::make_number(replay.loads)},
                {"copied_bytes", json::Value::make_number(replay.copied_bytes)},
                {"max_merged_relative_l2", json::Value::make_number(replay.error)},
                {"median_merged_relative_l2", json::Value::make_number(median(replay.errors))},
                {"p95_merged_relative_l2", json::Value::make_number(replay.errors.empty() ? 0 :
                    replay.errors[static_cast<size_t>((replay.errors.size() - 1) * 0.95)])},
                {"max_merged_absolute_error", json::Value::make_number(replay.absolute_error)},
                {"layers", json::Value::make_array(std::move(layer_rows))}
            });
            std::cout << "cpu=" << policy.first << " keep_hits=" << policy.second
                      << " frame_expert_ms=" << sum / (replay.times.size() / 40)
                      << " max_l2=" << replay.error << " wall_ms=" << elapsed(overall) << std::endl;
            reports.push_back(std::move(row));
            ++candidate;
            std::ofstream(argv[3]) << json::stringify(json::Value::make_object({
                {"cases", json::Value::make_number(cases.size())},
                {"rows", json::Value::make_array(reports)}
            }));
        }
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
