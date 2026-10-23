// Serial JSONL worker. A session survives independent requests; preparation
// defaults are reset for each request. Fatal execution errors end the process.
#include "app/cli/request.h"
#include "engine/community_models/fish_drama3/session.h"
#include "engine/framework/audio/output.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/json.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/errors.h"
#include <cuda_runtime_api.h>

#include <chrono>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace json = engine::io::json;
namespace runtime = engine::runtime;
using Clock = std::chrono::steady_clock;

static double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
static void emit(json::Value::Object fields) {
    std::cout << "@drama3 " << json::stringify(json::Value::make_object(std::move(fields))) << std::endl;
}
static void text_file(const std::filesystem::path & path, const std::string & value) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file << value;
    if (!file) throw std::runtime_error("Cannot write artifact: " + path.u8string());
}

int main(int argc, char ** argv) {
    std::string id;
    try {
        if (argc >= 2 && std::string(argv[1]) == "--self-test") {
            engine::core::BackendConfig config;
            config.type = engine::core::BackendType::Cuda;
            config.device = argc > 2 ? std::stoi(argv[2]) : 0;
            engine::core::ExecutionContext execution(config);
            std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
                ggml_init({1024 * 1024, nullptr, true}), ggml_free);
            if (!context) throw std::runtime_error("Self-test graph allocation failed");
            auto * weight = ggml_new_tensor_2d(context.get(), GGML_TYPE_BF16, 32, 32);
            auto * input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 32, 1);
            auto * output = ggml_mul_mat(context.get(), weight, input);
            auto * graph = ggml_new_graph_custom(context.get(), 64, false);
            ggml_build_forward_expand(graph, output);
            std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
                ggml_backend_alloc_ctx_tensors(context.get(), execution.backend()), ggml_backend_buffer_free);
            if (!buffer) throw std::runtime_error("Self-test GPU allocation failed");
            std::vector<ggml_bf16_t> weights(32 * 32, ggml_fp32_to_bf16(1.0F));
            std::vector<float> values(32, 1.0F), result(32);
            ggml_backend_tensor_set(weight, weights.data(), 0, weights.size() * sizeof(ggml_bf16_t));
            ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
            if (engine::core::compute_backend_graph(execution.backend(), graph) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("Native CUDA self-test compute failed");
            ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
            double error = 0;
            for (const auto value : result) {
                if (!std::isfinite(value)) throw std::runtime_error("Nonfinite native self-test result");
                error = std::max(error, static_cast<double>(std::abs(value - 32.0F)));
            }
            engine::core::release_backend_graph_resources(execution.backend(), graph, true);
            emit({{"event", json::Value::make_string("self_test")},
                  {"passed", json::Value::make_bool(error < 0.01)},
                  {"device", json::Value::make_number(config.device)},
                  {"bf16_matmul_max_abs", json::Value::make_number(error)}});
            return error < 0.01 ? 0 : 3;
        }
        if (argc == 2 && std::string(argv[1]) == "--probe") {
            int count = 0, driver = 0, runtime_version = 0;
            const auto status = cudaGetDeviceCount(&count);
            cudaDriverGetVersion(&driver);
            cudaRuntimeGetVersion(&runtime_version);
            json::Value::Array devices;
            size_t selected_free = 0, selected_total = 0;
            int selected = -1;
            for (int index = 0; status == cudaSuccess && index < count; ++index) {
                cudaDeviceProp properties{};
                size_t free = 0, total = 0;
                if (cudaSetDevice(index) != cudaSuccess ||
                    cudaGetDeviceProperties(&properties, index) != cudaSuccess ||
                    cudaMemGetInfo(&free, &total) != cudaSuccess) continue;
                const int cc = properties.major * 10 + properties.minor;
                // Advertise the code targets actually shipped by this
                // Windows RTX build, not every CUDA device with newer VRAM.
                const bool supported = cc == 75 || cc == 86 || cc == 89 || cc == 120;
                devices.push_back(json::Value::make_object({
                    {"index", json::Value::make_number(index)},
                    {"name", json::Value::make_string(properties.name)},
                    {"major", json::Value::make_number(properties.major)},
                    {"minor", json::Value::make_number(properties.minor)},
                    {"free_bytes", json::Value::make_number(static_cast<double>(free))},
                    {"total_bytes", json::Value::make_number(static_cast<double>(total))},
                    {"supported", json::Value::make_bool(supported)},
                }));
                if (supported && (selected < 0 || free > selected_free)) {
                    selected = index; selected_free = free; selected_total = total;
                }
            }
            emit({{"event", json::Value::make_string("memory")},
                  {"available", json::Value::make_bool(selected >= 0)},
                  {"free_bytes", json::Value::make_number(static_cast<double>(selected_free))},
                  {"total_bytes", json::Value::make_number(static_cast<double>(selected_total))},
                  {"selected_device", json::Value::make_number(selected)},
                  {"devices", json::Value::make_array(std::move(devices))},
                  {"driver_version", json::Value::make_number(driver)},
                  {"runtime_version", json::Value::make_number(runtime_version)},
                  {"error", json::Value::make_string(status == cudaSuccess ? "" : cudaGetErrorString(status))}});
            return selected >= 0 ? 0 : 2;
        }
        if (argc != 2) throw std::runtime_error("Usage: drama3_worker config.json | --probe");
        const auto config = json::parse_file(std::filesystem::u8path(argv[1]));
        engine::debug::configure_logging({true, json::require_string(config, "log")});
        const auto started = Clock::now();
        auto registry = runtime::make_default_registry();
        runtime::ModelLoadRequest load;
        load.model_path = std::filesystem::u8path(json::require_string(config, "model"));
        load.family_hint = "fish_drama3";
        auto model = registry.load(load);
        runtime::SessionOptions options;
        options.backend.type = engine::core::BackendType::Cuda;
        options.backend.device = json::optional_i32(config, "device", 0);
        if (cudaSetDevice(options.backend.device) != cudaSuccess)
            throw std::runtime_error("Selected CUDA device is unavailable");
        options.backend.threads = json::require_i32(config, "threads");
#ifdef _OPENMP
        omp_set_num_threads(options.backend.threads);
#endif
        options.options = minitts::cli::json_options_map(config.find("options"));
        // One streaming-capable session also implements offline run; both paths
        // reuse the same model rather than allocating a second inference bank.
        auto session = model->create_task_session({runtime::VoiceTaskKind::Tts, runtime::RunMode::Streaming}, options);
        runtime::finish_model_load_trace();
        auto * offline = dynamic_cast<runtime::IOfflineVoiceTaskSession *>(session.get());
        if (!offline) throw std::runtime_error("No offline Drama3 session");
        emit({{"event", json::Value::make_string("ready")},
              {"load_seconds", json::Value::make_number(elapsed(started))},
              {"context_resize", json::Value::make_bool(true)},
              {"protocol", json::Value::make_number(5)}});
        std::string line;
        while (std::getline(std::cin, line)) {
            id.clear();
            try {
                const auto envelope = json::parse(line);
                id = json::require_string(envelope, "id");
                const auto operation = minitts::cli::json_optional_string(envelope, "op").value_or("");
                if (operation == "close") break;
                if (operation == "reserve_context") {
                    const int capacity = json::require_i32(envelope, "capacity");
                    if (capacity < 256 || capacity > 8192)
                        throw runtime::InvalidRequestError("Context capacity must be in 256..8192");
                    auto * drama = dynamic_cast<engine::models::fish_drama3::Drama3Session *>(session.get());
                    if (!drama) throw runtime::InvalidRequestError("Session does not support context resizing");
                    const auto resize_started = Clock::now();
                    drama->reserve_context(capacity);
                    emit({{"event", json::Value::make_string("context_ready")},
                          {"id", json::Value::make_string(id)},
                          {"capacity", json::Value::make_number(drama->context_capacity())},
                          {"seconds", json::Value::make_number(elapsed(resize_started))}});
                    continue;
                }
                if (operation == "prewarm_reference") {
                    auto request = minitts::cli::build_request_from_json(
                        envelope.require("request"), std::filesystem::current_path());
                    const auto encode_only = request.options.find("encode_reference_only");
                    if (encode_only == request.options.end() ||
                        (encode_only->second != "true" && encode_only->second != "1")) {
                        throw runtime::InvalidRequestError(
                            "Reference preparation must not synthesize speech");
                    }
                    session->prepare(runtime::build_preparation_request(request));
                    auto result = offline->run(request);
                    if (result.audio_output) {
                        throw std::runtime_error(
                            "Native reference preparation unexpectedly generated speech");
                    }
                    json::Value::Array references;
                    for (const auto & artifact : result.output_artifacts) {
                        if (artifact.id.rfind("drama3_reference_", 0) != 0) continue;
                        references.push_back(json::Value::make_string(
                            std::string(reinterpret_cast<const char *>(artifact.payload.data()),
                                        artifact.payload.size())));
                    }
                    if (references.empty()) {
                        throw std::runtime_error("Native reference preparation returned no audio codes");
                    }
                    emit({{"event", json::Value::make_string("prewarmed")},
                          {"id", json::Value::make_string(id)},
                          {"references", json::Value::make_array(std::move(references))}});
                    continue;
                }
                const auto output = std::filesystem::u8path(json::require_string(envelope, "output"));
                if (!output.is_absolute() || output.extension() != ".wav")
                    throw runtime::InvalidRequestError("Output must be an absolute WAV path");
                auto request = minitts::cli::build_request_from_json(
                    envelope.require("request"), std::filesystem::current_path());
                const auto chunk_frames = json::optional_i32(envelope, "chunk_frames", 16);
                if (chunk_frames < 1 || chunk_frames > 2048)
                    throw runtime::InvalidRequestError("Audio chunk frame count must be in 1..2048");
                const auto request_started = Clock::now();
                // Clear defaults on every request: no voice/history leaks.
                session->prepare(runtime::build_preparation_request(request));
                runtime::TaskResult result;
                if (json::optional_bool(envelope, "streaming", false)) {
                    auto * streamed = dynamic_cast<runtime::IStreamingVoiceTaskSession *>(session.get());
                    if (!streamed) throw runtime::InvalidRequestError("Native audio streaming is unavailable");
                    runtime::AudioBuffer pending;
                    int64_t delivered = 0, sequence = 0;
                    const auto flush = [&]() {
                        if (pending.samples.empty()) return;
                        auto part = output.parent_path() / (output.stem().u8string() + ".part-" + std::to_string(++sequence) + ".wav");
                        engine::audio::WavPcm16Sink().write(part, {pending.sample_rate, pending.channels, pending.samples});
                        emit({{"event", json::Value::make_string("audio")}, {"id", json::Value::make_string(id)},
                              {"path", json::Value::make_string(part.u8string())},
                              {"index", json::Value::make_number(sequence - 1)},
                              {"start_sample", json::Value::make_number(delivered)},
                              {"samples", json::Value::make_number(pending.samples.size())},
                              {"sample_rate", json::Value::make_number(pending.sample_rate)}});
                        delivered += pending.samples.size();
                        pending.samples.clear();
                    };
                    std::filesystem::create_directories(output.parent_path());
                    streamed->set_stream_event_sink([&](const runtime::StreamEvent & event) {
                        if (event.audio_output) {
                            runtime::append_audio_buffer(pending, *event.audio_output);
                            if (pending.samples.size() >= static_cast<size_t>(chunk_frames) * 2048) flush();
                        }
                    });
                    try {
                        streamed->start_stream(request);
                        result = streamed->finalize();
                        flush();
                        streamed->set_stream_event_sink({});
                    } catch (...) { streamed->set_stream_event_sink({}); throw; }
                } else result = offline->run(request);
                if (!result.audio_output) throw std::runtime_error("No audio output");
                std::filesystem::create_directories(output.parent_path());
                const auto & audio = *result.audio_output;
                auto temporary = output;
                temporary += ".tmp";
                engine::audio::WavPcm16Sink().write(temporary, {audio.sample_rate, audio.channels, audio.samples});
                // Output paths are unique per UI job; replace is for CLI reuse.
                if (std::filesystem::exists(output)) std::filesystem::remove(output);
                std::filesystem::rename(temporary, output);
                auto take = output;
                take.replace_extension(".take.json");
                for (const auto & artifact : result.output_artifacts) {
                    if (artifact.id == "drama3_take") {
                        text_file(take, std::string(reinterpret_cast<const char *>(artifact.payload.data()),
                                                    artifact.payload.size()));
                    }
                }
                emit({{"event", json::Value::make_string("result")},
                      {"id", json::Value::make_string(id)},
                      {"output", json::Value::make_string(output.u8string())},
                      {"native_take_file", json::Value::make_string(take.u8string())},
                      {"request_seconds", json::Value::make_number(elapsed(request_started))},
                      {"audio_seconds", json::Value::make_number(
                          static_cast<double>(audio.samples.size()) / (audio.sample_rate * audio.channels))},
                      {"generation", result.custom_schema_output ? result.custom_schema_output->data : json::Value{}}});
            } catch (const runtime::CapacityError & error) {
                emit({{"event", json::Value::make_string("error")}, {"id", json::Value::make_string(id)},
                      {"recoverable", json::Value::make_bool(true)},
                      {"message", json::Value::make_string(error.what())}});
            } catch (const runtime::InvalidRequestError & error) {
                emit({{"event", json::Value::make_string("error")}, {"id", json::Value::make_string(id)},
                      {"recoverable", json::Value::make_bool(true)},
                      {"message", json::Value::make_string(error.what())}});
            } catch (const std::exception & error) {
                emit({{"event", json::Value::make_string("error")}, {"id", json::Value::make_string(id)},
                      {"recoverable", json::Value::make_bool(false)},
                      {"message", json::Value::make_string(error.what())}});
                return 1;
            }
        }
        return 0;
    } catch (const std::exception & error) {
        emit({{"event", json::Value::make_string("error")}, {"id", json::Value::make_string(id)},
              {"recoverable", json::Value::make_bool(false)},
              {"message", json::Value::make_string(error.what())}});
        return 1;
    }
}
