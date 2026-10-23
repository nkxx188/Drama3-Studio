// Decode an existing Drama3 take; never loads or executes the backbone.
#include "engine/community_models/fish_drama3/assets.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/codecs/fish_dac_codec_runtime.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/json.h"
#include "engine/framework/io/safetensors.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc, char ** argv) {
    try {
        if (argc == 7 && (std::string(argv[1]) == "--encode" || std::string(argv[1]) == "--encode-f32")) {
            namespace core = engine::core;
            namespace codecs = engine::codecs;
            using Clock = std::chrono::steady_clock;
            engine::debug::configure_logging({true, std::nullopt});
            const std::string backend(argv[5]);
            if (backend != "cpu" && backend != "cuda")
                throw std::runtime_error("encode backend must be cpu or cuda");
            core::BackendConfig config;
            config.type = backend == "cuda" ? core::BackendType::Cuda : core::BackendType::Cpu;
            config.threads = std::stoi(argv[6]);
            const auto started = Clock::now();
            const auto assets = engine::models::fish_drama3::load_drama3_assets(argv[2]);
            core::ExecutionContext execution(config);
            codecs::FishDacCodecRuntimeOptions options;
            options.weight_context_bytes = 16 * 1024 * 1024;
            if (std::string(argv[1]) == "--encode-f32") {
                options.matmul_weight_storage_type = options.conv_weight_storage_type =
                    engine::assets::TensorStorageType::F32;
            }
            const auto component = codecs::FishDacCodecComponent::load_from_tensor_source(
                assets->codec_weights, assets->codec, {}, execution.backend(), config.type, options);
            codecs::FishDacCodecRuntime codec(component, config, config.threads, options.graph_arena_bytes);
            assets->codec_weights->release_storage();
            assets->model_weights->release_storage();
            const auto wav = engine::audio::read_wav_f32(std::filesystem::path(argv[3]));
            const auto encode_started = Clock::now();
            const auto codes = codec.encode_codes({wav.sample_rate, wav.channels, wav.samples});
            const auto finished = Clock::now();
            std::vector<unsigned char> bytes(codes.codes.size() * sizeof(int32_t));
            std::memcpy(bytes.data(), codes.codes.data(), bytes.size());
            engine::io::write_safetensors_file(argv[4], {
                {"codes", "I32", {codes.codebooks, codes.frames}, std::move(bytes)}});
            std::cout << "probe.load_ms="
                      << std::chrono::duration<double, std::milli>(encode_started - started).count()
                      << " encode_ms="
                      << std::chrono::duration<double, std::milli>(finished - encode_started).count()
                      << " books=" << codes.codebooks << " frames=" << codes.frames << std::endl;
            return 0;
        }
        if (argc != 7) throw std::runtime_error("model codes.json output-dir cpu|cuda full|stream threads");
        namespace core = engine::core;
        namespace codecs = engine::codecs;
        namespace json = engine::io::json;
        using Clock = std::chrono::steady_clock;
        engine::debug::configure_logging({true, std::nullopt});
        const auto data = json::parse_file(argv[2]);
        codecs::FishDacCodes codes;
        const auto & books = data.require("vq_codes").as_array();
        codes.codebooks = static_cast<int64_t>(books.size());
        codes.frames = static_cast<int64_t>(books.at(0).as_array().size());
        if (codes.codebooks != 10 || codes.frames < 1) throw std::runtime_error("invalid probe codes");
        for (size_t book = 0; book < books.size(); ++book) {
            if (books[book].as_array().size() != static_cast<size_t>(codes.frames))
                throw std::runtime_error("unequal codebook lengths");
            for (const auto & item : books[book].as_array()) {
                const auto value = item.as_i64();
                if (value < 0 || value >= (book == 0 ? 4096 : 1024))
                    throw std::runtime_error("out of range codes");
                codes.codes.push_back(static_cast<int32_t>(value));
            }
        }
        const std::filesystem::path output(argv[3]);
        std::filesystem::create_directories(output);
        const std::string backend(argv[4]), mode(argv[5]);
        if ((backend != "cpu" && backend != "cuda") || (mode != "full" && mode != "stream"))
            throw std::runtime_error("unsupported probe mode");
        core::BackendConfig config;
        config.type = backend == "cuda" ? core::BackendType::Cuda : core::BackendType::Cpu;
        config.threads = std::stoi(argv[6]);
        const auto started = Clock::now();
        const auto assets = engine::models::fish_drama3::load_drama3_assets(argv[1]);
        core::ExecutionContext execution(config);
        codecs::FishDacCodecRuntimeOptions options;
        options.weight_context_bytes = 16 * 1024 * 1024;
        const auto component = codecs::FishDacCodecComponent::load_from_tensor_source(
            assets->codec_weights, assets->codec, {}, execution.backend(), config.type, options);
        codecs::FishDacCodecRuntime codec(component, config, config.threads, options.graph_arena_bytes);
        assets->codec_weights->release_storage();
        assets->model_weights->release_storage();
        std::cout << "probe.load_ms=" <<
            std::chrono::duration<double, std::milli>(Clock::now() - started).count() << '\n';
        for (int iteration = 0; iteration < 3; ++iteration) {
            const auto begun = Clock::now();
            engine::runtime::AudioBuffer audio;
            if (mode == "full") audio = codec.decode_codes(codes);
            else {
                codec.reset_decode_stream();
                for (int64_t frame = 0; frame < codes.frames; ++frame) {
                    codecs::FishDacCodes one;
                    one.codebooks = codes.codebooks;
                    one.frames = 1;
                    for (int64_t book = 0; book < codes.codebooks; ++book)
                        one.codes.push_back(codes.codes[book * codes.frames + frame]);
                    auto chunk = codec.decode_stream(one);
                    if (frame == 0) { audio.sample_rate = chunk.sample_rate; audio.channels = chunk.channels; }
                    audio.samples.insert(audio.samples.end(), chunk.samples.begin(), chunk.samples.end());
                }
            }
            const auto milliseconds = std::chrono::duration<double, std::milli>(Clock::now() - begun).count();
            if (audio.samples.size() != static_cast<size_t>(codes.frames * 2048))
                throw std::runtime_error("unexpected probe audio length");
            for (const auto value : audio.samples)
                if (!std::isfinite(value)) throw std::runtime_error("nonfinite probe waveform");
            const auto stem = output / ("decode-" + std::to_string(iteration));
            engine::audio::write_pcm16_wav(stem.string() + ".wav", audio.sample_rate, audio.channels, audio.samples);
            std::ofstream raw(stem.string() + ".f32", std::ios::binary);
            raw.write(reinterpret_cast<const char *>(audio.samples.data()),
                      static_cast<std::streamsize>(audio.samples.size() * sizeof(float)));
            std::cout << "probe.iteration=" << iteration << " ms=" << milliseconds
                      << " frames=" << codes.frames << " samples=" << audio.samples.size() << std::endl;
        }
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
