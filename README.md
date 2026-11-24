# Drama 3 Studio

**English** | [简体中文](README.zh-CN.md)

**Run Fish Audio Drama 3 locally for speech generation, voice cloning, multi-speaker dialogue, and audio editing.**

Built with Fish Audio.

Drama 3 Studio brings model inference, voice references, script editing, and generated audio into one WebUI. Model weights are downloaded separately. This branch is a community Linux preview; Windows users can follow the [main branch](https://github.com/nkxx188/Drama3-Studio/tree/main).

## Linux NVIDIA Preview

### Native Installation

Requires Linux x86_64, an NVIDIA GPU and driver, AVX2, Python 3.12, CMake, Ninja, a C++ compiler, and CUDA Toolkit 13.x. Install the system packages in the [native Linux guide](docs/linux.md), then run from the repository root:

```bash
sudo apt install git build-essential cmake ninja-build ffmpeg libsndfile1 libgomp1
bash tools/setup-linux.sh
.venv-linux/bin/python tools/build_linux_runtime.py --cuda-toolkit /usr/local/cuda-13.2
./start.sh
```

Change the CUDA Toolkit path to the location containing `bin/nvcc`. The worker and CUDA libraries are kept in `runtime/native-linux/`. The PR author tested native startup and GGUF speech generation on Ubuntu 26.04 with an RTX 5070. Other GPUs and workflows still need validation.

### Docker Deployment

Use this route to manage Studio as a Compose service. Build the CUDA 12 worker on the Linux host before building the image. The PR author reported testing generation, cloning, ASR, dialogue, and MCP on Ubuntu 26.04 with an RTX 4090; the Dockerfile still needs a clean build and end-to-end verification. See the [Docker deployment guide](docs/linux-docker.md).

## Features

- **Speech generation and cloning:** upload reference audio, enter or edit its transcript, and generate new lines; you can also choose a saved voice from the voice library.
- **Multi-speaker dialogue:** assign voices to speakers and write dialogue with speaker markers. Generate an entire conversation jointly, or generate individual turns and combine them.
- **Voice design:** describe the voice you want in natural language, then use the generated audio as a voice reference.
- **Audio editing:** upload a recording for automatic transcription, then revise the whole take or a selected passage on the same page. Audio preparation runs when you generate. Existing generation results retain their direct editing workflow.
- **Script controls:** insert delivery tags, pauses, and speaker markers, and add natural-language performance instructions.
- **Local management:** save voices, browse generation history, play and download results, and add model folders from other locations.
- **Model downloads:** choose a model and download source in Settings, with parallel downloads, resume support, and integrity checks.
- **Optional reference transcription:** automatically transcribe reference audio when a local ASR model is available. Transcripts remain editable and can also be entered manually.
- **Multilingual interface:** Simplified Chinese, English, Japanese, and Korean.

This is a third-party local application built around Fish Audio Drama 3. Speech capabilities and model weights are provided by the upstream model.

## Windows Portable Package

See the download and startup instructions on the main branch. [Windows portable package](https://github.com/nkxx188/Drama3-Studio/blob/main/README.md#windows-portable-package)

## Linux Model Setup and Operation

### Download and Select a Model

Open **Settings → Download a model** to download a GGUF model, or place an existing one under `checkpoints/` with its required supporting files. Refresh the model list and select the model you want to use.

### Reference Audio and Transcription

Use reference audio with clear speech and a transcript that matches what is actually said. Automatic transcription uses a separate Qwen3-ASR-0.6B model. Without it, you can enter the transcript manually and still use reference audio normally.

The default transcription model location is `checkpoints/Qwen3-ASR-0.6B-hf`. You can also set `DRAMA_ASR_MODEL_DIR` to a local model in the native Transformers format. ASR models do not appear in the speech generation model list.

### First Generation and Releasing Resources

Opening the page or restoring a draft does not load the generation model. The model loads and prepares the voice reference when you first generate, so the first task usually takes longer. While the model remains loaded, Studio reuses its caches and can automatically warm up new references.

When you are done generating for a while, click the plug icon to the left of Refresh in the top-right corner. Its tooltip reads **Release resources**. Studio cancels warm-up, unloads the generation model, and clears its caches while preserving model files, voices, drafts, and history. The next generation loads the model again. Finish or cancel any running generation task before releasing resources.

### Updates and Troubleshooting

Back up `checkpoints/`, `ui_data/`, and `outputs/` before updating. After pulling the latest `linux-preview` changes, follow the installation guide for your setup to update dependencies and rebuild the worker.

The launcher shows service and task status, with detailed logs under `diagnostics/`. Export diagnostics from Settings or run `./start.sh doctor` for the native installation.

## Run from Source and Build

The repository contains application source, frontend assets, dependency versions, upstream patches, and build scripts. It **does not include complete model weights, a Python runtime, CUDA binaries, or user audio**. A roughly 4 KiB task-control output row is retained as a supporting file for some quantized variants; see [NOTICE](NOTICE) for its source and license.

See the [development and build guide (Chinese)](docs/development.md) for source setup and packaging instructions. GitHub's **Source code** ZIP is not the complete, ready-to-run portable package.

## Project and Author

Created by **nkxx188 / 那颗星星188**. If you find Studio useful, follow the author for updates.

- [Bilibili · Follow the author](https://b23.tv/86MY5Ia)
- [Fish Audio · Official service](https://fish.audio/?fpr=nkxx188)
- [YouTube · Follow the author](https://www.youtube.com/@nkxx-188)
- [GitHub · Source and feedback](https://github.com/nkxx188/Drama3-Studio)

## Linux Contributions

- `@ghdt5tdd` contributed Docker Compose deployment and the CUDA 12 worker build for the container. [PR #4](https://github.com/nkxx188/Drama3-Studio/pull/4)
- `@helloHub1212` contributed the native Linux setup, launcher, and CUDA worker build. [PR #6](https://github.com/nkxx188/Drama3-Studio/pull/6)

## Credits and Licenses

Thanks to [Fish Audio / Fish Speech](https://github.com/fishaudio/fish-speech), [audio.cpp](https://github.com/0xShug0/audio.cpp), [ggml](https://github.com/ggml-org/ggml), CodeMirror, Lucide, and the other dependencies that make this project possible.

Original material owned by this project's author is licensed under the [MIT License](LICENSE). This does not cover third-party or adapted code. For example, [codec.py](src/drama_low_vram/codec.py) is adapted from Fish Speech and remains subject to the Fish Audio Community License. Native modifications based on audio.cpp retain their Apache-2.0 license obligations.

Other third-party code, icons, and runtime libraries retain their respective licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [NOTICE](NOTICE). Complete model weights are downloaded separately. Read each model's license before using it.
