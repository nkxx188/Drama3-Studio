# Drama 3 Studio

**English** | [简体中文](README.zh-CN.md)

**Run Fish Audio Drama 3 locally on Windows for speech generation, voice cloning, multi-speaker dialogue, and audio editing.**

Built with Fish Audio.

Drama 3 Studio brings model inference, voice references, script editing, and generated audio into one WebUI. The Windows NVIDIA portable package includes its own runtime, so normal use requires no separate Python, Node.js, or CUDA Toolkit setup. Model weights are downloaded separately.

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

1. Download the [Windows NVIDIA portable package from Releases](https://github.com/nkxx188/Drama3-Studio/releases) and fully extract it to a writable local folder with enough free space. An SSD is recommended.
2. Double-click **`start.bat`** to open Studio in your browser. Keep the launcher window open while using it.
3. Go to **Settings → Download a model** and choose a model and source. Once downloaded, select it under **Available models** and click **Use this model**. If you already have a complete model, place it under `checkpoints` or add its existing folder.
4. Select a model at the top of the page, upload reference audio or choose a voice from the library, enter your text, and generate.

The default address is `http://127.0.0.1:8879/`. If the port is occupied, the launcher automatically chooses another local port. Running the launcher again opens the existing instance of this installation.

For split archives named `.zip.001`, `.zip.002`, and so on, download **all volumes** into the same folder, then open `.zip.001` with **7-Zip** and extract the package. A complete, single-file ZIP can be extracted directly.

### Download Models

Settings offers FP8 / BF16, INT4, INT8 ConvRot, W6A8, W4A8, and several GGUF variants. The downloader prepares the supporting files required by each variant and installs them under the portable package's `checkpoints` folder.

- FP8 / BF16 and INT4 offer both Hugging Face and ModelScope sources, with an automatic speed-test option.
- Other listed quantized variants currently offer Hugging Face only. Make sure your network or proxy can access the selected source. The UI shows which sources are available for each model.
- The download panel shows connection, download, and verification status, along with progress, size, and speed. After an interruption or cancellation, choose the same model again to resume using previously completed chunks.

When downloading manually or using models from another tool, preserve the directory structure and supporting files required by that variant. Whether a single model file works directly depends on its format and the availability of its supporting files.

### Model Locations

The default `checkpoints` folder is the only model location you need. Studio scans its subfolders automatically; you do not need to add each model directory individually.

```text
Drama3 Studio/
├─ start.bat
├─ checkpoints/
│  ├─ model-a/
│  └─ model-b/
├─ runtime/
└─ ...
```

If you already keep models elsewhere, add that folder under **Settings → Model locations**. Models from additional locations and the default folder appear in one shared list. Existing model files are not copied or moved. Check each model's instructions for its required supporting files.

Supported formats include the original FP8 / BF16 model, INT4 / W4A16, INT8, W6A8, W4A8, and compatible GGUF models. Format, quantization, and system memory configuration affect speed, resource use, and generated audio.

### Requirements

- Windows x64, an NVIDIA GPU, an AVX2-capable CPU, and a compatible NVIDIA driver.
- The portable package includes Python, the required Python dependencies, and CUDA runtime libraries.
- The native backend includes compilation targets for RTX 20 / 30 / 40 / 50 series GPUs. Full hardware validation was performed on an **RTX 4070 Laptop GPU with 8 GB VRAM**; other hardware still needs validation.
- RAM and storage requirements depend on the model. Low-VRAM operation keeps some model data in system RAM or memory-mapped files, so VRAM capacity alone does not determine the system requirements.

### Reference Audio and Transcription

Use reference audio with clear speech and a transcript that matches what is actually said. Automatic transcription uses a separate Qwen3-ASR-0.6B model. Without it, you can enter the transcript manually and still use reference audio normally.

The default transcription model location is `checkpoints/Qwen3-ASR-0.6B-hf`. You can also set `DRAMA_ASR_MODEL_DIR` to a local model in the native Transformers format. ASR models do not appear in the speech generation model list.

### First Generation and Releasing Resources

Opening the page or restoring a draft does not load the generation model. The model loads and prepares the voice reference when you first generate, so the first task usually takes longer. While the model remains loaded, Studio reuses its caches and can automatically warm up new references.

When you are done generating for a while, click the plug icon to the left of Refresh in the top-right corner. Its tooltip reads **Release resources**. Studio cancels warm-up, unloads the generation model, and clears its caches while preserving model files, voices, drafts, and history. The next generation loads the model again. Finish or cancel any running generation task before releasing resources.

### Updates and Troubleshooting

Models, voices, drafts, generation history, and caches are stored locally. Before updating, back up your `checkpoints`, `ui_data`, and `outputs` folders, then follow the release instructions to update the application files.

The launcher window shows service and task status; detailed logs are stored under `diagnostics`. For troubleshooting, export diagnostics from Settings or run `start.bat --diagnostics` in a terminal.

## Run from Source and Build

The repository contains application source, frontend assets, dependency versions, upstream patches, and build scripts. It **does not include complete model weights, a Python runtime, CUDA binaries, or user audio**. A roughly 4 KiB task-control output row is retained as a supporting file for some quantized variants; see [NOTICE](NOTICE) for its source and license.

See the [development and build guide (Chinese)](docs/development.md) for source setup and packaging instructions. GitHub's **Source code** ZIP is not the complete, ready-to-run portable package.

## Project and Author

Created by **nkxx188 / 那颗星星188**. If you find Studio useful, follow the author for updates.

- [Bilibili · Follow the author](https://b23.tv/86MY5Ia)
- [Fish Audio · Official service](https://fish.audio/?fpr=nkxx188)
- [YouTube · Follow the author](https://www.youtube.com/@nkxx-188)
- [GitHub · Source and feedback](https://github.com/nkxx188/Drama3-Studio)

## Credits and Licenses

Thanks to [Fish Audio / Fish Speech](https://github.com/fishaudio/fish-speech), [audio.cpp](https://github.com/0xShug0/audio.cpp), [ggml](https://github.com/ggml-org/ggml), CodeMirror, Lucide, and the other dependencies that make this project possible.

Original material owned by this project's author is licensed under the [MIT License](LICENSE). This does not cover third-party or adapted code. For example, [codec.py](src/drama_low_vram/codec.py) is adapted from Fish Speech and remains subject to the Fish Audio Community License. Native modifications based on audio.cpp retain their Apache-2.0 license obligations.

Other third-party code, icons, and runtime libraries retain their respective licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [NOTICE](NOTICE). Complete model weights are downloaded separately. Read each model's license before using it.
