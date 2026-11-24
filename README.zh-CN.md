# Drama 3 Studio

[English](README.md) | **简体中文**

**在本地使用 Fish Audio Drama 3，完成语音生成、声音克隆、多人对话和音频编辑。**

Built with Fish Audio.

Drama 3 Studio 将模型推理、参考音频、剧本编辑和生成结果整理在同一个 WebUI 中。模型权重单独下载。本分支为社区 Linux 预览版；Windows 用户请查看[主线](https://github.com/nkxx188/Drama3-Studio/tree/main)。

## Linux NVIDIA 预览

### 原生安装

需要 Linux x86_64、NVIDIA 显卡和驱动、AVX2、Python 3.12、CMake、Ninja、C++ 编译器及 CUDA 13.x Toolkit。先按[原生安装指南](docs/linux.md)安装系统依赖，再在仓库根目录运行：

```bash
sudo apt install git build-essential cmake ninja-build ffmpeg libsndfile1 libgomp1
bash tools/setup-linux.sh
.venv-linux/bin/python tools/build_linux_runtime.py --cuda-toolkit /usr/local/cuda-13.2
./start.sh
```

请把 CUDA Toolkit 路径换成包含 `bin/nvcc` 的实际目录。原生 worker 和 CUDA 动态库保存在 `runtime/native-linux/`。PR 作者在 Ubuntu 26.04 和 RTX 5070 上验证了原生启动及 GGUF 语音生成。其他显卡和工作流仍需验证。

### Docker 部署

习惯用 Compose 管理服务时可选此方式。先在宿主机上构建 CUDA 12 推理程序，再构建镜像。PR 作者报告在 Ubuntu 26.04 和 RTX 4090 上测试了生成、克隆、转写、多人对话和 MCP；当前 Dockerfile 仍需从干净环境重新构建并端到端验证。依赖和数据持久化见 [Docker 部署指南](docs/linux-docker.md)。

## 功能

- **语音生成与克隆**：上传参考音频，填写或修改对应文字，生成新的台词；也可从音色库选择已保存的声音。
- **多人对话**：给说话人分配声音，编写带角色标记的对话；支持整段联合生成，以及逐段生成后拼接。
- **声音设计**：用自然语言描述想要的声音，生成后可用作参考音频。
- **音频编辑**：上传已有音频后自动转写，在同一页修改整段表达或重新生成选中片段；点击生成时自动准备原音频。常规生成结果原有的直接编辑流程保持不变。
- **剧本控制**：插入演绎标签、停顿和说话人标记，使用自然语言补充演绎要求。
- **本地管理**：保存音色，查看生成历史，直接播放和下载结果；支持添加其他位置的模型文件夹。
- **模型下载**：在设置中选择模型版本和下载渠道，支持并行下载、断点续传及完整性校验。
- **可选参考音频转写**：检测到本地 ASR 模型时自动转写，文字保持可编辑；也可手动填写。
- **多语言界面**：简体中文、English、日本語、한국어。

这是基于 Fish Audio Drama 3 构建的第三方本地工具。语音能力和模型权重由上游模型提供。

## Windows 一键包

下载和启动步骤见主线的 [Windows 一键包说明](https://github.com/nkxx188/Drama3-Studio/blob/main/README.zh-CN.md#windows-一键包)。

## Linux 模型设置与使用

### 下载并选择模型

打开 **设置 → 下载模型** 下载 GGUF。也可以把已有的 GGUF 模型及其配套文件放到 `checkpoints/`。刷新模型列表并选择要用的模型。

### 参考音频与转写

参考音频应包含清晰的人声，转写文字应与音频实际内容一致。自动转写使用单独的 Qwen3-ASR-0.6B 模型；未安装时仍可手动填写，不影响使用参考音频。

转写模型的默认位置为 `checkpoints/Qwen3-ASR-0.6B-hf`，也可通过 `DRAMA_ASR_MODEL_DIR` 指定本地原生 Transformers 格式模型。ASR 模型不属于语音生成模型列表。

### 首次生成与释放资源

打开页面或恢复草稿不会提前加载生成模型。首次点击生成时，程序才会加载模型并准备参考音频，因此首次任务的等待时间通常更长。模型驻留期间会继续复用缓存，更换参考音频时可自动预热。

暂时不再生成时，可点击右上角刷新按钮左侧的插头图标，悬停提示为 **“释放资源”**。程序会取消预热、卸载生成模型并清理其缓存，保留模型文件、音色、草稿和历史记录；下次生成会重新加载。生成任务进行中需要先完成或取消任务。

### 更新与排查

更新前备份 `checkpoints/`、`ui_data/` 和 `outputs/`。拉取 `linux-preview` 的最新代码后，按所选安装方式的指南更新依赖并重新构建推理程序。

启动窗口显示服务和任务状态，详细日志保存在 `diagnostics/`。可在设置页导出诊断日志，原生安装也可在终端运行 `./start.sh doctor` 排查问题。

## 从源码运行与构建

GitHub 仓库提供程序源码、前端资源、依赖版本、上游补丁及构建脚本，**不包含完整模型权重、Python 运行环境、CUDA 二进制库或用户音频**。仓库保留一个约 4 KiB 的任务控制输出行，供部分量化版本配套使用，其来源和许可见 [NOTICE](NOTICE)。

源码运行与打包步骤见 [开发与构建](docs/development.md)。仅下载 GitHub 的 Source code ZIP 并不能直接获得完整的一键包。

## 项目与作者

作者：**那颗星星188**。如果这个工具对你有帮助，欢迎关注，了解后续更新。

- [B站 · 关注作者](https://b23.tv/86MY5Ia)
- [Fish Audio 官方服务](https://fish.audio/?fpr=nkxx188)
- [YouTube · 关注作者](https://www.youtube.com/@nkxx-188)
- [GitHub · 源码与反馈](https://github.com/nkxx188/Drama3-Studio)

## Linux 适配贡献

- `@ghdt5tdd` 提供了 Docker Compose 部署和容器所用的 CUDA 12 推理程序构建流程。[PR #4](https://github.com/nkxx188/Drama3-Studio/pull/4)
- `@helloHub1212` 提供了 Linux 原生安装、启动脚本和 CUDA 推理程序构建流程。[PR #6](https://github.com/nkxx188/Drama3-Studio/pull/6)

## 上游与许可

感谢 [Fish Audio / Fish Speech](https://github.com/fishaudio/fish-speech)、[audio.cpp](https://github.com/0xShug0/audio.cpp)、[ggml](https://github.com/ggml-org/ggml)、CodeMirror、Lucide 及其他依赖项目。

本项目作者原创并拥有版权的部分使用 [MIT License](LICENSE)。该许可不覆盖第三方或改编代码：例如，[codec.py](src/drama_low_vram/codec.py) 改编自 Fish Speech，仍受 Fish Audio Community License 约束；基于 audio.cpp 的原生修改继续遵循其 Apache-2.0 许可义务。

其他第三方代码、图标和运行库遵循各自许可，详见 [第三方说明](THIRD_PARTY_NOTICES.md) 和 [NOTICE](NOTICE)。完整模型权重单独下载；使用模型前请阅读对应模型附带的许可。
