# Linux 原生安装

`linux-preview` 提供从源码安装的 NVIDIA 预览版。原始 PR 在 Ubuntu 26.04 和 RTX 5070 上验证了启动和 GGUF 生成。[验证记录](https://github.com/nkxx188/Drama3-Studio/pull/6)

## 环境要求

- Linux x86_64、支持 AVX2 的处理器、NVIDIA 显卡和与 CUDA 13 兼容的驱动。驱动版本以 [NVIDIA 兼容性说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)为准。
- Python 3.12 和 `venv` 支持。发行版仓库没有 3.12 时，可用 `uv python install 3.12`，然后确认 `python3.12 --version` 在当前终端可用。[uv 文档](https://docs.astral.sh/uv/guides/install-python/)
- Git、CMake、Ninja、C++17 编译器和 CUDA Toolkit 13.x。以下以 CUDA 13.2 的安装目录为例。构建完成后，Studio 用生成目录里的 CUDA 动态库和系统 NVIDIA 驱动运行。
- FFmpeg、libsndfile 和用于系统 C++/OpenMP 运行的共享库。在 Debian 或 Ubuntu 可从软件包管理器安装。

```bash
sudo apt install git build-essential cmake ninja-build ffmpeg libsndfile1 libgomp1
```

## 安装和启动

```bash
git clone --branch linux-preview https://github.com/nkxx188/Drama3-Studio.git
cd Drama3-Studio
bash tools/setup-linux.sh
.venv-linux/bin/python tools/build_linux_runtime.py --cuda-toolkit /usr/local/cuda-13.2
./start.sh doctor
./start.sh
```

替换 `--cuda-toolkit` 后面的路径，使它指向本机 CUDA Toolkit 根目录，其中应有 `bin/nvcc`。编译器自动选择失败时，用 `--cuda-host-compiler` 指定与该 CUDA 版本兼容的 C++ 编译器。服务器上可用 `./start.sh --no-browser`，在启动终端查看实际地址和端口。

## 模型和数据

- 在 Studio 的设置中下载 GGUF，或把已有 GGUF 放到 `checkpoints/` 后选择。已有的 Windows 模型权重也可以放到这里，再在界面中选择一次。
- 本地转写模型放在 `checkpoints/Qwen3-ASR-0.6B-hf/`，具体要求见[参考音频与转写](../README.zh-CN.md#参考音频与转写)。
- `ui_data/` 保存音色、草稿和历史。Windows 历史里的绝对路径依赖原系统目录，从 Windows 换到 Linux 时，可在原环境导出要带走的成品音频。

## 构建和排查

- `.venv-linux/` 保存 Linux Python 依赖。
- `runtime/native-linux/` 保存原生 worker、CUDA 动态库和 `manifest-linux.json`。更换系统时，请考虑目标发行版的 glibc 版本，并在目标环境或兼容的构建机上构建。
- `research/phase37-linux-build.log` 保存编译输出。启动或 CUDA 检查失败时，先查看 `./start.sh doctor` 的结果和 `diagnostics/` 下的日志。
- 需要容器时，按 [Docker 部署指南](linux-docker.md)构建对应的 CUDA 12 worker。
