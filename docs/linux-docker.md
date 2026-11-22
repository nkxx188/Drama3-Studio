# Linux 与 Docker 部署

Windows 一键包仍是首选路径。本页给出在 Linux 上用容器运行同一份源码、同一个原生 worker 的方法，适用于服务器、常开机器或需要多人经局域网访问的场景。

应用层没有平台分支：`src/drama_low_vram/`、`ui/` 和 `drama3-mcp/` 与 Windows 包完全相同，差异只在原生 worker 的编译方式和启动入口。

## 与 Windows 包的对应关系

| Windows | Linux |
| --- | --- |
| `start.bat` | `docker/entrypoint.sh` |
| `tools/build_windows_runtime.py` → `runtime/native/drama3_worker.exe` | `tools/build_linux_runtime.py` → `runtime/native/drama3_worker` |
| `runtime/native/manifest.json`（`platform: windows-x86_64`） | 同一文件，`platform: linux-x86_64` |
| 包内自带 CUDA 运行库 DLL | torch 的 `nvidia-*-cu12` 轮子，由入口脚本拼 `LD_LIBRARY_PATH` |

`manifest.json` 的其余字段（`protocol`、`features`、`sha256`）保持一致，应用据此加载 worker，无需区分平台。

## 前置条件

- Linux x86_64 宿主机，已装 NVIDIA 驱动与 `nvidia-container-toolkit`
- Docker 与 Docker Compose v2
- 编译 worker 需要：C++17 编译器、CMake、Ninja、CUDA Toolkit 12.8 或 12.9（仅构建机需要，运行镜像不含 SDK）
- 模型文件：`.gguf` 放在 `checkpoints/`，或启动后在设置里添加目录

## 三步

```bash
git clone https://github.com/nkxx188/Drama3-Studio.git
cd Drama3-Studio

# 1. 还原固定提交的 Fish Speech，并应用仓库内校验过的补丁
python tools/prepare_sources.py

# 2. 编译 Linux 原生 worker -> runtime/native/drama3_worker
python tools/build_linux_runtime.py --cuda-toolkit /usr/local/cuda --jobs 6

# 3. 构建镜像并启动
docker compose -f docker/compose.yaml up -d --build
```

默认监听 `http://<服务器IP>:8879/`。首次加载模型较慢属正常。

`--architectures` 决定编译哪些 SM 目标，默认 `89`（RTX 40 系 / Ada）。需要覆盖多代显卡时按 CMake 语法传入，例如 `--architectures "75;86;89;120"`；目标越多编译越久。

## 数据放在哪里

| 宿主机 | 容器 | 说明 |
| --- | --- | --- |
| `checkpoints/` | `/app/checkpoints` | 模型权重，只读挂载 |
| `ui_data/` | `/app/ui_data` | **状态根**：`jobs/`、`uploads/`、`voices/`、`transcripts/`、`settings.json` |
| `diagnostics/` | `/app/diagnostics` | 启动诊断 |

`ui_data` 不挂载等于不保存：生成结果落在 `ui_data/jobs/<job_id>/result.wav`，容器重建即丢失。

## 已知限制

**基础镜像的 glibc 不能比构建机旧。** worker 与宿主 glibc 动态链接，在 Ubuntu 26.04（glibc 2.43）上编译的产物需要 26.04 或更新的运行镜像，因此 `RUNTIME_IMAGE` 默认 `ubuntu:26.04`；在 Ubuntu 24.04（glibc 2.39）上编译的产物两者都能跑，可改成 `ubuntu:24.04`。

```bash
docker build --build-arg RUNTIME_IMAGE=ubuntu:24.04 -f docker/Dockerfile .
```

**glibc 2.43 与 nvcc 的头文件冲突。** 该版本按 C23 给 `rsqrtf` 等数学函数加了 `noexcept(true)`，与 nvcc 的 `__device__` 重声明冲突，device 编译阶段直接报异常规格不匹配。`build_linux_runtime.py` 检测到该守卫时，会把 `bits/mathcalls.h` 复制到 `.cache/sysinclude/` 并把对应 `#if` 改成 `#if 0`，再通过 `CMAKE_CUDA_FLAGS` 注入；系统头文件不动，旧 glibc 上该分支自动跳过。

**GPU 独占。** 24 GB 级 GGUF 模型常驻显存约 5~10 GB，峰值宿主内存较高。与 ComfyUI 一类常驻工作流同机时，先停一边再跑另一边，否则容易在对方跑批时爆显存。

**镜像体积。** torch 的 cu12 轮子约 4 GB，成镜像约 13 GB。worker 已静态链接 CUDA 运行时，镜像内不需要 toolkit。

## 验证记录

在 Ubuntu 26.04 + RTX 4090（驱动 580 系）+ CUDA 12.9.1 + GCC 13.4 上实测：

| 项目 | 结果 |
| --- | --- |
| worker `--probe` | `runtime_version 12090`，设备识别正常 |
| 首次生成（冷启） | 21.1 s → 4.74 s 音频 |
| 热态生成 | 8.22 s |
| 音色克隆 | 8.2 s → 6.08 s 音频 |
| ASR 转写 | 加载 1.30 s / 推理 0.69 s / 显存 1.56 GB |
| 前端资源 | 20/20 返回 200，7 个视图无 JS 报错 |
| MCP（`drama3-mcp/`） | 8/8 工具可用，端到端出声 6.46 s / 44.1 kHz |

`SM89` 之外的架构未在实机上验证；`manifest.json` 里的 `scope` 字段已注明这是"编译目标"而非"逐卡实测"。

## 排查

| 现象 | 处理 |
| --- | --- |
| 启动即报 `libcublas.so.12: cannot open shared object file` | 入口脚本未生效或 torch 未装进 `/opt/venv`；`docker compose logs` 看 `[drama3]` 前缀行 |
| `worker exited` / `protocol mismatch` | worker 与 `manifest.json` 不匹配，重跑 `tools/build_linux_runtime.py` |
| 页面报 `crypto.randomUUID is not a function` | `ui/index.html` 的安全上下文兜底被覆盖；确认镜像用的是仓库版本 |
| 生成时 OOM | 先停掉同机其他 GPU 任务；必要时降低 `checkpoints` 里模型的量化档位 |
| `nvidia-smi` 在容器内不存在 | compose 的 `deploy.resources.reservations.devices` 未生效，检查 toolkit 安装 |
