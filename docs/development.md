# 开发与构建

本页面向需要修改源码或重新构建一键包的开发者。普通使用者请使用完整的 Windows NVIDIA 一键包。

## 目录

| 目录 / 文件 | 内容 |
| --- | --- |
| `src/drama_low_vram/` | Python 推理、内存管理和工作流 |
| `ui/` | 本地 HTTP 服务及前端资源 |
| `ui/editor-source/` | CodeMirror 剧本编辑器源代码 |
| `native/` | 原生 worker、可选 CPU 模块及 audio.cpp 补丁 |
| `third_party/` | Fish Speech 的固定版本和本地补丁 |
| `tools/` | 启动、构建、打包和开发检查脚本 |
| `tests/` | 单元与集成测试 |
| `requirements-windows.lock.txt` | 已验证 Windows Python 环境的依赖版本 |

`vendor`、`runtime`、`checkpoints`、`dist`、缓存、研究输出和用户数据不提交到 Git。`src/drama_low_vram/data/control-head.safetensors` 是下载器需要的 4 KiB 配套数据，作为特例保留，并由 CI 校验 SHA256；完整模型权重仍排除。

## 准备源码环境

需要 Git、Windows x64 Python 3.12 和可用的 NVIDIA 驱动。以下命令在仓库根目录执行：

```powershell
git clone https://github.com/nkxx188/Drama3-Studio.git
cd Drama3-Studio
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-windows.lock.txt
.\.venv\Scripts\python.exe tools\prepare_sources.py
```

`prepare_sources.py` 恢复指定提交的 Fish Speech，并应用仓库中校验过的补丁。它不会下载模型、升级上游版本或重置已有修改；已应用过的补丁不会重复应用。发生版本或补丁冲突时会停止并说明原因。

模型放到本地 `checkpoints` 目录，或启动后在设置中添加其他模型位置。源码目录和一键包的模型识别方式相同。

完成 Python 环境和模型准备后，可运行：

```powershell
.\start.bat
```

使用原生后端的模型还需要先完成下面的原生运行库构建。源码 ZIP 不包含已编译 worker；Python 后端也不会替代该 worker 的职责。

## 构建前端编辑器

已提交的 `ui/script-editor.js` 可直接运行。只有修改编辑器源代码时才需要 Node.js 和 npm：

```powershell
cd ui\editor-source
npm ci --include=optional --ignore-scripts
npm run build
cd ..\..
```

前端其余资源由本地服务直接提供，不依赖 CDN。重新生成的编辑器文件需要和源代码一同提交。

## 构建 Windows NVIDIA 原生运行库

另外需要：

- Visual Studio 2022 Build Tools 的 x64 C++ 工具链、Windows SDK、CMake 和 Ninja。
- CUDA Toolkit **12.8**。该 SDK 仅在构建机器上需要，最终用户使用打包的运行库。
- 足够的本地磁盘空间。编译过程使用 `.cache` 保存上游源码、构建目录和临时文件。

```powershell
.\.venv\Scripts\python.exe tools\build_windows_runtime.py `
  --cuda-toolkit "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8" `
  --jobs 2
```

脚本按固定提交恢复 audio.cpp，校验并应用 `native/audio-cpp-runtime.patch`，生成 `runtime/native` 和相应 manifest。首次构建需要连接上游仓库。目标已有其他版本或冲突修改时会停止，保留原文件。

可选的 CPU 实验模块单独构建：

```powershell
.\.venv\Scripts\python.exe tools\build_native.py
```

## 打包 Windows 一键包

确认原生运行库已构建，并使用安装了锁定依赖的 Python 执行：

```powershell
.\.venv\Scripts\python.exe tools\package_windows.py --zip
```

默认输出 `dist/Drama3 Windows NVIDIA` 及 ZIP、SHA256 文件。打包脚本从当前 Python 的基础安装和已安装依赖复制运行环境；因此应使用正常的 Windows Python 3.12 安装和上述虚拟环境，不要使用缺少标准库、开发头文件或导入库的裁剪环境。

发布包保留运行 DLL、Python 头文件和 Triton 所需导入库，排除本项目推理不使用的 `torch/lib/*.lib` 开发链接库。该运行环境面向应用推理；需要编译 PyTorch C++ 扩展时请另建开发环境。

在已有完整运行环境的包中更新应用文件：

```powershell
.\.venv\Scripts\python.exe tools\package_windows.py --refresh-app
```

刷新应用保留本地模型及用户数据；创建发布 ZIP 则只打入应用目录和发布文件，不打入 `checkpoints`、草稿、音色、生成结果、诊断记录或外部目录链接。

发布前应在独立解压目录中验证启动、模型选择、克隆、多人对话和音频编辑。源码检查与编译通过不能替代真实模型和硬件上的验证。

## 检查

无需模型或 GPU 的发布基础检查：

```powershell
.\.venv\Scripts\python.exe -m pip install pytest==9.1.1 packaging==24.1
.\.venv\Scripts\python.exe -m pytest `
  tests\test_release_sources.py tests\test_release_archive_parts.py tests\test_ui_assets.py `
  tests\test_windows_package.py tests\test_prepare_sources.py -q
```

安装完整依赖后，可运行应用集成检查：

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\test_ui_server.py tests\test_ui_model_discovery.py `
  tests\test_studio.py tests\test_reference_asr.py `
  tests\test_windows_launcher.py tests\test_windows_runtime.py -q
```

这些检查使用临时目录和模拟工作流；真实生成、CUDA/Triton 内核和音频质量需要单独验证。`tools` 中的历史 benchmark / validation 工具可能还需要本地研究输入，不能替代发布测试。

GitHub Actions 的 `Source checks` 运行源码编译检查、必需配套数据校验、发布基础测试和编辑器重建检查。该工作流不构建 NVIDIA worker，也不宣称完成 GPU 推理验证。

发布文件的生成与独立解压验收见 [发布准备](release-preparation.md)。
