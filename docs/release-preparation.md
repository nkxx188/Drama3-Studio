# 发布准备

本页供维护者整理 GitHub 源码和 Windows 一键包，日常使用见根目录 README。

## 源码检查

确认 `git status` 已包含最新代码、模型下载清单和必需的任务控制配套文件。完整模型、用户数据和运行环境继续由 `.gitignore` 排除。

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\test_release_sources.py tests\test_release_archive_parts.py tests\test_ui_assets.py `
  tests\test_windows_package.py tests\test_prepare_sources.py -q

.\.venv\Scripts\python.exe -m pytest `
  tests\test_ui_server.py tests\test_ui_model_discovery.py `
  tests\test_studio.py tests\test_studio_recovery.py tests\test_reference_asr.py `
  tests\test_windows_launcher.py tests\test_windows_runtime.py `
  tests\test_model_download.py tests\test_model_catalog.py -q
```

在独立检出目录执行前端的 `npm ci --include=optional --ignore-scripts` 和 `npm run build`，确认生成的 `ui/script-editor.js` 与提交版本一致。GitHub Actions 也执行该检查。

## 生成发布文件

在完成原生运行库和 Python 环境构建后，更新已有一键包并创建 ZIP：

```powershell
.\.venv\Scripts\python.exe tools\package_windows.py --refresh-app --zip
```

输出为 `dist/Drama3 Windows NVIDIA.zip` 和 `.zip.sha256`。该命令保留现有包目录中的模型和用户数据，创建的发布 ZIP 会排除这些内容。

GitHub 每个 Release 附件需小于 2 GiB。完整 ZIP 较大时，生成每份最多 1900 MiB 的分卷：

```powershell
.\.venv\Scripts\python.exe tools\split_release_archive.py `
  "dist/Drama3 Windows NVIDIA.zip" --output dist/github-release
```

上传该目录中的所有 `.zip.001`、`.zip.002` 等分卷及 `SHA256SUMS.txt`、`release-parts.json`。所有分卷应来自同一次生成；脚本要求新的输出目录以防旧文件混用。用户将全部分卷放在同一文件夹，用 7-Zip 打开第一卷解压。完整 ZIP 和源码 ZIP 仍可单独用于其他分发渠道。

源码压缩包应从最终 Git 提交或已检查的暂存树生成，避免混入工作目录的模型、缓存和旧产物。完成暂存后可执行：

```powershell
$releaseTree = git write-tree
git archive --format=zip --prefix=Drama3-Studio/ `
  --output=dist/Drama3-Studio-source.zip $releaseTree
(Get-FileHash dist/Drama3-Studio-source.zip -Algorithm SHA256).Hash.ToLowerInvariant() |
  Set-Content dist/Drama3-Studio-source.zip.sha256 -Encoding ascii
```

发布时使用新生成的文件；名称带 `preview` 的历史压缩包不作为当前版本交付。

## 独立验收

使用尚不存在的解压目录，校验 SHA256、ZIP 内容、解压 CRC、独立 Python/CUDA 环境和原生自检：

```powershell
.\.venv\Scripts\python.exe tools\verify_windows_archive.py `
  --extract-to dist/release-acceptance
```

再用解压包启动 WebUI，确认空模型目录可正常打开设置、下载目录正确、首次打开不会加载生成模型，实际生成后可以释放资源并再次生成。克隆、多人对话和音频编辑应使用本地已验证模型分别检查。

验收时的模型和参考音频只作为外部测试输入，不重新加入发布 ZIP。测试结果记录所用模型、硬件和实际执行范围；网络失败或未测的格式保留为未验证状态。

## GitHub

先提交已检查的源码，再推送到项目仓库。当前 `Source checks` 负责源码与编辑器检查；完整 NVIDIA 一键包由维护者构建、验收后上传，不会由现有 CI 自动发布。
