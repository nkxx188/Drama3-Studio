# Drama 3 Studio MCP

[English](#english) | [简体中文](#简体中文)

## English

This is an optional local companion for Drama 3 Studio. It uses the same running Studio instance and selected speech model as the WebUI. Your existing Windows portable package does not need to be modified, and the adapter does not load another model.

**Writing for Drama 3:** The server instructions and speech tools point the AI to `get_script_guide` for requests with a scene, emotion, or performance direction. The [guide](script-prompt.zh-CN.md) covers natural-language tags, paired tags, the Fish Audio S2 tag reference, and Studio's `S1/S2` format. The AI uses it to prepare the tool inputs, not as text for the speech model. Verbatim requests and prepared scripts keep their wording and tags unless the user asks for changes.

1. Start Studio with `start.bat` and leave it running. Select a ready speech model for generation or editing; standalone transcription only needs the local ASR model.
2. Have [uv](https://docs.astral.sh/uv/getting-started/installation/) available on the MCP client's machine. The first launch downloads the adapter's small Python dependencies into a separate environment.
3. Add a **local stdio** server to your MCP client. Set its command to `uv` and use the example below, replacing the two paths with the actual adapter script and Studio folder. If your client cannot find `uv`, use its absolute executable path.

```json
{
  "mcpServers": {
    "drama3": {
      "command": "uv",
      "args": [
        "run", "--no-project", "--script", "D:/path/to/drama3-mcp/server.py",
        "--studio-dir", "D:/path/to/Drama3 Windows NVIDIA"
      ]
    }
  }
}
```

For the Codex CLI, the equivalent is:

```powershell
codex mcp add drama3 -- uv run --no-project --script "D:/path/to/drama3-mcp/server.py" --studio-dir "D:/path/to/Drama3 Windows NVIDIA"
```

`--studio-dir` reads the launcher's current port and verifies that the server belongs to that installation. For a manually started server, replace `--studio-dir` and its value with `--studio-url http://127.0.0.1:8879`, using the address shown in Studio. The adapter accepts only local addresses and does not open an MCP network port.

| Tool | Purpose |
| --- | --- |
| `get_script_guide` | Read the complete Drama 3 script and performance guide. Does not require a running Studio. |
| `list_voices` | Get the names and IDs of saved voices. |
| `submit_speech` | Generate speech from `text` and optional `direction`. Use a saved `voice_id`, a local `reference_audio`, or neither. |
| `submit_voice_design` | Generate an audition using `voice_description`, `text`, and optional `direction`. No reference audio or automatic voice-library entry. |
| `submit_dialogue` | Generate a script with `S1:`, `S2:`, etc. Bind every speaker to a saved voice or local reference. Choose `joint` (default) or `per_line`. Both return a combined WAV. |
| `submit_audio_edit` | Edit a single-speaker `audio_path` or completed `source_job_id`. Give both time bounds for a local replacement, or omit both for a whole-recording edit. |
| `transcribe_audio` | Transcribe a local `audio_path`, or just a selection with `start_seconds` and `end_seconds`. |
| `get_job` | Check a job. Returns `wav_path` for generated audio or `text` for standalone transcription. |

Use `{"audio_path": "D:/audio/reference.wav"}` for a local `reference_audio`, or as a speaker's value in the dialogue `voices` map. An optional `transcript` skips ASR for that reference. Files are sent only to the local Studio, and uploads do not create voice-library entries.

For a timed edit, try: **“In `D:/audio/take.wav`, replace 8.2 to 10.6 seconds with ‘See you tomorrow.’”** Only the selected clip is cropped for recognition; Studio edits the full source and saves a new WAV. For a whole edit, provide `direction` to change delivery or `text` with the full new transcript. To edit a multi-speaker take with its role bindings, use the WebUI dialogue editor.

Automatic transcription uses Studio's installed local ASR. An edit can use an existing transcript or transcribe the recording; it also transcribes the selection when `original_excerpt` is omitted. Ambiguous matches return `needs_input`. A verified `original_transcript`, `original_excerpt`, and, for a repeated phrase, its 1-based `occurrence` can be supplied instead. Manual excerpts also allow selections shorter than the ASR minimum of 0.5 seconds. Studio limits imported recordings for editing to 120 seconds. Replacement speech may have a different duration than the original selection.

Standalone transcription runs locally without changing the source file. The ASR input (whole file or selection) must be 0.5 to 120 seconds long. For a recording longer than 120 seconds, use a WAV source and specify a shorter selection. Missing ASR weights are reported rather than downloaded. The result includes the recognized text, language, and duration when available.

Use the returned `job_id` with `get_job` regardless of whether the job is `preparing`, `queued`, or `running`. Preparation IDs are local to the current MCP process; for generation or editing, a returned `studio_job_id` also works after reconnecting. Generated WAVs appear in Studio's history.

## 简体中文

这是独立的本地适配器，复用正在运行的 Drama 3 Studio 和 WebUI 当前选中的模型。不用替换一键包文件，不会启动第二份推理引擎。

**情绪剧本指南：** 用户提供场景、情绪或表演意图时，MCP 会提示 AI 先调用 `get_script_guide`，按[完整指南](script-prompt.zh-CN.md)组织演绎要求、自然语言标签和成对标签。单人把整体要求放在 `direction`，多人把演绎写在各自的 `S1/S2` 台词内。指南供 AI 编排使用，不作为台词发给 Drama 3；要求照稿念或已给定稿时，保留原文和现有标签。

1. 双击一键包的 `start.bat`，保持 Studio 运行。配音或编辑需要选好生成模型；单独转写只需要已安装的本地 ASR。
2. 在运行 MCP 客户端的电脑上准备 [uv](https://docs.astral.sh/uv/getting-started/installation/)。首次启动会在独立环境安装适配器的 Python 依赖，不影响一键包。之后可复用，无需重新下载模型。
3. 按上方 JSON 示例或 `codex mcp add` 命令，在客户端添加一个 **本地 stdio** MCP 服务。将两个示例路径分别改为实际的 `server.py` 和一键包根目录。若客户端找不到 `uv`，使用 `uv.exe` 的绝对路径。

`--studio-dir` 会读取启动器记录的端口，并确认连的是指定的那一份 Studio。如果是手动启动的服务，可以改用 `--studio-url http://127.0.0.1:8879`，端口以 Studio 实际地址为准。适配器只连接本机，不开放 MCP 网络端口。

| 工具 | 用途 |
| --- | --- |
| `get_script_guide` | 读取完整情绪剧本指南（含 Fish Audio S2 标签表），无需启动 Studio。 |
| `list_voices` | 获取音色库的名称和 ID。 |
| `submit_speech` | 单人配音。支持音色库 `voice_id`、本地参考 `reference_audio` 或无参考生成。演绎要求 `direction` 选填。 |
| `submit_voice_design` | 按声音描述 `voice_description`、试听文本 `text` 和选填的演绎要求 `direction` 生成试听，不用参考音频，不自动加入音色库。 |
| `submit_dialogue` | 生成 `S1：`、`S2：` 格式的多人对话。每个角色可绑定音色库或本地参考，默认联合生成，也可选逐句生成，均返回合成后的整段 WAV。 |
| `submit_audio_edit` | 编辑本地单人录音或已有的单人生成结果。给出起止秒数时局部替换；不指定时间时可整段换词或调整演绎。 |
| `transcribe_audio` | 转写本地 `audio_path`。可选 `start_seconds` 和 `end_seconds`，只识别指定选区。 |
| `get_job` | 按任务 ID 查询准备、生成或转写；生成返回本地 `wav_path`，转写返回 `text`。 |

配置后可以问：「用 `D:/素材/旁白.wav` 的声音读这段稿子」「给 `S1` 和 `S2` 各选一个音色，联合生成这段对话」「把 `D:/素材/录音.wav` 的 8.2 到 10.6 秒换成‘明天见’」。MCP 只把指定文件交给本机 Studio，不会自动加入音色库。参考台词未提供时，会调用 Studio 已安装的本地 ASR；有参考台词就直接用。

局部编辑会用已有原稿，或先转写原录音，再裁出选区识别旧台词。能确定原句才会自动提交；对不上或无法区分重复台词时，返回 `needs_input`。可以补充完整原稿 `original_transcript`、确认过的旧台词 `original_excerpt`，以及重复时的第几处 `occurrence`（从 1 数）。自动裁剪转写需要选区至少 0.5 秒，编辑源音频最长 120 秒。**结果另存为新 WAV，不覆盖原文件；新台词不保证和选区等长。** 需要保留多人角色绑定的音频编辑，目前仍在 WebUI 操作。

单独转写使用本地 ASR，不修改原文件。整段或选区送入 ASR 的长度须在 0.5 到 120 秒之间。超过 120 秒的录音可用 WAV 格式并指定较短选区。缺少模型时会提示安装，不自动下载。完成时返回识别文字，有可用的语言与时长信息时一并返回。

已有配置无需变更，重新连接 MCP 即可加载新工具。带本地参考的生成、编辑和转写可能先返回准备编号，可以用同一个 `job_id` 跟到完成。`prep-` 开头的编号只在当前 MCP 进程有效；生成或编辑提交后，`get_job` 返回的 `studio_job_id` 也可在重连后查询。
