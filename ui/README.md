# Studio UI development

The user-facing project introduction and installation instructions are in the
[root README](../README.md). Source setup and native builds are documented in
[development.md](../docs/development.md).

`server.py` serves the local UI and serializes work through the existing
inference engine. It binds to localhost by default. `start.bat` selects the
project or bundled Python and opens the matching instance.

The UI has separate generation, dialogue, voice-library, history and settings
views. Voice design and audio editing are distinct creation workspaces, with
independent result state. Drafts, saved voices and generated files live under
`ui_data` and are excluded from Git and release archives.

Plain JavaScript, CSS, icons, translations and the bundled editor are served
locally; daily use requires no Node.js or remote frontend service.

The CodeMirror editor source is in `editor-source`. Rebuild after changing it:

```powershell
cd ui\editor-source
npm ci --ignore-scripts
npm run build
```

Commit the resulting `ui/script-editor.js` together with its source. Preserve
the license files in `ui/licenses`. Do not commit `node_modules`, local caches,
model files, user audio, drafts or service logs.

Tests in `tests/test_ui_assets.py` cover the shipped UI structure. Server,
model-discovery, reference-transcription and studio tests exercise application
behavior with temporary data. Those checks do not establish voice similarity
or replace real-model generation tests.
