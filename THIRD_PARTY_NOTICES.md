# Third-party components

The root [MIT License](LICENSE) applies only to original project material owned
by the author. It does not relicense the following components or adapted files.

| Component | Source | Included license / notice |
| --- | --- | --- |
| Fish Speech / Drama 3 | <https://github.com/fishaudio/fish-speech> | [Fish Audio Community License](licenses/fish-audio/LICENSE), [NOTICE](NOTICE) |
| audio.cpp | <https://github.com/0xShug0/audio.cpp> | [Apache-2.0 and upstream attribution](licenses/audio-cpp/LICENSE) |
| ggml | <https://github.com/ggml-org/ggml> | [MIT](licenses/ggml/LICENSE) |
| Bilibili site favicon | <https://www.bilibili.com/> | Official favicon used only to identify the Bilibili link; rights to its mark remain with Bilibili. |
| CodeMirror and editor dependencies | <https://codemirror.net/> | [Per-package license files](ui/licenses/) |
| Lucide | <https://lucide.dev/> | [License](ui/licenses/lucide.txt) |

`src/drama_low_vram/codec.py` adapts code from Fish Speech's
`fish_speech/models/text2semantic/inference.py` and remains under the Fish Audio
Community License. `vendor/fish-speech/` is recovered from the exact commit in
`third_party/fish-speech.json`; the local generator and sampler changes are
recorded in `third_party/fish-speech.patch`.

The exact Fish Speech revision and local patch are recorded in
`third_party/fish-speech.json`. The native audio.cpp revision and patch checksum
are recorded in `native/audio-cpp-runtime.json`.

`ui/drama3-fish-mark.svg` and `ui/favicon.svg` use the rounded shark waveform
design supplied by the project author. They are project branding assets and
are not Fish Audio logos.

Windows packages also include Python and installed Python dependencies, CUDA
runtime libraries, and Microsoft runtime libraries. The dependency distributions
retain their own license files; the native runtime retains upstream notices
under `runtime/native/licenses/`. The package manifest records the bundled
dependency versions.

The full model weights are downloaded separately. The package includes only a
4 KiB Drama 3 `<|audio_separator|>` output row under the Fish Audio Community
License, documented in [NOTICE](NOTICE). Consult the license supplied with each
model; source-code notices do not grant additional rights to model weights,
third-party names, or logos.
