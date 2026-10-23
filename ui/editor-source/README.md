# Local Script Editor

The shipped `ui/script-editor.js` is a self-contained CodeMirror 6 editor with
Lucide tool icons. It needs no Node.js, CDN, remote font or network service at
runtime. The underlying script stays plain Drama 3 text; capsule decorations
are not sent as HTML to the model.

Developer rebuild:

```powershell
npm ci --include=optional --cache ../../.cache/npm --ignore-scripts
npm run build
```

Versions are pinned in `package-lock.json`. Runtime distributions include the
bundle and `ui/licenses/`, not this development directory or `node_modules`.
The editor keeps CodeMirror's selection, history, composition and clipboard
machinery rather than implementing its own contenteditable document engine.

`tag-catalog.js` keeps the official English emotion presets and legacy search
aliases. Captions and natural-language cue text use the existing `ui/i18n.js`
catalog; paired scope tags keep their fixed protocol names. Search spans every
category and all four UI translations. Existing script text is never rewritten
when the UI language changes.

Validate the catalog with `node tools/validate_tag_catalog.mjs` from the project
root. Browser selection and responsive tests run with
`python -I tools/validate_studio_ui.py --tag-checks`. Product and source notes
are in [development.md](../../docs/development.md).
