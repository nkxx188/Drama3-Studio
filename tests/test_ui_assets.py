"""Static UI asset contract for the first Windows workbench."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "ui/index.html").read_text(encoding="utf-8")
JS = (ROOT / "ui/app.js").read_text(encoding="utf-8")
CSS = (ROOT / "ui/app.css").read_text(encoding="utf-8")


def test_generate_page_keeps_required_internal_workflow_modes():
    for mode in ("tts", "revision", "edit", "infill", "audio_stream"):
        assert f'value="{mode}"' in HTML
    for label in ("Dialogue", "Voices", "History", "Settings"):
        assert label in HTML


def test_workspace_navigation_keeps_history_adjacent_to_settings():
    nav = HTML[HTML.index('<nav class="workspace-nav"'):HTML.index("</nav>", HTML.index('<nav class="workspace-nav"'))]
    assert nav.index('data-view="takes"') < nav.index('data-view="settings"')
    assert 'data-view="docs"' not in HTML
    assert 'id="view-docs"' not in HTML
    assert 'src="/assets/guide.js"' not in HTML
    assert not (ROOT / "ui/guide.js").exists()


def test_settings_start_with_author_links_and_keep_required_attribution():
    studio = (ROOT / "ui/studio.js").read_text(encoding="utf-8")
    about = HTML[HTML.index('id="studio-about"'):HTML.index("</section>", HTML.index('id="studio-about"'))]
    assert HTML.index('id="studio-about"') < HTML.index('class="panel model-panel"')
    assert "runtime.after($('studio-about'))" not in studio
    assert "$('view-settings').prepend($('studio-about'))" in studio
    for url in (
        "https://b23.tv/86MY5Ia",
        "https://github.com/nkxx188/Drama3-Studio",
        "https://fish.audio/?fpr=nkxx188",
        "https://www.youtube.com/@nkxx-188",
    ):
        assert url in about
    for icon_name in ("i-github", "i-youtube"):
        assert f'href="#{icon_name}"' in about
    assert 'src="/assets/bilibili-logo.ico"' in about
    assert 'src="/assets/drama3-fish-mark.svg"' in about
    assert 'src="/assets/drama3-fish-mark.svg"' in HTML
    assert (ROOT / "ui/drama3-fish-mark.svg").is_file()
    assert not (ROOT / "ui/fish-audio-logo.svg").exists()
    assert about.count("Built with Fish Audio") == 1
    assert '<small class="about-attribution">Built with Fish Audio</small>' in about
    for label in ("About Drama 3 Studio", "Created by", "Source code and releases",
                  "Official Fish Audio service", "Follow the author"):
        assert label in about


def test_project_license_is_standard_mit_with_separate_upstream_notices():
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    third_party = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    normalized_license = " ".join(license_text.split())
    assert normalized_license.startswith("MIT License Copyright (c) 2026 那颗星星188")
    assert 'obtaining a copy of this software and associated documentation files (the "Software")' in normalized_license
    assert "Original Project Materials" not in license_text
    assert "src/drama_low_vram/codec.py" in third_party
    assert "Fish Audio Community License" in third_party
    assert "audio-cpp-runtime.json" in third_party


def test_ui_exposes_explicit_scene_fallback_and_history_edit_controls():
    studio = (ROOT / "ui/studio.js").read_text(encoding="utf-8")
    assert "['joint', 'Joint dialogue']" in studio and "['per_line', 'Per-turn & assemble']" in studio
    assert 'id="previous-take"' in HTML
    assert 'id="source-take"' in HTML
    assert 'id="cancel-job"' in HTML
    assert "useTake" in JS and "take_file" in JS
    assert "download-link" in JS and "download" in JS


def test_ui_assets_are_local_and_nonempty():
    assert "/assets/app.css" in HTML and "/assets/app.js" in HTML
    assert len(CSS) > 4000
    assert len(JS) > 6000


def test_model_settings_keep_local_setup_and_add_model_catalog_downloader():
    server = (ROOT / "ui/server.py").read_text(encoding="utf-8")
    catalog = (ROOT / "src/drama_low_vram/model_catalog.py").read_text(encoding="utf-8")
    i18n = (ROOT / "ui/i18n.js").read_text(encoding="utf-8")
    assert 'id="browse-model-directory"' in HTML and "modelPickerAvailable" in JS
    assert "api('/api/model/browse'" in JS
    assert "api('/api/model/roots/add'" in JS
    assert "api('/api/model/roots/remove'" in JS
    assert 'id="model-root-list"' in HTML and 'id="installed-model"' in HTML
    assert 'id="browse-model-directory"' in HTML and 'aria-label="Add model folder"' in HTML
    assert "const defaultLocation = locations.find(location => !location.removable);" in JS
    assert "...locations.filter(location => location.removable)" in JS
    for marker in ('id="model-directory"', 'id="inspect-model"', 'id="model-active-folder"',
                   'id="model-folder-status"', 'id="model-cache-status"'):
        assert marker not in HTML
    assert "$('browse-model-directory').hidden" not in JS
    assert 'if path == "/api/model/browse":' in server
    assert 'path in ("/api/model/roots/add", "/api/model/roots/remove")' in server
    assert "fallbackModelLocations" not in JS
    assert "filedialog.askdirectory" in server
    assert "GGUF" not in HTML and "GGUF" in i18n
    for marker in ('id="model-download-choice"', 'id="model-download-sources"',
                   'id="download-model"', 'id="model-download-task"',
                   'id="model-download-progress"', 'id="cancel-model-download"'):
        assert marker in HTML
    for marker in ("renderModelTask", "pollModelTask", "modelDownloadActive",
                   "downloadSelectedModel", "/api/model/catalog", "/api/model/tasks"):
        assert marker in JS
    for marker in ('path == "/api/model/catalog"', 'model_id=model_id',
                   'checkpoint_root=ROOT / "checkpoints"'):
        assert marker in server
    assert "download-catalog.json" in catalog
    assert "parallel chunks and SHA256 verification" in i18n


def test_shared_visual_tokens_define_type_boundaries_and_control_geometry():
    studio = (ROOT / "ui/studio.css").read_text(encoding="utf-8")
    for name in ("--type-title", "--type-editor", "--control-height", "--tool-size",
                 "--control-line", "--line-hover", "--focus-ring", "--radius-control"):
        assert name in CSS and f"var({name})" in CSS + studio
    assert ":root {" not in studio
    assert "font-size: var(--type-editor)" in studio
    assert "prefers-reduced-motion" in studio


def test_completed_output_is_quiet_without_hiding_download_or_timing():
    studio = (ROOT / "ui/studio.css").read_text(encoding="utf-8")
    assert 'class="secondary-btn download-link"' in JS
    assert "$('job-badge').closest('.monitor-panel').dataset.state = job.status" in JS
    assert 'monitor-panel:not([data-state="queued"]):not([data-state="running"]) .progress-track' in studio
    assert "generationTimeLabel(job)" in JS


def test_result_no_longer_exposes_diagnostic_or_version_sections():
    studio = (ROOT / "ui/studio.js").read_text(encoding="utf-8")
    for marker in ('id="job-log"', 'id="job-events"', 'id="job-timing-details"'):
        assert marker not in HTML
    for marker in ("candidates.id = 'candidate-section'", "exportButton.id = 'export-work'",
                   "function renderVersions(", "command('Adopt version'"):
        assert marker not in studio


def test_unified_editor_contains_full_scope_without_composer_jumps():
    editor = (ROOT / "ui/audio-editor.js").read_text(encoding="utf-8")
    assert "['whole', 'Whole audio']" in editor
    assert "['local', 'Selected passage']" in editor
    assert "hooks.editScript" not in editor
    assert "['continue', 'Continue audio']" not in editor


def test_tag_palette_preserves_first_open_selection_and_has_no_recent_section():
    editor = (ROOT / "ui/editor-source/editor.js").read_text(encoding="utf-8")
    assert "view.dom.ownerDocument.getSelection()" in editor
    assert "pendingSelection || captureSelection()" in editor
    assert "textarea.selectionStart" in editor
    assert "drama3.recentTags" not in editor and "Recently used" not in editor


def test_tag_palette_is_categorized_localized_and_keyboard_accessible():
    editor = (ROOT / "ui/editor-source/editor.js").read_text(encoding="utf-8")
    assert "./tag-catalog.js" in editor
    assert "window.I18n.translations(caption)" in editor
    assert "window.I18n.locale === 'en' ? value : t(caption)" in editor
    assert "navigation.setAttribute('role', 'tablist')" in editor
    assert "ArrowLeft" in editor and "ArrowRight" in editor
    assert "No matching tags" in editor


def test_tag_palette_categories_share_a_continuous_scroll_list():
    editor = (ROOT / "ui/editor-source/editor.js").read_text(encoding="utf-8")
    assert "tag-category-section" in editor
    assert "query || section.category === category" not in editor
    assert "choices.addEventListener('scroll'" in editor or "choices.onscroll" in editor


def test_regular_text_entry_has_no_placeholder_but_infill_keeps_its_guidance():
    import re
    from html.parser import HTMLParser

    class TextAreaParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.attrs = None

        def handle_starttag(self, tag, attrs):
            if tag == "textarea" and dict(attrs).get("id") == "text":
                self.attrs = dict(attrs)

    parser = TextAreaParser()
    parser.feed(HTML)
    assert parser.attrs is not None
    assert "placeholder" not in parser.attrs
    assert "data-i18n-placeholder" not in parser.attrs
    assert "Write a line, a paragraph, or a full scene fragment…" not in JS
    assert re.search(
        r"const textPlaceholder = mode === 'infill'\s*\?\s*"
        r"'The complete revised transcript for the infilled take…'\s*:\s*'';",
        JS,
    )
    assert "delete $('text').dataset.i18nPlaceholder;" in JS
    assert "$('text').removeAttribute('placeholder');" in JS


def test_library_is_a_compact_grid_with_a_reusable_add_voice_dialog():
    studio = (ROOT / "ui/studio.js").read_text(encoding="utf-8")
    css = (ROOT / "ui/studio.css").read_text(encoding="utf-8")
    assert 'id="model-quality"' not in HTML and "$('model-quality')" not in JS
    assert 'id="add-voice"' in HTML and 'id="voice-library-dialog"' in HTML
    assert "dialog.showModal()" in studio and "dialog.querySelectorAll('audio')" in studio
    assert "voice-tile-preview" in JS and "data-preview-clip" in JS
    assert "StudioAudio.enhance($('voice-list'))" not in JS
    assert ".voice-list { display: grid;" in css
    assert "makeDock('save-voice'" not in studio


def test_homepage_direction_precedes_the_expanding_writing_area():
    studio = (ROOT / "ui/studio.js").read_text(encoding="utf-8")
    css = (ROOT / "ui/studio.css").read_text(encoding="utf-8")
    assert "scriptLabel.before(performance)" in studio
    assert "fitWritingArea(compose, editorArea)" in studio
    assert "editor.top + scrollY" in studio
    assert "ResizeObserver(schedule)" in studio
    assert ".editor-area .cm-scroller { max-height: none;" in css


def test_sidebar_omits_technical_model_type_without_losing_product_name():
    assert 'id="model-quality"' not in HTML
    assert "$('model-quality')" not in JS
    assert "Drama 3 Studio" in HTML


def test_voice_library_has_reusable_add_dialog_and_compact_preview_tiles():
    assert 'id="add-voice"' in HTML
    assert 'id="voice-library-dialog"' in HTML
    assert 'id="close-voice-library"' in HTML
    dialog = HTML.index('<dialog id="voice-library-dialog"')
    assert dialog < HTML.index('id="voice-form"') < HTML.index('</dialog>', dialog)
    assert "voice-tile-preview" in JS


def test_voice_design_reuses_the_physical_script_direction_and_submit_controls():
    studio = (ROOT / "ui/studio.js").read_text(encoding="utf-8")
    for identifier in ("design-audition", "design-direction", "generate-designed-voice"):
        assert identifier not in studio
    assert "field('Voice description', description)" in studio
    assert "return { mode: 'design', text: payload.text, direction: payload.direction," in studio
    assert "voice_description: description" in studio
    assert "back-to-generation" in studio


def test_voice_design_request_is_built_before_any_reference_preparation():
    studio = (ROOT / "ui/studio.js").read_text(encoding="utf-8")
    request = studio[studio.index("async function prepareRequest(payload)"):]
    design = request[request.index("if (design)"):request.index("const operation = copy(flow.operation)")]
    assert "return { mode: 'design'" in design
    for reference in ("await singleBindings", "prepareCloneReference", "voice_ids:", "uploaded_references:"):
        assert reference not in design
    assert "active: voiceDesignActive()" in studio
    assert "references.hidden = design" in studio


def test_creation_no_longer_builds_new_draft_menus_or_text_stream_tools():
    studio = (ROOT / "ui/studio.js").read_text(encoding="utf-8")
    for retired in ("menu('New draft'", "command('New draft'", "menu('Tools'",
                    "streamSwitch.id = 'toggle-text-stream'", "streamStatus.id = 'text-stream-status'"):
        assert retired not in studio
    assert "back-to-generation" in studio
    assert "voice_description: description" in studio


def test_normal_and_voice_design_writing_are_persisted_independently():
    studio = (ROOT / "ui/studio.js").read_text(encoding="utf-8")
    assert "Object.assign(form, flow.writing.reference)" in studio
    assert "contentVersion: 2, ...copy(flow.writing.design)" in studio
    assert "flow.writing = { reference: writingOf(value.form), design: writingOf(value.design) }" in studio
    assert "ScriptEditor.reset($('text'))" in studio


def test_normal_header_and_permanent_save_success_status_are_not_shown():
    studio = (ROOT / "ui/studio.js").read_text(encoding="utf-8")
    assert "title.closest('.panel-head').hidden = !design" in studio
    assert "saved.id = 'draft-status'" not in studio
    assert "$('draft-status')" not in studio
    assert "fail(error, 'save')" in studio
    assert "dataset.source === 'save'" in studio


def test_settings_omit_capability_listing_and_open_local_machine_by_default():
    studio = (ROOT / "ui/studio.js").read_text(encoding="utf-8")
    assert 'id="capability-list"' not in HTML and 'data-i18n="Engine gates"' not in HTML
    assert "renderCapabilities" not in JS and "$('capability-list')" not in JS
    assert "state.capabilities = capabilities" in JS
    assert "runtime.open = true" in studio
    assert "localized('summary', '', 'Local machine')" in studio


def test_dialogue_uses_one_editor_and_a_compact_speaker_sidebar():
    studio = (ROOT / "ui/studio.js").read_text(encoding="utf-8")
    assert 'id="dialogue-script" data-dialogue="true"' in HTML
    assert 'class="dialogue-speakers"' in HTML
    for old in ('id="scene-direction"', 'id="scene-description"', 'id="scene-lines"', 'id="add-line"'):
        assert old not in HTML
    assert "sceneRole('S1', 0)" in studio and "sceneRole('S2', 1)" in studio
    assert "dialogue-voice-dialog" in studio and "speaker-change" in studio
    assert "ScriptEditor.dialogue.reconcile" in studio


def test_dialogue_play_icons_use_the_same_renderer_before_and_after_playback():
    studio = (ROOT / "ui/studio.js").read_text(encoding="utf-8")
    assert "symbol === 'play' || symbol === 'pause'" in studio
    assert "button.innerHTML = icon(symbol)" in studio
    assert "function renderPreviewButton(button, playing)" in JS
    assert "renderPreviewButton(button, true)" in JS


def test_reference_transcription_status_is_prominent_and_accessible():
    references = (ROOT / "ui/references.js").read_text(encoding="utf-8")
    css = (ROOT / "ui/studio.css").read_text(encoding="utf-8")
    assert "field-help reference-status" in references
    assert "transcriptHeading, status, transcript" in references
    assert "status.dataset.state = value.pending || value.busy ? 'busy'" in references
    assert "status.setAttribute('aria-live', 'polite')" in references
    assert '.reference-status[data-state="busy"]::before' in css
    assert "reference-pending" in css and "prefers-reduced-motion" in css
    assert HTML.index('id="clone-asr-status"') < HTML.index('id="clone-transcript"')


def test_reference_transcription_is_automatic_optional_and_local_only():
    references = (ROOT / "ui/references.js").read_text(encoding="utf-8")
    studio = (ROOT / "ui/studio.js").read_text(encoding="utf-8")
    assert 'id="asr-mode"' not in HTML and "$('asr-mode')" not in JS + studio
    assert "local_only: true" in JS and "local_only: true" in references
    assert "task.status === 'skipped'" in JS and "task.status === 'skipped'" in references
    assert "reference.asrMode === 'manual'" not in references
    assert "transcriptHeading.append(transcriptLabel, recognize)" in references


def test_selected_library_voice_has_no_preparation_or_more_menu():
    studio = (ROOT / "ui/studio.js").read_text(encoding="utf-8")
    render = studio[studio.index("function renderLibraryReferences("):studio.index("async function singleBindings(")]
    assert "selected-library-voice" in render and "selected-voice-clear" in render
    assert "ReferencePrep.mount(" not in render and "save-current-voice" not in studio
    assert "voice-more-options" not in studio + JS
    assert "libraryTitle.dataset.i18n = 'Choose from voice library'" in studio


def test_speaker_voice_picker_no_longer_offers_additional_references():
    studio = (ROOT / "ui/studio.js").read_text(encoding="utf-8")
    picker = studio[studio.index("function chooseSpeakerVoice(role)") : studio.index("function renderScene()")]
    assert "Additional references · same voice" not in picker
    assert "command('Add reference'" not in picker
    assert "draft.extraReferences.shift()" in picker


def test_history_uses_text_previews_and_direct_download_without_a_view_button():
    assert "renderHistoryTitle(job, row.querySelector('.job-title'))" in JS
    assert "job.recipe?.text, job.result?.text" in JS
    assert "job.result?.native_take_file" in JS
    assert "historyTranscripts" in JS
    assert "inspect-job" not in JS
    assert "translated('Play')" in JS
    assert 'download-link history-download' in JS
    assert '<a class="menu-command download-link"' not in JS


def test_collapsed_upload_has_a_remove_control_that_reuses_existing_cleanup():
    assert 'id="clone-summary-remove"' in HTML
    handler = JS[JS.index("$('clone-summary-remove').onclick"):JS.index("function updateCloneSelectionActions()")]
    assert "event.preventDefault(); event.stopPropagation();" in handler
    assert "$('remove-clone-audio').click()" in handler


def test_result_workspaces_keep_independent_ids_and_reject_cross_page_renders():
    studio = (ROOT / "ui/studio.js").read_text(encoding="utf-8")
    assert "resultJobs: { generate: null, scene: null }" in JS
    assert "resultWorkspaceFor(job) !== state.resultWorkspace" in JS
    assert "active_job_id: state.resultJobs.generate?.id || null" in studio
    assert "active_job_id: state.resultJobs.scene?.id || null" in studio
    assert "await restoreWorkspaceResults(draft?.active_job_id, savedScene?.active_job_id)" in studio
    assert "showWorkspaceResult(event.detail)" in studio


def test_history_play_starts_audio_directly_without_a_player_popover():
    assert 'class="text-btn history-play"' in JS
    assert "previewVoice({ audio_url: fileUrl(job.output) }, play)" in JS
    assert '<details class="history-preview">' not in JS
    assert "resume ? audition : new Audio(voice.audio_url)" in JS
    assert "function audioDurationLabel(job)" in JS
    assert "audioDurationLabel(job)" in JS
    assert "history-waveform" not in JS
    player = (ROOT / "ui/player.js").read_text(encoding="utf-8")
    assert "miniWave" not in player and "waveform:" not in player
    assert "history-waveform" not in (ROOT / "ui/studio.css").read_text(encoding="utf-8")


def test_web_ui_rounds_display_to_two_decimals_without_rounding_request_state():
    player = (ROOT / "ui/player.js").read_text(encoding="utf-8")
    studio = (ROOT / "ui/studio.js").read_text(encoding="utf-8")
    assert "toFixed(2)" in JS
    assert "toFixed(3)" not in JS + player + studio
    assert "String(Number(Number(value).toFixed(2)))" in JS
    assert "return previous && previous.display === input.value ? String(previous.raw ?? '') : input.value;" in JS


def test_audio_edit_uses_a_dedicated_view_and_keeps_its_own_result():
    editor = (ROOT / "ui/audio-editor.js").read_text(encoding="utf-8")
    studio = (ROOT / "ui/studio.js").read_text(encoding="utf-8")
    css = (ROOT / "ui/studio.css").read_text(encoding="utf-8")
    assert "result_scope === 'audio_editor'" in JS
    assert "function renderResults(job, box = $('result-box'))" in JS
    assert "if (workspace !== 'editor') state.resultJobs[workspace] = job" in studio
    assert "studio.result_scope = 'audio_editor'; studio.edit_source = identity" in editor
    assert "document.querySelector('#view-audio-edit').replaceChildren(workspace)" in editor
    assert "hooks.enterEditor?.(returnView)" in editor
    assert "hooks.leaveEditor?.(returnView)" in editor
    assert 'id="view-audio-edit"' in HTML
    assert "const footer = el('footer', 'compose-footer action-dock audio-edit-dock')" in editor
    assert "draft.result_job_id = job.id" in editor
    assert "renderResults(job, resultBox)" in editor
    assert "await hooks.submit(payload); cleanup()" not in editor
    assert ".compact-studio .main:has(#view-audio-edit.active)" in css
    assert "delete draftStudio.result_scope; delete draftStudio.edit_source" in studio


def test_searchable_model_lists_expand_to_the_picker_width():
    controls = (ROOT / "ui/controls.js").read_text(encoding="utf-8")
    assert "control.search ? Math.max(rect.width, 760)" in controls
    assert "Math.min(innerWidth - 24, preferredWidth)" in controls


def test_dialogue_uses_measured_cursors_for_its_multiline_placeholder():
    editor = (ROOT / "ui/editor-source/editor.js").read_text(encoding="utf-8")
    assert "placeholder, drawSelection" in editor
    assert "textarea.dataset.dialogue === 'true' ? [speakers, drawSelection()]" in editor


def test_active_page_survives_refresh_per_browser_tab():
    studio = (ROOT / "ui/studio.js").read_text(encoding="utf-8")
    assert "const viewStorageKey = 'drama3.ui.view'" in JS
    assert "sessionStorage.getItem('drama3.ui.view')" in HTML
    assert "sessionStorage.setItem(viewStorageKey, name)" in JS
    assert "Object.hasOwn(viewTitles, name)" in JS
    assert "restoreView();" in JS
    assert "view(state.view);" in studio
    assert "await AudioEditor.restoreActive();" in studio
    assert "await StudioStartup.commit();" in studio


def test_initial_page_is_chosen_before_styles_or_body_can_paint():
    assert HTML.index("sessionStorage.getItem('drama3.ui.view')") < HTML.index('rel="stylesheet"')
    assert '<h1 id="view-title"></h1>' in HTML
    assert '<section class="view active" id="view-generate">' not in HTML
    assert '<button class="nav active" data-view="generate">' not in HTML
    assert "document.documentElement.dataset.view = name" in JS
    for name in ("generate", "scene", "voices", "takes", "settings"):
        assert f'html[data-view="{name}"] #view-{name}' in CSS
        assert f'html[data-view="{name}"] .nav[data-view="{name}"]' in CSS


def test_startup_uses_the_final_theme_and_a_single_hydration_commit():
    studio = (ROOT / "ui/studio.js").read_text(encoding="utf-8")
    startup = (ROOT / "ui/startup.js").read_text(encoding="utf-8")
    assert 'class="compact-studio"' in HTML
    assert "classList.add('compact-studio')" not in studio
    assert 'id="workspace-content" aria-busy="true" inert' in HTML
    assert "configureControls(); configureScene(); configureWorkspaceLayout();" in studio
    assert "api('/api/studio'), loadHealth(), loadVoices(), loadModel(), loadCapabilities(), loadJobs()" in studio
    assert "loadHealth(); loadCapabilities(); loadVoices(); loadJobs(); loadModel();" not in JS
    assert "await document.fonts.ready" in startup
    assert "root.dataset.startup = 'ready'; root.dataset.studioReady = 'true'" in startup
    assert "setTimeout" not in startup
    assert "StudioStartup.fail(error)" in studio


def test_navigation_clicks_cannot_bind_to_the_document_page_marker():
    assert "document.querySelectorAll('.nav, button[data-view=\"voices\"]')" in JS
    assert "document.querySelectorAll('.nav,[data-view=\"voices\"]')" not in JS
