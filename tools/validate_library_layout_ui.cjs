/* Compact library and writing layout: real uploads/store/audio, fixture ASR; no inference. */
const fs = require('node:fs'), path = require('node:path'), { chromium } = require('playwright');
const root = path.resolve(__dirname, '..'), base = process.env.DRAMA_UI_URL;
const folder = process.env.DRAMA_STUDIO_FOLDER;
const report = { scope: 'Actual writing layout and compact voice library, reusable add dialog, real uploads/audio/store; synthetic ASR, no model generation.',
  checks: {}, errors: [], screenshots: [], layout: [] };
(async () => {
  let browser, page;
  try {
    browser = await chromium.launch({ channel: 'chrome', headless: true });
    const context = await browser.newContext({ viewport: { width: 1440, height: 960 }, locale: 'zh-CN' });
    await context.addInitScript(() => localStorage.setItem('drama3.ui.language', 'zh-CN'));
    page = await context.newPage();
    page.on('pageerror', error => report.errors.push(error.stack || error.message));
    let asrCalls = 0, asrAvailable = false;
    await page.route('**/api/transcriptions', route => {
      asrCalls++;
      return route.fulfill({ json: asrAvailable
        ? { id: 'library-fixture-asr', status: 'complete', result: { text: '这是自动转写的附加参考台词。' } }
        : { id: 'library-fixture-asr', status: 'skipped', reason: 'model_unavailable' } });
    });
    const api = async (url, data) => {
      const response = data === undefined ? await context.request.get(base + url)
        : await context.request.post(base + url, { data });
      if (!response.ok()) throw new Error(url + ': ' + await response.text());
      return response.json();
    };
    const check = (name, value) => { report.checks[name] = Boolean(value); if (!value) throw new Error(name); };
    const ready = () => page.waitForFunction(() => document.documentElement.dataset.studioReady === 'true');
    const settle = () => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    const choose = async (id, value) => {
      await page.locator(`#${id}-trigger`).click();
      await page.locator(`#${id}-listbox [data-value=${JSON.stringify(value)}]`).click();
    };
    const navigate = async name => { await page.locator(`.nav[data-view="${name}"]`).click(); await settle(); };
    const editor = () => page.locator('.compose-panel > .editor-area .cm-content');
    const editorHost = () => page.locator('.compose-panel > .editor-area .script-editor');
    const dialog = () => page.locator('#voice-library-dialog');
    const openDialog = async () => {
      await page.locator('#add-voice').click(); await dialog().waitFor({ state: 'visible' }); await settle();
    };
    const note = () => page.locator('#voice-form-note');
    const shot = async name => {
      const file = path.join(folder, name + '.png'); await page.screenshot({ path: file }); report.screenshots.push(file);
    };
    const writingLayout = () => page.evaluate(() => {
      const rect = node => {
        const r = node.getBoundingClientRect();
        return { top: r.top, bottom: r.bottom, height: r.height, left: r.left, right: r.right };
      };
      const area = document.querySelector('.compose-panel > .editor-area');
      return { viewport: { width: innerWidth, height: innerHeight }, editor: rect(area),
        direction: rect(document.querySelector('#direction')), label: rect(document.querySelector('label[for="text"]')),
        canvas: rect(area.querySelector('.cm-editor')), toolbar: rect(area.querySelector('.script-toolbar')),
        dock: rect(document.querySelector('#submit-job').closest('.action-dock')) };
    });
    await page.goto(base, { waitUntil: 'networkidle' }); await ready();
    check('sidebar_has_no_model_type_or_format_element', await page.locator('#model-quality').count() === 0);
    check('sidebar_keeps_product_and_machine_information', (await page.locator('.side-version').textContent()).includes('Drama 3 Studio')
      && await page.locator('.local-card').isVisible() && Boolean((await page.locator('#hero-hardware').textContent()).trim()));
    check('direction_precedes_text_label_and_editor', await page.evaluate(() => {
      const direction = document.getElementById('direction'), label = document.querySelector('label[for="text"]');
      const area = document.querySelector('.compose-panel > .editor-area');
      return Boolean(direction.compareDocumentPosition(label) & Node.DOCUMENT_POSITION_FOLLOWING)
        && Boolean(label.compareDocumentPosition(area) & Node.DOCUMENT_POSITION_FOLLOWING);
    }));
    await editor().fill('简洁的创作界面，也要保留完整的声音创作能力。');
    for (const size of [{ width: 1920, height: 1080 }, { width: 1440, height: 960 }, { width: 1271, height: 712 }]) {
      await page.setViewportSize(size); await page.evaluate(() => scrollTo(0, 0)); await settle();
      const layout = await writingLayout(); report.layout.push(layout);
      const gap = layout.dock.top - layout.editor.bottom;
      check(`writing_surface_uses_remaining_height_${size.width}`, gap >= 0 && gap <= 64);
      check(`rich_editor_fills_its_surface_${size.width}`, layout.canvas.height >= layout.editor.height - layout.toolbar.height - 5
        && layout.editor.bottom - layout.toolbar.bottom <= 3);
      check(`direction_and_label_do_not_overlap_${size.width}`, layout.direction.bottom <= layout.label.top + 1
        && layout.label.bottom <= layout.editor.top + 1);
      check(`generate_stays_visible_${size.width}`, await page.locator('#submit-job').evaluate(node => {
        const r = node.getBoundingClientRect(); return r.top >= 0 && r.bottom <= innerHeight && r.right <= innerWidth;
      }));
    }
    await page.setViewportSize({ width: 1440, height: 960 }); await settle();
    const beforeLongText = await writingLayout();
    await editor().fill(Array.from({ length: 90 }, (_, index) => `第${index + 1}行：编辑区内部滚动，不应把页面不断撑高。`).join('\n'));
    await settle();
    check('long_script_scrolls_inside_stable_editor', await page.locator('.compose-panel > .editor-area .cm-scroller').evaluate(node =>
      node.scrollHeight > node.clientHeight + 100) && Math.abs((await writingLayout()).editor.height - beforeLongText.editor.height) <= 2);
    await page.evaluate(() => scrollTo(0, document.documentElement.scrollHeight)); await settle();
    const afterScroll = await writingLayout();
    check('document_scroll_does_not_increase_editor_height', Math.abs(afterScroll.editor.height - beforeLongText.editor.height) <= 2);
    await editorHost().locator('[data-label="Source text"]').click(); await settle();
    check('raw_text_mode_remains_available', await page.locator('#text').isVisible());
    check('raw_text_consumes_the_same_writing_surface', await page.locator('#text').evaluate(node => {
      const r = node.getBoundingClientRect(), area = node.closest('.editor-area').getBoundingClientRect();
      const toolbar = node.closest('.editor-area').querySelector('.script-toolbar').getBoundingClientRect();
      return r.height >= area.height - toolbar.height - 6 && r.bottom <= toolbar.top + 2;
    }));
    await page.locator('#text').fill('原始文本也能选中标签。');
    await page.locator('#text').evaluate(node => { node.focus(); node.setSelectionRange(5, 7); });
    await editorHost().locator('[data-label="Tags"]').click();
    await editorHost().locator('[data-tag="whisper"][data-kind="scope"]').click();
    check('expanded_writing_surface_keeps_raw_selection_tagging', await page.inputValue('#text') === '原始文本也<whisper>能选</whisper>中标签。');
    if (await page.locator('#text').isVisible()) await editorHost().locator('[data-label="Source text"]').click();
    await editor().fill('保持这份尚未提交的创作草稿。');
    await page.locator('#direction').fill('自然，轻松。');
    await page.evaluate(() => scrollTo(0, 0)); await settle(); await shot('writing-surface-desktop');
    await navigate('voices');
    check('voice_library_has_add_entry_not_permanent_form', await page.locator('#add-voice').isVisible()
      && !await page.locator('#voice-form').isVisible() && !await dialog().getAttribute('open'));
    check('persistent_voice_form_is_inside_native_modal', await dialog().count() === 1
      && await page.locator('#voice-form').evaluate(form => form.closest('dialog')?.id === 'voice-library-dialog'));
    check('voice_library_uses_grid_tiles', await page.locator('#voice-list').evaluate(node => getComputedStyle(node).display === 'grid')
      && await page.locator('#voice-list .voice-card').count() === 2);
    check('library_tiles_have_no_waveform_or_dom_audio', !await page.locator('#voice-list .studio-player,#voice-list audio,#voice-list canvas').count());
    const tiles = page.locator('#voice-list .voice-card');
    const boxes = await tiles.evaluateAll(nodes => nodes.map(node => {
      const r = node.getBoundingClientRect(); return { top: r.top, width: r.width, height: r.height };
    }));
    check('desktop_tiles_share_a_row_and_remain_compact', Math.abs(boxes[0].top - boxes[1].top) < 2
      && boxes.every(box => box.width <= 460 && box.height <= 260));
    check('tiles_keep_use_and_download_actions', await tiles.first().locator('.use-voice').isVisible()
      && await tiles.first().locator('a[download]').count() >= 1);
    const previews = page.locator('#voice-list .voice-tile-preview');
    await previews.nth(0).click();
    await page.waitForFunction(() => document.querySelector('#voice-list .voice-tile-preview').getAttribute('aria-pressed') === 'true');
    check('tile_play_uses_real_audio_without_waveform', await page.evaluate(() => audition && !audition.paused && audition.duration > 0)
      && !await page.locator('#voice-list audio').count());
    await previews.nth(1).click();
    await page.waitForFunction(() => document.querySelectorAll('#voice-list .voice-tile-preview')[1].getAttribute('aria-pressed') === 'true');
    check('only_one_tile_previews_at_a_time', await previews.nth(0).getAttribute('aria-pressed') === 'false'
      && await previews.nth(1).getAttribute('aria-pressed') === 'true');
    await previews.nth(1).click();
    check('tile_play_button_also_pauses', await page.evaluate(() => !audition || audition.paused)
      && await previews.nth(1).getAttribute('aria-pressed') === 'false');
    await previews.nth(0).click();
    await page.locator('#voice-search').fill('not-a-real-voice-name');
    check('search_hides_nonmatching_tiles_and_stops_preview', await page.locator('#voice-list .voice-card:visible').count() === 0
      && await page.evaluate(() => !audition || audition.paused));
    await page.locator('#voice-search').fill('');
    await previews.nth(0).click(); await navigate('generate');
    check('leaving_library_stops_tile_audio', await page.evaluate(() => !audition || audition.paused));
    check('browsing_voices_does_not_change_writing_draft', await page.inputValue('#text') === '保持这份尚未提交的创作草稿。'
      && await page.inputValue('#direction') === '自然，轻松。');
    await navigate('voices'); await openDialog();
    check('add_voice_opens_one_native_modal', await page.locator('dialog[open]').count() === 1
      && await dialog().evaluate(node => node.open && node.contains(document.activeElement)));
    check('add_modal_does_not_use_fixed_generation_dock', !await dialog().locator('.action-dock').count());
    await page.locator('#voice-name').fill('保留的音色草稿');
    await page.locator('#save-voice').click();
    await page.waitForFunction(() => document.getElementById('voice-form-note').textContent.trim().length > 0);
    check('missing_audio_error_is_visible_inside_modal', await note().isVisible()
      && await note().evaluate(node => Boolean(node.closest('dialog[open]')))
      && await page.locator('#save-voice').isEnabled() && (await api('/api/voices')).voices.length === 2);
    await page.keyboard.press('Escape');
    await dialog().waitFor({ state: 'hidden' });
    check('escape_closes_but_keeps_modal_and_draft', await dialog().count() === 1
      && await page.inputValue('#voice-name') === '保留的音色草稿'
      && await page.locator('#add-voice').evaluate(node => document.activeElement === node));
    await openDialog();
    check('reopening_keeps_voice_name_and_preparation', await page.inputValue('#voice-name') === '保留的音色草稿'
      && await page.locator('#voice-library-preparation').count() === 1);
    await page.locator('#library-new-upload').setInputFiles(process.env.DRAMA_STUDIO_REFERENCE);
    await page.waitForFunction(() => Boolean(StudioFlow.flow.voiceDraft.main.upload?.id));
    await page.locator('#library-new-transcript').waitFor({ state: 'visible' });
    check('reference_preparation_keeps_waveform_only_inside_modal', await page.locator('#voice-library-clips .studio-player').count() === 1
      && !await page.locator('#voice-list .studio-player').count() && asrCalls === 1);
    await page.locator('#save-voice').click();
    await page.waitForFunction(() => document.getElementById('voice-form-note').textContent.includes(t('Enter the exact words spoken in the reference audio.')));
    check('manual_reference_requires_transcript_without_losing_upload', await dialog().isVisible()
      && await page.evaluate(() => Boolean(StudioFlow.flow.voiceDraft.main.upload?.id))
      && (await api('/api/voices')).voices.length === 2);
    await page.locator('#library-new-transcript').fill('这是我人工核对的主参考台词。');
    await page.evaluate(() => StudioFlow.save());
    await page.locator('#close-voice-library').click();
    await page.reload({ waitUntil: 'networkidle' }); await ready(); await navigate('voices'); await openDialog();
    check('voice_modal_draft_recovers_after_page_reload', await page.inputValue('#voice-name') === '保留的音色草稿'
      && await page.inputValue('#library-new-transcript') === '这是我人工核对的主参考台词。'
      && !await page.locator('#library-new-asr').count()
      && await page.evaluate(() => Boolean(StudioFlow.flow.voiceDraft.main.upload?.id)));
    asrAvailable = true;
    const additional = page.locator('#voice-library-preparation > .additional-references');
    await additional.locator(':scope > summary').click();
    await additional.getByRole('button', { name: await page.evaluate(() => t('Add reference')), exact: true }).click();
    const extra = page.locator('#voice-library-extras > .reference-preparation').first();
    await extra.locator('input[type="file"]').setInputFiles(process.env.DRAMA_STUDIO_REFERENCE);
    await page.waitForFunction(() => StudioFlow.flow.voiceDraft.extraReferences[0]?.transcript === '这是自动转写的附加参考台词。');
    check('additional_reference_uses_automatic_transcription', asrCalls === 2
      && await extra.locator('textarea').inputValue() === '这是自动转写的附加参考台词。');
    await page.locator('#voice-name').fill('Compact multi-reference voice');
    let rejectSave = true;
    await page.route('**/api/voices/from-references', route => {
      if (rejectSave) { rejectSave = false; return route.fulfill({ status: 400, json: { error: 'Fixture rejected save.' } }); }
      return route.continue();
    });
    await page.locator('#save-voice').click();
    await page.waitForFunction(() => document.getElementById('voice-form-note').textContent.includes('Fixture rejected save.'));
    check('server_error_stays_in_modal_and_preserves_ready_references', await note().isVisible()
      && await dialog().isVisible() && await page.locator('#save-voice').isEnabled()
      && await page.inputValue('#library-new-transcript') === '这是我人工核对的主参考台词。'
      && await extra.locator('textarea').inputValue() === '这是自动转写的附加参考台词。');
    await page.locator('#save-voice').click(); await dialog().waitFor({ state: 'hidden' });
    await page.waitForFunction(() => document.querySelectorAll('#voice-list .voice-card').length === 3);
    const savedVoice = (await api('/api/voices')).voices.find(voice => voice.name === 'Compact multi-reference voice');
    check('successful_save_closes_modal_and_persists_all_references', savedVoice?.references?.length === 2
      && savedVoice.references[0].transcript.includes('这是我人工核对的主参考台词。')
      && savedVoice.references[1].transcript.includes('这是自动转写的附加参考台词。'));
    check('successful_save_resets_only_add_voice_draft', await page.inputValue('#voice-name') === ''
      && await page.inputValue('#text') === '保持这份尚未提交的创作草稿。'
      && await page.evaluate(() => !StudioFlow.flow.voiceDraft.main.upload && !StudioFlow.flow.voiceDraft.extraReferences.length));
    for (const name of ['清晰旁白', '나긋한 설명', 'A long voice name for layout checking without horizontal overflow', '落ち着いたナレーション']) {
      await api('/api/voices/from-references', { name,
        references: [{ voice_id: savedVoice.id, reference_index: 0, speaker: 0, text: '这是我人工核对的主参考台词。' }] });
    }
    await page.evaluate(() => loadVoices());
    const savedTile = page.locator('#voice-list .voice-card').filter({ has: page.locator('.use-voice[data-id=' + JSON.stringify(savedVoice.id) + ']') });
    const more = savedTile.locator('.voice-reference-clips');
    check('multi_reference_tile_keeps_compact_extra_clips', await more.count() === 1);
    await more.locator(':scope > summary').click(); await settle();
    check('extra_reference_has_preview_and_download_without_waveform', await more.locator('button').count() >= 1
      && await more.locator('a[download]').count() === 1 && !await more.locator('audio,.studio-player').count());
    await more.locator('button').first().click();
    await page.waitForFunction(() => audition && !audition.paused && audition.duration > 0);
    check('extra_reference_preview_uses_real_saved_clip', await page.evaluate(() => audition.duration > 0));
    await more.locator(':scope > summary').click(); await settle();
    check('collapsing_extra_clips_stops_their_audio', await page.evaluate(() => !audition || audition.paused));
    await savedTile.locator('.use-voice').click();
    await page.locator('#view-generate.active').waitFor();
    check('use_voice_returns_to_creation_with_correct_saved_voice', await page.evaluate(id => selectedVoices().includes(id), savedVoice.id)
      && await page.inputValue('#text') === '保持这份尚未提交的创作草稿。');
    await navigate('voices'); await shot('compact-voice-library-desktop');
    const names = { 'zh-CN': '添加音色', en: 'Add a voice', ja: 'ボイスを追加', ko: '음성 추가' };
    for (const locale of ['zh-CN', 'en', 'ja', 'ko']) {
      await page.evaluate(locale => I18n.setLocale(locale), locale);
      for (const size of [{ width: 1440, height: 960 }, { width: 390, height: 844 }]) {
        await page.setViewportSize(size); await navigate('voices');
        check(`library_fits_${locale}_${size.width}`, await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
        check(`add_voice_is_localized_and_visible_${locale}_${size.width}`, (await page.locator('#add-voice').textContent()).trim() === names[locale]
          && await page.locator('#add-voice').isVisible());
        await openDialog();
        check(`add_voice_dialog_fits_${locale}_${size.width}`, await dialog().evaluate(node => {
          const r = node.getBoundingClientRect();
          return r.left >= 8 && r.top >= 8 && r.right <= innerWidth - 8 && r.bottom <= innerHeight - 8
            && node.scrollWidth <= node.clientWidth + 1;
        }));
        check(`modal_has_no_transcription_mode_selector_${locale}_${size.width}`, !await page.locator('#library-new-asr').count());
        check(`modal_keeps_its_upload_entry_${locale}_${size.width}`, await page.locator('#library-new-upload').count() === 1
          && await page.locator('#voice-library-clips .reference-upload-button').isVisible());
        await page.locator('#close-voice-library').click();
        await navigate('generate');
        check(`writing_layout_fits_${locale}_${size.width}`, await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
        check(`direction_order_kept_${locale}_${size.width}`, await page.locator('#direction').evaluate(node =>
          Boolean(node.compareDocumentPosition(document.querySelector('label[for="text"]')) & Node.DOCUMENT_POSITION_FOLLOWING)));
        check(`generate_visible_${locale}_${size.width}`, await page.locator('#submit-job').evaluate(node => {
          const r = node.getBoundingClientRect(); return r.top >= 0 && r.bottom <= innerHeight && r.right <= innerWidth;
        }));
      }
    }
    await page.evaluate(() => I18n.setLocale('zh-CN')); await page.setViewportSize({ width: 390, height: 844 });
    await navigate('voices'); await shot('compact-voice-library-mobile'); await openDialog(); await shot('add-voice-modal-mobile');
    await page.locator('#close-voice-library').click();
    check('no_model_generation_was_submitted', (await api('/api/jobs')).jobs.length === 0);
    check('no_browser_exceptions', report.errors.length === 0);
  } catch (error) {
    report.errors.push(error.stack || String(error));
    if (page) {
      report.playback = await page.evaluate(() => ({
        source: audition?.src, paused: audition?.paused, duration: audition?.duration,
        readyState: audition?.readyState, error: audition?.error?.message,
        view: state.view,
        status: document.querySelector('#startup-state')?.textContent,
      })).catch(() => null);
      const file = path.join(folder, 'library-layout-failure.png');
      await page.screenshot({ path: file, fullPage: true }).catch(() => {}); report.screenshots.push(file);
    }
  } finally {
    await browser?.close();
    report.passed = !report.errors.length && Object.values(report.checks).every(Boolean);
    fs.writeFileSync(path.join(root, 'research/2026-10-20-library-layout-ui-acceptance.json'), JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify(report, null, 2)); if (!report.passed) process.exitCode = 1;
  }
})();
