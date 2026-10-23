/* Real local uploads, storage and audio; optional ASR and synthetic TTS acceptance. */
const fs = require('node:fs'), path = require('node:path'), { chromium } = require('playwright');
const root = path.resolve(__dirname, '..'), base = process.env.DRAMA_UI_URL, folder = process.env.DRAMA_STUDIO_FOLDER;
const report = { scope: 'Compact saved-voice selection, automatic local-only ASR, silent missing-model handling, editable transcripts and shared preparation; synthetic inference and controlled ASR.',
  checks: {}, errors: [], screenshots: [], asrRequests: [] };
(async () => {
  let browser, page;
  try {
    browser = await chromium.launch({ channel: 'chrome', headless: true });
    const context = await browser.newContext({ viewport: { width: 1440, height: 960 }, locale: 'zh-CN' });
    await context.addInitScript(() => localStorage.setItem('drama3.ui.language', 'zh-CN'));
    page = await context.newPage();
    page.on('pageerror', error => report.errors.push(error.stack || error.message));
    let behavior = 'real-missing', completed = false, serial = 0, lastJob = '';
    await page.route('**/api/transcriptions', route => {
      report.asrRequests.push(route.request().postDataJSON());
      if (behavior === 'real-missing') return route.continue();
      return route.fulfill({ json: behavior === 'pending'
        ? { id: `reference-flow-${++serial}`, status: 'running', stage: 'transcribing' }
        : { id: `reference-flow-${++serial}`, status: 'complete', cache_hit: behavior === 'cached',
          result: { text: '自动识别出的参考台词。' } } });
    });
    await page.route('**/api/transcriptions/reference-flow-*', route => route.fulfill({
      json: completed ? { id: 'reference-flow-complete', status: 'complete', result: { text: '稍后返回的识别结果。' } }
        : { id: 'reference-flow-pending', status: 'running', stage: 'transcribing' },
    }));
    const api = async (url, data) => {
      const response = data === undefined ? await context.request.get(base + url)
        : await context.request.post(base + url, { data });
      if (!response.ok()) throw new Error(url + ': ' + await response.text());
      return response.json();
    };
    const check = (name, value) => { report.checks[name] = Boolean(value); if (!value) throw new Error(name); };
    const ready = () => page.waitForFunction(() => document.documentElement.dataset.studioReady === 'true');
    const waitClone = () => page.waitForFunction(() => state.cloneUpload && !state.cloneAsrPromise);
    const chooseVoice = async id => {
      await page.locator('#saved-voice-options > summary').click();
      await page.locator(`#voice-picks [data-voice="${id}"]`).click();
    };
    const selected = () => page.locator('#selected-reference-details .selected-library-voice');
    const editor = () => page.locator('.compose-panel > .editor-area .cm-content');
    const finish = async () => {
      await page.waitForFunction(old => StudioFlow.flow.lastSubmit?.id !== old
        && document.querySelector('#result-box').dataset.jobId === StudioFlow.flow.lastSubmit?.id
        && document.querySelector('#job-badge').dataset.status === 'complete', lastJob);
      lastJob = await page.locator('#result-box').getAttribute('data-job-id');
      return api('/api/jobs/' + lastJob);
    };
    const shot = async (name, selector = null) => {
      const target = selector ? page.locator(selector) : page;
      const file = path.join(folder, name + '.png'); await target.screenshot({ path: file });
      report.screenshots.push(file);
    };
    await page.goto(base, { waitUntil: 'networkidle' }); await ready();
    const voices = (await api('/api/voices')).voices;
    check('library_entry_uses_requested_name', (await page.locator('#saved-voice-options > summary').innerText()).includes('从音色库选取'));
    check('no_more_menu_or_transcription_mode_selector', !await page.locator('#voice-more-options,#asr-mode,#save-current-voice').count());
    await editor().fill('保持这份正文，不被参考音色选取改变。');
    await page.locator('#direction').fill('自然清晰。');
    await chooseVoice(voices[0].id);
    check('selection_shows_voice_name_and_play_button', await selected().locator('strong').textContent() === voices[0].name
      && await selected().locator('.reference-compact-play').isVisible());
    check('library_selection_hides_upload_and_reference_preparation', !await page.locator('#clone-upload-zone').isVisible()
      && !await page.locator('#clone-upload-preview').isVisible() && !await selected().locator('details,textarea,.studio-player').count());
    check('library_selection_has_no_save_voice_action', !await page.locator('#view-generate').getByRole('button', { name: '保存音色', exact: true }).isVisible());
    check('library_selection_does_not_run_asr', report.asrRequests.length === 0);
    await selected().locator('.reference-compact-play').click();
    await page.waitForFunction(() => audition && !audition.paused && audition.duration > 0);
    check('selected_voice_preview_plays_the_saved_audio', await page.evaluate(() => audition.currentSrc === new URL(state.voices[0].audio_url, location.origin).href));
    await selected().locator('.reference-compact-play').click();
    check('selected_voice_preview_can_be_stopped', await page.evaluate(() => !audition || audition.paused));
    await page.evaluate(id => StudioFlow.flow.singleReferences.push(ReferencePrep.forVoice(id)), voices[0].id);
    await chooseVoice(voices[1].id);
    check('choosing_another_voice_replaces_the_compact_binding', await selected().count() === 1
      && await selected().locator('strong').textContent() === voices[1].name);
    check('choosing_a_voice_clears_hidden_references_from_older_drafts',
      await page.evaluate(() => StudioFlow.flow.singleReferences.length === 0));
    await page.evaluate(() => StudioFlow.save());
    await page.reload({ waitUntil: 'networkidle' }); await ready();
    check('saved_voice_restores_without_expanding_preparation', await selected().locator('strong').textContent() === voices[1].name
      && !await page.locator('#clone-upload-zone').isVisible() && !await selected().locator('details,textarea').count());
    check('selection_and_reload_preserve_target_text_and_direction', await page.inputValue('#text') === '保持这份正文，不被参考音色选取改变。'
      && await page.inputValue('#direction') === '自然清晰。');
    await page.locator('#submit-job').click(); const libraryJob = await finish();
    check('compact_selection_still_submits_the_correct_voice', libraryJob.recipe.reference_snapshot.length === 1
      && libraryJob.recipe.reference_snapshot[0].text.includes('参考音频的台词。') && report.asrRequests.length === 0);
    await page.evaluate(() => { state.jobSubmitting = true; renderModelPicker(); });
    check('busy_generation_does_not_show_model_switch_warning', !await page.locator('#active-model-status').isVisible()
      && await page.locator('#apply-active-model').isDisabled());
    await page.evaluate(() => { state.jobSubmitting = false; renderModelPicker(); });
    await selected().locator('.selected-voice-clear').click();
    check('clear_restores_upload_without_losing_script', await page.locator('#clone-upload-zone').isVisible()
      && !await selected().count() && await page.inputValue('#text') === libraryJob.recipe.text);

    await page.locator('#clone-audio').setInputFiles(process.env.DRAMA_STUDIO_REFERENCE); await waitClone();
    check('missing_local_asr_leaves_an_empty_editable_transcript', await page.locator('#clone-transcript').isEditable()
      && await page.inputValue('#clone-transcript') === '' && !await page.locator('#clone-asr-status').isVisible());
    check('automatic_transcription_requests_are_local_only', report.asrRequests.at(-1).local_only === true
      && report.asrRequests.at(-1).language === 'auto');
    check('missing_model_has_no_error_banner', !await page.locator('#studio-error').isVisible()
      && !await page.locator('#result-error').isVisible());
    check('retry_is_beside_transcript_heading', await page.locator('#transcribe-clone').evaluate(node =>
      node.parentElement.classList.contains('transcription-heading') && Boolean(node.parentElement.querySelector('label[for="clone-transcript"]'))));
    await page.locator('#clone-transcript').fill('没有识别模型时自己填写的台词。');
    await page.locator('#transcribe-clone').click(); await waitClone();
    check('retry_without_model_is_silent_and_keeps_manual_text', !await page.locator('#clone-asr-status').isVisible()
      && await page.inputValue('#clone-transcript') === '没有识别模型时自己填写的台词。'
      && report.asrRequests.at(-1).force === true && report.asrRequests.at(-1).local_only === true);
    await page.locator('#submit-job').click(); const manualJob = await finish();
    check('missing_asr_does_not_prevent_manually_prepared_generation', manualJob.recipe.reference_snapshot[0].text === '没有识别模型时自己填写的台词。');
    behavior = 'cached';
    await page.locator('#transcribe-clone').click(); await waitClone();
    check('installed_or_cached_asr_populates_the_editable_transcript', await page.inputValue('#clone-transcript') === '自动识别出的参考台词。'
      && await page.locator('#clone-transcript').isEditable());
    check('completed_transcription_has_no_success_chatter', !await page.locator('#clone-asr-status').isVisible()
      && !await page.getByText('识别完成，可检查或修改文字。', { exact: true }).count());
    await page.locator('#clone-transcript').fill('识别后可以正常修改。');
    check('recognized_transcript_remains_manually_editable', await page.inputValue('#clone-transcript') === '识别后可以正常修改。');
    behavior = 'pending'; completed = false;
    await page.locator('#transcribe-clone').click();
    await page.locator('#clone-asr-status[data-state="busy"]').waitFor();
    check('active_transcription_keeps_a_prominent_status', await page.locator('#clone-asr-status').evaluate(node =>
      getComputedStyle(node).fontWeight === '600' && node.getAttribute('aria-live') === 'polite'));
    await page.locator('#clone-transcript').fill('识别期间改过的文字不能被覆盖。');
    completed = true; await waitClone();
    check('late_recognition_preserves_user_edits', await page.inputValue('#clone-transcript') === '识别期间改过的文字不能被覆盖。'
      && await page.locator('#apply-asr-text').isVisible() && !await page.locator('#clone-asr-status').isVisible());
    await page.locator('#apply-asr-text').click();
    check('user_can_explicitly_apply_the_recognized_alternative', await page.inputValue('#clone-transcript') === '稍后返回的识别结果。'
      && !await page.locator('#clone-asr-status').isVisible());
    await page.evaluate(() => StudioFlow.save());
    const saved = (await api('/api/studio')).drafts.single.value;
    saved.cloneRecognition.status = 'Transcribed. Review or edit the text.';
    saved.form['asr-mode'] = 'manual';
    await api('/api/studio/draft', { key: 'single', value: saved });
    await page.reload({ waitUntil: 'networkidle' }); await ready();
    await waitClone();
    check('legacy_success_status_and_manual_mode_do_not_reappear', !await page.locator('#clone-asr-status').isVisible()
      && !await page.locator('#asr-mode').count() && await page.inputValue('#clone-transcript') === '稍后返回的识别结果。');
    completed = false; await page.locator('#transcribe-clone').click();
    await page.locator('#clone-asr-status[data-state="busy"]').waitFor();
    await chooseVoice(voices[0].id); completed = true;
    await page.waitForTimeout(800);
    check('switching_to_a_library_voice_ignores_old_asr_results', await selected().locator('strong').textContent() === voices[0].name
      && !await page.locator('#clone-upload-zone').isVisible() && !await page.locator('#clone-upload-preview').isVisible()
      && await page.inputValue('#clone-transcript') === '');

    behavior = 'real-missing';
    await page.locator('.nav[data-view="voices"]').click(); await page.locator('#add-voice').click();
    await page.locator('#library-new-upload').setInputFiles(process.env.DRAMA_STUDIO_REFERENCE);
    await page.waitForFunction(() => StudioFlow.flow.voiceDraft.main.upload
      && document.querySelector('#library-new-transcript').getAttribute('aria-busy') === 'false');
    check('library_add_uses_the_same_optional_automatic_transcription', !await page.locator('#library-new-asr').count()
      && !await page.locator('#voice-library-clips .reference-status').isVisible()
      && await page.locator('#library-new-transcript').isEditable());
    check('shared_retry_is_beside_the_transcript_label', await page.locator('#library-new-transcript').evaluate(node => {
      const heading = node.previousElementSibling.previousElementSibling;
      return heading.classList.contains('transcription-heading') && Boolean(heading.querySelector('button'));
    }));
    await page.locator('#library-new-transcript').fill('音色库手动准备的参考台词。');
    await page.locator('#voice-name').fill('Optional ASR voice');
    await page.locator('#save-voice').click();
    await page.locator('#voice-library-dialog').waitFor({ state: 'hidden' });
    check('voice_library_can_save_without_an_asr_model', (await api('/api/voices')).voices.some(voice => voice.name === 'Optional ASR voice'));
    await page.locator('.nav[data-view="scene"]').click();
    await page.locator('.dialogue-speaker').first().locator('.speaker-change').click();
    await page.locator('#dialogue-voice-tab-upload').click();
    const upload = page.locator('#dialogue-voice-upload input[type="file"]').first();
    await upload.setInputFiles(process.env.DRAMA_STUDIO_REFERENCE);
    await page.waitForFunction(() => document.querySelector('#dialogue-voice-upload textarea')?.getAttribute('aria-busy') === 'false'
      && document.querySelector('#dialogue-voice-upload audio')?.getAttribute('src'));
    check('dialogue_upload_also_skips_missing_asr_without_error', !await page.locator('#dialogue-voice-upload select[id$="-asr"]').count()
      && !await page.locator('#dialogue-voice-upload .reference-status').isVisible()
      && await page.locator('#dialogue-voice-upload textarea').isEditable());
    await page.locator('#dialogue-voice-upload textarea').fill('多人对话手动准备的参考台词。');
    await page.locator('#dialogue-voice-apply').click();
    await page.locator('#dialogue-voice-dialog').waitFor({ state: 'detached' });
    check('dialogue_can_use_a_manually_prepared_reference', await page.locator('.dialogue-speaker').first().locator('.speaker-audio-name').textContent() === 'reference.wav');
    await page.locator('.nav[data-view="generate"]').click();
    const names = { 'zh-CN': '从音色库选取', en: 'Choose from voice library', ja: 'ライブラリから選択', ko: '음성 라이브러리에서 선택' };
    for (const locale of ['zh-CN', 'en', 'ja', 'ko']) {
      await page.evaluate(locale => I18n.setLocale(locale), locale);
      for (const size of [{ width: 1440, height: 960 }, { width: 390, height: 844 }]) {
        await page.setViewportSize(size);
        check(`library_entry_localized_${locale}_${size.width}`, (await page.locator('#saved-voice-options > summary').textContent()).includes(names[locale]));
        check(`voice_source_entries_share_typography_${locale}_${size.width}`, await page.evaluate(() => {
          const library = getComputedStyle(document.querySelector('#saved-voice-options > summary'));
          const design = getComputedStyle(document.querySelector('#describe-voice'));
          return library.fontWeight === '500'
            && ['fontFamily', 'fontSize', 'fontWeight', 'fontStyle', 'lineHeight', 'letterSpacing']
              .every(property => library[property] === design[property]);
        }));
        check(`compact_voice_is_visible_and_fits_${locale}_${size.width}`, await selected().evaluate(node => {
          const rect = node.getBoundingClientRect();
          return rect.width > 0 && rect.left >= 0 && rect.right <= innerWidth && node.scrollWidth <= node.clientWidth + 1;
        }));
        check(`reference_ui_has_no_horizontal_overflow_${locale}_${size.width}`, await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
        check(`removed_controls_stay_absent_${locale}_${size.width}`, !await page.locator('#asr-mode,#voice-more-options,#save-current-voice').count());
        if (size.width === 1440 && ['en', 'ja'].includes(locale)) await shot(`matching-voice-entry-type-${locale}`, '.voice-source-bar');
      }
    }
    await page.evaluate(() => I18n.setLocale('zh-CN'));
    await page.setViewportSize({ width: 1440, height: 960 }); await shot('compact-home-reference-desktop');
    await page.setViewportSize({ width: 390, height: 844 }); await shot('compact-home-reference-mobile');
    check('all_reference_asr_requests_are_local_only', report.asrRequests.every(request => request.local_only === true && request.language === 'auto'));
    check('no_browser_errors', report.errors.length === 0);
  } catch (error) {
    report.errors.push(error.stack || String(error));
    if (page) {
      const file = path.join(folder, 'reference-flow-failure.png');
      await page.screenshot({ path: file }).catch(() => {}); report.screenshots.push(file);
    }
  } finally {
    if (browser) await browser.close();
    report.passed = !report.errors.length && Object.values(report.checks).every(Boolean);
    fs.writeFileSync(path.join(root, 'research/2026-10-20-reference-workflow-ui-acceptance.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report, null, 2));
    if (!report.passed) process.exitCode = 1;
  }
})();
