/* Separate normal/design drafts in one physical composer; synthetic inference and ASR. */
const fs = require('node:fs'), path = require('node:path'), { chromium } = require('playwright');
const root = path.resolve(__dirname, '..'), base = process.env.DRAMA_UI_URL;
const folder = process.env.DRAMA_STUDIO_FOLDER;
const report = { scope: 'Voice-design workspace: one physical editor and submit, independent normal/design scripts and directions, Back navigation, reference-free requests, draft/history recovery, four-language responsive UI; synthetic inference and ASR only.',
  checks: {}, errors: [], screenshots: [], requests: [] };
(async () => {
  let browser, page;
  try {
    browser = await chromium.launch({ channel: 'chrome', headless: true });
    const context = await browser.newContext({ viewport: { width: 1440, height: 960 }, locale: 'zh-CN' });
    await context.addInitScript(() => localStorage.setItem('drama3.ui.language', 'zh-CN'));
    page = await context.newPage();
    page.on('pageerror', error => report.errors.push(error.stack || error.message));
    const submissions = [];
    let referenceUploads = 0;
    page.on('request', request => {
      if (request.method() === 'POST' && new URL(request.url()).pathname === '/api/reference-audio') referenceUploads++;
      if (request.method() === 'POST' && new URL(request.url()).pathname === '/api/jobs') {
        const value = request.postDataJSON(); submissions.push(value);
        report.requests.push({ mode: value.mode, text: value.text, direction: value.direction,
          voice_description: value.voice_description, voice_ids: value.voice_ids,
          voice_bindings: value.voice_bindings, uploaded_references: value.uploaded_references,
          condition_source: value.condition_source, model: value.model });
      }
    });
    let pendingAsr = false, asrCalls = 0, asrAvailable = false;
    await page.route('**/api/transcriptions', route => {
      asrCalls++;
      return route.fulfill({ json: !asrAvailable ? { id: 'voice-design-asr', status: 'skipped', reason: 'model_unavailable' } : pendingAsr
        ? { id: 'voice-design-pending-asr', status: 'running', stage: 'transcribing' }
        : { id: 'voice-design-complete-asr', status: 'complete', result: { text: '自动转写应当只影响参考音频。' } } });
    });
    await page.route('**/api/transcriptions/voice-design-pending-asr', route => {
      return route.fulfill({ json: pendingAsr
        ? { id: 'voice-design-pending-asr', status: 'running', stage: 'transcribing' }
        : { id: 'voice-design-pending-asr', status: 'complete', result: { text: '自动转写应当只影响参考音频。' } } });
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
    const editor = () => page.locator('.compose-panel > .editor-area .cm-content');
    const editorHost = () => page.locator('.compose-panel > .editor-area .script-editor');
    const assertUndoIsolation = async (name, expected) => {
      await editor().press('Control+z'); await settle();
      check(`${name}_keyboard_undo_does_not_restore_other_workspace`, await page.inputValue('#text') === expected);
      await editorHost().locator('[data-label="Undo"]').click(); await settle();
      check(`${name}_toolbar_undo_does_not_restore_other_workspace`, await page.inputValue('#text') === expected);
      await editorHost().locator('[data-label="Source text"]').click();
      await page.locator('#text').press('Control+z'); await settle();
      check(`${name}_raw_keyboard_undo_does_not_restore_other_workspace`, await page.inputValue('#text') === expected);
      await editorHost().locator('[data-label="Source text"]').click(); await settle();
      check(`${name}_raw_return_keeps_rich_editor_in_sync`, await editor().isVisible()
        && await page.inputValue('#text') === expected
        && await page.evaluate(expected => ScriptEditor.get(document.getElementById('text')).view.state.doc.toString() === expected, expected));
    };
    const fillScript = async value => {
      await editor().fill(value);
      await page.waitForFunction(value => document.getElementById('text').value === value
        && ScriptEditor.get(document.getElementById('text')).view.state.doc.toString() === value, value);
    };
    const activeDesign = () => page.evaluate(() => StudioFlow.voiceDesignActive());
    const setDesign = async value => {
      if (await activeDesign() !== value) await page.locator(value ? '#describe-voice' : '#back-to-generation').click();
      await page.waitForFunction(value => StudioFlow.voiceDesignActive() === value, value);
      await settle();
    };
    const choose = async (id, value) => {
      await page.locator(`#${id}-trigger`).click();
      await page.locator(`#${id}-listbox [data-value=${JSON.stringify(value)}]`).click();
    };
    const navigate = async view => { await page.locator(`.nav[data-view="${view}"]`).click(); await settle(); };
    const shot = async name => {
      const file = path.join(folder, name + '.png'); await page.screenshot({ path: file }); report.screenshots.push(file);
    };
    let latest = '';
    const finished = async previous => {
      await page.waitForFunction(old => {
        const id = StudioFlow.flow.lastSubmit?.id;
        return id && id !== old && document.getElementById('result-box').dataset.jobId === id
          && document.getElementById('job-badge').dataset.status === 'complete';
      }, previous ?? latest);
      latest = await page.locator('#result-box').getAttribute('data-job-id');
      return api('/api/jobs/' + latest);
    };
    const generate = async () => {
      const before = latest; await page.locator('#submit-job').click(); return finished(before);
    };
    const restoreHistory = async id => {
      await page.evaluate(() => loadJobs()); await navigate('takes');
      const card = page.locator(`.job-card[data-job-id="${id}"]`);
      await card.locator('.history-actions > summary').click();
      await card.locator('.restore-job').click();
      await page.locator('#view-generate.active').waitFor(); await settle();
    };
    const requestHasNoReferences = payload => ['voice_ids', 'voice_bindings', 'uploaded_references', 'reference_snapshot',
      'previous_take', 'previous_sources', 'source_take'].every(key => !payload[key]?.length);
    const noIndependentDesignControls = () => page.locator('#design-audition,#design-direction,#generate-designed-voice').count().then(count => count === 0);
    const noRetiredCommands = () => page.locator('#toggle-text-stream,#exit-text-stream,#text-stream-status,'
      + '#view-generate [data-i18n="Tools"],#view-generate [data-i18n="New draft"],'
      + '#view-generate [data-i18n="New with this voice"],#view-scene [data-i18n="New draft"]').count().then(count => count === 0);

    await page.goto(base, { waitUntil: 'networkidle' }); await ready();
    check('independent_audition_direction_and_submit_are_removed', await noIndependentDesignControls());
    check('voice_design_starts_as_an_explicit_unselected_source', !await activeDesign() && !await page.locator('#voice-design').isVisible());
    check('removed_new_creation_and_tools_commands_do_not_exist', await noRetiredCommands());
    check('ordinary_composer_has_no_redundant_heading_or_divider', !await page.locator('.compose-panel > .panel-head').isVisible());
    await page.evaluate(() => StudioFlow.save());
    check('successful_autosave_has_no_permanent_saved_badge', !await page.locator('#draft-status:visible').count());
    let rejectSave = true;
    const rejectDraftOnce = route => {
      if (rejectSave) { rejectSave = false; return route.fulfill({ status: 500, json: { error: 'Fixture draft save failed.' } }); }
      return route.continue();
    };
    await page.route('**/api/studio/draft', rejectDraftOnce);
    await page.evaluate(() => StudioFlow.save());
    check('autosave_failure_remains_visible_without_a_saved_badge', await page.locator('#studio-error').isVisible()
      && (await page.locator('#studio-error').textContent()).includes('Fixture draft save failed.')
      && !await page.locator('#draft-status:visible').count());
    await page.evaluate(() => StudioFlow.save()); await page.unroute('**/api/studio/draft', rejectDraftOnce);
    check('successful_retry_clears_only_the_recovered_save_failure', !await page.locator('#studio-error').isVisible()
      && !await page.locator('#draft-status:visible').count());
    await page.evaluate(() => document.dispatchEvent(new CustomEvent('drama:editor-notice', { detail: 'Fixture independent editor notice.' })));
    await page.evaluate(() => StudioFlow.save());
    check('successful_autosave_does_not_dismiss_an_unrelated_editor_error', await page.locator('#studio-error').isVisible()
      && (await page.locator('#studio-error').textContent()).includes('Fixture independent editor notice.'));
    const voices = (await api('/api/voices')).voices;
    await navigate('voices');
    await page.locator(`#voice-list .use-voice[data-id="${voices[0].id}"]`).click();
    await page.locator('#view-generate.active').waitFor();
    await fillScript('这是普通生成自己的台词。');
    await page.locator('#direction').fill('轻松自然，末尾稍作停顿。');
    await page.locator('#advanced-settings > summary').click();
    await page.locator('#temperature').fill('.71'); await page.locator('#top-p').fill('.89');
    await page.locator('#advanced-settings > summary').click();
    const physicalEditor = await editor().elementHandle();
    const beforeToggle = (await api('/api/jobs')).jobs.length;
    await setDesign(true);
    check('selecting_design_does_not_submit_a_generation', (await api('/api/jobs')).jobs.length === beforeToggle && submissions.length === 0);
    check('first_voice_design_does_not_copy_normal_text_or_direction', await page.inputValue('#text') === ''
      && await page.inputValue('#direction') === '');
    await assertUndoIsolation('entering_empty_design', '');
    check('design_keeps_shared_sampling_settings', Number(await page.inputValue('#temperature')) === .71
      && Number(await page.inputValue('#top-p')) === .89);
    check('design_title_replaces_the_hidden_ordinary_heading', await page.locator('.compose-panel > .panel-head').isVisible()
      && await page.locator('.compose-panel > .panel-head h2').textContent() === await page.evaluate(() => t('Voice design')));
    check('design_removes_redundant_source_row', !await page.locator('.voice-source-bar').isVisible()
      && !await page.locator('#describe-voice').isVisible());
    check('design_back_is_in_top_right_header', await page.locator('#back-to-generation').isVisible()
      && await page.locator('#back-to-generation').evaluate(node => {
        const head = node.closest('.panel-head'), title = head?.querySelector('h2');
        return Boolean(head && title && node.getBoundingClientRect().left > title.getBoundingClientRect().right);
      }));
    check('design_reveals_only_its_voice_description', await page.locator('#voice-description').isVisible()
      && await noIndependentDesignControls() && !await page.locator('#voice-design .script-editor,#voice-design button').count());
    check('design_hides_upload_and_selected_reference_controls', !await page.locator('#clone-upload-zone').isVisible()
      && !await page.locator('#clone-upload-preview').isVisible()
      && !await page.locator('#selected-reference-details').isVisible()
      && !await page.locator('#additional-references').isVisible());
    check('design_reuses_one_physical_editor_direction_and_submit', await page.locator('#view-generate .editor-area .script-editor:visible').count() === 1
      && await page.locator('#direction').isVisible() && await page.locator('#submit-job').isVisible()
      && await page.locator('#view-generate label[for="direction"]:visible').count() === 1
      && await physicalEditor.evaluate(node => node === document.querySelector('.compose-panel > .editor-area .cm-content')));
    await fillScript('按描述设计的声音，用这段台词。');
    await page.locator('#direction').fill('清晰坚定，语速稍慢。');
    await page.locator('#voice-description').fill('   ');
    if (await page.locator('#submit-job').isEnabled()) await page.locator('#submit-job').click();
    await settle();
    check('description_is_required_before_design_can_submit', submissions.length === 0
      && (await page.locator('#submit-job').isDisabled() || await page.locator('#result-error').isVisible() || await page.locator('#studio-error').isVisible()));
    await page.locator('#voice-description').fill('年轻女声，清亮，带一点沙哑。');
    const designed = await generate(), designRequest = submissions.at(-1);
    check('shared_submit_builds_design_mode_request', designed.mode === 'design' && designRequest.mode === 'design');
    check('active_design_script_and_direction_reach_design_recipe', designed.recipe.text === '按描述设计的声音，用这段台词。'
      && designed.recipe.direction === '清晰坚定，语速稍慢。'
      && designed.recipe.temperature === .71 && designed.recipe.top_p === .89);
    check('voice_description_reaches_design_recipe', designed.recipe.voice_description === '年轻女声，清亮，带一点沙哑。');
    check('selected_library_voice_is_not_sent_in_design_mode', requestHasNoReferences(designRequest)
      && designed.recipe.reference_snapshot.length === 0);
    check('reference_free_design_does_not_start_phantom_uploads', referenceUploads === 0);
    check('design_result_keeps_main_playback_download_and_edit', await page.locator('.main-result .studio-player').count() === 1
      && await page.locator('.main-result .download-link').isVisible() && await page.locator('.main-result .creative-edit').isVisible());
    await setDesign(false);
    check('leaving_design_restores_previous_library_selection', await page.evaluate(id => selectedVoices().includes(id), voices[0].id)
      && await page.locator('#selected-reference-details').isVisible());
    check('back_restores_normal_script_and_direction_without_overwriting_design', await page.inputValue('#text') === '这是普通生成自己的台词。'
      && await page.inputValue('#direction') === '轻松自然，末尾稍作停顿。'
      && await page.inputValue('#voice-description') === '年轻女声，清亮，带一点沙哑。');
    await assertUndoIsolation('back_to_normal_after_design_edits', '这是普通生成自己的台词。');
    check('normal_source_returns_without_heading_or_back', !await page.locator('.compose-panel > .panel-head').isVisible()
      && await page.locator('#describe-voice').isVisible()
      && !await page.locator('#back-to-generation').isVisible() && await page.locator('.voice-source-bar').isVisible()
      && !await page.locator('#draft-status:visible').count());
    const ordinary = await generate();
    check('regular_submit_after_design_uses_original_library_voice', ordinary.mode === 'tts'
      && ordinary.recipe.reference_snapshot.length === 1 && !submissions.at(-1).voice_description
      && ordinary.recipe.text === '这是普通生成自己的台词。' && ordinary.recipe.direction === '轻松自然，末尾稍作停顿。');
    await setDesign(true);
    check('reentering_design_recovers_its_independent_script_and_direction', await page.inputValue('#text') === '按描述设计的声音，用这段台词。'
      && await page.inputValue('#direction') === '清晰坚定，语速稍慢。');
    await setDesign(false);

    await page.locator('#selected-reference-details .selected-voice-clear').click();
    await page.locator('#clone-audio').setInputFiles(process.env.DRAMA_STUDIO_REFERENCE);
    await page.waitForFunction(() => state.cloneUpload && !state.cloneAsrPromise);
    await page.locator('#clone-transcript').fill('这份上传参考音频应当完整保留。');
    await page.waitForFunction(() => Boolean(state.cloneFile));
    await page.evaluate(() => StudioFlow.save());
    const uploadId = await page.evaluate(() => state.cloneUpload.id);
    const beforeUploadToggle = submissions.length, uploadsBeforeToggle = referenceUploads;
    await setDesign(true); await setDesign(false);
    check('upload_source_round_trip_does_not_generate', submissions.length === beforeUploadToggle);
    check('source_round_trip_does_not_reupload_unchanged_audio', referenceUploads === uploadsBeforeToggle);
    check('uploaded_audio_and_transcript_survive_design_round_trip', await page.evaluate(id => state.cloneUpload?.id === id, uploadId)
      && await page.inputValue('#clone-transcript') === '这份上传参考音频应当完整保留。'
      && await page.locator('#clone-upload-preview').isVisible());
    check('missing_asr_is_silently_skipped', asrCalls === 1 && !await page.locator('#clone-asr-status').isVisible());

    pendingAsr = asrAvailable = true;
    await page.locator('#transcribe-clone').click();
    await page.waitForFunction(() => Boolean(state.cloneAsrPromise && state.cloneAsrTask));
    await page.locator('#clone-transcript').fill('');
    await fillScript('普通模式的参考转写还在进行。');
    await page.locator('#direction').fill('普通说话自然。');
    await setDesign(true);
    check('reference_asr_does_not_copy_ordinary_words_into_design', await page.inputValue('#text') === '按描述设计的声音，用这段台词。'
      && await page.inputValue('#direction') === '清晰坚定，语速稍慢。');
    await fillScript('参考转写还没完成，也能直接设计声音。');
    await page.locator('#direction').fill('语速适中，清楚地说。');
    const beforePendingDesign = submissions.length;
    await page.locator('#submit-job').click();
    await page.waitForFunction(() => StudioFlow.flow.lastSubmit?.payload.mode === 'design'
      && StudioFlow.flow.lastSubmit.payload.text === '参考转写还没完成，也能直接设计声音。', null, { timeout: 6000 });
    const notWaiting = await finished();
    check('design_submission_does_not_wait_for_unrelated_reference_asr', pendingAsr && asrCalls > 0
      && submissions.length === beforePendingDesign + 1 && notWaiting.status === 'complete');
    check('unfinished_uploaded_reference_is_not_sent_in_design_request', requestHasNoReferences(submissions.at(-1))
      && notWaiting.recipe.reference_snapshot.length === 0);
    check('pending_asr_does_not_replace_target_script', await page.inputValue('#text') === '参考转写还没完成，也能直接设计声音。');
    pendingAsr = false;
    await setDesign(false);
    check('back_after_pending_asr_design_restores_normal_draft', await page.inputValue('#text') === '普通模式的参考转写还在进行。'
      && await page.inputValue('#direction') === '普通说话自然。');
    await page.waitForFunction(() => !state.cloneAsrPromise);
    if (!(await page.inputValue('#clone-transcript')).trim()) await page.locator('#clone-transcript').fill('自动转写应当只影响参考音频。');
    const retainedTranscript = await page.inputValue('#clone-transcript');
    await setDesign(true);
    await page.evaluate(() => StudioFlow.save());
    const saved = (await api('/api/studio')).drafts.single.value;
    check('saved_draft_keeps_normal_fields_canonical_and_design_fields_separate',
      saved.form.text === '普通模式的参考转写还在进行。' && saved.form.direction === '普通说话自然。'
      && saved.design.text === '参考转写还没完成，也能直接设计声音。'
      && saved.design.direction === '语速适中，清楚地说。' && saved.design.active);
    await page.reload({ waitUntil: 'networkidle' }); await ready();
    check('saved_design_draft_recovers_selected_source_and_description', await activeDesign()
      && await page.locator('#voice-description').isVisible() && await page.inputValue('#voice-description') === '年轻女声，清亮，带一点沙哑。');
    check('saved_design_draft_recovers_its_editor_and_direction', await page.inputValue('#text') === saved.design.text
      && await page.inputValue('#direction') === saved.design.direction && await noIndependentDesignControls());
    check('recovered_design_still_hides_reference_ui', !await page.locator('#clone-upload-preview').isVisible()
      && !await page.locator('#clone-upload-zone').isVisible());
    await setDesign(false);
    check('recovered_hidden_upload_remains_available_when_returning_to_reference', await page.evaluate(id => state.cloneUpload?.id === id, uploadId)
      && await page.inputValue('#clone-transcript') === retainedTranscript && await page.locator('#clone-upload-preview').isVisible());
    check('reload_preserves_the_inactive_normal_script_and_direction', await page.inputValue('#text') === '普通模式的参考转写还在进行。'
      && await page.inputValue('#direction') === '普通说话自然。');
    await page.evaluate(() => StudioFlow.save());
    await page.reload({ waitUntil: 'networkidle' }); await ready();
    check('normal_reload_keeps_normal_source_and_normal_script', !await activeDesign()
      && await page.inputValue('#text') === '普通模式的参考转写还在进行。');
    await setDesign(true);
    check('normal_reload_also_preserves_the_inactive_design_script', await page.inputValue('#text') === '参考转写还没完成，也能直接设计声音。'
      && await page.inputValue('#direction') === '语速适中，清楚地说。');
    await setDesign(false);

    let heldUpload;
    const delayUpload = route => { heldUpload = route; };
    await page.route('**/api/reference-audio', delayUpload);
    await page.locator('#clone-audio').setInputFiles(process.env.DRAMA_STUDIO_REFERENCE);
    await page.waitForFunction(() => Boolean(state.cloneUploadPromise && !state.cloneUpload));
    await setDesign(true);
    await fillScript('上传尚未完成，不应阻挡无参考的声音设计。');
    const beforeUploadingDesign = submissions.length;
    await page.locator('#submit-job').click();
    await page.waitForFunction(() => StudioFlow.flow.lastSubmit?.payload.text === '上传尚未完成，不应阻挡无参考的声音设计。',
      null, { timeout: 6000 });
    const withoutUploadWait = await finished();
    check('design_submission_does_not_wait_for_unrelated_pending_upload', Boolean(heldUpload)
      && await page.evaluate(() => Boolean(state.cloneUploadPromise && !state.cloneUpload))
      && submissions.length === beforeUploadingDesign + 1 && withoutUploadWait.status === 'complete');
    check('pending_upload_does_not_leak_a_reference_into_design', requestHasNoReferences(submissions.at(-1))
      && withoutUploadWait.recipe.reference_snapshot.length === 0);
    await heldUpload.continue();
    await page.unroute('**/api/reference-audio', delayUpload);
    await page.waitForFunction(() => Boolean(state.cloneUpload));

    const historical = await api('/api/jobs', { mode: 'design', text: '历史声音设计的实际正文。',
      voice_description: 'A bright mature voice.', direction: 'Steady and reassuring.', character_name: 'Narrator',
      scene: 'Quiet room.', seed: 321, max_new_tokens: 192, temperature: .83, top_p: .91, top_k: 21,
      studio: { work_id: 'design-history', segment_id: 'main', layout: 'single' } });
    await page.waitForFunction(async id => {
      const response = await fetch('/api/jobs/' + id); return (await response.json()).status === 'complete';
    }, historical.id);
    const historicalRecord = await api('/api/jobs/' + historical.id);
    await restoreHistory(historical.id);
    check('history_design_restores_as_the_shared_design_composer', await activeDesign()
      && await page.inputValue('#text') === '历史声音设计的实际正文。'
      && await page.inputValue('#direction') === 'Steady and reassuring.'
      && await page.inputValue('#voice-description') === 'A bright mature voice.');
    check('history_design_restores_sampling_without_duplicate_editor', Number(await page.inputValue('#temperature')) === .83
      && Number(await page.inputValue('#top-p')) === .91 && Number(await page.inputValue('#top-k')) === 21
      && await noIndependentDesignControls());
    await setDesign(false);
    check('restoring_design_history_does_not_overwrite_normal_draft', await page.inputValue('#text') === '普通模式的参考转写还在进行。'
      && await page.inputValue('#direction') === '普通说话自然。');
    check('back_keeps_restored_shared_sampling_instead_of_restoring_old_mode_settings', Number(await page.inputValue('#temperature')) === .83
      && Number(await page.inputValue('#top-p')) === .91);
    await setDesign(true);
    check('returning_to_restored_design_keeps_its_historical_words', await page.inputValue('#text') === '历史声音设计的实际正文。'
      && await page.inputValue('#direction') === 'Steady and reassuring.');
    const restoredDesign = await generate(), restoredRequest = submissions.at(-1);
    check('restored_design_keeps_source_model_and_character_metadata', restoredRequest.model === historicalRecord.recipe.model
      && restoredRequest.character_name === 'Narrator' && restoredRequest.scene === 'Quiet room.'
      && restoredRequest.condition_source?.job_id === historical.id);
    check('restored_design_remains_reference_free', restoredDesign.mode === 'design' && requestHasNoReferences(restoredRequest)
      && restoredDesign.recipe.reference_snapshot.length === 0);
    await restoreHistory(ordinary.id);
    check('ordinary_history_restore_exits_design_and_restores_references', !await activeDesign()
      && !await page.locator('#voice-design').isVisible() && await page.locator('#clone-upload-preview').isVisible()
      && await page.inputValue('#text') === ordinary.recipe.text);
    await setDesign(true);
    check('ordinary_history_restore_does_not_overwrite_design_draft', await page.inputValue('#text') === '历史声音设计的实际正文。'
      && await page.inputValue('#direction') === 'Steady and reassuring.');
    check('design_reentry_keeps_shared_settings_from_ordinary_history', Number(await page.inputValue('#temperature')) === ordinary.recipe.temperature
      && Number(await page.inputValue('#top-p')) === ordinary.recipe.top_p);
    await navigate('voices'); await page.locator(`#voice-list .use-voice[data-id="${voices[1].id}"]`).click();
    await page.locator('#view-generate.active').waitFor();
    check('use_library_voice_explicitly_returns_to_regular_generation', !await activeDesign()
      && await page.evaluate(id => selectedVoices().includes(id), voices[1].id)
      && !await page.locator('#voice-design').isVisible()
      && await page.inputValue('#text') === ordinary.recipe.text && await page.inputValue('#direction') === ordinary.recipe.direction);
    await fillScript('从音色库明确选择的声音。');
    const selectedVoiceJob = await generate();
    check('library_selection_does_not_reuse_design_metadata_or_source', selectedVoiceJob.mode === 'tts'
      && selectedVoiceJob.recipe.reference_snapshot.length === 1 && !submissions.at(-1).voice_description
      && submissions.at(-1).condition_source?.job_id !== historical.id);
    await fillScript('普通生成的草稿应当独立保存。');
    await page.locator('#direction').fill('正常、自然。'); await setDesign(true);
    await fillScript('同一个编辑器，声音设计使用自己的草稿。');
    await page.locator('#direction').fill('设计声音的演绎要求。');
    await page.locator('#voice-description').fill('柔和、明亮的年轻声音。');
    const beforeResponsive = submissions.length;
    for (const locale of ['zh-CN', 'en', 'ja', 'ko']) {
      await page.evaluate(locale => I18n.setLocale(locale), locale);
      for (const size of [{ width: 1440, height: 960 }, { width: 390, height: 844 }]) {
        await page.setViewportSize(size); await page.evaluate(() => scrollTo(0, 0)); await settle();
        check(`design_workspace_fits_${locale}_${size.width}`, await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
        check(`design_has_one_editor_and_direction_${locale}_${size.width}`, await page.locator('#view-generate .editor-area .script-editor:visible').count() === 1
          && await page.locator('#direction').isVisible() && await noIndependentDesignControls());
        check(`description_precedes_direction_and_script_${locale}_${size.width}`, await page.locator('#voice-description').evaluate(node =>
          Boolean(node.compareDocumentPosition(document.getElementById('direction')) & Node.DOCUMENT_POSITION_FOLLOWING)
          && Boolean(node.compareDocumentPosition(document.getElementById('text')) & Node.DOCUMENT_POSITION_FOLLOWING)));
        check(`design_submit_remains_visible_${locale}_${size.width}`, await page.locator('#submit-job').evaluate(node => {
          const r = node.getBoundingClientRect(); return r.left >= 0 && r.right <= innerWidth && r.top >= 0 && r.bottom <= innerHeight;
        }));
        check(`description_label_is_localized_${locale}_${size.width}`, await page.locator('label[for="voice-description"]').textContent()
          === await page.evaluate(() => t('Voice description')));
        check(`design_title_and_back_are_localized_${locale}_${size.width}`, await page.locator('.compose-panel > .panel-head h2').textContent()
          === await page.evaluate(() => t('Voice design')) && await page.locator('#back-to-generation').isVisible()
          && await page.locator('.compose-panel > .panel-head').isVisible()
          && !await page.locator('.voice-source-bar').isVisible() && await noRetiredCommands()
          && !await page.locator('#draft-status:visible').count());
        check(`reference_upload_hidden_in_design_${locale}_${size.width}`, !await page.locator('#clone-upload-zone').isVisible()
          && !await page.locator('#clone-upload-preview').isVisible());
        await setDesign(false);
        check(`normal_creation_returns_${locale}_${size.width}`, !await page.locator('#voice-design').isVisible()
          && await page.locator('#selected-reference-details .selected-library-voice').isVisible()
          && !await page.locator('#clone-upload-zone').isVisible() && await page.inputValue('#text') === '普通生成的草稿应当独立保存。'
          && await page.inputValue('#direction') === '正常、自然。'
          && !await page.locator('.compose-panel > .panel-head').isVisible()
          && !await page.locator('#draft-status:visible').count());
        await setDesign(true);
        check(`design_draft_returns_independently_${locale}_${size.width}`, await page.inputValue('#text') === '同一个编辑器，声音设计使用自己的草稿。'
          && await page.inputValue('#direction') === '设计声音的演绎要求。');
        if (locale === 'zh-CN') await shot(`voice-design-workspace-${size.width}`);
      }
    }
    check('layout_language_and_source_switches_never_submit_generation', submissions.length === beforeResponsive);
    check('no_browser_exceptions', report.errors.length === 0);
  } catch (error) {
    report.errors.push(error.stack || String(error));
    if (page) {
      const file = path.join(folder, 'voice-design-failure.png');
      await page.screenshot({ path: file, fullPage: true }).catch(() => {}); report.screenshots.push(file);
    }
  } finally {
    await browser?.close();
    report.passed = !report.errors.length && Object.values(report.checks).every(Boolean);
    fs.writeFileSync(path.join(root, 'research/2026-10-20-voice-design-ui-acceptance.json'), JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify(report, null, 2)); if (!report.passed) process.exitCode = 1;
  }
})();
