/* Current product flow, real HTTP/store/audio. Inference and ASR are explicit fixtures. */
const fs = require('node:fs'), path = require('node:path'), { chromium } = require('playwright');
const root = path.resolve(__dirname, '..'), base = process.env.DRAMA_UI_URL, folder = process.env.DRAMA_STUDIO_FOLDER;
const report = { scope: 'Simplified product UI: real uploads, requests, references, history, dialogue, streaming and recovery; synthetic inference and ASR.',
  checks: {}, errors: [], screenshots: [] };
(async () => {
  let browser, page;
  try {
    browser = await chromium.launch({ channel: 'chrome', headless: true });
    const context = await browser.newContext({ viewport: { width: 1440, height: 960 }, locale: 'zh-CN' });
    page = await context.newPage(); page.on('pageerror', error => report.errors.push(error.stack || error.message));
    await page.addInitScript(() => {
      window.initialPageFrames = [];
      const record = () => {
        if (document.documentElement?.dataset.studioReady === 'true') return;
        const styled = [...document.styleSheets].some(sheet => sheet.href?.endsWith('/assets/app.css'));
        if (styled) {
          const visible = [...document.querySelectorAll('.view')]
            .filter(node => getComputedStyle(node).display !== 'none'
              && getComputedStyle(node).visibility === 'visible' && node.getClientRects().length)
            .map(node => node.id);
          window.initialPageFrames.push(visible);
        }
        requestAnimationFrame(record);
      };
      requestAnimationFrame(record);
    });
    let asrCalls = 0, asrAvailable = false;
    await page.route('**/api/transcriptions', route => {
      asrCalls++;
      return route.fulfill({ json: asrAvailable
        ? { id: 'fixture-asr', status: 'complete', result: { text: '参考音频中的准确台词。' } }
        : { id: 'fixture-asr', status: 'skipped', reason: 'model_unavailable' } });
    });
    const api = async (url, data) => {
      const response = data === undefined ? await context.request.get(base + url)
        : await context.request.post(base + url, { data });
      if (!response.ok()) throw new Error(url + ': ' + await response.text());
      return response.json();
    };
    const check = (name, value) => { report.checks[name] = Boolean(value); if (!value) throw new Error(name); };
    const ready = () => page.waitForFunction(() => document.documentElement.dataset.studioReady === 'true');
    const choose = async (id, value) => {
      await page.locator(`#${id}-trigger`).click();
      await page.locator(`#${id}-listbox [data-value=${JSON.stringify(value)}]`).click();
    };
    const editor = () => page.locator('.compose-panel > .editor-area .cm-content');
    let latest = '';
    const finished = async (previous = latest) => {
      await page.waitForFunction(old => {
        const id = StudioFlow.flow.lastSubmit?.id;
        return id && id !== old && document.querySelector('#result-box').dataset.jobId === id
          && document.querySelector('#job-badge').dataset.status === 'complete';
      }, previous);
      latest = await page.locator('#result-box').getAttribute('data-job-id');
      return api('/api/jobs/' + latest);
    };
    const show = async id => {
      await page.evaluate(id => { view('generate'); monitorJob(id); }, id);
      await page.waitForFunction(id => document.querySelector('#result-box').dataset.jobId === id, id);
    };
    const noRetiredUi = async () => !await page.locator(
      '#candidate-section,#candidate-list,#work-parts,#export-work,#composition-result,#job-log,#job-events,#job-timing-details,.creative-adopt,.context-update,[data-edit-mode="continue"]'
    ).count();
    const shot = async name => {
      const file = path.join(folder, name + '.png'); await page.screenshot({ path: file });
      report.screenshots.push(file);
    };
    await page.goto(base, { waitUntil: 'networkidle' }); await ready();
    check('studio_initializes_without_error', !await page.locator('#studio-error').isVisible());
    check('first_open_defaults_to_generation', await page.locator('#view-generate').isVisible()
      && await page.evaluate(() => state.view === 'generate'));
    await editor().fill('刷新页面也不能丢失的生成草稿。');
    await page.locator('#direction').fill('保持当前演绎要求。');
    await page.locator('.nav[data-view="scene"]').click();
    await page.locator('.dialogue-editor-area .cm-content').fill('S1:刷新前的第一段。\nS2:刷新前的第二段。');
    await page.evaluate(() => StudioFlow.save());
    for (const viewName of ['scene', 'voices', 'takes', 'settings', 'generate']) {
      await page.locator(`.nav[data-view="${viewName}"]`).click();
      let releaseApp;
      const gate = new Promise(resolve => { releaseApp = resolve; });
      const holdApp = async route => { await gate; await route.continue(); };
      await page.route('**/assets/app.js', holdApp);
      const reloading = page.reload({ waitUntil: 'networkidle' });
      try {
        await page.waitForFunction(() => typeof window.restoreView === 'undefined'
          && document.querySelectorAll('.view').length === 5 && window.initialPageFrames?.length >= 2);
        check(`correct_route_and_boot_state_before_application_script_${viewName}`, await page.evaluate(name => {
          const visible = [...document.querySelectorAll('.view')]
            .filter(node => getComputedStyle(node).display !== 'none'
              && getComputedStyle(node).visibility === 'visible').map(node => node.id);
          const nav = document.querySelector(`.nav[data-view="${name}"]`);
          const status = document.querySelector('#startup-state');
          return visible.length === 0 && document.documentElement.dataset.view === name
            && getComputedStyle(status).display === 'flex'
            && getComputedStyle(nav).backgroundColor === 'rgb(255, 241, 235)'
            && document.querySelector('#workspace-content').inert;
        }, viewName));
      } finally {
        releaseApp(); await reloading; await page.unroute('**/assets/app.js', holdApp);
      }
      await ready();
      check(`no_homepage_or_other_page_in_any_initial_frame_${viewName}`, await page.evaluate(name =>
        window.initialPageFrames.length >= 2
          && window.initialPageFrames.every(frame => frame.length === 0 || frame.length === 1 && frame[0] === `view-${name}`), viewName));
      check(`refresh_restores_the_current_page_${viewName}`, await page.locator(`#view-${viewName}`).isVisible()
        && await page.evaluate(name => state.view === name
          && document.querySelector(`.nav[data-view="${name}"]`).getAttribute('aria-current') === 'page'
          && document.querySelectorAll('.view.active').length === 1
          && document.querySelector('#view-title').textContent === t(viewTitles[name]), viewName));
      check(`page_refresh_keeps_both_writing_drafts_${viewName}`,
        await page.inputValue('#text') === '刷新页面也不能丢失的生成草稿。'
        && await page.inputValue('#direction') === '保持当前演绎要求。'
        && await page.inputValue('#dialogue-script') === 'S1:刷新前的第一段。\nS2:刷新前的第二段。');
      check(`restored_page_has_the_correct_result_container_${viewName}`, await page.evaluate(name =>
        document.querySelector('.monitor-panel').parentElement.matches(name === 'scene' ? '.scene-grid' : '.generation-grid'), viewName));
    }
    await page.locator('.nav[data-view="settings"]').click();
    const otherTab = await context.newPage();
    await otherTab.goto(base, { waitUntil: 'networkidle' });
    await otherTab.waitForFunction(() => document.documentElement.dataset.studioReady === 'true');
    check('separate_tabs_do_not_override_each_others_current_page',
      await otherTab.locator('#view-generate').isVisible() && await page.locator('#view-settings').isVisible());
    await otherTab.close();
    await page.evaluate(() => sessionStorage.setItem('drama3.ui.view', 'retired-edit-screen'));
    await page.reload({ waitUntil: 'networkidle' }); await ready();
    check('invalid_stored_page_falls_back_to_a_usable_homepage',
      await page.locator('#view-generate').isVisible() && !await page.locator('#studio-error').isVisible());
    check('retired_version_and_log_sections_are_removed_not_hidden', await noRetiredUi());
    check('normal_creation_omits_redundant_compose_header', !await page.locator('.compose-panel > .panel-head').isVisible());
    await page.evaluate(() => StudioFlow.save());
    check('autosave_does_not_add_a_permanent_success_status', !await page.locator('#draft-status:visible').count());
    await page.locator('.nav[data-view="settings"]').click();
    check('settings_remove_capability_listing_instead_of_hiding_it',
      !await page.locator('#capability-list,[data-i18n="Engine gates"]').count());
    check('local_machine_details_are_expanded_by_default', await page.locator('#runtime-details').evaluate(node => node.open)
      && await page.locator('#hardware-card').isVisible() && await page.locator('#hardware-card .metric').count() > 0);
    check('local_machine_uses_the_full_settings_width', await page.locator('.settings-grid').evaluate(node => {
      const pane = node.getBoundingClientRect(), content = node.firstElementChild.getBoundingClientRect();
      return Math.abs(pane.width - content.width) < 1;
    }));
    await page.locator('#runtime-details > summary').click();
    check('local_machine_can_still_be_collapsed', !await page.locator('#hardware-card').isVisible());
    await page.locator('#runtime-details > summary').click();
    check('local_machine_can_be_reopened', await page.locator('#hardware-card').isVisible());
    check('capability_gating_is_still_loaded', await page.evaluate(() => Boolean(state.capabilities?.ordinary_tts
      && state.capabilities?.voice_cloning && state.capabilities?.infill)));
    for (const size of [{ width: 1440, height: 900 }, { width: 390, height: 844 }]) {
      await page.setViewportSize(size);
      for (const [viewName, submit] of [['generate', 'submit-job'], ['scene', 'submit-scene'], ['voices', 'add-voice']]) {
        await page.locator(`.nav[data-view="${viewName}"]`).click();
        check(`primary_action_in_view_${viewName}_${size.width}`, await page.locator(`#${submit}`).evaluate(node => {
          const rect = node.getBoundingClientRect();
          return rect.top >= 0 && rect.bottom <= innerHeight && rect.left >= 0 && rect.right <= innerWidth;
        }));
      }
      for (const viewName of ['takes', 'settings']) {
        await page.locator(`.nav[data-view="${viewName}"]`).click();
        check(`view_has_no_horizontal_overflow_${viewName}_${size.width}`,
          await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
      }
    }
    await page.setViewportSize({ width: 1440, height: 960 });
    await page.locator('.nav[data-view="generate"]').click();
    await editor().fill('大家好，欢迎来找我玩。'); await editor().press('End'); await editor().press('ArrowLeft');
    await editor().press('Shift+ArrowLeft'); await editor().press('Shift+ArrowLeft');
    await page.locator('.compose-panel > .editor-area [data-label="Tags"]').click();
    await page.locator('.compose-panel > .editor-area [data-tag="whisper"][data-kind="scope"]').click();
    check('tag_shortcut_wraps_only_selected_words', await page.inputValue('#text') === '大家好，欢迎来找<whisper>我玩</whisper>。'
      && await page.locator('.compose-panel .script-scope').isVisible());
    await page.locator('.compose-panel > .editor-area [data-label="Undo"]').click();
    check('tag_edit_can_be_undone', await page.inputValue('#text') === '大家好，欢迎来找我玩。');
    await page.locator('#clone-audio').setInputFiles(process.env.DRAMA_STUDIO_REFERENCE);
    await page.waitForFunction(() => state.cloneUpload && !state.cloneAsrPromise);
    await page.locator('#clone-transcript').fill('人工核对的参考台词。');
    check('reference_upload_keeps_waveform_and_manual_transcript',
      await page.locator('#clone-upload-preview .studio-player').isVisible()
      && await page.inputValue('#clone-transcript') === '人工核对的参考台词。');
    await editor().fill('这是第一份生成的台词。');
    await page.locator('#direction').fill('温暖自然。');
    await page.locator('#submit-job').click();
    const first = await finished();
    check('actual_generate_preserves_target_direction_and_reference', first.recipe.text === '这是第一份生成的台词。'
      && first.recipe.direction === '温暖自然。' && first.recipe.reference_snapshot[0].text === '人工核对的参考台词。');
    check('missing_asr_is_skipped_without_error_or_a_mode_selector', asrCalls === 1
      && !await page.locator('#asr-mode,#voice-more-options').count() && !await page.locator('#clone-asr-status').isVisible());
    check('result_has_single_player_and_basic_actions', await page.locator('.main-result .studio-player').count() === 1
      && await page.locator('.main-result .creative-edit').isVisible()
      && await page.locator('.main-result .creative-retry').isVisible()
      && await page.locator('.main-result .download-link').isVisible());
    check('completed_result_keeps_only_generation_time', (await page.locator('#progress-count').textContent()).includes('生成耗时')
      && await page.locator('#progress-stage').textContent() === '' && await noRetiredUi());
    const timed = { ...first, status: 'running', timing: { schema: 1, phase: 'model_load',
      generation_seconds: 0, model_load_seconds: 18.543, initialization_seconds: 0, total_seconds: 18.543 } };
    const display = value => page.evaluate(job => { state.activeJob = job; renderJob(job); }, value);
    await display(timed);
    check('loading_is_only_a_transient_stage_under_progress', await page.locator('#progress-stage').textContent() === '加载模型'
      && await page.locator('#progress-count').textContent() === '' && !await page.locator('#job-log').count());
    timed.timing.phase = 'initialization'; await display(timed);
    check('initialization_replaces_loading_stage', await page.locator('#progress-stage').textContent() === '正在初始化');
    timed.timing.phase = 'generation'; timed.timing.generation_seconds = 7.25; await display(timed);
    check('generation_time_excludes_loading_and_initialization', await page.locator('#progress-count').textContent() === '生成耗时 7.25 秒');
    timed.status = 'complete'; timed.timing.phase = 'finished'; timed.timing.generation_seconds = 67.805;
    await display(timed);
    check('finished_generation_has_no_persistent_load_or_total_breakdown', await page.locator('#progress-stage').textContent() === ''
      && await page.locator('#progress-count').textContent() === '生成耗时 67.805 秒'
      && !await page.locator('#result-box').textContent().then(text => /模型加载|初始化|请求总耗时/.test(text)));
    await display({ ...first, timing: undefined, result: { ...first.result, backend: 'native GGUF CUDA',
      total_seconds: 87.726, client_request_seconds: 67.8051732, session_reused: false, load_seconds: 18.543 } });
    check('older_native_results_still_display_only_actual_request_generation',
      await page.locator('#progress-count').textContent() === '生成耗时 67.805 秒');
    await display(first);
    await shot('simplified-result');
    await editor().fill('另一份没有提交的创作草稿。');
    await page.locator('.main-result .creative-retry').click(); const retry = await finished();
    check('one_click_regenerate_uses_source_not_current_unsent_text',
      retry.recipe.operation === 'retry' && retry.recipe.text === first.recipe.text
      && await page.inputValue('#text') === '另一份没有提交的创作草稿。');
    await page.locator('.main-result .result-more > summary').click();
    check('result_more_only_contains_voice_reuse_and_save',
      await page.locator('.main-result .result-more .menu-command').count() === 2
      && await page.locator('.main-result .result-more').getByRole('button', { name: '用作参考音色', exact: true }).isVisible()
      && await page.locator('.main-result .result-more').getByRole('button', { name: '保存音色', exact: true }).isVisible());
    await page.locator('.main-result .result-more').getByRole('button', { name: '保存音色', exact: true }).click();
    await page.locator('#save-reference-transcript').fill('这是第一份生成的台词。');
    await page.locator('#save-reference-name').fill('Saved result voice');
    await page.locator('#reference-checked').check();
    await page.locator('dialog[open]').getByRole('button', { name: '保存音色', exact: true }).click();
    await page.locator('dialog[open]').waitFor({ state: 'hidden' });
    check('saving_result_as_voice_still_persists_checked_audio',
      (await api('/api/voices')).voices.some(voice => voice.name === 'Saved result voice'));
    await page.locator('.nav[data-view="takes"]').click();
    check('history_is_one_list_without_candidate_or_work_hierarchy', !await page.locator('#studio-works,#generation-archive').count()
      && await page.locator('.job-card').count() >= 2 && await page.locator('.history-panel').isVisible());
    const firstHistory = page.locator(`.job-card[data-job-id="${first.id}"]`);
    check('history_title_shows_generated_words_instead_of_the_mode',
      await firstHistory.locator('.job-title').textContent() === first.recipe.text
      && !await page.locator('#job-list .job-script').count());
    check('history_has_no_view_button_and_play_is_named_play',
      !await page.locator('#job-list .inspect-job').count()
      && (await firstHistory.locator('.history-play').textContent()).trim() === '播放');
    check('history_download_is_visible_outside_the_more_menu',
      await firstHistory.locator('.job-actions > .history-download').isVisible()
      && !await firstHistory.locator('.history-actions .download-link').count());
    const download = await Promise.all([page.waitForEvent('download'), firstHistory.locator('.history-download').click()]);
    check('direct_history_download_returns_the_audio_file', download[0].suggestedFilename() === 'result.wav'
      && !await download[0].failure());
    check('history_previews_remove_control_tags_without_splitting_unicode', await page.evaluate(() => {
      const heading = document.createElement('strong');
      renderHistoryTitle({ recipe: { text: '<|speaker:0|> [happy]<whisper>你好，欢迎使用。</whisper>' } }, heading);
      if (heading.textContent !== '你好，欢迎使用。') return false;
      renderHistoryTitle({ recipe: { text: '😀'.repeat(65) } }, heading);
      if (heading.textContent !== '😀'.repeat(64) + '…') return false;
      renderHistoryTitle({ recipe: { text: '<img src=x onerror=alert(1)>正文' } }, heading);
      return !heading.children.length && heading.textContent === '<img src=x onerror=alert(1)>正文';
    }));
    const historical = await api('/api/jobs');
    const legacyResult = { ...first, id: 'legacy-result-title', recipe: undefined, result: { text: '旧版结果里保存的实际生成正文。' } };
    const legacyTake = { ...first, id: 'legacy-take-title', recipe: undefined, take_file: null,
      result: { native_take_file: first.take_file } };
    const historyRoute = route => route.request().method() === 'GET'
      ? route.fulfill({ json: { jobs: [legacyResult, legacyTake, ...historical.jobs] } }) : route.continue();
    let titleInspections = 0;
    const countTitleInspection = request => {
      if (request.url().endsWith('/api/takes/inspect') && request.postDataJSON()?.path === first.take_file) titleInspections++;
    };
    page.on('request', countTitleInspection);
    await page.route('**/api/jobs', historyRoute); await page.evaluate(() => loadJobs());
    await page.waitForFunction(text => document.querySelector('.job-card[data-job-id="legacy-take-title"] .job-title')?.textContent === text,
      first.recipe.text);
    check('legacy_result_records_show_their_saved_body',
      await page.locator('.job-card[data-job-id="legacy-result-title"] .job-title').textContent() === legacyResult.result.text);
    check('legacy_native_take_records_recover_text_without_loading_a_model',
      await page.locator('.job-card[data-job-id="legacy-take-title"] .job-title').textContent() === first.recipe.text);
    await page.evaluate(() => loadJobs());
    check('legacy_take_text_is_cached_across_history_refreshes', titleInspections === 1);
    await shot('history-text-titles-desktop');
    await page.unroute('**/api/jobs', historyRoute); page.off('request', countTitleInspection);
    await page.evaluate(() => loadJobs());
    await firstHistory.locator('.history-play').click();
    await page.waitForFunction(id => document.querySelector(`.job-card[data-job-id="${id}"] .history-play`).getAttribute('aria-pressed') === 'true', first.id);
    check('history_plays_directly_without_opening_another_player',
      !await firstHistory.locator('audio,.studio-player,.history-preview').count()
      && (await firstHistory.locator('.history-play').textContent()).trim() === '暂停播放');
    await firstHistory.locator('.history-play').click();
    check('history_play_button_can_pause_audio', await firstHistory.locator('.history-play').getAttribute('aria-pressed') === 'false');
    await firstHistory.locator('.history-actions > summary').click();
    check('history_keeps_download_and_one_unified_edit_entry',
      await firstHistory.locator('.download-link').isVisible() && await firstHistory.locator('.use-take').count() === 1);
    await firstHistory.locator('.use-take').click();
    await page.locator('#audio-edit-workspace').waitFor();
    check('history_edit_opens_the_same_inline_whole_editor',
      await page.locator('[data-edit-mode="whole"]').getAttribute('aria-selected') === 'true'
      && !await page.locator('dialog[open]').count() && await page.inputValue('#text') === '另一份没有提交的创作草稿。');
    await page.locator('#audio-edit-close').click(); await page.locator('#audio-edit-workspace').waitFor({ state: 'detached' });
    if (!await page.locator('#transcribe-clone').isVisible()) await page.locator('#clone-upload-preview > summary').click();
    asrAvailable = true;
    await page.locator('#transcribe-clone').click();
    await page.waitForFunction(() => document.querySelector('#clone-transcript').value === '参考音频中的准确台词。');
    check('automatic_reference_transcription_is_preserved', asrCalls === 2
      && !await page.locator('#clone-asr-status').isVisible());
    const voices = (await api('/api/voices')).voices;
    await page.locator('.nav[data-view="scene"]').click();
    await page.locator('.dialogue-editor-area .cm-content').fill('S1：我们准备好了吗？\nS2：准备好了。');
    const roles = await page.locator('.dialogue-speaker').evaluateAll(rows => rows.map(row => row.dataset.roleId));
    for (const [index, role] of roles.entries()) {
      await page.locator(`.dialogue-speaker[data-role-id="${role}"] .speaker-change`).click();
      await page.locator(`.dialogue-voice-option[data-voice-id="${voices[index].id}"] .dialogue-voice-pick`).click();
      await page.locator('#dialogue-voice-apply').click();
      await page.locator('#dialogue-voice-dialog').waitFor({ state: 'detached' });
    }
    await page.locator('#submit-scene').click(); const joint = await finished();
    check('joint_dialogue_keeps_multiple_voices_and_joint_request', joint.recipe.scene_strategy === 'joint'
      && joint.recipe.lines.length === 2 && joint.recipe.reference_snapshot.length === 2);
    check('joint_dialogue_has_combined_download_without_manual_adoption',
      await page.locator('.main-result .download-link').isVisible() && await noRetiredUi());
    await page.locator('[data-strategy="per_line"]').click();
    await page.locator('#submit-scene').click(); const perLine = await finished();
    check('per_line_dialogue_still_generates_complete_combined_audio', perLine.result.segments.length === 2
      && perLine.output && await page.locator('.main-result .download-link').isVisible());
    check('per_line_editing_does_not_reintroduce_adopt_or_context_update_controls',
      await page.locator('.segment-creative-actions .creative-edit').count() === 2 && await noRetiredUi());
    await page.locator('.nav[data-view="generate"]').click();
    check('tools_text_stream_and_new_creation_commands_are_removed',
      !await page.locator('#toggle-text-stream,#exit-text-stream,#text-stream-status,'
        + '#view-generate [data-i18n="Tools"],#view-generate [data-i18n="New draft"],'
        + '#view-generate [data-i18n="New with this voice"],#view-scene [data-i18n="New draft"]').count());
    await page.evaluate(() => StudioFlow.save());
    const previous = (await api('/api/studio')).drafts.single.value;
    const legacyStream = structuredClone(previous);
    legacyStream.mode = 'stream'; legacyStream.form.text = '旧版普通创作的正文应当保留。';
    legacyStream.form.chunks = '旧版流式第一句话。\n旧版流式第二句话。';
    legacyStream.operation = null;
    await api('/api/studio/draft', { key: 'single', value: legacyStream });
    await page.reload({ waitUntil: 'networkidle' }); await ready();
    check('legacy_stream_draft_migrates_to_usable_ordinary_generation',
      await page.inputValue('#mode') === 'tts' && await page.inputValue('#text') === legacyStream.form.chunks
      && await page.locator('#submit-job').isEnabled() && !await page.locator('#stream-fields').isVisible());
    await page.evaluate(() => StudioFlow.save());
    const saved = (await api('/api/studio')).drafts.single.value;
    check('legacy_stream_does_not_reappear_after_saving', saved.mode === 'tts' && saved.form.text === legacyStream.form.chunks
      && !await page.locator('#toggle-text-stream,#exit-text-stream').count());
    const retired = structuredClone(saved);
    retired.mode = 'edit'; retired.operation = { action: 'continue', source: { job_id: first.id }, model: first.recipe.model };
    retired.form.text = '从旧续写草稿保留下来的新台词。'; retired.form['previous-take'] = first.take_file;
    retired.form['source-take'] = first.take_file;
    await api('/api/studio/draft', { key: 'single', value: retired });
    await page.reload({ waitUntil: 'networkidle' }); await ready();
    check('old_continuation_draft_migrates_to_normal_creation_without_losing_words',
      await page.inputValue('#mode') === 'tts' && await page.inputValue('#text') === retired.form.text
      && !await page.inputValue('#previous-take') && !await page.inputValue('#source-take')
      && await page.evaluate(() => !StudioFlow.flow.operation));
    await page.evaluate(() => StudioFlow.save());
    const migrated = (await api('/api/studio')).drafts.single.value;
    check('retired_continuation_cannot_reappear_after_saving', migrated.mode === 'tts' && !migrated.operation
      && migrated.form.text === retired.form.text);
    await page.locator('#submit-job').click(); const migratedJob = await finished('');
    check('migrated_draft_no_longer_sends_previous_take_or_continue_operation',
      migratedJob.recipe.mode === 'tts' && !migratedJob.recipe.previous_take && !migratedJob.recipe.previous_sources
      && migratedJob.recipe.operation !== 'continue');
    const failure = { ...migratedJob, status: 'error', output: null, error: 'Synthetic generation failure.',
      result: {}, timing: { schema: 1, generation_seconds: 2, phase: 'finished' } };
    await display(failure);
    check('errors_remain_visible_without_a_log_panel',
      await page.locator('#result-error').isVisible()
      && (await page.locator('#result-error').textContent()).includes('Synthetic generation failure.')
      && !await page.locator('#job-log').count());
    await display(migratedJob);
    await api('/__test__/archive', { source: first.id }); await page.evaluate(() => StudioFlow.refresh());
    await page.locator('.nav[data-view="takes"]').click();
    check('older_history_remains_accessible_after_removing_work_browser', await page.locator('#load-older-results').isVisible());
    await page.locator('#load-older-results').click();
    await page.waitForFunction(id => Boolean(document.querySelector(`.job-card[data-job-id="${id}"]`)), first.id);
    check('older_generations_are_loaded_into_same_history_list',
      await page.locator(`.job-card[data-job-id="${first.id}"]`).isVisible()
      && await page.locator('.job-card').count() > 100);
    const other = (await api('/api/model')).available_models.find(model => model.runtime === 'gguf' && /q4/i.test(model.name));
    await page.locator('.nav[data-view="generate"]').click();
    if (other) {
      await choose('active-model', other.directory); await page.locator('#apply-active-model').click();
      await page.waitForFunction(model => state.model.selected_directory === model && !state.switchingModel, other.directory);
      check('model_picker_still_switches_actual_model_without_erasing_draft',
        await page.inputValue('#text') === retired.form.text && await page.locator('#active-model-trigger').isVisible());
    }
    for (const locale of ['zh-CN', 'en', 'ja', 'ko']) {
      await choose('ui-language', locale);
      for (const size of [{ width: 1440, height: 900 }, { width: 390, height: 844 }]) {
        await page.setViewportSize(size);
        for (const viewName of ['generate', 'takes', 'scene', 'voices', 'settings']) {
          await page.locator(`.nav[data-view="${viewName}"]`).click();
          check(`simplified_view_fits_${locale}_${size.width}_${viewName}`,
            await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
          check(`saved_status_stays_absent_${locale}_${size.width}_${viewName}`, !await page.locator('#draft-status:visible').count());
          if (viewName === 'generate') check(`normal_compose_heading_stays_hidden_${locale}_${size.width}`,
            !await page.locator('.compose-panel > .panel-head').isVisible());
          if (viewName === 'settings') check(`local_machine_stays_expanded_without_capability_list_${locale}_${size.width}`,
            await page.locator('#runtime-details').evaluate(node => node.open)
            && await page.locator('#hardware-card').isVisible()
            && !await page.locator('#capability-list').count());
          if (viewName === 'takes') {
            check(`history_play_label_is_localized_${locale}_${size.width}`,
              (await page.locator('.history-play').first().textContent()).trim() === await page.evaluate(() => t('Play')));
            check(`history_titles_and_actions_fit_${locale}_${size.width}`, await page.locator('.job-card').first().evaluate(node => {
              const title = node.querySelector('.job-title'), copy = node.querySelector('.job-copy').getBoundingClientRect();
              const buttons = node.querySelector('.job-actions').getBoundingClientRect(), r = title.getBoundingClientRect();
              return r.width > 0 && r.right <= copy.right + 1 && r.left >= copy.left
                && (r.right <= buttons.left + 1 || r.bottom <= buttons.top) && buttons.right <= innerWidth;
            }));
            check(`history_metadata_stays_aligned_${locale}_${size.width}`, await page.locator('.job-card').first().evaluate(node => {
              const card = node.getBoundingClientRect();
              const meta = node.querySelector('.job-meta').getBoundingClientRect();
              const actions = node.querySelector('.job-actions').getBoundingClientRect();
              const radius = parseFloat(getComputedStyle(node).borderTopLeftRadius);
              return meta.left >= card.left && meta.right <= card.right
                && (meta.right <= actions.left + 1 || meta.bottom <= actions.top + 1)
                && radius <= 8;
            }));
            if (locale === 'zh-CN' && size.width === 390) await shot('history-text-titles-mobile');
          }
        }
        await page.locator('.nav[data-view="generate"]').click();
        await page.locator('#active-model-trigger').click();
        check(`model_list_is_wide_and_viewport_bounded_${locale}_${size.width}`,
          await page.locator('.select-popover-active-model').evaluate(node => {
            const r = node.getBoundingClientRect(), trigger = document.querySelector('#active-model-trigger').getBoundingClientRect();
            return Math.abs(r.width - Math.min(innerWidth - 24, Math.max(trigger.width, 760))) < 1
              && r.left >= 12 && r.right <= innerWidth - 11 && node.scrollWidth <= node.clientWidth + 1;
          }));
        if (locale === 'zh-CN') await shot('wide-model-list-' + size.width);
        await page.keyboard.press('Escape');
      }
    }
    await page.route('**/api/capabilities', route => route.fulfill({
      status: 503, json: { error: 'Fixture capability lookup failed.' },
    }));
    await page.evaluate(() => loadCapabilities());
    check('capability_lookup_failure_is_visible_without_the_retired_panel',
      (await page.locator('#engine-status').textContent()).includes('Fixture capability lookup failed.')
      && !report.errors.length);
    await page.unroute('**/api/capabilities');
    await page.evaluate(() => loadCapabilities());
    check('capability_lookup_recovers_without_the_retired_panel', await page.evaluate(() => Boolean(state.capabilities?.ordinary_tts))
      && !(await page.locator('#engine-status').textContent()).includes('Fixture capability lookup failed.')
      && await page.locator('#engine-dot').evaluate(node => node.classList.contains('ready'))
      && !report.errors.length);
    check('no_browser_exceptions', !report.errors.length);
  } catch (error) {
    report.errors.push(error.stack || String(error));
    if (page) {
      const file = path.join(folder, 'simplified-ui-failure.png');
      await page.screenshot({ path: file, fullPage: true }).catch(() => {}); report.screenshots.push(file);
    }
  } finally {
    if (browser) await browser.close();
    report.passed = !report.errors.length && Object.values(report.checks).every(Boolean);
    fs.writeFileSync(path.join(root, 'research/2026-10-20-studio-simplification-ui-acceptance.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report, null, 2)); if (!report.passed) process.exitCode = 1;
  }
})();
