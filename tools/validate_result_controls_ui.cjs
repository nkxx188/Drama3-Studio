/* Real local APIs/audio/storage, with synthetic inference and delayed-response fixtures. */
const fs = require('node:fs'), path = require('node:path'), { chromium } = require('playwright');
const root = path.resolve(__dirname, '..'), base = process.env.DRAMA_UI_URL, folder = process.env.DRAMA_STUDIO_FOLDER;
const report = { scope: 'Collapsed-upload removal, independent result workspaces, reload and delayed-response races, direct history playback and audio duration. Synthetic inference only.',
  checks: {}, errors: [], screenshots: [] };
(async () => {
  let browser, page, releasePoll;
  try {
    browser = await chromium.launch({ channel: 'chrome', headless: true });
    const context = await browser.newContext({ viewport: { width: 1440, height: 960 }, locale: 'zh-CN' });
    await context.addInitScript(() => {
      localStorage.setItem('drama3.ui.language', 'zh-CN');
      const NativeAudio = window.Audio;
      window.testAudios = [];
      window.Audio = function(...args) {
        const audio = new NativeAudio(...args); window.testAudios.push(audio); return audio;
      };
      window.Audio.prototype = NativeAudio.prototype;
    });
    page = await context.newPage();
    page.on('pageerror', error => report.errors.push(error.stack || error.message));
    const api = async (url, body) => {
      const response = body === undefined ? await context.request.get(base + url)
        : await context.request.post(base + url, { data: body });
      if (!response.ok()) throw new Error(url + ': ' + await response.text());
      return response.json();
    };
    const check = (name, value) => { report.checks[name] = Boolean(value); if (!value) throw new Error(name); };
    const ready = () => page.waitForFunction(() => document.documentElement.dataset.studioReady === 'true');
    const nav = name => page.locator(`.nav[data-view="${name}"]`).click();
    const editor = () => page.locator('.compose-panel > .editor-area .cm-content');
    const visibleResult = () => page.locator('#result-box').getAttribute('data-job-id');
    const finished = async old => {
      await page.waitForFunction(old => {
        const id = StudioFlow.flow.lastSubmit?.id;
        return id && id !== old && document.querySelector('#result-box').dataset.jobId === id
          && document.querySelector('#job-badge').dataset.status === 'complete';
      }, old);
      return api('/api/jobs/' + await visibleResult());
    };
    const shot = async name => {
      const file = path.join(folder, name + '.png');
      await page.screenshot({ path: file, animations: 'disabled' }); report.screenshots.push(file);
    };
    await page.goto(base, { waitUntil: 'networkidle' }); await ready();
    await editor().fill('移除参考音频不能清空这段目标文本。');
    await page.locator('#direction').fill('自然、清晰。');
    for (const width of [1440, 390]) {
      await page.setViewportSize({ width, height: width === 390 ? 844 : 960 });
      await page.locator('#clone-audio').setInputFiles(process.env.DRAMA_STUDIO_REFERENCE);
      await page.waitForFunction(() => state.cloneUpload && !state.cloneAsrPromise);
      await page.locator('#clone-transcript').fill('核对后的参考台词。');
      await page.locator('#clone-upload-preview > summary').click();
      check(`remove_is_after_play_and_fits_collapsed_upload_${width}`, await page.locator('#clone-summary-remove').evaluate(node => {
        const play = document.querySelector('#clone-summary-play').getBoundingClientRect(), box = node.getBoundingClientRect();
        return node.previousElementSibling.id === 'clone-summary-play' && box.width === 32 && box.height === 32
          && box.left >= play.right && box.right <= innerWidth && !node.closest('details').open;
      }));
      await page.locator('#clone-summary-play').click();
      await page.waitForFunction(() => !document.querySelector('#clone-file-audio').paused);
      check(`play_does_not_expand_the_upload_${width}`, !await page.locator('#clone-upload-preview').evaluate(node => node.open));
      if (width === 1440) await shot('collapsed-upload-with-remove');
      await page.locator('#clone-summary-remove').focus();
      await page.keyboard.press('Enter');
      check(`remove_stops_audio_and_restores_upload_${width}`, await page.locator('#clone-upload-zone').isVisible()
        && !await page.locator('#clone-upload-preview').isVisible()
        && await page.evaluate(() => !state.cloneFile && !state.cloneUpload && document.querySelector('#clone-file-audio').paused));
      check(`remove_preserves_target_and_direction_${width}`, await page.inputValue('#text') === '移除参考音频不能清空这段目标文本。'
        && await page.inputValue('#direction') === '自然、清晰。');
    }
    await page.evaluate(() => StudioFlow.save());
    await page.reload({ waitUntil: 'networkidle' }); await ready();
    check('removed_upload_stays_removed_after_reload', await page.locator('#clone-upload-zone').isVisible()
      && !await page.locator('#clone-upload-preview').isVisible());
    await page.setViewportSize({ width: 1440, height: 960 });
    await editor().fill('这段音频属于首页生成。'); await page.locator('#submit-job').click();
    const single = await finished();
    await nav('scene');
    check('first_dialogue_visit_has_no_homepage_audio', !await page.locator('#result-box').isVisible()
      && !await page.locator('#result-box audio').count() && await page.locator('#job-badge').getAttribute('data-status') === 'idle');
    await nav('generate');
    check('homepage_keeps_its_result_after_visiting_empty_dialogue', await visibleResult() === single.id);
    await nav('scene');
    await page.locator('.dialogue-editor-area .cm-content').fill('S1: 这是多人对话。\nS2: 和首页结果分开保存。');
    const voices = (await api('/api/voices')).voices;
    for (let index = 0; index < 2; index++) {
      await page.locator('.dialogue-speaker').nth(index).locator('.speaker-change').click();
      await page.locator(`.dialogue-voice-option[data-voice-id="${voices[index].id}"] .dialogue-voice-pick`).click();
      await page.locator('#dialogue-voice-apply').click();
    }
    await page.locator('#submit-scene').click();
    const joint = await finished(single.id);
    check('joint_dialogue_result_has_its_own_audio', joint.id !== single.id && joint.output !== single.output
      && joint.recipe.scene_strategy === 'joint');
    for (const [workspace, job] of [['generate', single], ['scene', joint], ['generate', single]]) {
      await nav(workspace);
      check(`workspace_${workspace}_${job.id}_shows_only_its_audio`, await visibleResult() === job.id
        && await page.locator('.main-result audio').getAttribute('src') === await page.evaluate(output => fileUrl(output), job.output));
    }
    let nextId, pollSeen;
    const seen = new Promise(resolve => { pollSeen = resolve; });
    const gate = new Promise(resolve => { releasePoll = resolve; });
    const createRoute = async route => {
      if (route.request().method() !== 'POST') return route.continue();
      const response = await route.fetch(); nextId = (await response.json()).id;
      await route.fulfill({ response });
    };
    const delayedPoll = async route => {
      if (route.request().method() === 'GET' && route.request().url().endsWith('/' + nextId)) {
        pollSeen(); await gate;
      }
      await route.continue();
    };
    await page.route('**/api/jobs', createRoute);
    await page.route('**/api/jobs/*', delayedPoll);
    await editor().fill('等待响应时切换页面，也不能把这个结果放进多人对话。');
    await page.locator('#submit-job').click(); await seen;
    await nav('scene');
    check('switching_during_an_inflight_homepage_poll_keeps_dialogue_result', await visibleResult() === joint.id);
    releasePoll();
    await page.waitForFunction(id => state.resultJobs.generate?.id === id && state.resultJobs.generate?.status === 'complete', nextId);
    check('late_homepage_response_does_not_replace_dialogue_result', await visibleResult() === joint.id);
    await page.unroute('**/api/jobs', createRoute); await page.unroute('**/api/jobs/*', delayedPoll);
    await nav('generate');
    check('background_homepage_job_is_not_lost_after_switching_away', await visibleResult() === nextId);
    await page.evaluate(() => StudioFlow.save());
    const drafts = (await api('/api/studio')).drafts;
    check('drafts_save_two_independent_result_ids', drafts.single.value.active_job_id === nextId
      && drafts.scene.value.active_job_id === joint.id);
    for (const [workspace, id] of [['generate', nextId], ['scene', joint.id]]) {
      await nav(workspace); await page.reload({ waitUntil: 'networkidle' }); await ready();
      check(`reload_restores_${workspace}_with_its_own_result`, await visibleResult() === id
        && await page.locator(`#view-${workspace}`).isVisible());
    }
    const legacySingle = structuredClone(drafts.single.value), legacyScene = structuredClone(drafts.scene.value);
    legacySingle.active_job_id = joint.id; delete legacyScene.active_job_id;
    await api('/api/studio/draft', { key: 'single', value: legacySingle });
    await api('/api/studio/draft', { key: 'scene', value: legacyScene });
    await nav('generate'); await page.reload({ waitUntil: 'networkidle' }); await ready();
    check('legacy_shared_pointer_is_classified_instead_of_cross_rendered', await visibleResult() === nextId);
    await nav('scene');
    check('legacy_dialogue_result_remains_available', await visibleResult() === joint.id);
    await page.locator('[data-strategy="per_line"]').click(); await page.locator('#submit-scene').click();
    const perLine = await finished(nextId);
    await nav('generate');
    check('per_line_dialogue_does_not_replace_homepage_audio', await visibleResult() === nextId && perLine.result.segments.length === 2);
    await nav('scene');
    check('dialogue_keeps_the_latest_per_line_combined_result', await visibleResult() === perLine.id);
    await page.evaluate(() => StudioFlow.save());
    const explicitClear = structuredClone((await api('/api/studio')).drafts.single.value);
    explicitClear.active_job_id = null;
    await api('/api/studio/draft', { key: 'single', value: explicitClear });
    await nav('generate'); await page.reload({ waitUntil: 'networkidle' }); await ready();
    check('explicitly_cleared_result_is_not_resurrected_from_history', !await page.locator('#result-box').isVisible());
    await nav('takes');
    const first = page.locator(`.job-card[data-job-id="${single.id}"]`), second = page.locator(`.job-card[data-job-id="${joint.id}"]`);
    await second.locator('.history-actions > summary').click();
    await second.locator('.use-take').click();
    await page.locator('#audio-edit-workspace').waitFor();
    check('history_dialogue_edit_opens_in_the_dialogue_workspace',
      await page.locator('#view-audio-edit').isVisible()
      && await page.locator('.nav[data-view="scene"]').getAttribute('aria-current') === 'page');
    await page.locator('#audio-edit-back').click();
    await page.locator('#audio-edit-workspace').waitFor({ state: 'detached' });
    await nav('takes');
    const expectedDuration = await page.evaluate(duration => I18n.t('Audio · {seconds}s', {
      seconds: UiNumber.format(duration),
    }), single.result.audio.duration_seconds);
    check('history_omits_waveforms_and_shows_audio_duration',
      !await page.locator('.history-waveform,.job-card .player-wave').count()
      && (await first.locator('.job-size').textContent()).includes(expectedDuration));
    await page.evaluate(() => loadJobs());
    check('history_refresh_keeps_duration_without_creating_a_waveform',
      !await page.locator('.history-waveform,.job-card .player-wave').count()
      && (await first.locator('.job-size').textContent()).includes(expectedDuration));
    await shot('history-duration');
    await first.locator('.history-play').click();
    await page.waitForFunction(id => window.testAudios.at(-1)?.duration > 0 && !window.testAudios.at(-1).paused
      && document.querySelector(`.job-card[data-job-id="${id}"] .history-play`).getAttribute('aria-pressed') === 'true', single.id);
    check('history_starts_playing_on_the_first_click', await first.locator('.history-play').getAttribute('aria-pressed') === 'true'
      && (await first.locator('.history-play').textContent()).trim() === '暂停播放');
    check('history_does_not_open_a_player_or_dialog', !await first.locator('audio,.studio-player,.history-preview').count()
      && !await page.locator('dialog[open]').count());
    await page.evaluate(() => { window.testAudios.at(-1).currentTime = 1; });
    await first.locator('.history-play').click();
    check('history_second_click_pauses_without_resetting_position', await page.evaluate(() =>
      window.testAudios.at(-1).paused && window.testAudios.at(-1).currentTime >= 1));
    const audioCount = await page.evaluate(() => window.testAudios.length);
    await first.locator('.history-play').click();
    await page.waitForFunction(() => !window.testAudios.at(-1).paused);
    check('history_play_resumes_the_same_audio', await page.evaluate(count =>
      window.testAudios.length === count && window.testAudios.at(-1).currentTime >= 1, audioCount));
    await page.evaluate(() => loadJobs());
    check('history_refresh_keeps_current_playback', await first.locator('.history-play').getAttribute('aria-pressed') === 'true'
      && await page.evaluate(() => !window.testAudios.at(-1).paused));
    await second.locator('.history-play').click();
    await page.waitForFunction(() => !window.testAudios.at(-1).paused);
    check('playing_another_record_stops_the_previous_one', await first.locator('.history-play').getAttribute('aria-pressed') === 'false'
      && await page.evaluate(() => window.testAudios.at(-2).paused));
    await page.evaluate(() => {
      const audio = window.testAudios.at(-1);
      if (Number.isFinite(audio.duration)) audio.currentTime = Math.max(0, audio.duration - .05);
    });
    await page.waitForFunction(id => document.querySelector(`.job-card[data-job-id="${id}"] .history-play`).getAttribute('aria-pressed') === 'false', joint.id);
    check('ended_audio_restores_the_play_button', (await second.locator('.history-play').textContent()).trim() === '播放');
    await first.locator('.history-play').click();
    await page.waitForFunction(() => !window.testAudios.at(-1).paused);
    await nav('generate');
    check('leaving_history_stops_playback', await page.evaluate(() => window.testAudios.at(-1).paused));
    await nav('takes');
    check('history_download_stays_direct', await first.locator('.job-actions > .history-download').isVisible());
    await page.setViewportSize({ width: 390, height: 844 });
    check('direct_history_controls_fit_mobile', await first.locator('.job-actions').evaluate(node => {
      const r = node.getBoundingClientRect(); return r.left >= 0 && r.right <= innerWidth;
    }));
    await shot('history-direct-play-mobile');
    const failureJob = await api('/api/jobs', { text: '用于验证播放失败重试的测试音频。' });
    let failureResult;
    for (let attempt = 0; attempt < 100; attempt++) {
      failureResult = await api('/api/jobs/' + failureJob.id);
      if (failureResult.status === 'complete') break;
      await new Promise(resolve => setTimeout(resolve, 50));
    }
    if (!failureResult.output) throw new Error('Playback failure fixture did not finish.');
    await page.evaluate(() => loadJobs());
    const failureRow = page.locator(`.job-card[data-job-id="${failureJob.id}"]`);
    const audioUrl = base + await page.evaluate(output => fileUrl(output), failureResult.output);
    const unavailable = route => route.fulfill({ status: 404, headers: { 'Cache-Control': 'no-store' }, body: 'missing' });
    await page.route(audioUrl, unavailable);
    await failureRow.locator('.history-play').click();
    await failureRow.locator('.history-play-error').waitFor();
    check('playback_failure_is_visible_beside_the_record', (await failureRow.locator('.history-play-error').textContent()).includes('无法播放')
      && await failureRow.locator('.history-play').getAttribute('aria-pressed') === 'false');
    check('playback_error_does_not_overflow_mobile', await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
    await page.unroute(audioUrl, unavailable);
    check('retry_source_is_available_from_the_real_server', (await context.request.get(audioUrl)).ok());
    await failureRow.locator('.history-play').click();
    await page.waitForFunction(id => document.querySelector(`.job-card[data-job-id="${id}"] .history-play`).getAttribute('aria-pressed') === 'true', failureJob.id);
    check('playback_can_retry_after_a_media_error', !await failureRow.locator('.history-play-error').count()
      && await page.evaluate(() => !window.testAudios.at(-1).paused));
    check('no_browser_errors', !report.errors.length);
  } catch (error) {
    report.errors.push(error.stack || String(error));
    if (page) {
      report.playback = await page.evaluate(() => window.testAudios?.slice(-3).map(audio => ({
        src: audio.src, paused: audio.paused, error: audio.error?.code, errorMessage: audio.error?.message,
        networkState: audio.networkState, readyState: audio.readyState,
      }))).catch(() => null);
      const file = path.join(folder, 'result-controls-failure.png');
      await page.screenshot({ path: file }).catch(() => {}); report.screenshots.push(file);
    }
  } finally {
    releasePoll?.();
    if (browser) await browser.close();
    report.passed = !report.errors.length && Object.values(report.checks).every(Boolean);
    fs.writeFileSync(path.join(root, 'research/2026-10-21-result-controls-ui-acceptance.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report, null, 2));
    if (!report.passed) process.exitCode = 1;
  }
})();
