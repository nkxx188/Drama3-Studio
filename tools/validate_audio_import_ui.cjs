/* Real upload/store/editor requests; only ASR, codec and synthesis are fixtures. */
const fs = require('node:fs'), path = require('node:path'), { chromium } = require('playwright');
const base = process.env.DRAMA_UI_URL, folder = process.env.DRAMA_STUDIO_FOLDER;
const report = { scope: 'One-page upload editing, automatic ASR, deferred codec, cache and original-flow isolation.',
  checks: {}, errors: [], screenshots: [] };
(async () => {
  let browser;
  try {
    browser = await chromium.launch({ channel: 'chrome', headless: true });
    const context = await browser.newContext({ viewport: { width: 1440, height: 960 }, locale: 'zh-CN' });
    const page = await context.newPage();
    page.on('pageerror', error => report.errors.push(error.stack || error.message));
    const check = (name, value) => {
      report.checks[name] = Boolean(value);
      if (!value) throw new Error(name);
    };
    const ready = () => page.waitForFunction(() => document.documentElement.dataset.studioReady === 'true');
    const shot = async name => {
      const file = path.join(folder, name + '.png');
      await page.screenshot({ path: file, fullPage: true }); report.screenshots.push(file);
    };
    const api = async (url, data) => {
      const response = data === undefined ? await context.request.get(base + url)
        : await context.request.post(base + url, { data });
      if (!response.ok()) throw new Error(url + ': ' + await response.text());
      return response.json();
    };
    let asrCalls = 0, rejectAsr = false;
    await page.route('**/api/transcriptions', route => {
      asrCalls++;
      return route.fulfill({ json: rejectAsr ? { status: 'skipped' }
        : { id: 'fixture-asr', status: 'complete', result: { text: '明天我们一起出发。' } } });
    });
    const finish = async () => {
      const previous = await page.locator('#audio-import-result-box').getAttribute('data-job-id');
      await page.locator('#audio-import-submit').click();
      await page.waitForFunction(previous => {
        const box = document.querySelector('#audio-import-result-box');
        return box?.dataset.jobId && box.dataset.jobId !== previous
          && document.querySelector('#audio-import-result-badge')?.dataset.status === 'complete';
      }, previous);
      return api('/api/jobs/' + await page.locator('#audio-import-result-box').getAttribute('data-job-id'));
    };
    const wholeText = () => page.locator('#view-audio-import .audio-edit-line .cm-content');
    const replacement = () => page.locator('#view-audio-import .audio-edit-replacement .cm-content');
    await page.goto(base, { waitUntil: 'networkidle' }); await ready();
    await page.locator('.compose-panel > .editor-area .cm-content').fill('保留原有创作草稿。');
    await page.locator('.nav[data-view="audio-import"]').click();
    await page.locator('#audio-import-dropzone').waitFor({ state: 'visible' });
    check('editor_is_on_upload_page_from_the_start',
      await wholeText().isVisible() && await page.locator('#audio-import-submit').isVisible()
      && !await page.locator('#audio-import-prepare, #audio-import-open-editor').count());
    await shot('audio-import-empty-desktop');
    await page.locator('#audio-import-file').setInputFiles(process.env.DRAMA_STUDIO_REFERENCE);
    await page.waitForFunction(() => document.querySelector('#audio-import-text').value === '明天我们一起出发。');
    await page.waitForFunction(() => document.querySelector('#audio-import-player').duration > 0);
    check('upload_automatically_transcribes_without_model_edit_jobs',
      asrCalls === 1 && (await api('/api/jobs')).jobs.length === 0);
    check('unchanged_audio_requires_an_edit', await page.locator('#audio-import-submit').isDisabled());
    await wholeText().fill('后天我们一起出发。');
    let forcedPhase = 'model_load', heldResult = null;
    await page.route('**/api/jobs/*', async route => {
      if (route.request().method() !== 'GET' || !forcedPhase) return route.continue();
      const response = await route.fetch();
      const actual = await response.json();
      if (!actual.recipe?.audio_edit) return route.fulfill({ response });
      heldResult = actual;
      return route.fulfill({ response, json: {
        ...actual, status: 'running', output: null, take_file: null, result: {}, error: null,
        timing: { schema: 1, phase: forcedPhase, generation_seconds: forcedPhase === 'generation' ? 6.25 : 0,
          model_load_seconds: 18, initialization_seconds: 8, queue_seconds: 0, total_seconds: 32.25 },
        events: [{ stage: 'runtime', message: 'Growing context while keeping model weights' }],
      } });
    });
    await page.locator('#audio-import-submit').click();
    await page.waitForFunction(() => document.querySelector('#audio-import-stage').textContent === '加载模型');
    forcedPhase = 'initialization';
    await page.waitForFunction(() => document.querySelector('#audio-import-stage').textContent === '正在准备原音频');
    forcedPhase = 'generation';
    await page.waitForFunction(() => document.querySelector('#audio-import-stage').textContent === '正在生成');
    check('context_runtime_event_cannot_override_generation_phase',
      await page.locator('#audio-import-progress-count').textContent() === '生成耗时 6.25 秒');
    const stagedId = heldResult.id;
    forcedPhase = null;
    await page.waitForFunction(() => document.querySelector('#audio-import-result-badge').dataset.status === 'complete');
    check('completion_clears_loading_stage_and_busy_state',
      !await page.locator('#audio-import-stage').isVisible()
      && await page.locator('#audio-import-stage').textContent() === ''
      && await page.locator('#audio-import-results').getAttribute('aria-busy') === 'false');
    const whole = await api('/api/jobs/' + stagedId);
    check('one_click_prepares_and_regenerates_in_same_page',
      whole.status === 'complete' && whole.recipe.mode === 'tts' && whole.recipe.audio_edit
      && whole.source_preparation.codec_cache_hit === false
      && await page.locator('#view-audio-import').isVisible()
      && !await page.locator('#view-audio-edit').isVisible());
    check('whole_edit_uses_original_recording_and_transcript',
      whole.recipe.reference_snapshot[0].text === '明天我们一起出发。'
      && whole.recipe.reference_snapshot[0].audio.endsWith('source.wav'));
    await page.locator('[data-import-mode="local"]').click();
    await page.locator('#audio-import-player').evaluate(audio => StudioAudio.setSelection(audio, 1, 2, { emit: true }));
    await page.locator('#audio-import-quote').fill('明天');
    await replacement().fill('今天');
    const local = await finish();
    check('local_edit_reuses_codes_and_preserves_unselected_codes',
      local.recipe.mode === 'infill' && local.recipe.text === '<|speaker:0|> 今天我们一起出发。'
      && local.source_preparation.codec_cache_hit === true
      && local.result.infill.prefix_codes_equal && local.result.infill.suffix_codes_equal);
    check('only_edit_results_appear_in_history',
      (await api('/api/jobs')).jobs.length === 2 && !(await api('/api/jobs')).jobs.some(job => job.mode === 'audio_import'));
    await shot('audio-import-local-desktop');
    await page.reload({ waitUntil: 'networkidle' }); await ready();
    await page.locator('#audio-import-result-box audio').waitFor({ state: 'attached' });
    check('refresh_keeps_source_scope_edits_and_result_without_extra_asr',
      await page.locator('#view-audio-import').isVisible() && asrCalls === 1
      && await page.inputValue('#audio-import-quote') === '明天'
      && await page.inputValue('#audio-import-replacement') === '今天'
      && await page.locator('#audio-import-result-box').getAttribute('data-job-id') === local.id);
    await page.locator('[data-import-mode="whole"]').click();
    check('switching_scopes_keeps_whole_edit', await page.inputValue('#audio-import-text') === '后天我们一起出发。');
    await page.locator('#audio-import-original > summary').click();
    await page.locator('#audio-import-transcript').fill('明天我们一同出发。');
    await wholeText().fill('后天我们一同出发。');
    const corrected = await finish();
    check('corrected_transcript_does_not_force_codec_reencoding',
      corrected.source_preparation.codec_cache_hit === true
      && corrected.recipe.reference_snapshot[0].text === '明天我们一同出发。');
    await page.locator('#audio-import-result-box .main-result .creative-retry').click();
    await page.waitForFunction(previous =>
      document.querySelector('#audio-import-result-box')?.dataset.jobId !== previous
      && document.querySelector('#audio-import-result-badge')?.dataset.status === 'complete', corrected.id);
    const retried = await api('/api/jobs/' + await page.locator('#audio-import-result-box').getAttribute('data-job-id'));
    check('regenerate_stays_in_uploaded_editor_and_reuses_codes',
      retried.recipe.text === corrected.recipe.text && retried.source_preparation.codec_cache_hit === true
      && await page.locator('#view-audio-import').isVisible());
    await page.locator('#audio-import-result-box .main-result .creative-edit').click();
    await page.waitForFunction(() => document.querySelector('#audio-import-transcript').value === '后天我们一同出发。');
    check('result_edit_becomes_source_in_same_page',
      await page.locator('#view-audio-import').isVisible() && asrCalls === 1
      && !await page.locator('#audio-import-result-box .main-result').count());
    for (const locale of ['zh-CN', 'en', 'ja', 'ko']) {
      await page.evaluate(locale => I18n.setLocale(locale), locale);
      for (const width of [1440, 390]) {
        await page.setViewportSize({ width, height: 900 });
        check(`layout_${locale}_${width}`, await page.evaluate(() =>
          document.documentElement.scrollWidth <= innerWidth
          && document.querySelector('#audio-import-submit').getBoundingClientRect().right <= innerWidth));
      }
    }
    await page.evaluate(() => I18n.setLocale('zh-CN'));
    await shot('audio-import-mobile');
    await page.setViewportSize({ width: 1440, height: 960 });
    await page.locator('#audio-import-remove').click();
    rejectAsr = true;
    await page.locator('#audio-import-file').setInputFiles(process.env.DRAMA_STUDIO_REFERENCE);
    await page.waitForFunction(() => document.querySelector('#audio-import-asr-status').textContent.includes('未安装'));
    check('asr_unavailable_opens_manual_transcript', await page.locator('#audio-import-transcript').isVisible()
      && await page.locator('#audio-import-submit').isDisabled());
    await page.locator('#audio-import-transcript').fill('手动填写的原文。');
    await wholeText().fill('修改后的内容。');
    check('manual_transcript_can_generate', await page.locator('#audio-import-submit').isEnabled());
    await page.locator('.nav[data-view="generate"]').click();
    check('creation_draft_and_result_pointer_are_untouched',
      await page.locator('.compose-panel > .editor-area .cm-content').innerText() === '保留原有创作草稿。'
      && await page.evaluate(() => state.resultJobs.generate === null));
    check('no_browser_errors', report.errors.length === 0);
  } catch (error) {
    report.errors.push(error.stack || error.message); process.exitCode = 1;
  } finally {
    if (browser) await browser.close();
    const output = path.join(folder, 'audio-import-report.json');
    fs.writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
    console.log(output); console.log(JSON.stringify(report, null, 2));
  }
})();
