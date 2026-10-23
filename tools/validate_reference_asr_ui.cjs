/* One real GPU ASR request; later task responses are controlled race fixtures. */
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');
const root = path.resolve(__dirname, '..');
const base = process.env.DRAMA_UI_URL;
const folder = process.env.DRAMA_ASR_TEST_FOLDER;
const report = {
  scope: 'Actual upload, Qwen3-ASR GPU transcription and disk cache. Later race fixtures stub ASR tasks and TTS submission; no TTS inference.',
  checks: {}, errors: [], screenshots: [],
};
(async () => {
  let browser;
  try {
    browser = await chromium.launch({ channel: 'chrome', headless: true });
    const context = await browser.newContext({ locale: 'zh-CN', viewport: { width: 1440, height: 1000 } });
    const page = await context.newPage();
    page.on('pageerror', error => report.errors.push(error.message));
    const requests = [], tasks = [];
    page.on('response', async response => {
      if (response.url().endsWith('/api/transcriptions')) {
        try { tasks.push(await response.json()); } catch (_) {}
      }
    });
    await page.route('**/api/jobs', route => {
      if (route.request().method() !== 'POST') return route.continue();
      requests.push(route.request().postDataJSON());
      return route.fulfill({ status: 202, json: { id: 'asr-ui-tts-fixture' } });
    });
    await page.route('**/api/jobs/asr-ui-tts-fixture', route => route.fulfill({
      json: { id: 'asr-ui-tts-fixture', mode: 'tts', status: 'cancelled', events: [] },
    }));
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.locator('#active-model:not([disabled])').waitFor();
    const model = await (await context.request.get(base + '/api/model')).json();
    report.selected_model = model.selected_directory;
    report.checks.asr_not_in_tts_model_list = !model.available_models.some(m => m.name.includes('Qwen3-ASR'));
    const began = Date.now();
    await page.locator('#clone-audio').setInputFiles(process.env.DRAMA_ASR_TEST_AUDIO);
    await page.waitForFunction(() => state.cloneFile && !state.cloneAsrPromise
      && document.querySelector('#clone-transcript').value.length > 0, undefined, { timeout: 120000 });
    report.first_transcript = await page.inputValue('#clone-transcript');
    report.first_request_seconds = (Date.now() - began) / 1000;
    const id = tasks[0]?.id;
    const actual = await (await context.request.get(base + '/api/transcriptions/' + id)).json();
    report.actual_asr = actual;
    report.checks.upload_automatically_transcribed = actual.status === 'complete'
      && report.first_transcript === actual.result.text;
    report.checks.gpu_used = actual.result.device.startsWith('cuda');
    report.checks.no_comfyui_runtime_required = actual.model === 'Qwen/Qwen3-ASR-0.6B-hf';
    report.checks.transcript_editable = await page.locator('#clone-transcript').isEditable();
    const afterFirst = Date.now();
    await page.click('#remove-clone-audio');
    await page.locator('#clone-audio').setInputFiles(process.env.DRAMA_ASR_TEST_AUDIO);
    await page.waitForFunction(() => !state.cloneAsrPromise
      && document.querySelector('#clone-transcript').value.length > 0);
    report.checks.same_file_uses_cached_transcript = tasks.at(-1).cache_hit === true;
    report.cached_request_seconds = (Date.now() - afterFirst) / 1000;
    const manual = '手工核对后的参考逐字稿。';
    await page.fill('#clone-transcript', manual);
    await page.fill('#text', '这是新台词。');
    await page.click('#submit-job');
    await page.waitForFunction(() => !document.querySelector('#submit-job').disabled);
    report.checks.tts_uses_manually_corrected_transcript = requests.at(-1).uploaded_references[0].text === manual;
    report.checks.tts_does_not_retranscribe = tasks.length === 2;

    let completed = false;
    let serial = 0;
    let lastPayload;
    await page.route('**/api/transcriptions', async route => {
      lastPayload = route.request().postDataJSON();
      return route.fulfill({ status: 202, json: {
        id: `fixture-${++serial}`, status: 'running', stage: 'transcribing',
      } });
    });
    await page.route('**/api/transcriptions/fixture-*', route => route.fulfill({
      json: completed ? { id: 'fixture', status: 'complete', result: { text: '受控识别结果。' } }
        : { id: 'fixture', status: 'running', stage: 'transcribing' },
    }));
    await page.click('#transcribe-clone');
    await page.waitForFunction(() => Boolean(state.cloneAsrPromise));
    await page.waitForTimeout(150);
    report.checks.retry_always_uses_auto_language = lastPayload.language === 'auto' && lastPayload.force === true;
    await page.fill('#clone-transcript', '识别过程中手改的新文字。');
    completed = true;
    await page.waitForFunction(() => !state.cloneAsrPromise);
    report.checks.late_asr_does_not_overwrite_edits = await page.inputValue('#clone-transcript') === '识别过程中手改的新文字。';
    report.checks.can_apply_recognized_result_explicitly = await page.locator('#apply-asr-text').isVisible();
    await page.click('#apply-asr-text');
    report.checks.apply_result_button_works = await page.inputValue('#clone-transcript') === '受控识别结果。';

    completed = false;
    await page.click('#remove-clone-audio');
    await page.locator('#clone-audio').setInputFiles(process.env.DRAMA_ASR_TEST_AUDIO);
    await page.waitForFunction(() => Boolean(state.cloneAsrPromise));
    await page.click('#submit-job');
    await page.waitForTimeout(200);
    const count = requests.length;
    report.checks.generate_waits_for_pending_asr = await page.locator('#submit-job').isDisabled();
    completed = true;
    await page.waitForFunction(() => !document.querySelector('#submit-job').disabled);
    report.checks.generate_continues_after_asr = requests.length === count + 1
      && requests.at(-1).uploaded_references[0].text === '受控识别结果。';

    completed = false;
    await page.click('#transcribe-clone');
    await page.waitForFunction(() => Boolean(state.cloneAsrPromise));
    await page.click('#remove-clone-audio');
    completed = true;
    await page.waitForTimeout(900);
    report.checks.removed_audio_ignores_late_result = await page.inputValue('#clone-transcript') === ''
      && await page.locator('#clone-upload-preview').isHidden();
    await page.locator('#clone-audio').setInputFiles(process.env.DRAMA_ASR_TEST_AUDIO);
    await page.waitForFunction(() => !state.cloneAsrPromise
      && document.querySelector('#clone-transcript').value.length > 0);
    for (const lang of ['en', 'ja', 'ko', 'zh-CN']) {
      await page.click('#ui-language-trigger');
      await page.locator(`#ui-language-listbox [data-value="${lang}"]`).click();
      report.checks[`locale_preserves_asr_${lang}`] = await page.inputValue('#clone-transcript') === '受控识别结果。';
    }
    for (const [name, width, height] of [['desktop', 1440, 1000], ['mobile', 390, 844]]) {
      await page.setViewportSize({ width, height });
      report.checks[`${name}_no_horizontal_overflow`] = await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1);
      const screenshot = path.join(folder, `${name}.png`);
      await page.screenshot({ path: screenshot, fullPage: true });
      report.screenshots.push(screenshot);
    }
    await page.route('**/api/transcriptions/fixture-*', route => route.fulfill({
      json: { id: 'fixture', status: 'error', error: 'Controlled recognition failure' },
    }));
    await page.click('#transcribe-clone');
    await page.waitForFunction(() => !state.cloneAsrPromise
      && state.cloneAsrStatus.startsWith('Transcription failed'));
    report.checks.failure_keeps_manual_transcript = await page.inputValue('#clone-transcript') === '受控识别结果。';
    await page.fill('#clone-transcript', '识别失败后手工填写。');
    await page.click('#submit-job');
    await page.waitForFunction(() => !document.querySelector('#submit-job').disabled);
    report.checks.manual_generation_works_after_asr_failure =
      requests.at(-1).uploaded_references[0].text === '识别失败后手工填写。';
    report.checks.translations_complete = await page.evaluate(() =>
      [...document.querySelectorAll('[data-i18n]')].every(e => I18n.has(e.dataset.i18n)));
    report.checks.no_browser_script_errors = report.errors.length === 0;
  } catch (error) {
    report.errors.push(error.stack || String(error));
  } finally {
    if (browser) await browser.close();
    report.passed = !report.errors.length && Object.values(report.checks).every(Boolean);
    fs.writeFileSync(path.join(root, 'research/qwen3-asr-ui-validation.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report, null, 2));
    if (!report.passed) process.exitCode = 1;
  }
})();
