/* Real requests and per-chunk retries; controlled synthesis/ASR, no GPU needed. */
const fs = require('node:fs'), path = require('node:path'), { chromium } = require('playwright');
const base = process.env.DRAMA_UI_URL, folder = process.env.DRAMA_STUDIO_FOLDER;
const report = { checks: {}, errors: [], screenshots: [] };
(async () => {
  let browser, page;
  try {
    browser = await chromium.launch({ channel: 'chrome', headless: true });
    const context = await browser.newContext({ viewport: { width: 1440, height: 960 }, locale: 'zh-CN' });
    await context.addInitScript(() => localStorage.setItem('drama3.ui.language', 'zh-CN'));
    page = await context.newPage();
    page.on('pageerror', error => report.errors.push(error.stack || error.message));
    await page.route('**/api/transcriptions', route => route.fulfill({
      json: { status: 'complete', id: 'fixture-asr', result: { text: 'Original source line.' } },
    }));
    const api = async (url, data) => {
      const response = data === undefined ? await context.request.get(base + url)
        : await context.request.post(base + url, { data });
      if (!response.ok()) throw new Error(await response.text());
      return response.json();
    };
    const check = (name, value) => { report.checks[name] = Boolean(value); if (!value) throw new Error(name); };
    const ready = () => page.waitForFunction(() => document.documentElement.dataset.studioReady === 'true');
    const absent = () => page.locator('#progressive-audio,#target-seconds,#duration,#max-frames').count();
    const showSettings = async id => page.locator(id).evaluate(node => { node.open = true; });
    const shot = async (name, selector) => {
      const file = path.join(folder, name + '.png');
      await page.locator(selector).screenshot({ path: file }); report.screenshots.push(file);
    };
    const editorOptionsFit = prefix => page.evaluate(prefix => {
      const seed = document.getElementById(prefix + '-seed').getBoundingClientRect();
      const random = document.getElementById(prefix + '-random').closest('label').getBoundingClientRect();
      const retry = document.getElementById(prefix + '-silence-retry').closest('label').getBoundingClientRect();
      return random.left > seed.right && Math.abs(random.bottom - seed.bottom) <= 1
        && retry.top > seed.bottom && Math.abs(retry.left - seed.left) <= 1;
    }, prefix);
    await page.goto(base, { waitUntil: 'networkidle' }); await ready();
    check('retired_stream_and_duration_controls_are_removed', !await absent());
    await showSettings('#advanced-settings');
    check('silence_retry_is_visible_and_on_by_default', await page.locator('#silence-retry').isVisible()
      && await page.locator('#silence-retry').isChecked());
    check('silence_retry_uses_requested_chinese_label',
      (await page.locator('#silence-retry').locator('..').textContent()).trim() === '异常静音时自动重试');
    check('seed_uses_short_chinese_label', await page.locator('label[for="seed"]').textContent() === '种子');
    check('seed_and_random_option_are_left_of_candidates', await page.evaluate(() => {
      const seed = document.getElementById('seed').getBoundingClientRect();
      const candidates = document.getElementById('candidate-count').getBoundingClientRect();
      const random = document.getElementById('random-seed').closest('label').getBoundingClientRect();
      return seed.right < candidates.left && Math.abs(seed.top - candidates.top) <= 1
        && Math.abs(random.left - seed.left) <= 1 && random.top > seed.bottom;
    }));
    await page.locator('#silence-retry').uncheck();
    await page.evaluate(() => StudioFlow.save()); await page.reload({ waitUntil: 'networkidle' }); await ready();
    check('explicitly_disabled_silence_retry_stays_off', !await page.locator('#silence-retry').isChecked());
    await showSettings('#advanced-settings');
    await page.locator('#silence-retry').check();
    await page.locator('#max-chars').fill('80');
    await page.locator('.compose-panel > .editor-area .cm-content').fill(
      'First accepted chunk. '.repeat(4) + 'retry-fixture needs a retry. '.repeat(4) + 'Final chunk. '.repeat(6));
    await page.locator('#submit-job').click();
    await page.waitForFunction(() => document.querySelector('#job-badge').dataset.status === 'complete');
    const sourceId = await page.locator('#result-box').getAttribute('data-job-id');
    const source = await api('/api/jobs/' + sourceId);
    check('normal_generation_uses_automatic_length_without_duration_hint',
      source.recipe.length_policy === 'automatic-v1' && !source.recipe.duration_frames
      && source.mode === 'tts' && !source.recipe.single_chunk);
    check('server_retries_each_silent_chunk_before_acceptance', source.recipe.silence_retry
      && source.result.silence_retry.retries > 0 && !source.result.silence_retry.exhausted
      && source.result.chunks.length > 1
      && source.events.some(event => event.stage === 'silence_retry'));
    await page.locator('.compose-panel > .editor-area .cm-content').fill('exhausted-fixture');
    await page.locator('#submit-job').click();
    await page.waitForFunction(old => document.querySelector('#result-box').dataset.jobId !== old
      && document.querySelector('#job-badge').dataset.status === 'incomplete', sourceId);
    const exhausted = await api('/api/jobs/' + await page.locator('#result-box').getAttribute('data-job-id'));
    check('exhausted_retries_are_bounded_and_explained', exhausted.result.silence_retry.retries === 2
      && exhausted.result.silence_retry.exhausted
      && await page.locator('#result-error').isVisible()
      && (await page.locator('#result-error').textContent()).includes('重试两次'));
    await page.evaluate(id => monitorJob(id), sourceId);
    await page.waitForFunction(id => document.querySelector('#result-box').dataset.jobId === id
      && document.querySelector('#job-badge').dataset.status === 'complete', sourceId);
    await page.evaluate(() => StudioFlow.save()); await page.reload({ waitUntil: 'networkidle' }); await ready();
    check('silence_switch_and_chunk_length_persist', await page.locator('#silence-retry').isChecked()
      && await page.inputValue('#max-chars') === '80' && !await absent());
    for (const locale of ['zh-CN', 'en', 'ja', 'ko']) {
      await page.evaluate(locale => I18n.setLocale(locale), locale);
      for (const width of [1440, 390]) {
        await page.setViewportSize({ width, height: 960 });
        await showSettings('#advanced-settings');
        check(`settings_fit_${locale}_${width}`, await page.locator('#silence-retry').isVisible()
          && await page.locator('#advanced-settings .generation-settings-popover').evaluate(node => {
            const box = node.getBoundingClientRect();
            return box.left >= -1 && box.right <= innerWidth + 1 && node.scrollWidth <= node.clientWidth + 1;
          }));
      }
    }
    await page.evaluate(() => I18n.setLocale('zh-CN'));
    await showSettings('#advanced-settings');
    await shot('generation-settings-mobile', '#advanced-settings .generation-settings-popover');
    await page.setViewportSize({ width: 1440, height: 960 });
    await showSettings('#advanced-settings');
    await shot('generation-settings-desktop', '#advanced-settings .generation-settings-popover');
    await page.locator('#advanced-settings').evaluate(node => { node.open = false; });
    await page.locator('#result-box .main-result .creative-edit').click();
    await page.locator('#audio-edit-workspace').waitFor();
    await showSettings('#audio-edit-workspace .audio-edit-parameters');
    check('generated_audio_editor_has_retry_without_manual_duration_controls',
      await page.locator('#audio-edit-silence-retry').isVisible()
      && await page.locator('#audio-edit-silence-retry').isChecked()
      && !await page.locator('#audio-edit-duration,#audio-edit-maximum').count());
    for (const width of [1440, 390]) {
      await page.setViewportSize({ width, height: 960 });
      check(`generated_editor_options_fit_${width}`, await editorOptionsFit('audio-edit'));
      await shot(`generated-editor-settings-${width}`, '#audio-edit-workspace .generation-settings-popover');
    }
    await page.setViewportSize({ width: 1440, height: 960 });
    await page.locator('#audio-edit-back').click();
    await page.locator('.nav[data-view="scene"]').click();
    await showSettings('#scene-generation-settings');
    check('joint_dialogue_retry_is_on_by_default', await page.locator('#scene-silence-retry').isChecked());
    await page.locator('.dialogue-editor-area .cm-content').fill('S1:First line.\nS2:Second line.');
    await page.locator('#submit-scene').click();
    await page.waitForFunction(() => document.querySelector('#job-badge').dataset.status === 'complete'
      && state.resultJobs.scene?.status === 'complete');
    const scene = await api('/api/jobs/' + await page.locator('#result-box').getAttribute('data-job-id'));
    check('joint_dialogue_passes_the_retry_option', scene.recipe.silence_retry && scene.result.silence_retry
      && scene.result.silence_retry.retries === 0);
    await page.locator('.nav[data-view="audio-import"]').click();
    await page.locator('#audio-import-file').setInputFiles(process.env.DRAMA_STUDIO_REFERENCE);
    await page.waitForFunction(() => document.querySelector('#audio-import-transcript').value === 'Original source line.');
    await page.locator('#view-audio-import .audio-edit-line .cm-content').fill('Changed source line.');
    await showSettings('#view-audio-import .audio-edit-parameters');
    check('upload_editor_has_retry_without_manual_duration_controls',
      await page.locator('#audio-import-silence-retry').isVisible()
      && await page.locator('#audio-import-silence-retry').isChecked()
      && !await page.locator('#audio-import-duration,#audio-import-maximum').count());
    for (const width of [1440, 390]) {
      await page.setViewportSize({ width, height: 960 });
      check(`uploaded_editor_options_fit_${width}`, await editorOptionsFit('audio-import'));
      await shot(`uploaded-editor-settings-${width}`, '#view-audio-import .generation-settings-popover');
    }
    await page.setViewportSize({ width: 1440, height: 960 });
    await page.locator('#audio-import-submit').click();
    await page.waitForFunction(() => document.querySelector('#audio-import-result-badge').dataset.status === 'complete');
    const uploaded = await api('/api/jobs/' + await page.locator('#audio-import-result-box').getAttribute('data-job-id'));
    check('uploaded_edit_uses_automatic_length_and_retry', uploaded.recipe.silence_retry
      && uploaded.recipe.length_policy === 'automatic-v1' && !uploaded.recipe.duration_frames);
    check('no_browser_errors', !report.errors.length);
  } catch (error) {
    report.errors.push(error.stack || String(error));
    if (page) await page.screenshot({ path: path.join(folder, 'generation-settings-failure.png') }).catch(() => {});
  } finally {
    if (browser) await browser.close();
    report.passed = !report.errors.length && Object.values(report.checks).every(Boolean);
    fs.writeFileSync(path.join(folder, 'generation-settings-report.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report, null, 2));
    if (!report.passed) process.exitCode = 1;
  }
})();
