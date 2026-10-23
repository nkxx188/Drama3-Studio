/* Real reference storage and UI; controlled warm-up progress, no GPU inference. */
const fs = require('node:fs'), path = require('node:path'), { chromium } = require('playwright');
const root = path.resolve(__dirname, '..'), base = process.env.DRAMA_UI_URL;
const folder = process.env.DRAMA_STUDIO_FOLDER;
const report = { scope: 'Automatic reference warm-up, replacement, cancellation, localization and generation access. Controlled warm-up backend; real local reference uploads.',
  checks: {}, errors: [], screenshots: [], requests: [] };
(async () => {
  let browser, page;
  try {
    browser = await chromium.launch({ channel: 'chrome', headless: true });
    const context = await browser.newContext({ viewport: { width: 1440, height: 960 } });
    await context.addInitScript(() => localStorage.setItem('drama3.ui.language', 'zh-CN'));
    page = await context.newPage();
    page.on('pageerror', error => report.errors.push(error.stack || error.message));
    let serial = 0, latest = null, unsupported = false;
    const tasks = new Map();
    await page.route('**/api/reference-prewarm', async route => {
      const payload = route.request().postDataJSON();
      report.requests.push(payload);
      if (!payload.voice_bindings?.length && !payload.uploaded_references?.length && !payload.voice_ids?.length) {
        latest = null;
        return route.fulfill({ json: { status: 'idle' } });
      }
      if (unsupported) return route.fulfill({ json: { status: 'unsupported' } });
      if (latest) Object.assign(latest, { status: 'cancelled', stage: 'cancelled' });
      latest = { id: `warm-${++serial}`, status: 'running', stage: 'warming_prefix' };
      tasks.set(latest.id, latest);
      return route.fulfill({ json: latest });
    });
    await page.route('**/api/reference-prewarm/*', route => {
      const id = route.request().url().split('/').at(-1);
      if (id === 'cancel') {
        if (!latest || !['queued', 'running'].includes(latest.status)) {
          latest = null;
          return route.fulfill({ json: { status: 'idle' } });
        }
        latest.status = latest.stage = 'cancelled';
        return route.fulfill({ json: latest });
      }
      return route.fulfill({ json: tasks.get(id) || { status: 'idle' } });
    });
    const check = (name, value) => {
      report.checks[name] = Boolean(value);
      if (!value) throw new Error(name);
    };
    const choose = async index => {
      await page.locator('#saved-voice-options > summary').click();
      await page.locator('#voice-picks [data-voice]').nth(index).click();
    };
    const status = () => page.locator('#reference-prewarm-status');
    const started = async previous => {
      await page.waitForFunction(previous => state.referencePrewarmTask?.id
        && state.referencePrewarmTask.id !== previous
        && state.referencePrewarmTask.status === 'running', previous);
      return page.evaluate(() => state.referencePrewarmTask.id);
    };
    const shot = async name => {
      const file = path.join(folder, name + '.png');
      await page.screenshot({ path: file }); report.screenshots.push(file);
    };
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.waitForFunction(() => document.documentElement.dataset.studioReady === 'true');
    await choose(0);
    let id = await started(null);
    check('selecting_a_library_voice_starts_warmup', report.requests.at(-1).voice_bindings[0].voice_id);
    check('warmup_can_be_cancelled_and_does_not_disable_generation',
      await page.locator('#reference-prewarm-cancel').isVisible()
      && await page.locator('#submit-job').isEnabled());
    for (const locale of ['en', 'ja', 'ko', 'zh-CN']) {
      await page.evaluate(locale => I18n.setLocale(locale), locale);
      for (const width of [1440, 390]) {
        await page.setViewportSize({ width, height: width === 390 ? 844 : 960 });
        check(`warmup_status_fits_${locale}_${width}`, await status().isVisible()
          && await status().evaluate(element =>
          element.clientWidth > 0 && element.scrollWidth <= element.clientWidth + 1
          && document.documentElement.scrollWidth <= innerWidth + 1));
      }
    }
    await shot('reference-prewarm-mobile');
    await page.setViewportSize({ width: 1440, height: 960 });
    await shot('reference-prewarm-desktop');
    await choose(1);
    const old = id; id = await started(old);
    check('changing_the_voice_replaces_the_old_task', tasks.get(old).status === 'cancelled');
    latest.status = 'ready'; latest.stage = 'complete';
    await page.waitForFunction(() => state.referencePrewarmTask?.status === 'ready');
    check('ready_status_is_clear_and_hides_cancel',
      !await page.locator('#reference-prewarm-cancel').isVisible()
      && await status().evaluate(element =>
        element.dataset.state === 'ready'
        && element.querySelector('#reference-prewarm-message').textContent
          === I18n.t('Voice reference ready')));
    check('warmup_hint_uses_prominent_inline_treatment',
      await status().evaluate(element => {
        const style = getComputedStyle(element);
        return parseFloat(style.fontSize) >= 15
          && parseInt(style.fontWeight, 10) >= 600
          && parseFloat(style.borderWidth) === 0
          && style.backgroundColor === 'rgba(0, 0, 0, 0)'
          && getComputedStyle(element, '::before').width === '8px';
      }));
    await page.locator('.selected-library-voice .selected-voice-clear').click();
    await status().waitFor({ state: 'hidden' });
    check('removing_the_voice_clears_ready_status', !await status().isVisible());
    await page.locator('#clone-audio').setInputFiles(process.env.DRAMA_STUDIO_REFERENCE);
    await page.waitForFunction(() => state.cloneUpload && !state.cloneAsrPromise);
    await page.locator('#clone-transcript').fill('参考音频的台词。');
    id = await started(id);
    const binding = report.requests.at(-1).voice_bindings[0];
    check('uploaded_audio_is_warmed_with_its_current_transcript',
      /^[a-f0-9]{32}$/.test(binding.id) && binding.text === '参考音频的台词。');
    const count = report.requests.length;
    await page.locator('.compose-panel > .editor-area .cm-content').fill('更改正文不会重新预热参考。');
    await page.waitForTimeout(1100);
    check('changing_target_text_does_not_rewarm', count === report.requests.length);
    await page.locator('#reference-prewarm-cancel').click();
    await status().waitFor({ state: 'hidden' });
    check('cancellation_clears_progress', !await status().isVisible());
    await page.evaluate(() => ReferencePrewarm.schedule(0));
    const beforeLeave = await started(id);
    await page.evaluate(() => view('scene'));
    await page.waitForFunction(() => state.referencePrewarmTask?.status === 'cancelled');
    check('opening_dialogue_cancels_single_voice_warmup', tasks.get(beforeLeave).status === 'cancelled');
    await page.evaluate(() => view('generate'));
    id = await started(beforeLeave);
    check('returning_to_generation_warms_its_current_reference', id !== beforeLeave);
    await page.evaluate(() => switchModel(state.model.selected_directory));
    const beforeModel = id; id = await started(beforeModel);
    check('model_selection_restarts_reference_warmup', tasks.get(beforeModel).status === 'cancelled');
    await page.locator('#remove-clone-audio').click();
    unsupported = true;
    await choose(0);
    await page.waitForFunction(() => state.referencePrewarmTask?.status === 'unsupported');
    check('unsupported_backend_adds_no_extra_status', !await status().isVisible());
    check('no_browser_errors', report.errors.length === 0);
  } catch (error) {
    report.errors.push(error.stack || String(error));
    if (page) {
      const file = path.join(folder, 'reference-prewarm-failure.png');
      await page.screenshot({ path: file }).catch(() => {}); report.screenshots.push(file);
    }
  } finally {
    if (browser) await browser.close();
    report.passed = !report.errors.length && Object.values(report.checks).every(Boolean);
    fs.writeFileSync(path.join(root, 'research/2026-10-22-reference-prewarm-ui.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report, null, 2));
    if (!report.passed) process.exitCode = 1;
  }
})();
