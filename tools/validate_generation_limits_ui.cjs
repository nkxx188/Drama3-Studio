/* Real UI requests and responsive controls; inference is explicitly intercepted. */
const fs = require('node:fs'), path = require('node:path'), { chromium } = require('playwright');
const root = path.resolve(__dirname, '..'), base = process.env.DRAMA_UI_URL;
const out = path.join(root, process.env.DRAMA_UI_REPORT || '.cache/generation-limits-ui/report.json');
const screenshots = path.dirname(out);
const report = { checks: {}, errors: [], screenshots: [], requests: [] };
const check = (name, value) => {
  report.checks[name] = Boolean(value);
  if (!value) throw new Error(name);
};
(async () => {
  let browser;
  try {
    fs.mkdirSync(screenshots, { recursive: true });
    browser = await chromium.launch({ channel: 'chrome', headless: true });
    const context = await browser.newContext({ locale: 'zh-CN', viewport: { width: 1440, height: 960 } });
    const page = await context.newPage();
    page.on('pageerror', error => report.errors.push(error.stack || error.message));
    await page.route('**/api/jobs', async route => {
      if (route.request().method() !== 'POST') return route.continue();
      report.requests.push(route.request().postDataJSON());
      return route.fulfill({ status: 202, json: { id: `length-check-${report.requests.length}` } });
    });
    await page.route('**/api/jobs/length-check-*', route => route.fulfill({
      json: { id: route.request().url().split('/').at(-1), mode: 'tts', status: 'cancelled',
        created_at: Date.now() / 1000, events: [], recipe: {}, result: {}, payload_summary: {} },
    }));
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.waitForFunction(() => document.documentElement.dataset.studioReady === 'true');
    const model = await context.request.get(base + '/api/model');
    report.selected_model = (await model.json()).selected_directory;
    const chooseLanguage = async locale => {
      await page.locator('#ui-language-trigger').click();
      await page.locator(`#ui-language-listbox [data-value="${locale}"]`).click();
    };
    const openSettings = async selector => page.locator(selector).evaluate(node => { node.open = true; });
    for (const locale of ['zh-CN', 'en', 'ja', 'ko']) {
      await chooseLanguage(locale);
      await page.locator('.nav[data-view="generate"]').click();
      await openSettings('#advanced-settings');
      check(`no_regular_duration_cap_${locale}`,
        !await page.locator('#max-seconds').count() && !await page.locator('#max-frames').isVisible());
      check(`regular_chunk_target_kept_${locale}`, await page.locator('#max-chars').isVisible());
      check(`target_duration_kept_${locale}`, await page.locator('#target-seconds').isVisible());
      await page.locator('.nav[data-view="scene"]').click();
      await openSettings('#scene-generation-settings');
      check(`no_joint_duration_cap_${locale}`, !await page.locator('#scene-max-seconds').count());
      await page.locator('[data-strategy="per_line"]').click();
      check(`no_per_turn_duration_cap_${locale}`, !await page.locator('#scene-max-seconds').count());
      await openSettings('#scene-generation-settings');
      check(`scene_chunk_target_kept_${locale}`, await page.locator('#scene-max-chars').isVisible());
      await page.locator('[data-strategy="joint"]').click();
    }
    await chooseLanguage('zh-CN');
    await page.locator('.nav[data-view="generate"]').click();
    await page.locator('.compose-panel > .editor-area .cm-content').fill('This is a full spoken sentence.');
    await page.locator('#submit-job').click();
    await page.waitForFunction(() => StudioFlow.flow.lastSubmit?.id === 'length-check-1');
    check('regular_request_uses_automatic_budget',
      report.requests[0].automatic_length === true && !Object.hasOwn(report.requests[0], 'max_new_tokens'));
    await page.locator('.nav[data-view="scene"]').click();
    await page.locator('.dialogue-editor-area .cm-content').fill('S1:Hello.\nS2:Nice to meet you.');
    await page.locator('#submit-scene').click();
    await page.waitForFunction(() => StudioFlow.flow.lastSubmit?.id === 'length-check-2');
    check('joint_request_uses_automatic_budget', report.requests[1].automatic_length === true
      && report.requests[1].scene_strategy === 'joint' && !Object.hasOwn(report.requests[1], 'max_new_tokens'));
    await page.locator('[data-strategy="per_line"]').click();
    await page.locator('#submit-scene').click();
    await page.waitForFunction(() => StudioFlow.flow.lastSubmit?.id === 'length-check-3');
    check('per_turn_request_uses_automatic_budget', report.requests[2].automatic_length === true
      && report.requests[2].scene_strategy === 'per_line' && !Object.hasOwn(report.requests[2], 'max_new_tokens'));
    for (const [width, height, name] of [[1440, 960, 'desktop'], [390, 844, 'mobile']]) {
      await page.setViewportSize({ width, height });
      for (const view of ['generate', 'scene']) {
        await page.locator(`.nav[data-view="${view}"]`).click();
        await openSettings(view === 'scene' ? '#scene-generation-settings' : '#advanced-settings');
        check(`settings_fit_${name}_${view}`, await page.evaluate(() =>
          document.documentElement.scrollWidth <= innerWidth + 1));
        const file = path.join(screenshots, `${name}-${view}.png`);
        await page.screenshot({ path: file, animations: 'disabled' });
        report.screenshots.push(file);
      }
    }
    check('no_page_errors', report.errors.length === 0);
  } catch (error) {
    report.errors.push(error.stack || error.message);
  } finally {
    await browser?.close();
    report.passed = !report.errors.length && Object.values(report.checks).every(Boolean);
    fs.writeFileSync(out, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify(report, null, 2));
    if (!report.passed) process.exitCode = 1;
  }
})();
