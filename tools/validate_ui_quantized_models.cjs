/* Real Chrome model discovery/selection against the local engine UI. */
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');
const root = path.resolve(__dirname, '..');
const base = process.env.DRAMA_UI_URL || 'http://127.0.0.1:8877';
const cache = path.join(root, '.cache', 'ui-browser');
fs.mkdirSync(cache, { recursive: true });
const profile = fs.mkdtempSync(path.join(cache, 'quant-profile-'));
process.env.TEMP = process.env.TMP = path.join(root, '.cache', 'tmp');

(async () => {
  const report = { checked_at: new Date().toISOString(), checks: {}, errors: [], models: [] };
  let context;
  let selected;
  try {
    const jobs = await (await fetch(`${base}/api/jobs`)).json();
    if (jobs.jobs.some(job => ['queued', 'running'].includes(job.status))) {
      throw new Error('Model selection audit requires an idle serialized job queue');
    }
    selected = (await (await fetch(`${base}/api/model`)).json()).selected_directory;
    context = await chromium.launchPersistentContext(profile, {
      channel: 'chrome', headless: true, viewport: { width: 1440, height: 1000 },
      args: ['--disable-gpu'],
    });
    const page = await context.newPage();
    page.on('pageerror', error => report.errors.push(error.message));
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.click('.nav[data-view="settings"]');
    await page.waitForFunction(() => document.querySelector('#model-badge').textContent === 'READY');
    const models = (await (await fetch(`${base}/api/model`)).json()).available_models;
    report.checks.installed_models_match_options = (
      await page.locator('#installed-model option').count()) === models.length + 1;
    report.checks.w4a8_installed = models.some(model => model.directory.endsWith('drama-3-w4a8'));
    for (const model of models) {
      await page.selectOption('#installed-model', model.directory);
      await page.waitForFunction(directory => (
        state.model?.inspection?.directory === directory
      ), model.directory);
      const preview = await page.evaluate(() => state.model);
      if (preview.selected_directory !== model.directory) {
        await page.click('#select-model');
        await page.waitForFunction(directory => (
          state.model?.selected_directory === directory
        ), model.directory);
      }
      const modelState = await page.evaluate(() => api('/api/model'));
      const row = {
        label: model.label, directory: model.directory,
        ready: modelState.inspection.ready,
        selected: modelState.selected_directory === model.directory,
        direct: modelState.uses_checkpoint_directly,
        selection_metadata_matches_model: Boolean(modelState.selected_format)
          && modelState.available_models.some(available => available.directory === model.directory
            && available.name === model.name),
        no_duplicate_cache: modelState.uses_checkpoint_directly,
      };
      report.models.push(row);
      report.checks[model.label] = row.ready && row.selected && row.direct
        && row.selection_metadata_matches_model && row.no_duplicate_cache;
    }
    await page.setViewportSize({ width: 390, height: 844 });
    report.checks.mobile_no_horizontal_overflow = await page.evaluate(() => (
      document.documentElement.scrollWidth <= window.innerWidth + 1
    ));
    const screenshot = path.join(root, 'research/ui-browser-screenshots/phase19-model-selection.png');
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.screenshot({ path: screenshot, fullPage: true });
    report.screenshot = screenshot;
  } catch (error) {
    report.errors.push(error.stack || String(error));
  } finally {
    if (selected) {
      const restored = await fetch(`${base}/api/model/select`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ directory: selected }),
      });
      report.checks.original_selection_restored = restored.ok;
    }
    if (context) await context.close();
    if (path.resolve(profile).startsWith(path.resolve(cache) + path.sep)) {
      fs.rmSync(profile, { recursive: true, force: true });
    }
    report.passed = !report.errors.length && Object.values(report.checks).every(Boolean);
    fs.writeFileSync(path.join(root, 'research/phase19-ui-quantized-models-validation.json'),
      JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify(report, null, 2));
    if (!report.passed) process.exitCode = 1;
  }
})();
