/* Browser model-folder discovery and local model selection.
   Reads the existing checkpoint and does not download model files. */
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');
const root = path.resolve(__dirname, '..');
const base = process.env.DRAMA_UI_URL || 'http://127.0.0.1:8877';
const cache = path.join(root, '.cache', 'ui-browser');
fs.mkdirSync(cache, { recursive: true });
const profile = fs.mkdtempSync(path.join(cache, 'model-profile-'));
process.env.TEMP = process.env.TMP = path.join(root, '.cache', 'tmp');

(async () => {
  const report = {
    scope: 'Real Chrome checks for local model discovery, model catalog rendering, and download controls. No model download.',
    base_url: base, checked_at: new Date().toISOString(), checks: {}, errors: [],
  };
  let context;
  try {
    context = await chromium.launchPersistentContext(profile, {
      channel: 'chrome', headless: true, viewport: { width: 1440, height: 1000 },
      args: ['--disable-gpu'],
    });
    const page = await context.newPage();
    page.on('pageerror', error => report.errors.push(error.message));
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.locator('.nav[data-view="settings"]').click();
    await page.waitForFunction(() => document.querySelector('#model-badge').textContent === 'READY');
    const selected = (await page.evaluate(() => api('/api/model'))).selected_directory;
    const catalog = await page.evaluate(() => api('/api/model/catalog'));
    report.model_directory = selected;
    report.download_catalog_entries = catalog.models?.length || 0;
    report.checks.selected_model_is_project_checkpoint = selected === path.join(root, 'checkpoints', 'drama-3');
    report.checks.default_model_location_listed = await page.locator('#model-root-list')
      .textContent().then(text => text.includes(path.join(root, 'checkpoints')));
    report.checks.folder_picker_visible = await page.locator('#browse-model-directory').isVisible();
    const missing = path.join(root, '.cache', 'ui-model-not-downloaded');
    if (fs.existsSync(missing)) throw new Error('The missing-folder fixture already exists');
    const inspection = await page.evaluate(async directory => (
      api('/api/model/inspect', { method: 'POST', body: JSON.stringify({ directory }) })
    ), missing);
    report.checks.missing_folder_detected_without_creation = !inspection.inspection.ready
      && !fs.existsSync(missing);
    report.checks.duplicate_path_controls_removed = await page.locator(
      '#model-directory, #inspect-model, #model-active-folder, #model-folder-status, #model-cache-status',
    ).count() === 0;
    report.checks.download_catalog_covers_requested_formats =
      ['fp8', 'int4', 'int8', 'w4a8', 'w6a8'].every(id => catalog.models.some(model => model.id === id))
      && catalog.models.some(model => model.id.startsWith('gguf-'));
    report.checks.download_controls_visible = await page.locator('#model-download-choice').isVisible()
      && await page.locator('#model-download-sources').isVisible()
      && await page.locator('#download-model').isVisible()
      && await page.locator('#model-download-progress').count() === 1;
    report.checks.source_availability_is_respected = await page.locator(
      '.model-source-option[data-source="modelscope"]',
    ).isEnabled();
    report.checks.model_selection_available = await page.locator('#installed-model').isVisible()
      && await page.locator('#select-model').isVisible();
    const screenshot = path.join(root, 'research', 'ui-browser-screenshots', 'desktop-model-setup.png');
    await page.screenshot({ path: screenshot, fullPage: true });
    report.screenshot = screenshot;
    report.checks.no_script_errors = report.errors.length === 0;
  } catch (error) {
    report.errors.push(error.stack || String(error));
  } finally {
    if (context) await context.close();
    if (path.resolve(profile).startsWith(path.resolve(cache) + path.sep)) {
      fs.rmSync(profile, { recursive: true, force: true });
    }
    report.passed = report.errors.length === 0 && Object.values(report.checks).every(value => value === true)
      && report.download_catalog_entries >= 10;
    fs.writeFileSync(path.join(root, 'research', 'ui-model-setup-validation.json'),
      JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({ passed: report.passed, checks: report.checks, errors: report.errors }, null, 2));
    if (!report.passed) process.exitCode = 1;
  }
})();
