/* Exercise real deletion routes against the isolated Studio fixture. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { setTimeout: delay } = require('node:timers/promises');
const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 390, height: 844 }, locale: 'zh-CN' });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const base = process.env.DRAMA_UI_URL;
    const request = async (route, data) => {
      const response = await (data ? page.request.post(base + route, { data }) : page.request.get(base + route));
      assert(response.ok(), `${route}: ${await response.text()}`);
      return response.json();
    };
    const createJob = async (data, status = 'complete') => {
      const { id } = await request('/api/jobs', data);
      for (let attempt = 0; attempt < 60; attempt++) {
        const job = await request(`/api/jobs/${id}`);
        if (job.status === status) return job;
        assert(['queued', 'running'].includes(job.status), job.error || job.status);
        await delay(100);
      }
      throw new Error(`Fixture job ${id} did not finish.`);
    };
    const ready = () => page.waitForSelector('html[data-startup="ready"]');
    const view = name => page.locator(`.nav[data-view="${name}"]`).click();
    const row = job => page.locator(`[data-job-id="${job.id}"].job-card`);
    const confirm = async () => {
      await page.locator('#delete-confirm').click();
      await page.locator('#delete-dialog').waitFor({ state: 'hidden' });
    };
    const singleDelete = async job => {
      await row(job).locator('summary').click();
      await row(job).locator('.history-delete').click();
      await confirm();
      assert.equal(await row(job).count(), 0);
    };
    let releaseLabels;
    const labels = new Promise(resolve => { releaseLabels = resolve; });
    const holdLabels = async route => { await labels; await route.continue(); };
    await page.route('**/assets/i18n.js', holdLabels);
    const starting = page.goto(base);
    try {
      await page.locator('#startup-state').waitFor({ state: 'visible', timeout: 5000 });
      assert(await page.locator('#workspace-content').evaluate(node => node.inert));
    } finally {
      releaseLabels(); await starting; await page.unroute('**/assets/i18n.js', holdLabels);
    }
    await ready();
    const { voices } = await request('/api/voices');
    const voice = voices.find(voice => voice.name === 'Fixture A');
    const original = await createJob({ text: 'Original fixture.', voice_ids: [voice.id] });
    const edit = await createJob({ source_version: { job_id: original.id }, operation: 'retry' });
    const failed = await createJob({ text: 'Failed fixture.' }, 'error');

    await view('voices');
    const voiceCard = page.locator(`[data-voice-id="${voice.id}"]`);
    await voiceCard.locator('.voice-delete').click();
    await page.locator('#delete-cancel').click();
    assert(await voiceCard.isVisible());
    await page.locator('#voices-select').click();
    await page.locator('#voice-search').fill('Fixture A');
    await page.locator('#voices-cleanup input').check();
    assert.equal(await page.locator('#voice-list .cleanup-check:checked').count(), 1);
    await page.locator('#voices-cleanup button').click();
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await confirm();
    assert.equal(await voiceCard.count(), 0);
    assert(fs.existsSync(voice.audio_path));
    await page.reload();
    await ready();
    assert.equal(await page.locator('#voice-list .voice-card').count(), 1);

    await view('takes');
    await page.locator('[data-history-filter="unfinished"]').click();
    await page.locator('#jobs-select').click();
    await page.locator('#jobs-cleanup input').check();
    assert.equal(await page.locator('#job-list .cleanup-check:checked').count(), 1);
    assert(await row(failed).locator('input').isChecked());
    assert(!await row(edit).isVisible());
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.screenshot({ path: path.join(process.env.DRAMA_STUDIO_FOLDER, 'cleanup-history-mobile.png') });
    await page.locator('#jobs-cleanup button').click();
    await confirm();
    assert(!fs.existsSync(path.join(process.env.DRAMA_STUDIO_FOLDER, 'ui_data', 'jobs', failed.id)));

    await page.setViewportSize({ width: 1440, height: 900 });
    await page.locator('[data-history-filter="all"]').click();
    const archiveSeed = await createJob({ text: 'Archive fixture seed.' });
    await request('/__test__/archive', { source: archiveSeed.id });
    await page.locator('#refresh-jobs').click();
    await page.locator('#load-older-results').click();
    await row(original).waitFor();
    assert(await page.locator('#job-list .job-card').count() > 100);
    await singleDelete(original);
    assert(fs.existsSync(original.output));
    await page.reload();
    await ready();
    await page.locator('#load-older-results').click();
    await row(edit).waitFor();
    assert.equal(await row(original).count(), 0);
    assert(await row(edit).isVisible());
    await singleDelete(edit);
    assert(!fs.existsSync(edit.output));
    assert(!fs.existsSync(original.output));
    assert(!fs.existsSync(voice.audio_path));
    assert.deepEqual(errors, []);
    console.log('Browser checks passed: startup feedback, filters, deletion, older results, and dependency retention.');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
