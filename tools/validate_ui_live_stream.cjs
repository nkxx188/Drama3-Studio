/* Exercise the production browser form, incremental HTTP input and real model.
   No inference is mocked. Playwright and Chrome are supplied by the host. */
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

const root = path.resolve(__dirname, '..');
const base = process.env.DRAMA_UI_URL || 'http://127.0.0.1:8877';
const output = path.join(root, 'research', 'ui-live-text-stream-validation.json');
const cache = path.join(root, '.cache', 'ui-browser');
fs.mkdirSync(cache, { recursive: true });
const profile = fs.mkdtempSync(path.join(cache, 'live-profile-'));
process.env.TEMP = process.env.TMP = path.join(root, '.cache', 'tmp');

(async () => {
  const report = {
    scope: 'Production Chrome form and real Drama3 model. Reference-conditioned live text input, late append after first delivered audio, tail flush and cancellation while waiting for lookahead.',
    base_url: base, started_at: new Date().toISOString(), checks: {}, errors: [],
  };
  let context, page;
  const waitFor = async (predicate, timeout = 300000) => {
    await page.waitForFunction(predicate, undefined, { timeout, polling: 500 });
  };
  try {
    context = await chromium.launchPersistentContext(profile, {
      channel: 'chrome', headless: true, viewport: { width: 1440, height: 1000 },
      args: ['--disable-gpu'],
    });
    page = await context.newPage();
    page.on('pageerror', error => report.errors.push(error.message));
    await page.goto(base, { waitUntil: 'networkidle' });
    await waitFor(() => state.voices.length > 0, 15000);
    const voice = await page.evaluate(() => state.voices.find(value => value.speaker === 0));
    if (!voice) throw new Error('A saved speaker 0 reference is required');
    report.voice_id = voice.id;
    await page.locator('#voice-picks .voice-pick').filter({ hasText: voice.name }).first().click();
    await page.selectOption('#mode', 'stream');
    await page.check('#stream-live');
    await page.fill('#chunks', '早上好。');
    await page.fill('#max-frames', '192');
    await page.click('#submit-job');
    await waitFor(() => Boolean(state.activeJob?.id), 15000);
    report.job_id = await page.evaluate(() => state.activeJob.id);
    console.log(JSON.stringify({ stage: 'created', job_id: report.job_id }));
    report.checks.live_controls_visible = await page.locator('#live-stream-controls').isVisible();
    await page.fill('#live-stream-text', '我们开始今天的测试。');
    await page.click('#push-stream');
    console.log(JSON.stringify({ stage: 'second_chunk_queued' }));
    let lastStage;
    const progress = setInterval(async () => {
      try {
        const value = await page.evaluate(() => ({
          status: state.activeJob?.status,
          stage: state.activeJob?.events?.at(-1)?.stage,
          frames: state.activeJob?.events?.at(-1)?.frames,
        }));
        const label = JSON.stringify(value);
        if (label !== lastStage) { console.log(label); lastStage = label; }
      } catch (_) { /* Browser may close after completion. */ }
    }, 10000);
    try {
      await waitFor(() => (
        state.activeJob?.status === 'error'
        || state.activeJob?.events?.some(event => event.stage === 'segment')
      ));
      const current = await page.evaluate(() => state.activeJob);
      if (current.status === 'error') throw new Error(current.error);
      report.checks.first_audio_before_input_finished = current.stream_input.finished === false
        && current.status === 'running';
      const audio = page.locator('#result-box .segment-item audio').first();
      await audio.evaluate(async element => { window.__liveAudio = element; await element.play(); });
      await waitFor(() => window.__liveAudio.currentTime > 0.05, 10000);
      report.checks.early_audio_playable = true;
      await page.fill('#live-stream-text', '所有内容都能按顺序听到。');
      await page.click('#push-stream');
      await page.click('#finish-stream');
      console.log(JSON.stringify({ stage: 'late_third_chunk_and_finish' }));
      await waitFor(() => ['complete', 'error', 'incomplete', 'cancelled'].includes(state.activeJob?.status));
      const done = await page.evaluate(() => state.activeJob);
      report.job = done;
      report.checks.complete = done.status === 'complete';
      report.checks.all_three_chunks_delivered = done.result?.segments?.length === 3;
      report.checks.lookahead_order = JSON.stringify(done.result?.segments?.map(value => value.lookahead))
        === JSON.stringify(['我们开始今天的测试。', '所有内容都能按顺序听到。', '']);
      report.checks.early_audio_element_preserved = await page.evaluate(() => (
        document.querySelector('#result-box .segment-item audio') === window.__liveAudio
      ));
      // A waiting one-chunk stream must cancel promptly and free the serial queue.
      await page.fill('#chunks', '这句话先等待。');
      await page.click('#submit-job');
      await page.waitForFunction(id => state.activeJob?.id !== id, done.id, { timeout: 15000 });
      const waitingId = await page.evaluate(() => state.activeJob.id);
      report.waiting_job_id = waitingId;
      await waitFor(() => state.activeJob.events?.some(event => event.stage === 'waiting_for_text'), 30000);
      await page.click('#cancel-job');
      await waitFor(() => state.activeJob.status === 'cancelled', 15000);
      report.cancelled_job = await page.evaluate(() => state.activeJob);
      report.checks.waiting_stream_cancelled = report.cancelled_job.status === 'cancelled'
        && report.cancelled_job.output === null;
      report.checks.no_browser_errors = report.errors.length === 0;
    } finally {
      clearInterval(progress);
    }
  } catch (error) {
    report.errors.push(error.stack || String(error));
    if (page) {
      report.last_job = await page.evaluate(() => state.activeJob).catch(() => null);
    }
  } finally {
    if (context) await context.close();
    if (path.resolve(profile).startsWith(path.resolve(cache) + path.sep)) {
      fs.rmSync(profile, { recursive: true, force: true });
    }
    report.passed = report.errors.length === 0
      && Object.values(report.checks).every(value => value === true);
    fs.writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({ passed: report.passed, checks: report.checks, errors: report.errors }, null, 2));
    if (!report.passed) process.exitCode = 1;
  }
})();
