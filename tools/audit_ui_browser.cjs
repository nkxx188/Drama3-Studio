/* Browser interaction audit. Uses an existing Playwright installation and Chrome.
   All browser profiles, screenshots and downloads stay on the project drive.
   This audits real saved audio; synthetic job states test polling controls only. */
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

const root = path.resolve(__dirname, '..');
const base = process.env.DRAMA_UI_URL || 'http://127.0.0.1:8876';
const reportFile = process.env.DRAMA_UI_BROWSER_REPORT
  || path.join(root, 'research', 'ui-browser-audit.json');
const artifactRoot = path.join(root, '.cache', 'ui-browser');
fs.mkdirSync(artifactRoot, { recursive: true });
const profile = fs.mkdtempSync(path.join(artifactRoot, 'profile-'));
const screenshots = path.join(root, 'research', 'ui-browser-screenshots');
fs.mkdirSync(screenshots, { recursive: true });
process.env.TEMP = process.env.TMP = path.join(root, '.cache', 'tmp');

(async () => {
  const report = {
    scope: 'Chrome click/layout and real saved-WAV playback. Synthetic job-state checks exercise UI polling without running inference.',
    base_url: base, checked_at: new Date().toISOString(), checks: {}, errors: [],
    screenshots: [],
  };
  let context;
  try {
    context = await chromium.launchPersistentContext(profile, {
      channel: 'chrome', headless: true, viewport: { width: 1440, height: 1000 },
      downloadsPath: path.join(artifactRoot, 'downloads'), args: ['--disable-gpu'],
    });
    report.browser = context.browser().version();
    const page = await context.newPage();
    page.on('pageerror', error => report.errors.push(error.message));
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.waitForFunction(() => document.querySelector('#engine-status').textContent !== 'Checking hardware…');
    for (const name of ['generate', 'scene', 'voices', 'takes', 'settings']) {
      await page.locator(`.nav[data-view="${name}"]`).click();
      report.checks[`desktop_${name}`] = await page.locator(`#view-${name}`).isVisible();
      const target = path.join(screenshots, `desktop-${name}.png`);
      await page.screenshot({ path: target, fullPage: true });
      report.screenshots.push(target);
    }
    const actualJobs = await page.evaluate(() => state.jobs);
    await page.locator('.nav[data-view="voices"]').click();
    const voiceAudio = page.locator('#voice-list audio').first();
    if (await voiceAudio.count()) {
      await voiceAudio.evaluate(async element => { window.__voiceAudio = element; await element.play(); });
      await page.waitForFunction(() => window.__voiceAudio.currentTime > 0.05, undefined, { timeout: 10000 });
      report.checks.voice_reference_playback = await voiceAudio.evaluate(element => (
        !element.error && Number.isFinite(element.duration)
      ));
      await voiceAudio.evaluate(element => element.pause());
    } else {
      report.checks.voice_reference_playback = false;
    }
    const completed = actualJobs.find(job => job.output && job.take_file);
    if (completed) {
      report.audio_job = completed.id;
      await page.locator('.nav[data-view="takes"]').click();
      const button = page.locator('.use-take[data-mode="infill"]').first();
      await button.click();
      report.checks.take_to_infill = await page.locator('#mode').inputValue() === 'infill'
        && Boolean(await page.locator('#source-take').inputValue())
        && await page.locator('#infill-fields').isVisible();
      await page.waitForFunction(() => state.sourceTake !== null);
      report.checks.infill_transcript_recovers_actual_take = await page.evaluate(() => (
        document.querySelector('#text').value === state.sourceTake.text
        && document.querySelector('#infill-source-preview').classList.contains('hidden') === false
      ));
      const initialFrames = await page.evaluate(() => ({
        start: document.querySelector('#start-frame').value,
        end: document.querySelector('#end-frame').value,
      }));
      await page.locator('#start-time').dispatchEvent('input');
      report.checks.infill_time_frame_roundtrip = await page.evaluate(expected => (
        document.querySelector('#start-frame').value === expected.start
        && document.querySelector('#end-frame').value === expected.end
      ), initialFrames);
      await page.fill('#start-time', '0.101');
      await page.fill('#end-time', '0.101');
      report.checks.infill_insertion_is_zero_width = await page.evaluate(() => (
        document.querySelector('#start-frame').value === document.querySelector('#end-frame').value
      ));
      await page.locator('#infill-source-audio').evaluate(async audio => {
        await audio.play();
        audio.pause();
        audio.currentTime = 0.5;
      });
      await page.click('#mark-infill-start');
      report.checks.infill_playhead_sets_frame = await page.locator('#start-frame').inputValue() === '10';
      await page.evaluate(job => {
        renderJob(job);
        window.__auditAudio = document.querySelector('#result-box .main-result audio');
      }, completed);
      await page.evaluate(async () => { await window.__auditAudio.play(); });
      await page.waitForFunction(() => window.__auditAudio.currentTime > 0.1);
      report.checks.saved_wav_playback = await page.evaluate(() => (
        !window.__auditAudio.paused && !window.__auditAudio.error
        && Number.isFinite(window.__auditAudio.duration)
      ));
      await page.evaluate(job => renderJob(job), completed);
      report.checks.poll_preserves_audio_element = await page.evaluate(() => (
        window.__auditAudio === document.querySelector('#result-box .main-result audio')
      ));
      await page.evaluate(() => window.__auditAudio.pause());
    } else {
      report.checks.saved_wav_playback = false;
    }
    await page.evaluate(() => {
      const button = document.querySelector('#cancel-job');
      button.disabled = true;
      renderJob({ id: 'audit-next-job', status: 'running', events: [] });
    });
    report.checks.next_job_cancel_enabled = await page.locator('#cancel-job').isEnabled();
    await page.evaluate(() => {
      state.voices = [{
        id: 'audit-voice', name: '<b id="audit-name">Voice & sample</b>',
        transcript: 'A reference.', speaker: 0, audio_url: '#',
      }];
      renderVoices();
    });
    report.checks.voice_names_are_literal_text = await page.locator('#audit-name').count() === 0
      && (await page.locator('#voice-picks').textContent()).includes('<b id="audit-name">');
    await page.reload({ waitUntil: 'networkidle' });
    await page.setViewportSize({ width: 390, height: 844 });
    report.checks.mobile_navigation_available = await page.locator('.nav[data-view="voices"]').isVisible();
    report.checks.mobile_no_horizontal_overflow = await page.evaluate(() => (
      document.documentElement.scrollWidth <= window.innerWidth
    ));
    const mobile = path.join(screenshots, 'mobile-generate.png');
    await page.screenshot({ path: mobile, fullPage: true });
    report.screenshots.push(mobile);
    report.checks.no_browser_script_errors = report.errors.length === 0;
  } catch (error) {
    report.errors.push(error.stack || String(error));
  } finally {
    if (context) await context.close();
    // Only remove this run's freshly-created browser profile, after closing it.
    const resolved = path.resolve(profile);
    if (resolved.startsWith(path.resolve(artifactRoot) + path.sep)) {
      fs.rmSync(resolved, { recursive: true, force: true });
    }
    report.passed = report.errors.length === 0
      && Object.values(report.checks).every(value => value === true);
    fs.writeFileSync(reportFile, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({ passed: report.passed, checks: report.checks, errors: report.errors }, null, 2));
    if (!report.passed) process.exitCode = 1;
  }
})();
