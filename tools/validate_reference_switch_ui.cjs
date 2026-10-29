/* Real uploads and job recipes; controlled upload/ASR delays, no GPU inference. */
const fs = require('node:fs'), path = require('node:path'), { chromium } = require('playwright');
const base = process.env.DRAMA_UI_URL, folder = process.env.DRAMA_STUDIO_FOLDER;
const report = { checks: {}, errors: [], requests: [] };
const check = (name, value) => {
  report.checks[name] = Boolean(value);
  if (!value) throw new Error(name);
};
const reference = frequency => {
  const samples = 22050, buffer = Buffer.alloc(44 + samples * 2);
  buffer.write('RIFF'); buffer.writeUInt32LE(buffer.length - 8, 4); buffer.write('WAVEfmt ', 8);
  buffer.writeUInt32LE(16, 16); buffer.writeUInt16LE(1, 20); buffer.writeUInt16LE(1, 22);
  buffer.writeUInt32LE(22050, 24); buffer.writeUInt32LE(44100, 28);
  buffer.writeUInt16LE(2, 32); buffer.writeUInt16LE(16, 34); buffer.write('data', 36);
  buffer.writeUInt32LE(samples * 2, 40);
  for (let i = 0; i < samples; i++) buffer.writeInt16LE(Math.round(4000 * Math.sin(i * 2 * Math.PI * frequency / 22050)), 44 + i * 2);
  return { name: 'same-name.wav', mimeType: 'audio/wav', buffer };
};
(async () => {
  let browser;
  try {
    browser = await chromium.launch({ channel: 'chrome', headless: true });
    const context = await browser.newContext({ viewport: { width: 1440, height: 960 } });
    const page = await context.newPage();
    page.on('pageerror', error => report.errors.push(error.stack || error.message));
    let uploadDelay = 0, asrDelay = 0, lastJob = '';
    await page.route('**/api/reference-audio', async route => {
      const response = await route.fetch();
      if (uploadDelay) await new Promise(resolve => setTimeout(resolve, uploadDelay));
      await route.fulfill({ response });
    });
    await page.route('**/api/transcriptions', async route => {
      const id = route.request().postDataJSON().upload_id;
      if (asrDelay) await new Promise(resolve => setTimeout(resolve, asrDelay));
      await route.fulfill({ json: { id: `asr-${id}`, status: 'complete', result: { text: `Reference ${id}.` } } });
    });
    await page.route('**/api/reference-prewarm', route => route.fulfill({ json: { status: 'idle' } }));
    await page.route('**/api/jobs', route => {
      if (route.request().method() === 'POST') report.requests.push(route.request().postDataJSON());
      return route.continue();
    });
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.waitForFunction(() => document.documentElement.dataset.studioReady === 'true');
    await page.locator('.compose-panel > .editor-area .cm-content').fill('Reference switching test.');
    const choose = async frequency => {
      await page.locator('#clone-audio').setInputFiles(reference(frequency));
      await page.waitForFunction(() => state.cloneUpload && !state.cloneAsrPromise);
      return page.evaluate(() => state.cloneUpload.id);
    };
    const generate = async () => {
      await page.locator('#submit-job').click();
      await page.waitForFunction(previous => StudioFlow.flow.lastSubmit?.id !== previous
        && document.querySelector('#result-box').dataset.jobId === StudioFlow.flow.lastSubmit?.id
        && document.querySelector('#job-badge').dataset.status === 'complete', lastJob);
      lastJob = await page.evaluate(() => StudioFlow.flow.lastSubmit.id);
      const response = await context.request.get(`${base}/api/jobs/${lastJob}`);
      if (!response.ok()) throw new Error(await response.text());
      return response.json();
    };
    const firstId = await choose(220), first = await generate();
    check('first_reference_is_submitted', first.recipe.voice_bindings[0].id === firstId);
    const secondId = await choose(440), second = await generate(), repeated = await generate();
    check('same_name_and_duration_have_distinct_uploads', firstId !== secondId);
    check('first_job_after_replacement_uses_new_reference', second.recipe.voice_bindings[0].id === secondId
      && second.recipe.reference_snapshot[0].audio.includes(secondId));
    check('second_job_keeps_the_same_new_reference', repeated.recipe.voice_bindings[0].id === secondId);
    uploadDelay = 700; asrDelay = 500;
    await page.locator('#clone-audio').setInputFiles(reference(660));
    const duringUpload = await generate();
    const thirdId = await page.evaluate(() => state.cloneUpload.id);
    check('generation_waits_for_current_upload_and_transcription', thirdId !== secondId
      && duringUpload.recipe.voice_bindings[0].id === thirdId
      && duringUpload.recipe.voice_bindings[0].text === `Reference ${thirdId}.`);
    await page.locator('#clone-audio').setInputFiles(reference(880));
    await page.locator('#clone-audio').setInputFiles(reference(1100));
    const rapid = await generate();
    const currentId = await page.evaluate(() => state.cloneUpload.id);
    check('rapid_replacement_ignores_previous_upload_results', rapid.recipe.voice_bindings[0].id === currentId
      && rapid.recipe.voice_bindings[0].text === `Reference ${currentId}.`);
    const updateFileInput = async (frequency, event = null) => page.evaluate(({ bytes, event }) => {
      const transfer = new DataTransfer();
      transfer.items.add(new File([Uint8Array.from(atob(bytes), value => value.charCodeAt(0))],
        'same-name.wav', { type: 'audio/wav' }));
      const input = document.querySelector('#clone-audio');
      input.files = transfer.files;
      if (event) input.dispatchEvent(new Event(event, { bubbles: true }));
    }, { bytes: reference(frequency).buffer.toString('base64'), event });
    await updateFileInput(1320, 'input');
    const inputOnly = await generate();
    check('file_input_event_cannot_submit_the_previous_reference', inputOnly.recipe.voice_bindings[0].id !== currentId);
    await updateFileInput(1540);
    const deferredChange = await generate();
    check('submission_reads_current_file_even_before_change_notification',
      deferredChange.recipe.voice_bindings[0].id !== inputOnly.recipe.voice_bindings[0].id);
    const latestId = deferredChange.recipe.voice_bindings[0].id;
    const revision = await page.evaluate(() => state.cloneRevision);
    await page.locator('#clone-audio').dispatchEvent('change');
    check('late_change_notification_keeps_the_prepared_reference',
      await page.evaluate(({ id, revision }) => state.cloneUpload.id === id && state.cloneRevision === revision,
        { id: latestId, revision }));
    await page.locator('#clone-upload-preview').evaluate(node => { node.open = true; });
    await page.waitForFunction(() => document.querySelector('#clone-file-audio').duration > 0);
    await page.evaluate(() => StudioAudio.setSelection(document.querySelector('#clone-file-audio'), 0, .75, { emit: true }));
    await page.locator('#crop-clone-audio').click();
    await page.waitForFunction(old => state.cloneUpload?.id && state.cloneUpload.id !== old && !state.cloneAsrPromise, latestId);
    const croppedId = await page.evaluate(() => state.cloneUpload.id);
    const cropped = await generate();
    check('submission_preserves_a_crop_instead_of_reverting_to_the_file_input',
      croppedId !== latestId && cropped.recipe.voice_bindings[0].id === croppedId);
    await page.locator('#restore-clone-audio').click();
    const restored = await generate();
    check('restoring_original_audio_preserves_its_upload_binding', restored.recipe.voice_bindings[0].id === latestId);
    await page.evaluate(() => StudioFlow.save());
    await page.reload({ waitUntil: 'networkidle' });
    await page.waitForFunction(() => document.documentElement.dataset.studioReady === 'true');
    check('restored_draft_reuses_the_matching_uploaded_reference', await page.evaluate(id =>
      state.cloneUpload?.id === id && state.cloneUploadSource?.file === state.cloneFile, latestId));
    const draft = await generate();
    check('generation_after_reload_uses_the_restored_reference', draft.recipe.voice_bindings[0].id === latestId);
    check('no_browser_errors', !report.errors.length);
  } catch (error) {
    report.errors.push(error.stack || String(error));
  } finally {
    if (browser) await browser.close();
    report.passed = !report.errors.length && Object.values(report.checks).every(Boolean);
    fs.writeFileSync(path.join(folder, 'reference-switch-report.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify({ ...report, requests: report.requests.map(request => ({
      voice_bindings: request.voice_bindings, mode: request.mode,
    })) }, null, 2));
    if (!report.passed) process.exitCode = 1;
  }
})();
