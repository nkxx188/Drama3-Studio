/* Real audio decode, pointer selection, crop HTTP and editable codec boundaries. */
const fs = require('node:fs'), path = require('node:path');
const { chromium } = require('playwright');
const root = path.resolve(__dirname, '..'), base = process.env.DRAMA_UI_URL;
const report = { scope: 'Synthetic real audio and codes; actual crop/inspect APIs; generation and ASR responses stubbed.',
  checks: {}, errors: [], screenshots: [] };
const resultPath = path.join(root, 'research/2026-10-19-waveform-creation-regression.json');
(async () => {
  let browser, page;
  try {
    browser = await chromium.launch({ channel: 'chrome', headless: true });
    const context = await browser.newContext({ viewport: { width: 1440, height: 1080 }, locale: 'zh-CN' });
    page = await context.newPage();
    page.on('pageerror', error => report.errors.push(error.stack || error.message));
    const submissions = [], transcriptions = [];
    await page.route('**/api/jobs', route => {
      if (route.request().method() !== 'POST') return route.continue();
      submissions.push(route.request().postDataJSON());
      return route.fulfill({ status: 202, json: { id: 'wave-submission-fixture' } });
    });
    await page.route('**/api/jobs/wave-submission-fixture', route => route.fulfill({
      json: { id: 'wave-submission-fixture', status: 'cancelled', mode: 'tts', events: [] },
    }));
    let asrAvailable = false;
    await page.route('**/api/transcriptions', route => {
      transcriptions.push(route.request().postDataJSON());
      return route.fulfill({ status: 202, json: asrAvailable ? {
        id: 'wave-asr-fixture', status: 'complete', result: { text: '新片段的逐字稿。' },
      } : { id: 'wave-asr-fixture', status: 'skipped', reason: 'model_unavailable' } });
    });
    const choose = async (id, value) => {
      await page.locator(`#${id}-trigger`).click();
      await page.locator(`#${id}-listbox .custom-option[data-value="${value}"]`).click();
    };
    const waitWave = async selector => {
      await page.locator(selector).locator('..').scrollIntoViewIfNeeded();
      await page.waitForFunction(selector => document.querySelector(selector)?.closest('.studio-player')
        ?.querySelectorAll('.player-wave rect').length === 80, selector);
    };
    const drag = async (selector, start, end) => {
      const node = page.locator(selector).locator('..').locator('.player-wave-surface');
      await node.scrollIntoViewIfNeeded();
      const box = await node.boundingBox();
      await page.mouse.move(box.x + box.width * start, box.y + box.height / 2);
      await page.mouse.down();
      await page.mouse.move(box.x + box.width * end, box.y + box.height / 2, { steps: 12 });
      await page.mouse.up();
      return page.locator(selector).evaluate(audio => StudioAudio.selection(audio));
    };
    const moveRegion = async (selector, distance) => {
      const player = page.locator(selector).locator('..');
      await player.locator('.player-wave-surface').scrollIntoViewIfNeeded();
      const surface = await player.locator('.player-wave-surface').boundingBox();
      const region = await player.locator('.player-region').boundingBox();
      const center = (region.x + region.width / 2 - surface.x) / surface.width;
      return drag(selector, center, center + distance);
    };
    const resizeRegion = async (selector, edge, target) => {
      const player = page.locator(selector).locator('..');
      await player.locator('.player-wave-surface').scrollIntoViewIfNeeded();
      const surface = await player.locator('.player-wave-surface').boundingBox();
      const handle = await player.locator(`.region-${edge}`).boundingBox();
      await page.mouse.move(handle.x + handle.width / 2, handle.y + handle.height / 2);
      await page.mouse.down();
      await page.mouse.move(surface.x + surface.width * target, surface.y + surface.height / 2, { steps: 12 });
      await page.mouse.up();
      return page.locator(selector).evaluate(audio => StudioAudio.selection(audio));
    };
    const span = region => region.end - region.start;
    const near = (a, b, tolerance = .00001) => Math.abs(a - b) < tolerance;
    const infillBoundaries = () => page.evaluate(() => ({
      start: Number(document.querySelector('#start-frame').value),
      end: Number(document.querySelector('#end-frame').value),
      startTime: UiNumber.read(document.querySelector('#start-time')),
      endTime: UiNumber.read(document.querySelector('#end-time')),
      region: StudioAudio.selection(document.querySelector('#infill-source-audio')),
    }));
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.waitForFunction(() => state.capabilities?.infill && state.model
      && document.documentElement.dataset.studioReady === 'true');
    report.selected_model = await page.evaluate(() => state.model.selected_directory);
    await page.locator('#clone-audio').setInputFiles(process.env.DRAMA_WAVEFORM_AUDIO);
    await waitWave('#clone-file-audio');
    report.checks.upload_waveform_visible = await page.locator('#clone-upload-preview .player-wave').isVisible();
    report.checks.stereo_right_channel_has_waveform = await page.locator('#clone-upload-preview .player-wave rect')
      .evaluateAll(bars => new Set(bars.map(bar => bar.getAttribute('height'))).size > 20);
    const selection = await drag('#clone-file-audio', 0.15, 0.55);
    const duration = await page.locator('#clone-file-audio').evaluate(a => a.duration);
    report.checks.drag_selects_real_audio_times = Math.abs(selection.start - duration * .15) < .02
      && Math.abs(selection.end - duration * .55) < .02;
    report.checks.crop_action_enabled = await page.locator('#crop-clone-audio').isEnabled();
    const right = await moveRegion('#clone-file-audio', .12);
    report.checks.reference_region_moves_right_without_resizing = near(span(right), span(selection))
      && near(right.start - selection.start, duration * .12, .02);
    const left = await moveRegion('#clone-file-audio', -.07);
    report.checks.reference_region_moves_left_without_resizing = near(span(left), span(selection))
      && near(right.start - left.start, duration * .07, .02);
    const atStart = await moveRegion('#clone-file-audio', -1);
    report.checks.reference_move_clamps_at_start = near(atStart.start, 0) && near(span(atStart), span(selection));
    const atEnd = await moveRegion('#clone-file-audio', 2);
    report.checks.reference_move_clamps_at_end = near(atEnd.end, duration) && near(span(atEnd), span(selection));
    report.checks.release_outside_clears_drag_cursor = !await page.locator(
      '#clone-upload-preview .player-wave-surface').evaluate(node => node.classList.contains('moving-region'));
    const fresh = await drag('#clone-file-audio', .1, .5);
    report.checks.drag_outside_region_creates_new_selection = near(fresh.start, duration * .1, .02)
      && near(fresh.end, duration * .5, .02);
    const resized = await resizeRegion('#clone-file-audio', 'start', .18);
    report.checks.start_handle_still_resizes = near(resized.start, duration * .18, .02) && near(resized.end, fresh.end);
    const resizedEnd = await resizeRegion('#clone-file-audio', 'end', .56);
    report.checks.end_handle_still_resizes = near(resizedEnd.start, resized.start)
      && near(resizedEnd.end, duration * .56, .02);
    const surface = page.locator('#clone-upload-preview .player-wave-surface');
    const surfaceBox = await surface.boundingBox();
    await surface.click({ position: { x: surfaceBox.width * .35, y: surfaceBox.height / 2 } });
    const clicked = await page.locator('#clone-file-audio').evaluate(audio => ({
      selection: StudioAudio.selection(audio), time: audio.currentTime,
    }));
    report.checks.click_inside_region_seeks_without_changing_selection = near(clicked.time, duration * .35, .02)
      && JSON.stringify(clicked.selection) === JSON.stringify(resizedEnd);
    await page.mouse.move(surfaceBox.x + surfaceBox.width * .35, surfaceBox.y + surfaceBox.height / 2);
    await page.mouse.down();
    await page.mouse.move(surfaceBox.x + surfaceBox.width * .4, surfaceBox.y + surfaceBox.height / 2);
    report.checks.region_uses_grabbing_cursor_while_moving = await surface.evaluate(node =>
      getComputedStyle(node).cursor === 'grabbing' && getComputedStyle(node.querySelector('.player-region')).cursor === 'grabbing');
    await surface.dispatchEvent('pointercancel', { pointerId: 1, pointerType: 'mouse' });
    await page.mouse.up();
    report.checks.cancel_clears_drag_cursor = !await surface.evaluate(node => node.classList.contains('moving-region'));
    const beforeKey = await page.locator('#clone-file-audio').evaluate(a => StudioAudio.selection(a));
    await page.locator('#clone-upload-preview .region-start').focus();
    await page.keyboard.press('ArrowRight');
    const keyboard = await page.locator('#clone-file-audio').evaluate(a => StudioAudio.selection(a));
    report.checks.keyboard_adjusts_selection = keyboard.start > beforeKey.start && keyboard.end === beforeKey.end;
    await page.locator('#clone-upload-preview .player-zoom-in').click();
    const zoomed = await page.locator('#clone-file-audio').evaluate(a => StudioAudio.selection(a));
    report.checks.zoom_preserves_selection = JSON.stringify(zoomed) === JSON.stringify(keyboard)
      && await page.locator('#clone-upload-preview .player-pan').isVisible();
    const zoomMoved = await moveRegion('#clone-file-audio', .06);
    report.checks.zoomed_region_moves_at_visible_time_scale = near(span(zoomMoved), span(zoomed))
      && near(zoomMoved.start - zoomed.start, duration / 2 * .06, .02);
    await page.locator('#clone-upload-preview .player-pan').evaluate(node => {
      node.value = Number(node.max) * .4; node.dispatchEvent(new Event('input', { bubbles: true }));
    });
    const pannedMove = await moveRegion('#clone-file-audio', -.06);
    report.checks.panned_region_preserves_move_scale = near(span(pannedMove), span(zoomed))
      && near(zoomMoved.start - pannedMove.start, duration / 2 * .06, .02);
    await page.locator('#clone-upload-preview .player-zoom-out').click();
    await page.fill('#clone-transcript', '原音频的完整逐字稿。');
    const cropResponse = page.waitForResponse(r => r.url().endsWith('/api/reference-audio/crop'));
    await page.click('#crop-clone-audio');
    report.crop = await (await cropResponse).json();
    await page.waitForFunction(() => state.cloneUpload?.id && state.cloneOriginal
      && document.querySelector('#clone-file-audio').duration < 3);
    await waitWave('#clone-file-audio');
    report.checks.crop_updates_actual_reference = Math.abs(
      await page.locator('#clone-file-audio').evaluate(a => a.duration) - report.crop.duration_seconds) < .001;
    report.checks.crop_clears_mismatched_transcript = await page.inputValue('#clone-transcript') === '';
    report.checks.missing_asr_leaves_crop_transcript_editable_without_errors = transcriptions.length === 2
      && !await page.locator('#clone-asr-status').isVisible();
    await page.fill('#clone-transcript', '裁剪片段的逐字稿。');
    await page.click('#submit-job');
    await page.waitForFunction(() => !document.querySelector('#submit-job').disabled);
    report.checks.generation_uses_cropped_reference = submissions.at(-1).voice_bindings[0].id === report.crop.id
      && submissions.at(-1).voice_bindings[0].text === '裁剪片段的逐字稿。'
      && submissions.at(-1).voice_bindings[0].speaker === 0;
    await page.click('#restore-clone-audio');
    await page.waitForFunction(() => !state.cloneOriginal && document.querySelector('#clone-file-audio').duration > 5);
    await waitWave('#clone-file-audio');
    report.checks.restore_original_audio_and_transcript = await page.inputValue('#clone-transcript') === '原音频的完整逐字稿。';
    await drag('#clone-file-audio', 0.1, 0.21);
    await page.locator('#clone-upload-preview .player-preview-selection').click();
    await page.waitForFunction(() => document.querySelector('#clone-file-audio').currentTime > .65);
    await page.waitForFunction(() => document.querySelector('#clone-file-audio').paused);
    const afterPreview = await page.locator('#clone-file-audio').evaluate(a => ({
      time: a.currentTime, selected: StudioAudio.selection(a),
    }));
    report.checks.selection_preview_stops_at_end = Math.abs(afterPreview.time - afterPreview.selected.end) < .04;
    await page.locator('#clone-upload-preview .player-clear-selection').click();
    report.checks.clear_selection_removes_crop = await page.locator('#crop-clone-audio').isDisabled();
    asrAvailable = true;
    await page.waitForFunction(() => !state.cloneAsrPromise);
    await drag('#clone-file-audio', .3, .6);
    const before = transcriptions.length;
    await page.click('#crop-clone-audio');
    await page.waitForFunction(() => state.cloneOriginal && !state.cloneAsrPromise
      && document.querySelector('#clone-transcript').value === '新片段的逐字稿。');
    report.checks.auto_crop_transcribes_only_new_reference = transcriptions.length === before + 1
      && transcriptions.at(-1).upload_id !== report.crop.id;
    const referenceImage = path.join(process.env.DRAMA_WAVEFORM_FOLDER, 'reference-waveform.png');
    await page.screenshot({ path: referenceImage, fullPage: true }); report.screenshots.push(referenceImage);
    await page.click('#remove-clone-audio');

    await page.click('.nav[data-view="takes"]');
    await page.evaluate(() => { view('generate'); monitorJob('waveform-fixture'); });
    await page.waitForFunction(() => state.activeJob?.id === 'waveform-fixture'
      && document.querySelector('#result-box').dataset.jobId === 'waveform-fixture');
    await waitWave('#result-box .main-result audio');
    const resultSelection = await drag('#result-box .main-result audio', .22, .64);
    await page.locator('.main-result .creative-edit').click();
    await page.locator('#audio-edit-workspace').waitFor();
    await page.waitForFunction(() => document.querySelector('#audio-edit-source').duration > 0);
    report.checks.result_opens_editor_with_selection = await page.locator('#audio-edit-source').evaluate(audio =>
      Boolean(StudioAudio.selection(audio)));
    await page.locator('#audio-edit-back').click();
    await page.evaluate(selection => {
      StudioFlow.derive({ job_id: 'waveform-fixture' }, 'infill', selection);
    }, resultSelection);
    await page.waitForFunction(() => state.sourceTake && document.querySelector('#mode').value === 'infill');
    await waitWave('#infill-source-audio');
    const originalFrames = await page.evaluate(() => [Number(document.querySelector('#start-frame').value),
      Number(document.querySelector('#end-frame').value)]);
    report.checks.result_selection_enters_infill = originalFrames[0] === Math.round(resultSelection.start * 44100 / 2048)
      && originalFrames[1] === Math.round(resultSelection.end * 44100 / 2048);
    report.checks.infill_recovers_full_transcript = await page.inputValue('#text') === '你好，今天的天气很好，我们一起出去走走吧。';
    await drag('#infill-source-audio', .1, .45);
    let boundaries = await infillBoundaries();
    report.checks.infill_waveform_snaps_to_codec_frames = Math.abs(boundaries.region.start * 44100 / 2048 - boundaries.start) < .00001
      && Math.abs(boundaries.region.end * 44100 / 2048 - boundaries.end) < .00001;
    const originalSpan = boundaries.end - boundaries.start;
    await moveRegion('#infill-source-audio', .08);
    await moveRegion('#infill-source-audio', -.04);
    await moveRegion('#infill-source-audio', .03);
    const movedInfill = await infillBoundaries();
    report.checks.infill_moves_repeatedly_without_frame_drift = movedInfill.end - movedInfill.start === originalSpan
      && movedInfill.start > boundaries.start;
    await moveRegion('#infill-source-audio', 1);
    const infillEnd = await infillBoundaries();
    report.checks.infill_move_clamps_at_last_frame = infillEnd.end === 128 && infillEnd.end - infillEnd.start === originalSpan;
    await moveRegion('#infill-source-audio', -2);
    const infillStart = await infillBoundaries();
    report.checks.infill_move_clamps_at_first_frame = infillStart.start === 0 && infillStart.end === originalSpan;
    await moveRegion('#infill-source-audio', .23);
    await page.locator('#infill-source-preview .player-zoom-in').click();
    await moveRegion('#infill-source-audio', .08);
    await moveRegion('#infill-source-audio', -.06);
    await page.locator('#infill-source-preview .player-zoom-out').click();
    boundaries = await infillBoundaries();
    report.checks.zoomed_infill_preserves_frame_count = boundaries.end - boundaries.start === originalSpan;
    report.checks.moved_infill_updates_seconds_and_frames = near(boundaries.region.start, boundaries.start * 2048 / 44100)
      && near(boundaries.region.end, boundaries.end * 2048 / 44100)
      && near(boundaries.startTime, boundaries.region.start, .000501) && near(boundaries.endTime, boundaries.region.end, .000501);
    await choose('ui-language', 'en');
    const afterLocale = await page.locator('#infill-source-audio').evaluate(a => StudioAudio.selection(a));
    report.checks.locale_preserves_region = JSON.stringify(boundaries.region) === JSON.stringify(afterLocale);
    await page.locator('.compose-panel > .editor-area .cm-content').fill('你好，今天的天气非常好，我们一起出去走走吧。');
    await page.click('#submit-job');
    await page.waitForFunction(() => !document.querySelector('#submit-job').disabled);
    report.checks.submitted_infill_uses_selected_codes = submissions.at(-1).mode === 'infill'
      && submissions.at(-1).start_frame === boundaries.start && submissions.at(-1).end_frame === boundaries.end
      && submissions.at(-1).source_take === process.env.DRAMA_WAVEFORM_TAKE;
    await page.fill('#start-time', '1.2');
    await page.fill('#end-time', '1.2');
    report.checks.zero_width_selection_remains_insertion = await page.inputValue('#start-frame') === await page.inputValue('#end-frame');
    const startHandle = page.locator('#infill-source-preview .region-start');
    await startHandle.focus(); await page.keyboard.press('ArrowLeft');
    const afterKey = await page.locator('#infill-source-audio').evaluate(a => StudioAudio.selection(a));
    report.checks.frame_keyboard_steps_are_one_frame = Math.abs((afterKey.end - afterKey.start) * 44100 / 2048 - 1) < .00001;
    for (const locale of ['zh-CN', 'en', 'ja', 'ko']) {
      await choose('ui-language', locale);
      const untranslated = await page.evaluate(() =>
        [...new Set([...document.querySelectorAll('[data-i18n]')]
          .filter(node => !node.closest('#model-download-choice')).map(node => node.dataset.i18n))]
          .filter(source => !I18n.has(source)));
      report.checks[`translations_${locale}`] = !untranslated.length;
      if (untranslated.length) console.log('Missing translations:', untranslated);
      for (const [name, width, height] of [['desktop', 1440, 1080], ['mobile', 390, 844]]) {
        await page.setViewportSize({ width, height });
        report.checks[`${name}_fits_${locale}`] = await page.evaluate(() =>
          document.documentElement.scrollWidth <= innerWidth + 1);
      }
    }
    await page.setViewportSize({ width: 1440, height: 1080 }); await choose('ui-language', 'zh-CN');
    const infillImage = path.join(process.env.DRAMA_WAVEFORM_FOLDER, 'infill-waveform.png');
    await page.screenshot({ path: infillImage, fullPage: true }); report.screenshots.push(infillImage);
    await page.evaluate(path => useTake(path, 'revision'), process.env.DRAMA_WAVEFORM_TAKE);
    await page.locator('#audio-edit-workspace').waitFor();
    await page.locator('[data-edit-mode="local"]').click();
    await waitWave('#audio-edit-source');
    await drag('#audio-edit-source', .6, .85);
    report.checks.director_source_selection_can_open_infill = await page.locator('#audio-edit-quote').isVisible();
    report.checks.director_region_is_infill_not_whole_revision =
      await page.locator('[data-edit-mode="local"]').getAttribute('aria-selected') === 'true'
      && await page.locator('#audio-edit-source').evaluate(audio => Boolean(StudioAudio.selection(audio)));
    report.checks.no_browser_errors = report.errors.length === 0;
  } catch (error) {
    report.errors.push(error.stack || String(error));
    if (page) {
      const image = path.join(process.env.DRAMA_WAVEFORM_FOLDER, 'failure.png');
      await page.screenshot({ path: image, fullPage: true }).catch(() => {}); report.screenshots.push(image);
    }
  } finally {
    if (browser) await browser.close();
    report.passed = !report.errors.length && Object.values(report.checks).every(Boolean);
    fs.writeFileSync(resultPath, JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report, null, 2));
    if (!report.passed) process.exitCode = 1;
  }
})();
