/* Missing material must not discard text, downgrade operations or unblock unconditioned generation. */
const fs = require('node:fs'), path = require('node:path');
const { chromium } = require('playwright');
const root = path.resolve(__dirname, '..'), base = process.env.DRAMA_UI_URL;
const folder = process.env.DRAMA_STUDIO_FOLDER;
const report = { scope: 'Real HTTP/store/browser with missing-source fixtures; synthetic inference, no ASR.',
  checks: {}, errors: [], screenshots: [] };
(async () => {
  let browser, page;
  try {
    browser = await chromium.launch({ channel: 'chrome', headless: true });
    const context = await browser.newContext({ viewport: { width: 1440, height: 1080 }, locale: 'zh-CN' });
    page = await context.newPage();
    page.on('pageerror', error => report.errors.push(error.stack || error.message));
    const api = async (url, body) => {
      const response = body === undefined ? await context.request.get(base + url)
        : await context.request.post(base + url, { data: body });
      if (!response.ok()) throw new Error(url + ': ' + await response.text());
      return response.json();
    };
    const check = (name, value) => {
      report.checks[name] = Boolean(value);
      if (!value) throw new Error(name);
    };
    const ready = () => page.waitForFunction(() => document.documentElement.dataset.studioReady === 'true');
    const reload = async () => { await page.reload({ waitUntil: 'networkidle' }); await ready(); };
    const editor = () => page.locator('.compose-panel > .editor-area .cm-content');
    const shot = async name => {
      const file = path.join(folder, name + '.png');
      await page.screenshot({ path: file, fullPage: true }); report.screenshots.push(file);
    };
    await page.goto(base, { waitUntil: 'networkidle' }); await ready();
    await page.locator('#clone-audio').setInputFiles(process.env.DRAMA_STUDIO_REFERENCE);
    await page.locator('#clone-transcript').fill('已经核对的参考台词。');
    await editor().fill('必须保留下来的未提交台词。');
    await page.locator('#advanced-settings > summary').click();
    await page.locator('#temperature').fill('.61');
    await page.locator('#max-seconds').fill('23');
    await page.evaluate(() => StudioFlow.save());
    const saved = (await api('/api/studio')).drafts.single.value;
    const missing = structuredClone(saved);
    missing.upload = { id: '1'.repeat(32), filename: 'missing-reference.wav' };
    await api('/api/studio/draft', { key: 'single', value: missing });
    await reload();
    check('missing_upload_does_not_abort_studio_initialization',
      await page.inputValue('#text') === saved.form.text && await page.inputValue('#clone-transcript') === saved.transcript
      && Number(await page.inputValue('#temperature')) === .61 && Number(await page.inputValue('#max-seconds')) === 23);
    check('missing_upload_keeps_identity_and_blocks_unconditioned_generation',
      await page.locator('#submit-job').isDisabled()
      && await page.locator('#default-voice').getAttribute('aria-pressed') === 'false'
      && (await page.locator('#clone-note').innerText()).includes('缺失')
      && await page.locator('#replace-clone-audio').isEnabled()
      && await page.locator('#remove-clone-audio').isEnabled());
    await page.evaluate(() => StudioFlow.save());
    const kept = (await api('/api/studio')).drafts.single.value;
    check('saving_missing_material_does_not_drop_its_id_or_transcript',
      kept.upload.id === missing.upload.id && kept.transcript === saved.transcript && kept.form.text === saved.form.text);
    await shot('missing-reference-retained');
    await page.locator('#clone-audio').setInputFiles(process.env.DRAMA_STUDIO_REFERENCE);
    await page.locator('#clone-transcript').fill('重新上传后的实际逐字稿。');
    check('replacing_missing_upload_restores_generation_without_losing_target',
      await page.locator('#submit-job').isEnabled() && await page.inputValue('#text') === saved.form.text
      && !(await page.locator('#clone-note').innerText()).includes('缺失'));
    await page.evaluate(() => StudioFlow.save());
    await reload();
    check('replacement_survives_refresh_without_stale_missing_notice',
      await page.locator('#submit-job').isEnabled() && await page.inputValue('#clone-transcript') === '重新上传后的实际逐字稿。'
      && !(await page.locator('#clone-note').innerText()).includes('缺失'));
    await api('/api/studio/draft', { key: 'single', value: missing }); await reload();
    await page.locator('#saved-voice-options > summary').click();
    const availableVoice = (await api('/api/voices')).voices[0];
    await page.locator(`#voice-picks [data-voice="${availableVoice.id}"]`).click();
    check('explicit_library_choice_can_replace_missing_upload',
      await page.locator('#submit-job').isEnabled() && await page.evaluate(() => !state.cloneMissing)
      && await page.inputValue('#text') === saved.form.text);
    await api('/api/studio/draft', { key: 'single', value: missing }); await reload();
    await page.locator('#remove-clone-audio').click();
    check('explicit_remove_allows_fresh_generation_and_keeps_target',
      await page.locator('#submit-job').isEnabled() && await page.inputValue('#text') === saved.form.text
      && await page.locator('#default-voice').getAttribute('aria-pressed') === 'true');
    await page.locator('#submit-job').click();
    await page.waitForFunction(() => document.querySelector('#job-badge').dataset.status === 'complete');
    const first = await api('/api/jobs/' + await page.locator('#result-box').getAttribute('data-job-id'));
    check('only_explicit_removal_results_in_reference_free_job', first.recipe.reference_snapshot.length === 0
      && first.recipe.text === saved.form.text);
    check('restored_valid_sampling_is_not_rejected_by_an_arbitrary_input_step',
      first.recipe.temperature === .61 && await page.locator('#temperature').evaluate(control => control.checkValidity()));

    const lostVoice = structuredClone(saved);
    lostVoice.upload = null; lostVoice.voices = ['missing-voice'];
    lostVoice.libraryReferences = [{ key: 'missing-library-reference', voiceId: 'missing-voice',
      libraryId: 'missing-voice', referenceIndex: 0, transcript: '原音色库的准确逐字稿。', asrMode: 'manual' }];
    await api('/api/studio/draft', { key: 'single', value: lostVoice }); await reload();
    check('missing_library_voice_does_not_abort_or_erase_transcript',
      await page.inputValue('#text') === saved.form.text
      && await page.evaluate(() => StudioFlow.flow.libraryReferences[0].transcript === '原音色库的准确逐字稿。')
      && await page.locator('#selected-reference-details .reference-compact-play').isDisabled());
    const count = (await api('/api/jobs')).jobs.length;
    await page.locator('#submit-job').click(); await page.waitForTimeout(250);
    check('missing_library_voice_is_not_silently_ignored', (await api('/api/jobs')).jobs.length === count
      && (await page.locator('#result-error').innerText()).includes('缺失'));
    await page.locator('#selected-reference-details .selected-voice-clear').click();

    await page.evaluate(source => StudioFlow.derive(source, 'revision'), { job_id: first.id });
    await page.evaluate(() => StudioFlow.save());
    const frozen = (await api('/api/studio')).drafts.single.value;
    delete frozen.operation.frozenReferences;
    await api('/api/studio/draft', { key: 'single', value: frozen });
    const unavailable = route => route.fulfill({ status: 400, json: { error: 'Source temporarily unavailable' } });
    await page.route('**/api/studio/prepare-restore', unavailable); await reload();
    check('unavailable_source_keeps_revision_instead_of_downgrading_to_legacy',
      await page.inputValue('#mode') === 'revision'
      && await page.locator('#text').evaluate(control => control.readOnly)
      && await page.locator('#submit-job').isDisabled()
      && await page.evaluate(() => !StudioFlow.flow.operation.legacy && Boolean(StudioFlow.flow.operation.restoreError)));
    check('unavailable_source_offers_explicit_reload_and_return_draft',
      await page.locator('#creative-operation').getByRole('button', { name: '重新加载来源', exact: true }).isVisible()
      && await page.locator('#return-draft').isVisible());
    await shot('source-restore-retry');
    await page.unroute('**/api/studio/prepare-restore', unavailable);
    await page.locator('#creative-operation').getByRole('button', { name: '重新加载来源', exact: true }).click();
    await page.waitForFunction(() => !StudioFlow.flow.operation.restoreError && !document.querySelector('#submit-job').disabled);
    check('source_reload_restores_exact_frozen_operation',
      await page.locator('#submit-job').isEnabled()
      && await page.inputValue('#text') === first.recipe.text
      && await page.locator('#text').evaluate(control => control.readOnly)
      && await page.inputValue('#previous-take') === first.take_file);

    await page.evaluate(source => StudioFlow.derive(source, 'infill', { start: 0, end: 1 }), { job_id: first.id });
    await page.evaluate(() => StudioFlow.save());
    await page.route('**/api/takes/inspect', unavailable); await reload();
    check('missing_infill_source_keeps_draft_and_does_not_show_another_take',
      await page.inputValue('#mode') === 'infill' && await page.inputValue('#text') === first.recipe.text
      && await page.locator('#submit-job').isDisabled() && !await page.locator('#infill-source-preview').isVisible());
    await page.unroute('**/api/takes/inspect', unavailable);
    await page.locator('#creative-operation').getByRole('button', { name: '重新加载来源', exact: true }).click();
    await page.waitForFunction(() => !StudioFlow.flow.operation.restoreError && !document.querySelector('#submit-job').disabled
      && !document.querySelector('#infill-source-preview').classList.contains('hidden'));
    check('infill_source_retry_restores_actual_take_and_selection',
      await page.locator('#submit-job').isEnabled() && await page.locator('#infill-source-preview').isVisible()
      && await page.inputValue('#source-take') === first.take_file
      && Number(await page.inputValue('#start-frame')) === 0 && Number(await page.inputValue('#end-frame')) > 0);
    const plain = structuredClone(saved);
    plain.operation = null; plain.mode = 'tts'; plain.returnDrafts = [];
    plain.form['previous-take'] = plain.form['source-take'] = '';
    const olderStream = { ...structuredClone(plain), mode: 'stream', streamLive: true };
    olderStream.form.chunks = '旧版流式草稿。\n这两句需要保留。';
    olderStream.returnDrafts = [plain, { ...structuredClone(plain), mode: 'stream' },
      { ...structuredClone(plain), mode: 'stream' }];
    await api('/api/studio/draft', { key: 'single', value: olderStream }); await reload();
    check('legacy_repeated_stream_entries_migrate_to_ordinary_draft_without_an_exit_button',
      await page.inputValue('#mode') === 'tts'
      && await page.inputValue('#text') === olderStream.form.chunks
      && !await page.locator('#toggle-text-stream,#exit-text-stream,#text-stream-status').count());
    await page.evaluate(() => StudioFlow.save()); await reload();
    check('migrated_ordinary_draft_stays_usable_after_refresh',
      await page.inputValue('#text') === olderStream.form.chunks
      && await page.inputValue('#mode') === 'tts' && await page.locator('#submit-job').isEnabled()
      && !await page.locator('#stream-fields').isVisible());
    const noParent = { ...structuredClone(olderStream), returnDrafts: [] };
    noParent.form.text = '';
    await api('/api/studio/draft', { key: 'single', value: noParent }); await reload();
    check('stream_without_normal_words_recovers_chunks_into_ordinary_text',
      await page.inputValue('#mode') === 'tts' && await page.inputValue('#text') === olderStream.form.chunks
      && !await page.locator('#stream-fields').isVisible()
      && !await page.inputValue('#previous-take') && !await page.inputValue('#source-take'));
    check('no_browser_exceptions', !report.errors.length);
  } catch (error) {
    report.errors.push(error.stack || String(error));
    if (page) {
      const file = path.join(folder, 'recovery-failure.png');
      await page.screenshot({ path: file, fullPage: true }).catch(() => {}); report.screenshots.push(file);
    }
  } finally {
    if (browser) await browser.close();
    report.passed = !report.errors.length && Object.values(report.checks).every(Boolean);
    fs.writeFileSync(path.join(root, 'research/2026-10-19-studio-recovery-ui-acceptance.json'),
      JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report, null, 2)); if (!report.passed) process.exitCode = 1;
  }
})();
