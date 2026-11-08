/* Real history, drafts and result routing; synthesis is provided by the isolated harness. */
const fs = require('node:fs'), path = require('node:path'), { chromium } = require('playwright');
const base = process.env.DRAMA_UI_URL, folder = process.env.DRAMA_STUDIO_FOLDER;
const report = { checks: {}, errors: [], screenshots: [] };
(async () => {
  let browser, page;
  try {
    browser = await chromium.launch({ channel: 'chrome', headless: true });
    const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, locale: 'zh-CN' });
    await context.addInitScript(() => localStorage.setItem('drama3.ui.language', 'zh-CN'));
    page = await context.newPage();
    page.on('pageerror', error => report.errors.push(error.stack || error.message));
    const api = async (url, data) => {
      const response = data === undefined ? await context.request.get(base + url)
        : await context.request.post(base + url, { data });
      if (!response.ok()) throw new Error(url + ': ' + await response.text());
      return response.json();
    };
    const check = (name, value) => { report.checks[name] = Boolean(value); if (!value) throw new Error(name); };
    const ready = () => page.waitForFunction(() => document.documentElement.dataset.studioReady === 'true');
    const editor = () => page.locator('.compose-panel > .editor-area .cm-content');
    const visibleId = () => page.locator('#result-box').getAttribute('data-job-id');
    const create = async payload => {
      const job = await api('/api/jobs', payload);
      for (let index = 0; index < 150; index++) {
        const row = await api('/api/jobs/' + job.id);
        if (row.status === 'complete') return row;
        if (['error', 'failed', 'incomplete'].includes(row.status)) throw new Error(JSON.stringify(row));
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      throw new Error('Synthetic job timed out');
    };
    const show = async job => {
      await page.evaluate(job => { view(resultWorkspaceFor(job)); monitorJob(job.id); }, job);
      await page.waitForFunction(id => state.activeJob?.id === id && state.activeJob.status === 'complete', job.id);
    };
    const restore = async source => {
      await page.evaluate(source => StudioFlow.restoreVersion(source), source);
      await page.waitForFunction(id => document.querySelector('#result-box').dataset.jobId === id, source.job_id);
    };
    const audioMatches = async output => await page.locator('#result-box .main-result audio').getAttribute('src')
      === await page.evaluate(output => fileUrl(output), output);
    const reload = async () => { await page.evaluate(() => StudioFlow.save()); await page.reload({ waitUntil: 'networkidle' }); await ready(); };
    const shot = async name => {
      const file = path.join(folder, name + '.png');
      await page.screenshot({ path: file, fullPage: true, animations: 'disabled' }); report.screenshots.push(file);
    };
    await page.goto(base, { waitUntil: 'networkidle' }); await ready();
    const voices = (await api('/api/voices')).voices;
    const basic = { mode: 'tts', text: 'A previous working result.', voice_ids: [voices[0].id], silence_retry: true,
      studio: { work_id: 'restore-single', segment_id: 'main', layout: 'single' } };
    const previous = await create(basic);
    const script = '[坚定] 背叛？<whisper>我只是拿回属于我的力量。</whisper>';
    const edited = await create({ ...basic, text: '<|speaker:0|> ' + script,
      studio: { work_id: 'restore-edit', segment_id: 'main', layout: 'single',
        result_scope: 'audio_editor', edit_source: 'history-fixture' } });
    const originalEdited = await api('/api/jobs/' + edited.id);
    const joint = await create({ mode: 'scene', scene_strategy: 'joint', scene: 'A conversation.',
      characters: [{ name: 'Ada', description: 'Calm.', speaker: 0 }, { name: 'Bo', description: 'Warm.', speaker: 1 }],
      character_ids: ['ada', 'bo'], line_ids: ['one', 'two'],
      lines: [{ character: 'Ada', text: '[好奇] 准备好了吗？' }, { character: 'Bo', text: '准备好了。' }],
      voice_bindings: [{ voice_id: voices[0].id, speaker: 0 }, { voice_id: voices[1].id, speaker: 1 }],
      studio: { work_id: 'restore-dialogue', segment_id: 'scene', layout: 'scene' } });
    await show(joint); await show(previous);
    await editor().fill('An unsent working draft.'); await page.locator('#direction').fill('Quietly.');
    await page.locator('#clone-audio').setInputFiles(process.env.DRAMA_STUDIO_REFERENCE);
    await page.waitForFunction(() => state.cloneUpload && !state.cloneAsrPromise);
    await page.locator('#clone-transcript').fill('Original working reference transcript.');
    await page.locator('#advanced-settings > summary').click();
    await page.locator('#temperature').fill('.64'); await page.locator('#top-p').fill('.87');
    await page.locator('#max-chars').fill('240'); await page.locator('#candidate-count').fill('2');
    await page.locator('#random-seed').uncheck(); await page.locator('#seed').fill('424242');
    await page.locator('#silence-retry').uncheck();
    await page.evaluate(() => StudioFlow.save());
    const originalDraft = (await api('/api/studio')).drafts.single.value;
    const checkOriginalCreation = async name => {
      await page.waitForFunction(id => document.querySelector('#result-box').dataset.jobId === id, previous.id);
      check(name, await page.inputValue('#text') === originalDraft.form.text
        && await page.inputValue('#direction') === originalDraft.form.direction
        && await page.inputValue('#clone-transcript') === originalDraft.transcript
        && await page.evaluate(id => state.cloneUpload?.id === id, originalDraft.upload.id)
        && Number(await page.inputValue('#temperature')) === .64 && Number(await page.inputValue('#top-p')) === .87
        && Number(await page.inputValue('#max-chars')) === 240 && Number(await page.inputValue('#candidate-count')) === 2
        && Number(await page.inputValue('#seed')) === 424242 && !await page.locator('#random-seed').isChecked()
        && !await page.locator('#silence-retry').isChecked() && await audioMatches(previous.output)
        && !await page.locator('#return-draft').isVisible()
        && await page.evaluate(() => !StudioFlow.flow.operation && !StudioFlow.flow.restoredFrom
          && !StudioFlow.flow.returnDrafts.length && !StudioFlow.voiceDesignActive()));
    };
    await page.locator('.nav[data-view="takes"]').click();
    await page.evaluate(() => loadJobs());
    const row = page.locator(`.job-card[data-job-id="${edited.id}"]`);
    await row.locator('.history-actions > summary').click(); await row.locator('.restore-job').click();
    await page.waitForFunction(id => document.querySelector('#result-box').dataset.jobId === id, edited.id);
    check('history_restore_hides_single_speaker_prefix_but_preserves_performance_tags',
      await page.inputValue('#text') === script && await editor().innerText() === script);
    check('history_restore_shows_the_exact_edit_result', await page.locator('#view-generate').isVisible()
      && await audioMatches(edited.output) && await visibleId() === edited.id);
    check('restoring_single_does_not_replace_dialogue_result', await page.evaluate(id => state.resultJobs.scene?.id === id, joint.id));
    await shot('restored-edit-desktop');
    await reload();
    check('restored_edit_result_and_clean_script_survive_reload', await visibleId() === edited.id
      && await audioMatches(edited.output) && await page.inputValue('#text') === script);
    const legacyDraft = (await api('/api/studio')).drafts.single.value;
    legacyDraft.form.text = '<|speaker:0|> ' + script;
    await api('/api/studio/draft', { key: 'single', value: legacyDraft });
    await page.reload({ waitUntil: 'networkidle' }); await ready();
    check('previously_saved_restored_draft_cleans_the_redundant_prefix', await page.inputValue('#text') === script);
    await page.evaluate(() => loadJobs());
    await page.locator('.nav[data-view="scene"]').click(); await page.locator('.nav[data-view="generate"]').click();
    check('refreshing_history_and_switching_pages_keeps_the_restored_result', await visibleId() === edited.id);
    await page.setViewportSize({ width: 390, height: 844 }); await shot('restored-edit-mobile');
    check('restored_page_fits_narrow_windows', await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.locator('#return-draft').click();
    await checkOriginalCreation('return_to_creation_restores_text_reference_settings_and_result');

    const alternate = await create({ ...basic, text: 'Another history draft.', voice_ids: [voices[1].id],
      studio: { work_id: 'restore-alternate', segment_id: 'main', layout: 'single' } });
    await restore({ job_id: edited.id });
    await editor().fill('Intermediate restored draft changes must not become the return destination.');
    for (const job of [alternate, edited, alternate, edited, alternate]) await restore({ job_id: job.id });
    check('repeated_history_restores_keep_one_original_creation', await page.evaluate(text => {
      const drafts = StudioFlow.flow.returnDrafts;
      return drafts.length === 1 && drafts[0].form.text === text && !drafts[0].returnDrafts;
    }, originalDraft.form.text) && await page.locator('#return-draft').textContent() === '返回创作');
    await reload();
    check('replacement_history_draft_and_single_return_survive_reload', await page.inputValue('#text') === alternate.recipe.text
      && await audioMatches(alternate.output) && await page.evaluate(() => StudioFlow.flow.returnDrafts.length === 1));
    await page.locator('#return-draft').click();
    await checkOriginalCreation('one_return_skips_all_intermediate_history_drafts_after_reload');
    await reload();
    await checkOriginalCreation('returned_creation_stays_out_of_restored_state_after_reload');

    await restore({ job_id: edited.id });
    await page.evaluate(() => StudioFlow.save());
    const firstRestored = (await api('/api/studio')).drafts.single.value;
    await restore({ job_id: alternate.id });
    await page.evaluate(() => StudioFlow.save());
    const legacyStack = (await api('/api/studio')).drafts.single.value;
    legacyStack.returnDrafts = [originalDraft, { ...firstRestored, returnDrafts: [] },
      { ...structuredClone(legacyStack), returnDrafts: [] }];
    await api('/api/studio/draft', { key: 'single', value: legacyStack });
    await page.reload({ waitUntil: 'networkidle' }); await ready();
    check('legacy_history_stack_is_flattened_to_original_creation', await page.evaluate(text => {
      const drafts = StudioFlow.flow.returnDrafts;
      return drafts.length === 1 && drafts[0].form.text === text && !drafts[0].restoredFrom && !drafts[0].returnDrafts;
    }, originalDraft.form.text));
    await page.locator('#return-draft').click();
    await checkOriginalCreation('legacy_stack_also_returns_in_one_click');

    const truncatedStack = structuredClone(legacyStack);
    truncatedStack.returnDrafts = [{ ...firstRestored, returnDrafts: [] }];
    await api('/api/studio/draft', { key: 'single', value: truncatedStack });
    await page.reload({ waitUntil: 'networkidle' }); await ready();
    await page.locator('#return-draft').click();
    await page.waitForFunction(id => document.querySelector('#result-box').dataset.jobId === id
      && !StudioFlow.flow.restoring, edited.id);
    check('truncated_legacy_stack_returns_to_editable_creation_without_source_lock', await page.inputValue('#text') === script
      && await page.inputValue('#mode') === 'tts' && !await page.locator('#return-draft').isVisible()
      && await page.evaluate(() => !StudioFlow.flow.restoredFrom && !StudioFlow.flow.operation
        && !document.querySelector('#text').readOnly));
    await api('/api/studio/draft', { key: 'single', value: originalDraft });
    await page.reload({ waitUntil: 'networkidle' }); await ready();
    await checkOriginalCreation('original_creation_fixture_is_restored_after_legacy_fallback_check');

    await editor().fill('A new working draft after returning.');
    await restore({ job_id: edited.id }); await restore({ job_id: alternate.id });
    await page.locator('#return-draft').click();
    await page.waitForFunction(id => !StudioFlow.flow.restoring && !StudioFlow.flow.returnDrafts.length
      && document.querySelector('#result-box').dataset.jobId === id, previous.id);
    check('next_restore_session_remembers_new_creation_not_an_old_snapshot',
      await page.inputValue('#text') === 'A new working draft after returning.' && await audioMatches(previous.output));
    await editor().fill(originalDraft.form.text);
    await restore({ job_id: edited.id });
    await page.locator('#submit-job').click();
    await page.waitForFunction(id => document.querySelector('#result-box').dataset.jobId !== id
      && document.querySelector('#job-badge').dataset.status === 'complete', edited.id);
    const generated = await api('/api/jobs/' + await visibleId());
    check('bottom_generate_keeps_tags_and_restored_voice_without_editor_routing', generated.recipe.text === script
      && generated.recipe.condition_source.job_id === edited.id && !generated.studio.result_scope
      && generated.recipe.reference_snapshot[0].text === edited.recipe.reference_snapshot[0].text);
    const unchanged = await api('/api/jobs/' + edited.id);
    check('restoring_does_not_modify_original_recipe_or_editor_scope', JSON.stringify(unchanged) === JSON.stringify(originalEdited)
      && unchanged.studio.result_scope === 'audio_editor');
    await restore({ job_id: edited.id });
    await page.locator('#result-box .creative-retry').click();
    await page.waitForFunction(id => document.querySelector('#result-box').dataset.jobId !== id
      && document.querySelector('#job-badge').dataset.status === 'complete', edited.id);
    const editRetry = await api('/api/jobs/' + await visibleId());
    check('regenerate_restored_editor_result_uses_current_workspace_and_unchanged_recipe',
      editRetry.recipe.source_version.job_id === edited.id && !editRetry.studio.result_scope
      && editRetry.recipe.text === edited.recipe.text);

    const multiText = '<|speaker:0|> First.\n<|speaker:1|> Second.';
    const multi = await create({ ...basic, text: multiText });
    await restore({ job_id: multi.id });
    check('normal_text_with_multiple_speakers_is_not_stripped', await page.inputValue('#text') === multiText);
    const nonzero = await create({ ...basic, text: '<|speaker:3|> A named speaker.' });
    await restore({ job_id: nonzero.id });
    check('nondefault_speaker_identity_is_not_silently_changed', await page.inputValue('#text') === nonzero.recipe.text);

    const perLine = await create({ ...joint.recipe, reference_snapshot: undefined, source_version: undefined,
      scene_strategy: 'per_line', voice_bindings: [{ voice_id: voices[0].id, speaker: 0 }, { voice_id: voices[1].id, speaker: 1 }],
      studio: { work_id: 'restore-turns', segment_id: 'scene', layout: 'scene' } });
    const secondLine = perLine.result.segments[1];
    await restore({ job_id: perLine.id, part_index: 1 });
    check('restoring_one_turn_shows_only_that_turns_audio', await page.locator('#view-generate').isVisible()
      && await audioMatches(secondLine.path) && !await page.locator('#result-box .segment-item').count());
    await page.evaluate(() => loadJobs()); await reload();
    check('selected_turn_audio_survives_history_refresh_and_reload', await audioMatches(secondLine.path)
      && await page.evaluate(() => state.resultJobs.generate.restoredSource.part_index === 1));
    await restore({ job_id: perLine.id, part_index: 0 });
    check('another_turn_in_the_same_job_replaces_audio_and_controls', await audioMatches(perLine.result.segments[0].path)
      && await page.inputValue('#text') === joint.recipe.lines[0].text);
    await restore({ job_id: perLine.id, part_index: 1 });
    await page.locator('#result-box .creative-retry').click();
    await page.waitForFunction(id => document.querySelector('#result-box').dataset.jobId !== id
      && document.querySelector('#job-badge').dataset.status === 'complete', perLine.id);
    const retried = await api('/api/jobs/' + await visibleId());
    check('regeneration_after_restoring_a_turn_targets_that_turn_not_whole_dialogue',
      retried.recipe.source_version.part_index === 1 && retried.recipe.text.includes('准备好了。'));

    await show(previous); await show(joint);
    await page.locator('.dialogue-composer .cm-content').fill('S1：Original unsent dialogue.\nS2：Keep this working scene.');
    await page.evaluate(() => StudioFlow.save());
    const originalScene = (await api('/api/studio')).drafts.scene.value;
    const otherJoint = await create({ ...joint.recipe, reference_snapshot: undefined, source_version: undefined,
      voice_bindings: [{ voice_id: voices[0].id, speaker: 0 }, { voice_id: voices[1].id, speaker: 1 }],
      studio: { work_id: 'restore-edited-dialogue', segment_id: 'scene', layout: 'scene',
        result_scope: 'audio_editor', edit_source: 'joint-fixture' } });
    await restore({ job_id: otherJoint.id });
    check('joint_edit_draft_restores_to_dialogue_with_its_result', await page.locator('#view-scene').isVisible()
      && await audioMatches(otherJoint.output) && (await page.inputValue('#dialogue-script')).includes('S1：')
      && !(await page.inputValue('#dialogue-script')).includes('<|speaker:'));
    check('restoring_dialogue_does_not_replace_single_result', await page.evaluate(id => state.resultJobs.generate?.id === id, previous.id));
    await reload();
    check('restored_joint_edit_result_survives_reload', await visibleId() === otherJoint.id && await audioMatches(otherJoint.output));
    await restore({ job_id: joint.id }); await restore({ job_id: otherJoint.id });
    check('repeated_dialogue_restores_keep_one_original_creation', await page.evaluate(script => {
      const drafts = StudioFlow.flow.scene.returnDrafts;
      return drafts.length === 1 && drafts[0].script === script && !drafts[0].returnDrafts;
    }, originalScene.script) && await page.locator('#return-scene-draft').textContent() === '返回对话');
    await page.evaluate(() => StudioFlow.save());
    const legacyScene = (await api('/api/studio')).drafts.scene.value;
    legacyScene.returnDrafts = [originalScene, { ...structuredClone(legacyScene), returnDrafts: [] }];
    await api('/api/studio/draft', { key: 'scene', value: legacyScene });
    await page.reload({ waitUntil: 'networkidle' }); await ready();
    check('legacy_dialogue_stack_is_flattened_without_switching_workspaces', await page.locator('#view-scene').isVisible()
      && await page.evaluate(script => {
        const drafts = StudioFlow.flow.scene.returnDrafts;
        return drafts.length === 1 && drafts[0].script === script && !drafts[0].restoredFrom && !drafts[0].returnDrafts;
      }, originalScene.script));
    await reload();
    await page.locator('#return-scene-draft').click();
    await page.waitForFunction(id => document.querySelector('#result-box').dataset.jobId === id, joint.id);
    check('returning_to_dialogue_creation_restores_text_and_result_without_another_back', await page.locator('#view-scene').isVisible()
      && !await page.locator('#view-generate').isVisible() && await audioMatches(joint.output)
      && await page.inputValue('#dialogue-script') === originalScene.script
      && !await page.locator('#return-scene-draft').isVisible()
      && await page.evaluate(() => !StudioFlow.flow.scene.restoredFrom && !StudioFlow.flow.scene.operation
        && !StudioFlow.flow.scene.returnDrafts.length));
    await page.evaluate(() => StudioFlow.save());
    const returnedScene = (await api('/api/studio')).drafts.scene.value;
    check('dialogue_return_preserves_original_speakers_and_generation_settings',
      JSON.stringify(returnedScene.roles) === JSON.stringify(originalScene.roles)
      && JSON.stringify(returnedScene.generation) === JSON.stringify(originalScene.generation)
      && returnedScene.strategy === originalScene.strategy);
    await shot('returned-normal-dialogue-desktop');

    const designed = await create({ mode: 'design', text: 'A voice design audition.',
      voice_description: 'Warm clear adult voice.', character_name: 'Voice', scene: '',
      studio: { work_id: 'restore-design', segment_id: 'main', layout: 'design' } });
    await restore({ job_id: designed.id });
    check('voice_design_draft_restores_its_result', await page.locator('#voice-design').isVisible()
      && await audioMatches(designed.output) && await page.inputValue('#text') === designed.recipe.text);
    await reload();
    check('voice_design_result_survives_reload', await visibleId() === designed.id && await audioMatches(designed.output));
    await page.locator('#back-to-generation').click();
    await page.waitForFunction(id => !StudioFlow.voiceDesignActive() && !StudioFlow.flow.restoring
      && state.resultJobs.generate?.id === id, previous.id);
    check('design_history_back_exits_history_without_a_second_return_button',
      !await page.locator('#return-draft').isVisible() && !await page.locator('#back-to-generation').isVisible()
      && await page.evaluate(() => !StudioFlow.flow.returnDrafts.length && !StudioFlow.flow.restoredFrom));
    await checkOriginalCreation('design_history_back_restores_original_creation_in_one_click');
    await reload();
    await checkOriginalCreation('design_history_back_does_not_restore_a_second_return_after_reload');
    await restore({ job_id: designed.id });
    await page.evaluate(() => StudioFlow.save());
    const olderDesignDraft = (await api('/api/studio')).drafts.single.value;
    delete olderDesignDraft.design.historyReturn;
    await api('/api/studio/draft', { key: 'single', value: olderDesignDraft });
    await page.reload({ waitUntil: 'networkidle' }); await ready();
    await page.locator('#back-to-generation').click();
    await checkOriginalCreation('previously_saved_design_history_also_returns_in_one_click');
    await restore({ job_id: alternate.id }); await restore({ job_id: designed.id });
    await restore({ job_id: designed.id });
    await page.locator('#back-to-generation').click();
    await checkOriginalCreation('design_back_after_repeated_and_mixed_history_restores_returns_once');
    await restore({ job_id: alternate.id }); await restore({ job_id: designed.id }); await restore({ job_id: edited.id });
    await page.locator('#return-draft').click();
    await checkOriginalCreation('mixed_design_and_normal_history_still_return_to_original_normal_creation');
    const missingAudio = async route => {
      const response = await route.fetch(), value = await response.json();
      const version = value.version || value;
      if (version.job_id === edited.id) version.audio_missing = true;
      return route.fulfill({ response, json: value });
    };
    await page.route('**/api/studio/prepare-restore', missingAudio);
    await page.route('**/api/studio/version', missingAudio);
    await restore({ job_id: edited.id });
    check('missing_restored_audio_does_not_leave_an_unrelated_result_visible', !await page.locator('#result-box').isVisible()
      && await page.locator('#output-placeholder').isVisible() && await page.inputValue('#text') === script);
    await reload();
    check('missing_audio_does_not_fall_back_to_another_job_after_reload', !await page.locator('#result-box').isVisible()
      && await page.evaluate(id => state.resultJobs.generate.id === id && !state.resultJobs.generate.output, edited.id));
    check('no_browser_errors', report.errors.length === 0);
  } catch (error) {
    report.errors.push(error.stack || String(error)); process.exitCode = 1;
    if (page) {
      const file = path.join(folder, 'draft-restore-failure.png');
      await page.screenshot({ path: file, fullPage: true }).catch(() => {}); report.screenshots.push(file);
    }
  } finally {
    if (browser) await browser.close();
    fs.writeFileSync(path.join(folder, 'draft-restore-report.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report, null, 2));
  }
})();
