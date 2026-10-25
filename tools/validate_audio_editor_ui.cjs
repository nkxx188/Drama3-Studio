/* Dedicated audio-edit page: real HTTP/store/audio/codes; synthetic inference only. */
const fs = require('node:fs'), path = require('node:path'), { chromium } = require('playwright');
const root = path.resolve(__dirname, '..'), base = process.env.DRAMA_UI_URL, folder = process.env.DRAMA_STUDIO_FOLDER;
const report = { scope: 'Dedicated audio-edit page, fixed edit dock, explicit snippet mapping, exact codec boundaries, references and persistence; synthetic inference.',
  checks: {}, errors: [], screenshots: [] };
(async () => {
  let browser, page, releasePoll;
  try {
    browser = await chromium.launch({ channel: 'chrome', headless: true });
    const context = await browser.newContext({ viewport: { width: 1440, height: 960 }, locale: 'zh-CN' });
    page = await context.newPage(); page.on('pageerror', error => report.errors.push(error.stack || error.message));
    const api = async (url, data) => {
      const response = data === undefined ? await context.request.get(base + url)
        : await context.request.post(base + url, { data });
      if (!response.ok()) throw new Error(url + ': ' + await response.text());
      return response.json();
    };
    const check = (name, value) => { report.checks[name] = Boolean(value); if (!value) throw new Error(name); };
    const ready = () => page.waitForFunction(() => document.documentElement.dataset.studioReady === 'true');
    const choose = async (id, value) => {
      await page.locator(`#${id}-trigger`).click();
      await page.locator(`#${id}-listbox [data-value="${value}"]`).click();
    };
    const shot = async name => {
      const file = path.join(folder, name + '.png'); await page.screenshot({ path: file });
      report.screenshots.push(file);
    };
    const replacement = () => page.locator('#audio-edit-workspace .audio-edit-replacement .cm-content');
    const close = async () => {
      if (!await page.locator('#audio-edit-workspace').isVisible()) {
        await page.evaluate(() => view('audio-edit'));
      }
      await page.locator('#audio-edit-back').click();
      await page.locator('#audio-edit-workspace').waitFor({ state: 'detached' });
      await page.waitForFunction(() => ['generate', 'scene'].includes(document.documentElement.dataset.view));
    };
    const show = async id => {
      const job = await api('/api/jobs/' + id);
      await page.evaluate(job => { view(resultWorkspaceFor(job)); monitorJob(job.id); }, job);
      await page.waitForFunction(id => document.querySelector('#result-box').dataset.jobId === id, id);
    };
    const open = async (id, mode = 'local', part = null) => {
      if (await page.locator('#audio-edit-workspace').count()) await close();
      await show(id);
      await page.locator(part === null ? '#result-box .main-result .creative-edit'
        : `#result-box .segment-item[data-part-index="${part}"] .creative-edit`).click();
      await page.locator('#audio-edit-workspace').waitFor();
      await page.waitForFunction(() => document.querySelector('#audio-edit-source')?.duration > 0);
      await page.locator(`#audio-edit-workspace [data-edit-mode="${mode}"]`).click();
    };
    const finish = async (selector = '#audio-edit-submit') => {
      const previous = await page.evaluate(() => StudioFlow.flow.lastSubmit?.id || '');
      const editing = selector === '#audio-edit-submit';
      const originalId = await page.locator('#result-box').getAttribute('data-job-id');
      await page.locator(selector).click();
      await page.waitForFunction(({ previous, editing }) => {
        const id = StudioFlow.flow.lastSubmit?.id;
        const box = document.querySelector(editing ? '#audio-edit-result-box' : '#result-box');
        const badge = document.querySelector(editing ? '#audio-edit-result-badge' : '#job-badge');
        return id && id !== previous && box?.dataset.jobId === id && badge?.dataset.status === 'complete';
      }, { previous, editing });
      const result = await api('/api/jobs/' + await page.evaluate(() => StudioFlow.flow.lastSubmit.id));
      if (editing) {
        check(`edited_result_${result.id}_does_not_replace_source`,
          await page.locator('#audio-edit-workspace').isVisible()
          && await page.locator('#audio-edit-results').isVisible()
          && await page.locator('#result-box').getAttribute('data-job-id') === originalId
          && await page.evaluate(id => !Object.values(state.resultJobs).some(job => job?.id === id), result.id)
          && await page.locator('#audio-edit-result-box .main-result audio').getAttribute('src')
            === await page.evaluate(output => fileUrl(output), result.output));
        await close();
      }
      (report.requests ||= []).push({ id: result.id, status: result.status, text: result.recipe.text,
        source_version: result.recipe.source_version, operation: result.recipe.operation, infill: result.result?.infill });
      return result;
    };
    const create = async payload => {
      const job = await api('/api/jobs', payload), deadline = Date.now() + 15000;
      do {
        const value = await api('/api/jobs/' + job.id);
        if (['failed', 'error'].includes(value.status)) throw new Error(value.error);
        const finished = payload.scene_strategy === 'per_line'
          ? value.result?.segments?.length && value.result.segments.every(row => row.take_file) : value.take_file;
        if (value.status === 'complete' && finished) return value;
        await new Promise(resolve => setTimeout(resolve, 80));
      } while (Date.now() < deadline);
      throw new Error('Synthetic source did not finish: ' + job.id);
    };
    const select = async (start, end) => {
      // A new range gesture must start outside an existing draggable handle.
      await page.locator('#audio-edit-source').evaluate(audio => StudioAudio.clearSelection(audio));
      const wave = page.locator('#audio-edit-workspace .audio-edit-source .player-wave-surface'); await wave.scrollIntoViewIfNeeded();
      const bounds = await wave.boundingBox();
      await page.mouse.move(bounds.x + bounds.width * start, bounds.y + bounds.height / 2);
      await page.mouse.down(); await page.mouse.move(bounds.x + bounds.width * end, bounds.y + bounds.height / 2, { steps: 10 });
      await page.mouse.up();
    };
    await page.goto(base, { waitUntil: 'networkidle' }); await ready();
    const untouched = '不能被局部编辑覆盖的创作草稿。';
    await page.locator('.compose-panel > .editor-area .cm-content').fill(untouched);
    const voices = (await api('/api/voices')).voices;
    const basic = { mode: 'tts', voice_ids: [voices[0].id], temperature: .812345678, top_p: .934567891,
      max_new_tokens: 777, studio: { work_id: 'editor-work', segment_id: 'first', layout: 'single' } };
    const first = await create({ ...basic, text: '明天下午我们一起出发。' });
    report.first = { id: first.id, take_file: first.take_file };
    await open(first.id);
    check('editing_uses_a_dedicated_view_with_two_scopes_and_no_continuation_or_jump_buttons',
      !await page.locator('dialog[open]').count() && await page.locator('#audio-edit-workspace .audio-edit-tabs > [role="tab"]').count() === 2
      && !await page.locator('#audio-edit-workspace [data-edit-mode="record"]').count()
      && !await page.locator('#audio-edit-workspace [data-edit-mode="continue"]').count()
      && !await page.locator('#audio-edit-full-script, #audio-edit-performance').count()
      && await page.locator('.audio-edit-heading-actions button').count() === 1
      && await page.locator('#audio-edit-workspace').evaluate(node =>
        getComputedStyle(node).position !== 'fixed' && node.parentElement.id === 'view-audio-edit')
      && await page.locator('#view-audio-edit').isVisible() && !await page.locator('#view-generate').isVisible());
    check('opening_editor_hides_but_preserves_the_original_result',
      !await page.locator('#result-box').isVisible()
      && !await page.locator('#audio-edit-results').isVisible()
      && !await page.locator('#audio-edit-result-box audio').count()
      && await page.locator('#result-box').getAttribute('data-job-id') === first.id);
    check('local_input_is_snippet_only_and_requires_mapping',
      await page.inputValue('#audio-edit-quote') === '' && await page.inputValue('#audio-edit-text-0') === ''
      && await page.locator('#audio-edit-submit').isDisabled());
    check('editor_has_its_own_fixed_settings_and_submit_dock',
      await page.locator('#audio-edit-dock').isVisible()
      && await page.locator('#audio-edit-dock .audio-edit-parameters').isVisible()
      && await page.locator('#audio-edit-submit').isVisible()
      && !await page.locator('#submit-job').isVisible());
    await page.reload({ waitUntil: 'networkidle' }); await ready();
    await page.locator('#audio-edit-workspace').waitFor();
    check('refresh_restores_the_active_edit_page_and_source',
      await page.locator('#view-audio-edit').isVisible()
      && await page.locator('#audio-edit-source').getAttribute('src') === await page.evaluate(output => fileUrl(output), first.output)
      && await page.locator('#audio-edit-dock').isVisible());
    check('local_regeneration_is_not_an_external_result_action', !await page.locator('.result-infill').isVisible());
    await close();
    check('back_returns_to_creation_with_the_original_result_intact',
      await page.locator('#view-generate').isVisible()
      && await page.locator('#result-box').getAttribute('data-job-id') === first.id);
    await page.locator('.main-result .result-more > summary').click();
    check('voice_reuse_and_save_stay_in_more',
      await page.locator('.result-more').getByRole('button', { name: '用作参考音色', exact: true }).isVisible()
      && await page.locator('.result-more').getByRole('button', { name: '保存音色', exact: true }).isVisible());
    await page.locator('.main-result .result-more > summary').click();
    await open(first.id); await select(.2, .5);
    await page.locator('#audio-edit-quote').fill('下午'); await replacement().fill('上午');
    check('diff_only_marks_the_changed_snippet_not_surrounding_transcript',
      await page.locator('#audio-edit-diff del').textContent() === '下'
      && !(await page.locator('#audio-edit-diff').textContent()).includes('我们一起出发')
      && await page.locator('#audio-edit-submit').isEnabled());
    const selected = await page.locator('#audio-edit-source').evaluate(audio => StudioAudio.selection(audio));
    await page.locator('#audio-edit-workspace .audio-edit-exact > summary').click();
    await page.locator('#audio-edit-workspace .audio-edit-parameters > summary').click();
    check('every_numeric_editor_field_displays_at_most_two_decimals',
      await page.locator('#audio-edit-workspace input[type=number]').evaluateAll(nodes => nodes.every(node =>
        !node.value || !node.value.includes('.') || node.value.split('.')[1].length <= 2)));
    await page.locator('#audio-edit-start').focus(); await page.locator('#audio-edit-start').press('Tab');
    check('display_rounding_and_focus_do_not_move_audio_boundaries',
      await page.locator('#audio-edit-source').evaluate((audio, range) =>
        JSON.stringify(StudioAudio.selection(audio)) === JSON.stringify(range), selected));
    check('inputs_and_panels_have_distinct_visible_borders',
      await page.locator('#audio-edit-quote').evaluate(node =>
        getComputedStyle(node).borderTopColor === 'rgb(185, 194, 206)')
      && await page.locator('#audio-edit-workspace').evaluate(node => getComputedStyle(node).borderTopColor === 'rgb(217, 222, 229)'));
    await page.locator('#audio-edit-workspace .audio-edit-exact > summary').click();
    await page.locator('#audio-edit-workspace .audio-edit-parameters > summary').click();
    await shot('audio-editor-inline-local');
    await close(); await page.reload({ waitUntil: 'networkidle' }); await ready(); await open(first.id);
    check('snippet_draft_and_exact_range_restore_without_overwriting_creation',
      await page.inputValue('#audio-edit-quote') === '下午' && await page.inputValue('#audio-edit-text-0') === '上午'
      && await page.inputValue('#text') === untouched
      && await page.locator('#audio-edit-source').evaluate((audio, range) =>
        JSON.stringify(StudioAudio.selection(audio)) === JSON.stringify(range), selected));
    const local = await finish();
    check('local_request_reconstructs_full_text_and_keeps_other_audio_codes',
      local.recipe.text === '明天上午我们一起出发。' && local.recipe.operation === 'infill'
      && local.recipe.source_take === first.take_file && local.result.infill.prefix_codes_equal
      && local.result.infill.suffix_codes_equal);
    check('sampling_and_exact_codec_frames_survive_two_decimal_display',
      local.recipe.temperature === basic.temperature && local.recipe.top_p === basic.top_p
      && local.recipe.max_new_tokens === 777
      && local.recipe.start_frame === Math.round(selected.start * 44100 / 2048)
      && local.recipe.end_frame === Math.round(selected.end * 44100 / 2048));
    await open(first.id);
    await page.waitForFunction(id => document.querySelector('#audio-edit-result-box').dataset.jobId === id
      && document.querySelector('#audio-edit-result-badge').dataset.status === 'complete', local.id);
    check('reopening_the_source_restores_only_its_edited_result',
      await page.locator('#result-box').getAttribute('data-job-id') === first.id
      && (await api('/api/studio')).drafts.audio_editor.value.entries[`${first.id}:all`].result_job_id === local.id);
    await page.waitForFunction(() =>
      document.querySelectorAll('#audio-edit-workspace .audio-edit-source .player-wave rect').length === 80
      && document.querySelectorAll('#audio-edit-result-box .main-result .player-wave rect').length === 80);
    for (const width of [1920, 1440, 390]) {
      await page.setViewportSize({ width, height: width === 390 ? 844 : 1080 });
      const layout = await page.locator('#audio-edit-workspace').evaluate(node => {
        const box = selector => node.querySelector(selector).getBoundingClientRect();
        const source = box('.audio-edit-source .player-wave-surface');
        const result = box('.audio-edit-results .player-wave-surface');
        const media = box('.audio-edit-source'), fields = box('.audio-edit-local-fields');
        return { sourceWidth: source.width, resultWidth: result.width,
          rightColumn: result.left > source.right, stacked: result.top > fields.bottom,
          fieldsBelowSource: fields.top >= media.bottom - 1, overflow: node.scrollWidth > node.clientWidth + 1 };
      });
      check(`editor_equal_waveforms_and_fields_below_source_${width}`,
        Math.abs(layout.sourceWidth - layout.resultWidth) < 1 && layout.fieldsBelowSource && !layout.overflow
        && (width > 1000 ? layout.rightColumn : layout.stacked));
      await page.locator('#audio-edit-workspace').scrollIntoViewIfNeeded();
      await shot('audio-editor-equal-columns-' + width);
    }
    await page.setViewportSize({ width: 1440, height: 960 });
    await shot('audio-editor-separated-results');
    await close(); await page.evaluate(() => StudioFlow.save());
    await page.reload({ waitUntil: 'networkidle' }); await ready();
    check('reload_keeps_original_workspace_result_after_editing',
      await page.locator('#result-box').getAttribute('data-job-id') === first.id
      && await page.evaluate(id => resultWorkspaceFor(state.jobs.find(job => job.id === id)) === 'editor', local.id));
    await open(first.id);
    await page.waitForFunction(id => document.querySelector('#audio-edit-result-box').dataset.jobId === id, local.id);
    const previousEdit = await page.evaluate(() => StudioFlow.flow.lastSubmit?.id || '');
    await page.locator('#audio-edit-result-box .creative-retry').click();
    await page.waitForFunction(previous => {
      const id = StudioFlow.flow.lastSubmit?.id;
      return id && id !== previous && document.querySelector('#audio-edit-result-box').dataset.jobId === id
        && document.querySelector('#audio-edit-result-badge').dataset.status === 'complete';
    }, previousEdit);
    check('edited_result_regeneration_stays_inside_the_editor',
      await page.locator('#result-box').getAttribute('data-job-id') === first.id);
    await close();
    await open(first.id); await page.locator('#audio-edit-quote').fill('不存在的原话');
    check('unmatched_original_words_cannot_replace_whole_transcript', await page.locator('#audio-edit-submit').isDisabled()
      && (await page.locator('#audio-edit-error').textContent()).includes('未在原文中找到'));
    await page.locator('#audio-edit-locator > summary').click();
    await page.locator('#audio-edit-original-0').focus(); await page.locator('#audio-edit-original-0').press('Control+Home');
    await page.locator('#audio-edit-original-0').press('ArrowRight'); await page.locator('#audio-edit-original-0').press('ArrowRight');
    await page.locator('#audio-edit-original-0').press('Shift+ArrowRight'); await page.locator('#audio-edit-original-0').press('Shift+ArrowRight');
    await page.locator('.audio-edit-original-line button').click();
    check('source_text_selection_sets_exact_original_quote', await page.inputValue('#audio-edit-quote') === '下午');
    await close();
    const repeated = await create({ ...basic, text: '等等，先等等，我们走吧。' });
    await open(repeated.id); await select(.3, .6); await page.locator('#audio-edit-quote').fill('等等');
    await replacement().fill('再等等');
    check('repeated_words_require_explicit_occurrence', await page.locator('.audio-edit-match').count() === 2
      && await page.locator('#audio-edit-submit').isDisabled());
    await page.locator('.audio-edit-match').nth(1).click();
    const chosen = await finish();
    check('only_the_chosen_repeat_is_replaced', chosen.recipe.text === '等等，先再等等，我们走吧。');
    const tagged = await create({ ...basic, text: '今天<whisper>下午</whisper>一起出发。' });
    await open(tagged.id); await select(.2, .5);
    await page.locator('#audio-edit-quote').fill('下午</whisper>一起'); await replacement().fill('晚上一起');
    check('local_range_cannot_break_surrounding_scope_tags', await page.locator('#audio-edit-submit').isDisabled());
    await page.locator('#audio-edit-quote').fill('下午');
    await replacement().press('Control+A'); await replacement().press('Backspace');
    await replacement().pressSequentially('晚上');
    const tagEdit = await finish();
    check('local_words_preserve_surrounding_tags_and_text', tagEdit.recipe.text === '今天<whisper>晚上</whisper>一起出发。');
    await open(first.id);
    await page.locator('#audio-edit-workspace .audio-edit-source .player-seek').focus();
    await page.locator('#audio-edit-workspace .audio-edit-source .player-seek').press('End');
    await page.locator('#audio-edit-insert').click();
    check('insertion_requires_an_explicit_text_position', await page.locator('#audio-edit-submit').isDisabled());
    await page.locator('#audio-edit-original-0').focus(); await page.locator('#audio-edit-original-0').press('Control+End');
    await page.locator('.audio-edit-original-line button').click(); await replacement().fill('请准时。');
    const insertion = await finish();
    check('insertion_keeps_source_words_and_submits_equal_codec_boundaries',
      insertion.recipe.text === first.recipe.text + '请准时。' && insertion.recipe.start_frame === insertion.recipe.end_frame);
    await show(first.id);
    check('whole_regeneration_is_a_single_result_button', await page.locator('#result-box .main-result .creative-retry').textContent() === '重新生成');
    const retry = await finish('#result-box .main-result .creative-retry');
    check('regenerate_button_uses_original_recipe', retry.recipe.operation === 'retry' && retry.recipe.text === first.recipe.text);
    await open(first.id, 'whole');
    check('whole_scope_edits_text_and_delivery_without_leaving_the_editor_page',
      await page.locator('#audio-edit-workspace').isVisible() && !await page.locator('dialog[open]').count()
      && await page.inputValue('#audio-edit-whole-0') === first.recipe.text
      && await page.locator('#audio-edit-direction').isVisible() && await page.inputValue('#text') === untouched);
    check('unchanged_full_script_is_not_an_extra_regenerate_partition',
      await page.locator('#audio-edit-submit').isDisabled());
    await page.locator('#audio-edit-direction').fill('轻声而自然。');
    await shot('audio-editor-dedicated-whole');
    const revision = await finish();
    check('director_revision_keeps_original_words_and_audio_context', revision.recipe.operation === 'revision'
      && revision.recipe.text === first.recipe.text && revision.recipe.previous_take === first.take_file
      && revision.recipe.direction === '轻声而自然。');
    await open(first.id, 'whole');
    await page.locator('#audio-edit-scripts .cm-content').fill('改到后天早上再出发。');
    await page.locator('[data-edit-mode="local"]').click();
    check('switching_scope_preserves_local_words_independently',
      await page.inputValue('#audio-edit-text-0') === '请准时。' && await page.inputValue('#text') === untouched);
    await page.locator('[data-edit-mode="whole"]').click(); await close();
    await page.reload({ waitUntil: 'networkidle' }); await ready(); await open(first.id, 'whole');
    check('whole_text_and_direction_restore_without_touching_home_draft',
      await page.inputValue('#audio-edit-whole-0') === '改到后天早上再出发。'
      && await page.inputValue('#audio-edit-direction') === '轻声而自然。' && await page.inputValue('#text') === untouched);
    const reword = await finish();
    check('full_script_edit_reuses_original_voice_without_old_word_history', reword.recipe.operation === 'reword'
      && reword.recipe.text === '改到后天早上再出发。' && !reword.recipe.previous_take
      && reword.recipe.reference_snapshot[0].audio === first.recipe.reference_snapshot[0].audio);
    check('original_take_remains_immutable', (await api('/api/jobs/' + first.id)).recipe.text === first.recipe.text);
    const joint = await create({ mode: 'scene', scene_strategy: 'joint', scene: 'A conversation.',
      characters: [{ name: '林', description: 'Calm', speaker: 0 }, { name: '周', description: 'Warm', speaker: 1 }],
      character_ids: ['lin', 'zhou'], lines: [{ character: '林', text: '我们准备好了吗？' }, { character: '周', text: '准备好了。' }],
      line_ids: ['one', 'two'], voice_bindings: [{ voice_id: voices[0].id, speaker: 0 }, { voice_id: voices[1].id, speaker: 1 }],
      studio: { work_id: 'editor-joint', segment_id: 'dialogue', layout: 'scene' } });
    await open(joint.id); await select(.4, .7); await page.locator('#audio-edit-quote').fill('准备好了。');
    check('a_different_source_starts_in_the_editor_page_without_another_sources_result',
      !await page.locator('#audio-edit-results').isVisible()
      && await page.locator('#view-audio-edit').isVisible()
      && await page.locator('.nav[data-view="scene"]').getAttribute('aria-current') === 'page');
    await replacement().fill('还没准备好。');
    const localJoint = await finish();
    check('joint_local_edit_preserves_other_turns_roles_and_codes',
      localJoint.recipe.lines[0].text === joint.recipe.lines[0].text && localJoint.recipe.lines[1].text === '还没准备好。'
      && localJoint.recipe.characters[1].speaker === 1 && localJoint.recipe.line_ids[1] === 'two'
      && localJoint.result.infill.prefix_codes_equal && localJoint.result.infill.suffix_codes_equal);
    check('back_from_dialogue_edit_returns_to_dialogue',
      await page.locator('#view-scene').isVisible() && await page.locator('#result-box').getAttribute('data-job-id') === joint.id);
    await open(joint.id, 'whole');
    check('whole_joint_dialogue_shows_named_roles_not_protocol_tokens',
      await page.inputValue('#audio-edit-role-0') === '林' && await page.inputValue('#audio-edit-role-1') === '周'
      && !await page.inputValue('#audio-edit-whole-0').then(text => text.includes('<|speaker:')));
    await page.locator('#audio-edit-add-line').click();
    await choose('audio-edit-role-2', '周');
    await page.locator('#audio-edit-scripts .cm-content').nth(2).fill('马上就好。');
    await page.locator('#audio-edit-scripts .audio-edit-line').first().getByRole('button', { name: '移除', exact: true }).click();
    const fullJoint = await finish();
    check('whole_joint_edit_preserves_roles_remaining_line_ids_and_frozen_voices',
      fullJoint.recipe.operation === 'reword' && fullJoint.recipe.lines.length === 2
      && fullJoint.recipe.line_ids[0] === 'two' && fullJoint.recipe.lines[1].character === '周'
      && fullJoint.recipe.lines[1].text === '马上就好。' && fullJoint.recipe.reference_snapshot.length === 2);
    const perLine = await create({ ...joint.recipe, reference_snapshot: undefined, source_version: undefined,
      mode: 'scene', scene_strategy: 'per_line',
      voice_bindings: [{ voice_id: voices[0].id, speaker: 0 }, { voice_id: voices[1].id, speaker: 1 }],
      studio: { work_id: 'editor-lines', segment_id: 'dialogue', layout: 'scene' } });
    await open(perLine.id, 'local', 1); await select(.3, .6); await page.locator('#audio-edit-quote').fill('准备好了');
    await replacement().fill('还没好');
    const changedLine = await finish();
    check('per_line_local_edit_preserves_speaker_and_line_identity', changedLine.recipe.text.includes('<|speaker:1|>')
      && changedLine.recipe.text.includes('还没好。') && changedLine.studio.line_id === 'two');
    const legacyRoute = async route => {
      if (route.request().postDataJSON().job_id !== first.id) return route.continue();
      const response = await route.fetch(), value = await response.json(); value.complete_recipe = false;
      delete value.recipe.schema; await route.fulfill({ json: value });
    };
    await page.route('**/api/studio/version', legacyRoute); await open(first.id);
    await select(.2, .5); await page.locator('#audio-edit-quote').fill('下午'); await replacement().fill('晚上');
    const legacy = await finish();
    check('legacy_local_edit_reconstructs_transcript_with_explicit_source_fallback', legacy.recipe.mode === 'infill'
      && legacy.recipe.text === '明天晚上我们一起出发。' && legacy.recipe.source_take === first.take_file
      && legacy.recipe.legacy_source.job_id === first.id && !legacy.recipe.source_version);
    await open(first.id, 'whole');
    await page.locator('#audio-edit-scripts .cm-content').fill('使用旧记录生成新的台词。');
    const legacyWhole = await finish();
    check('legacy_full_edit_preserves_source_voice_without_restoring_retired_continuation',
      legacyWhole.recipe.mode === 'tts' && legacyWhole.recipe.text === '使用旧记录生成新的台词。'
      && legacyWhole.recipe.reference_snapshot.length === 1 && !legacyWhole.recipe.previous_take
      && legacyWhole.recipe.legacy_source.job_id === first.id);
    await page.unroute('**/api/studio/version', legacyRoute);
    const designed = await create({ mode: 'design', text: '这是刚设计的声音。', voice_description: 'A warm voice.',
      studio: { work_id: 'inline-designed-voice', segment_id: 'voice', layout: 'single' } });
    await open(designed.id, 'whole'); await page.locator('#audio-edit-scripts .cm-content').fill('现在让这个声音说新的话。');
    await page.locator('#audio-edit-reference-0-transcript').waitFor({ state: 'visible' });
    check('missing_original_voice_is_prepared_inline_not_in_a_modal',
      !await page.locator('dialog[open]').count() && await page.locator('#audio-edit-source-voices').isVisible()
      && await page.inputValue('#audio-edit-reference-0-transcript') === designed.recipe.text
      && await page.inputValue('#text') === untouched);
    await page.locator('#audio-edit-submit').click();
    check('inline_voice_requires_reference_confirmation',
      (await page.locator('#audio-edit-error').textContent()).includes('核对'));
    await page.locator('#audio-edit-reference-checked').check();
    const fromDesign = await finish();
    check('new_words_from_designed_voice_use_checked_source_reference',
      fromDesign.recipe.text === '现在让这个声音说新的话。' && fromDesign.recipe.reference_transcripts_checked
      && fromDesign.recipe.reference_snapshot.length === 1 && !fromDesign.recipe.previous_take);
    const raceSource = await create({ ...basic, text: '编辑结果等待返回时，不能覆盖任何原始结果。' });
    await open(raceSource.id, 'whole'); await page.locator('#audio-edit-direction').fill('轻声、清楚。');
    const originalDialogueId = await page.evaluate(() => state.resultJobs.scene?.id);
    let nextId, pollSeen;
    const seen = new Promise(resolve => { pollSeen = resolve; });
    const gate = new Promise(resolve => { releasePoll = resolve; });
    const createRoute = async route => {
      if (route.request().method() !== 'POST') return route.continue();
      const response = await route.fetch(); nextId = (await response.json()).id;
      await route.fulfill({ response });
    };
    const delayedPoll = async route => {
      if (route.request().method() === 'GET' && route.request().url().endsWith('/' + nextId)) {
        pollSeen(); await gate;
      }
      await route.continue();
    };
    await page.route('**/api/jobs', createRoute); await page.route('**/api/jobs/*', delayedPoll);
    await page.locator('#audio-edit-submit').click(); await seen;
    check('editor_has_independent_progress_and_cancel_before_audio_arrives',
      await page.locator('#audio-edit-results').isVisible() && await page.locator('#audio-edit-cancel').isVisible()
      && !await page.locator('#audio-edit-result-box audio').isVisible()
      && await page.locator('#result-box').getAttribute('data-job-id') === raceSource.id);
    await page.locator('.nav[data-view="scene"]').click();
      check('switching_away_during_edit_keeps_dialogue_result',
      await page.locator('#result-box').getAttribute('data-job-id') === originalDialogueId);
    releasePoll();
    await page.waitForFunction(id => document.querySelector('#audio-edit-result-box').dataset.jobId === id
      && document.querySelector('#audio-edit-result-badge').dataset.status === 'complete', nextId);
    check('late_edit_response_stays_in_the_editor_without_navigating',
      await page.locator('#view-scene').isVisible()
      && await page.locator('#result-box').getAttribute('data-job-id') === originalDialogueId);
    await page.unroute('**/api/jobs', createRoute); await page.unroute('**/api/jobs/*', delayedPoll);
    await page.locator('.nav[data-view="generate"]').click();
    check('returning_to_creation_preserves_source_and_separate_completed_edit',
      await page.locator('#result-box').getAttribute('data-job-id') === raceSource.id
      && await page.locator('#audio-edit-result-box').getAttribute('data-job-id') === nextId);
    await close();
    await page.evaluate(() => StudioFlow.save());
    const savedDrafts = (await api('/api/studio')).drafts;
    check('edit_requests_do_not_rewrite_either_page_result_pointer',
      savedDrafts.single.value.active_job_id === raceSource.id
      && savedDrafts.scene.value.active_job_id === originalDialogueId);
    for (const locale of ['zh-CN', 'en', 'ja', 'ko']) {
      await choose('ui-language', locale); await open(first.id); await select(.2, .5);
      await page.locator('#audio-edit-quote').fill('下午'); await replacement().fill('上午');
      for (const size of [{ width: 1440, height: 960 }, { width: 390, height: 844 }]) {
        await page.setViewportSize(size); await page.locator('#audio-edit-submit').scrollIntoViewIfNeeded();
      check(`dedicated_editor_no_horizontal_overflow_${locale}_${size.width}`,
          await page.locator('#audio-edit-workspace').evaluate(node => {
            const rect = node.getBoundingClientRect(), button = node.querySelector('#audio-edit-submit').getBoundingClientRect();
            return rect.left >= 0 && rect.right <= innerWidth + 1 && node.scrollWidth <= node.clientWidth + 1
              && button.left >= 0 && button.right <= innerWidth && button.top >= 0 && button.bottom <= innerHeight;
          }));
      }
      await shot('audio-editor-dedicated-' + locale);
      await page.locator('[data-edit-mode="whole"]').click();
      check(`whole_editor_stays_on_dedicated_page_${locale}`, await page.locator('#audio-edit-direction').isVisible()
        && await page.inputValue('#text') === untouched && !await page.locator('dialog[open]').count());
      await close();
      await page.setViewportSize({ width: 1440, height: 960 });
    }
    await page.evaluate(async id => StudioFlow.restoreVersion({ job_id: id }), nextId);
    check('restoring_an_edit_as_regular_draft_drops_editor_result_routing',
      await page.evaluate(() => !StudioFlow.flow.single.result_scope && !StudioFlow.flow.single.edit_source));
    check('no_browser_exceptions', !report.errors.length);
  } catch (error) {
    report.errors.push(error.stack || String(error));
    if (page) {
      const file = path.join(folder, 'audio-editor-failure.png');
      await page.screenshot({ path: file }).catch(() => {}); report.screenshots.push(file);
    }
  } finally {
    releasePoll?.();
    if (browser) await browser.close();
    report.passed = !report.errors.length && Object.values(report.checks).every(Boolean);
    fs.writeFileSync(path.join(root, 'research/2026-10-19-audio-editor-ui-acceptance.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report, null, 2)); if (!report.passed) process.exitCode = 1;
  }
})();
