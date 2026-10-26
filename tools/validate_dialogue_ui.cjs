/* Real editor, file uploads, storage and audio; explicit synthetic inference/ASR. */
const fs = require('node:fs'), path = require('node:path'), { chromium } = require('playwright');
const root = path.resolve(__dirname, '..'), base = process.env.DRAMA_UI_URL, folder = process.env.DRAMA_STUDIO_FOLDER;
const report = { scope: 'Unified multi-speaker editor, voice switching, parsing, both generation strategies, persisted drafts and responsive localization; synthetic inference and ASR.',
  checks: {}, errors: [], screenshots: [] };
(async () => {
  let browser, page;
  try {
    browser = await chromium.launch({ channel: 'chrome', headless: true });
    const context = await browser.newContext({ viewport: { width: 1440, height: 960 }, locale: 'zh-CN' });
    await context.addInitScript(() => localStorage.setItem('drama3.ui.language', 'zh-CN'));
    page = await context.newPage(); page.on('pageerror', error => report.errors.push(error.stack || error.message));
    let asrCalls = 0, jobCalls = 0, asrReady = false, asrFailure = false, asrAvailable = false;
    await page.route('**/api/transcriptions', route => {
      asrCalls++;
      return route.fulfill({ json: asrAvailable ? { id: 'dialogue-asr-fixture', status: 'queued' }
        : { id: 'dialogue-asr-fixture', status: 'skipped', reason: 'model_unavailable' } });
    });
    await page.route('**/api/transcriptions/dialogue-asr-fixture', route => route.fulfill({
      json: asrFailure ? { id: 'dialogue-asr-fixture', status: 'failed', error: 'Synthetic ASR failure.' }
        : asrReady ? { id: 'dialogue-asr-fixture', status: 'complete', result: { text: '自动转写的参考台词。' } }
        : { id: 'dialogue-asr-fixture', status: 'running' },
    }));
    page.on('request', request => { if (request.url().endsWith('/api/jobs') && request.method() === 'POST') jobCalls++; });
    const check = (name, value) => { report.checks[name] = Boolean(value); if (!value) throw new Error(name); };
    const api = async (route, data) => {
      const response = data === undefined ? await context.request.get(base + route)
        : await context.request.post(base + route, { data });
      if (!response.ok()) throw new Error(route + ': ' + await response.text());
      return response.json();
    };
    const ready = () => page.waitForFunction(() => document.documentElement.dataset.studioReady === 'true');
    const settle = () => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    const editor = () => page.locator('.dialogue-editor-area .cm-content');
    const host = () => page.locator('.dialogue-editor-area .script-editor');
    const card = speaker => page.locator(`.dialogue-speaker[data-speaker="${speaker}"]`);
    const playStyle = locator => locator.evaluate(node => {
      const box = node.getBoundingClientRect(), image = node.querySelector('svg').getBoundingClientRect();
      return [box.width, box.height, getComputedStyle(node).borderRadius, image.width, image.height];
    });
    const choose = async (id, value) => {
      await page.locator(`#${id}-trigger`).click();
      await page.locator(`#${id}-listbox [data-value=${JSON.stringify(value)}]`).click();
    };
    const openVoice = async speaker => {
      await card(speaker).locator('.speaker-change').click();
      await page.locator('#dialogue-voice-dialog').waitFor({ state: 'visible' });
    };
    const closeVoice = async () => {
      await page.locator('#dialogue-voice-dialog .studio-dialog-heading .icon-btn').click();
      await page.locator('#dialogue-voice-dialog').waitFor({ state: 'detached' });
    };
    const applyVoice = async () => {
      await page.locator('#dialogue-voice-apply').click();
      await page.locator('#dialogue-voice-dialog').waitFor({ state: 'detached' });
    };
    const pickVoice = async (speaker, voice) => {
      await openVoice(speaker);
      await page.locator('#dialogue-voice-tab-library').click();
      await page.locator(`.dialogue-voice-option[data-voice-id="${voice}"] .dialogue-voice-pick`).click();
      await applyVoice();
    };
    const finish = async previous => {
      await page.waitForFunction(old => {
        const id = StudioFlow.flow.lastSubmit?.id;
        return id && id !== old && document.querySelector('#result-box').dataset.jobId === id
          && document.querySelector('#job-badge').dataset.status === 'complete';
      }, previous || '');
      return api('/api/jobs/' + await page.locator('#result-box').getAttribute('data-job-id'));
    };
    const shot = async name => {
      const file = path.join(folder, name + '.png'); await page.screenshot({ path: file });
      report.screenshots.push(file);
    };
    await page.goto(base, { waitUntil: 'networkidle' }); await ready();
    await page.locator('.nav[data-view="scene"]').click(); await settle();
    check('navigation_and_page_title_use_dialogue', (await page.locator('.nav[data-view="scene"]').textContent()).trim() === '多人对话'
      && await page.locator('#view-title').textContent() === '多人对话');
    check('one_script_editor_replaces_individual_line_forms', await editor().count() === 1
      && !await page.locator('#scene-lines,#add-line,.dialogue-line,.character-row').count());
    check('no_scene_character_or_line_direction_fields', !await page.locator(
      '#scene-direction,#scene-description,#scene-context-options,[id^="role-description-"],[id^="line-direction-"]'
    ).count());
    check('default_speakers_are_s1_and_s2', JSON.stringify(await page.locator('.speaker-badge').allTextContents()) === '["S1","S2"]');
    check('speaker_cards_expose_a_remove_action', await page.locator('.speaker-remove').count() === 2);
    check('speaker_card_action_icons_are_smaller_than_play',
      await card(0).locator('.speaker-play .icon').evaluate(node => node.getBoundingClientRect().width) === 17
      && await card(0).locator('.speaker-change .icon').evaluate(node => node.getBoundingClientRect().width) === 14
      && await card(0).locator('.speaker-remove .icon').evaluate(node => node.getBoundingClientRect().width) === 14);
    check('speaker_rows_have_no_waveforms_or_audio_forms', !await page.locator('.dialogue-speaker audio,.dialogue-speaker .studio-player,.dialogue-speaker input,.dialogue-speaker select').count());
    check('empty_speakers_have_disabled_preview', await card(0).locator('.speaker-play').isDisabled()
      && await card(1).locator('.speaker-play').isDisabled());
    check('unplayed_and_disabled_speakers_use_the_standard_play_icon',
      await page.locator('.speaker-play svg.icon use[href="#i-play"]').count() === 2);
    const standardPlayStyle = [32, 32, '4px', 17, 17];
    check('speaker_play_controls_have_stable_compact_dimensions',
      JSON.stringify(await playStyle(card(0).locator('.speaker-play'))) === JSON.stringify(standardPlayStyle)
      && JSON.stringify(await playStyle(card(1).locator('.speaker-play'))) === JSON.stringify(standardPlayStyle));
    check('editor_is_left_and_wider_than_speaker_sidebar', await page.evaluate(() => {
      const editor = document.querySelector('.dialogue-editor-area').getBoundingClientRect();
      const speakers = document.querySelector('.dialogue-speakers').getBoundingClientRect();
      return editor.right < speakers.left && editor.width > speakers.width * 1.5;
    }));
    await editor().focus(); await settle();
    check('empty_dialogue_keeps_the_two_line_hint_without_inserting_text',
      await editor().locator('.cm-placeholder').textContent() === 'S1:\nS2:'
      && await page.inputValue('#dialogue-script') === '');
    check('empty_dialogue_caret_stays_on_the_first_line', await page.evaluate(() => {
      const view = ScriptEditor.get(document.getElementById('dialogue-script')).view;
      const caret = view.scrollDOM.querySelector('.cm-cursor-primary')?.getBoundingClientRect();
      const placeholder = view.contentDOM.querySelector('.cm-placeholder');
      const first = document.createRange(); first.setStart(placeholder.firstChild, 0); first.setEnd(placeholder.firstChild, 3);
      const line = first.getBoundingClientRect(), lineHeight = parseFloat(getComputedStyle(view.contentDOM).lineHeight);
      return caret && caret.height > 0 && caret.height <= lineHeight
        && Math.abs(caret.top - line.top) < 2 && caret.bottom <= line.bottom + 2
        && getComputedStyle(view.contentDOM).caretColor === 'rgba(0, 0, 0, 0)';
    }));
    await shot('dialogue-single-line-empty-caret');
    await card(0).locator('.speaker-insert').click();
    check('speaker_click_inserts_a_first_label', await page.inputValue('#dialogue-script') === 'S1:');
    await page.keyboard.insertText('你好。'); await card(1).locator('.speaker-insert').click();
    await page.keyboard.insertText('很高兴见到你。'); await card(0).locator('.speaker-insert').click();
    await page.keyboard.insertText('我们开始吧。');
    check('speaker_click_appends_a_new_turn_without_replacing_text',
      await page.inputValue('#dialogue-script') === 'S1:你好。\nS2:很高兴见到你。\nS1:我们开始吧。');
    await settle();
    check('typed_dialogue_caret_follows_only_the_current_line', await page.evaluate(() => {
      const view = ScriptEditor.get(document.getElementById('dialogue-script')).view;
      const caret = view.scrollDOM.querySelector('.cm-cursor-primary').getBoundingClientRect();
      const line = [...view.contentDOM.querySelectorAll('.cm-line')].at(-1).getBoundingClientRect();
      return caret.height > 0 && caret.height <= parseFloat(getComputedStyle(view.contentDOM).lineHeight)
        && caret.top >= line.top - 1 && caret.bottom <= line.bottom + 1;
    }));
    check('speaker_labels_have_pale_capsule_decorations', await host().locator('.script-speaker').count() === 3
      && await host().locator('.script-speaker').first().evaluate(node => getComputedStyle(node).backgroundColor !== 'rgba(0, 0, 0, 0)'));
    check('click_insert_updates_structured_turn_order', await page.evaluate(() =>
      StudioFlow.flow.scene.lines.map(line => StudioFlow.flow.scene.roles.find(role => role.id === line.characterId).speaker).join() === '0,1,0'));
    await host().locator('[data-label="Source text"]').click();
    check('raw_dialogue_view_can_be_opened', await page.locator('#dialogue-script').isVisible());
    await page.locator('#dialogue-script').fill('S1: 手动填写。\nS2：另一位。\ns1 : 再来一句。');
    await card(1).locator('.speaker-insert').click();
    check('raw_mode_speaker_click_appends_and_keeps_raw_mode', await page.locator('#dialogue-script').isVisible()
      && (await page.inputValue('#dialogue-script')).endsWith('\nS2:'));
    await page.keyboard.insertText('最后一句。');
    await host().locator('[data-label="Source text"]').click();
    check('typed_ascii_fullwidth_and_lowercase_labels_are_recognized', await host().locator('.script-speaker').count() === 4
      && await page.evaluate(() => StudioFlow.flow.scene.lines.length === 4));
    await editor().fill('S1：重庆到了。\nS2：你听见了吗？');
    await page.evaluate(() => {
      const view = ScriptEditor.get(document.getElementById('dialogue-script')).view;
      view.focus(); view.dispatch({ selection: { anchor: 3, head: 4 } });
    });
    await host().locator('[data-label="Tags"]').click();
    await host().locator('[data-action="phoneme"]').click();
    await host().locator('.tag-pronunciation-form input').fill('chong2');
    await host().locator('.tag-pronunciation-form button[type="submit"]').click();
    check('pronunciation_quick_action_keeps_dialogue_speaker_and_other_hanzi',
      await page.inputValue('#dialogue-script') ===
        'S1：<|phoneme_start|>chong2<|phoneme_end|>庆到了。\nS2：你听见了吗？');
    await host().locator('[data-label="Undo"]').click();
    await editor().fill('S1：选中这几个字。\nS2：第二句。');
    await page.evaluate(() => {
      const view = ScriptEditor.get(document.getElementById('dialogue-script')).view;
      view.focus(); view.dispatch({ selection: { anchor: 3, head: 5 } });
    });
    await host().locator('[data-label="Tags"]').click();
    await host().locator('[data-tag="whisper"][data-kind="scope"]').click();
    check('scoped_tags_wrap_only_spoken_words_in_dialogue', await page.inputValue('#dialogue-script') === 'S1：<whisper>选中</whisper>这几个字。\nS2：第二句。');
    check('scoped_tags_do_not_change_speaker_binding', await page.evaluate(() => StudioFlow.flow.scene.lines[0].text.startsWith('<whisper>选中</whisper>')));
    await host().locator('[data-label="Undo"]').click();
    check('dialogue_tags_are_undoable', await page.inputValue('#dialogue-script') === 'S1：选中这几个字。\nS2：第二句。');
    await page.evaluate(() => {
      const view = ScriptEditor.get(document.getElementById('dialogue-script')).view;
      view.focus(); view.dispatch({ selection: { anchor: 0, head: 5 } });
    });
    await host().locator('[data-label="Tags"]').click();
    await host().locator('[data-tag="whisper"][data-kind="scope"]').click();
    check('scope_controls_do_not_wrap_or_destroy_speaker_labels',
      await page.inputValue('#dialogue-script') === 'S1：选中这几个字。\nS2：第二句。'
      && await page.locator('#studio-error').isVisible());
    await host().locator('.tag-search [data-label="Close"]').click();
    await editor().fill('S1：第一位。\nS3：手动添加第三位。\nS16：最后一个编号。');
    check('typing_a_new_valid_label_adds_its_speaker', await card(2).count() === 1 && await card(15).count() === 1);
    const beforeAdd = await page.inputValue('#dialogue-script');
    await page.locator('#add-character').click();
    check('add_speaker_opens_voice_picker_and_uses_a_free_id', await page.locator('#dialogue-voice-dialog').isVisible()
      && await card(3).count() === 1 && await page.inputValue('#dialogue-script') === beforeAdd);
    await page.locator('#dialogue-voice-dialog .studio-dialog-heading .icon-btn').click();
    await editor().fill(`${beforeAdd}\nS4:临时台词。`);
    await card(3).locator('.speaker-remove').click();
    check('speaker_with_dialogue_is_kept_and_explains_reassignment',
      await card(3).count() === 1 && await page.inputValue('#dialogue-script') === `${beforeAdd}\nS4:临时台词。`
      && (await page.locator('#studio-error').textContent()).includes('请先将这些台词改分配给其他说话人'));
    check('speaker_error_notice_matches_the_work_area_and_has_a_dismiss_button',
      await page.locator('#studio-error .icon-btn').count() === 1
      && await page.evaluate(() => Math.abs(
        document.querySelector('#studio-error').getBoundingClientRect().width
        - document.querySelector('#view-scene').getBoundingClientRect().width
      ) < 1));
    await page.locator('#studio-error .icon-btn').click();
    check('speaker_error_notice_can_be_dismissed', !await page.locator('#studio-error').isVisible());
    await editor().fill(`${beforeAdd}\nS4:`);
    await card(3).locator('.speaker-remove').click();
    check('empty_speaker_can_be_removed_without_deleting_dialogue', !await card(3).count()
      && await page.inputValue('#dialogue-script') === beforeAdd
      && await card(2).locator('.speaker-badge').textContent() === 'S3'
      && await card(15).locator('.speaker-badge').textContent() === 'S16');
    const voices = (await api('/api/voices')).voices;
    await pickVoice(0, voices[0].id);
    check('library_selection_updates_only_the_chosen_speaker',
      await card(0).locator('.speaker-audio-name').textContent() === voices[0].name
      && await card(1).locator('.speaker-audio-name').textContent() === '未选择声音');
    check('selected_library_voice_can_be_previewed', await card(0).locator('.speaker-play').isEnabled());
    const initialPlayIcon = await card(0).locator('.speaker-play svg').evaluate(node => node.outerHTML);
    check('enabled_and_disabled_play_buttons_keep_identical_icons_and_size',
      initialPlayIcon === await card(1).locator('.speaker-play svg').evaluate(node => node.outerHTML)
      && JSON.stringify(await playStyle(card(0).locator('.speaker-play'))) === JSON.stringify(standardPlayStyle));
    await card(0).locator('.speaker-play').click();
    await page.waitForFunction(() => document.querySelector('.dialogue-speaker[data-speaker="0"] .speaker-play').getAttribute('aria-pressed') === 'true');
    check('speaker_preview_plays_audio', await card(0).locator('.speaker-play').getAttribute('aria-pressed') === 'true');
    await card(0).locator('.speaker-play').click();
    check('speaker_preview_can_be_paused', await card(0).locator('.speaker-play').getAttribute('aria-pressed') === 'false');
    check('play_icon_and_dimensions_do_not_change_after_playback',
      initialPlayIcon === await card(0).locator('.speaker-play svg').evaluate(node => node.outerHTML)
      && JSON.stringify(await playStyle(card(0).locator('.speaker-play'))) === JSON.stringify(standardPlayStyle));
    await openVoice(0);
    check('voice_picker_preview_matches_the_speaker_play_button',
      JSON.stringify(await playStyle(page.locator('.dialogue-voice-preview').first())) === JSON.stringify(standardPlayStyle));
    check('voice_dialog_keeps_speaker_removal_on_speaker_cards',
      await page.locator('#dialogue-voice-apply').isVisible()
      && !await page.locator('#dialogue-voice-dialog .studio-dialog-footer .icon-btn').count());
    await page.locator('#dialogue-voice-search').fill('参考音频的台词');
    check('voice_picker_search_also_matches_saved_transcripts', await page.locator('.dialogue-voice-option').count() >= 2);
    await page.locator('#dialogue-voice-search').fill(voices[1].name);
    check('voice_picker_search_filters_names', await page.locator('.dialogue-voice-option').count() === 1);
    await page.locator('.dialogue-voice-pick').click(); await closeVoice();
    check('cancelled_voice_change_does_not_change_binding', await card(0).locator('.speaker-audio-name').textContent() === voices[0].name);
    await openVoice(1); await page.locator('#dialogue-voice-tab-upload').click();
    const roleB = await card(1).getAttribute('data-role-id'), prefix = `dialogue-voice-${roleB}`;
    await page.locator(`#${prefix}-upload`).setInputFiles(process.env.DRAMA_STUDIO_REFERENCE);
    await page.locator(`#${prefix}-transcript`).waitFor({ state: 'visible' });
    await page.locator('#dialogue-voice-upload .reference-status').waitFor({ state: 'hidden' });
    check('speaker_picker_has_no_additional_reference_foldout',
      !await page.locator('#dialogue-voice-upload .additional-references').count()
      && !await page.locator('#dialogue-voice-upload').getByRole('button', { name: '添加参考音频', exact: true }).count());
    check('waveform_play_button_matches_the_speaker_play_button',
      JSON.stringify(await playStyle(page.locator('#dialogue-voice-upload .player-play').first())) === JSON.stringify(standardPlayStyle));
    await page.locator('#dialogue-voice-apply').click();
    check('manual_upload_requires_a_transcript_before_commit', await page.locator('#dialogue-voice-dialog').isVisible()
      && (await page.locator('#dialogue-voice-error').textContent()).trim().length > 0);
    await page.locator(`#${prefix}-transcript`).fill('手动填写的参考台词。'); await applyVoice();
    check('manual_upload_is_committed_with_its_filename',
      await card(1).locator('.speaker-audio-name').textContent() === 'reference.wav' && asrCalls === 1);
    asrAvailable = true;
    await openVoice(2); await page.locator('#dialogue-voice-tab-upload').click();
    const roleC = await card(2).getAttribute('data-role-id'), autoPrefix = `dialogue-voice-${roleC}`;
    await page.locator(`#${autoPrefix}-upload`).setInputFiles(process.env.DRAMA_STUDIO_REFERENCE);
    const asrStatus = page.locator('#dialogue-voice-upload .reference-status');
    await page.locator('#dialogue-voice-upload .reference-status[data-state="busy"]').waitFor({ state: 'visible' });
    check('transcription_busy_state_is_prominent_above_the_transcript',
      await asrStatus.evaluate(node => {
        const style = getComputedStyle(node), transcript = node.nextElementSibling;
        return style.fontSize === '15px' && Number(style.fontWeight) >= 600
          && style.borderTopWidth === '1px' && style.backgroundColor === 'rgb(255, 243, 232)'
          && transcript.tagName === 'TEXTAREA' && transcript.getAttribute('aria-busy') === 'true'
          && node.getAttribute('aria-live') === 'polite';
      }));
    check('transcription_busy_state_has_an_animated_indicator',
      await asrStatus.evaluate(node => getComputedStyle(node, '::before').animationName === 'reference-pending'));
    await page.waitForFunction(() => document.querySelector('#dialogue-voice-upload .player-wave')?.childElementCount > 0);
    await asrStatus.scrollIntoViewIfNeeded(); await shot('dialogue-transcribing-desktop');
    await page.emulateMedia({ reducedMotion: 'reduce' });
    check('transcription_indicator_respects_reduced_motion',
      await asrStatus.evaluate(node => getComputedStyle(node, '::before').animationName === 'none'));
    await page.emulateMedia({ reducedMotion: 'no-preference' });
    await page.setViewportSize({ width: 390, height: 844 }); await settle();
    check('transcription_status_fits_the_mobile_voice_dialog', await asrStatus.evaluate(node => {
      const box = node.getBoundingClientRect(); return box.left >= 0 && box.right <= innerWidth
        && node.scrollWidth <= node.clientWidth + 1;
    }));
    await asrStatus.scrollIntoViewIfNeeded(); await shot('dialogue-transcribing-mobile');
    await page.setViewportSize({ width: 1440, height: 960 }); await settle();
    asrReady = true;
    await page.waitForFunction(id => document.getElementById(id)?.value === '自动转写的参考台词。', autoPrefix + '-transcript');
    check('completed_transcription_stops_the_busy_indicator', !await asrStatus.isVisible()
      && await page.locator(`#${autoPrefix}-transcript`).getAttribute('aria-busy') === 'false');
    asrFailure = true;
    await page.locator('#dialogue-voice-upload').getByRole('button', { name: '重新识别', exact: true }).click();
    await page.locator('#dialogue-voice-upload .reference-status[data-state="error"]').waitFor({ state: 'visible' });
    check('failed_transcription_has_a_visible_error_without_erasing_text',
      (await asrStatus.textContent()).includes('识别失败')
      && await page.inputValue(`#${autoPrefix}-transcript`) === '自动转写的参考台词。');
    asrFailure = false; asrReady = false;
    await page.locator('#dialogue-voice-upload').getByRole('button', { name: '重新识别', exact: true }).click();
    await page.locator('#dialogue-voice-upload .reference-status[data-state="busy"]').waitFor({ state: 'visible' });
    check('retry_restores_the_busy_style_and_clears_the_old_error', !await asrStatus.getAttribute('title'));
    asrReady = true;
    await page.locator('#dialogue-voice-upload .reference-status').waitFor({ state: 'hidden' });
    await applyVoice();
    check('automatic_reference_transcription_is_available_in_picker', asrCalls === 4
      && await card(2).locator('.speaker-audio-name').textContent() === 'reference.wav');
    await openVoice(2); await page.locator('#dialogue-voice-tab-library').click();
    await page.locator('.dialogue-no-reference').click(); await applyVoice();
    check('clearing_a_voice_removes_all_old_reference_state', await card(2).locator('.speaker-audio-name').textContent() === '未选择声音'
      && await page.evaluate(async () => {
        const role = StudioFlow.flow.scene.roles.find(role => role.speaker === 2);
        return !role.upload && !role.voiceId && !(await ReferencePrep.readAll(role, 2)).length;
      }));
    await page.evaluate(() => {
      const scene = StudioFlow.flow.scene, role = scene.roles.find(item => item.speaker === 2);
      role.extraReferences = [{ ...structuredClone(scene.roles.find(item => item.speaker === 1)),
        key: crypto.randomUUID(), transcript: '原有附加参考的台词。', extraReferences: [] }];
      return StudioFlow.save();
    });
    await page.reload({ waitUntil: 'networkidle' }); await ready(); await page.locator('.nav[data-view="scene"]').click();
    check('legacy_extra_only_reference_still_has_a_preview',
      await card(2).locator('.speaker-audio-name').textContent() === 'reference.wav'
      && await card(2).locator('.speaker-play').isEnabled());
    await openVoice(2); await page.locator('#dialogue-voice-tab-upload').click();
    check('legacy_extra_only_reference_becomes_an_editable_primary_clip',
      await page.locator(`#${autoPrefix}-transcript`).inputValue() === '原有附加参考的台词。'
      && !await page.locator('#dialogue-voice-upload .additional-references').count());
    await applyVoice();
    check('legacy_reference_migration_preserves_the_complete_audio_binding', await page.evaluate(async () => {
      const role = StudioFlow.flow.scene.roles.find(item => item.speaker === 2);
      const clips = await ReferencePrep.readAll(role, 2);
      return clips.length === 1 && clips[0].text === '原有附加参考的台词。' && !role.extraReferences.length;
    }));
    check('committed_speakers_still_do_not_show_waveforms', !await page.locator('.dialogue-speaker .studio-player,.dialogue-speaker audio').count());
    await page.setViewportSize({ width: 390, height: 844 }); await settle();
    check('four_speakers_wrap_into_two_mobile_columns', await page.locator('#characters').evaluate(node => {
      const rows = [...node.children].map(child => child.getBoundingClientRect()), frame = node.getBoundingClientRect();
      return getComputedStyle(node).display === 'grid' && rows.length === 4
        && Math.abs(rows[0].top - rows[1].top) < 1 && rows[2].top > rows[0].bottom
        && rows.every(row => row.left >= frame.left && row.right <= frame.right + 1);
    }));
    await page.setViewportSize({ width: 1440, height: 960 }); await settle();
    const script = 'S1：你准备好了吗？\nS2：准备好了，我们开始吧。\nS1：好的。';
    await editor().fill(script); await page.locator('[data-strategy="joint"]').click();
    await page.locator('#submit-scene').click(); const joint = await finish();
    check('joint_generation_uses_all_turns_in_one_request', joint.recipe.scene_strategy === 'joint'
      && joint.recipe.lines.length === 3 && joint.recipe.characters.length === 2);
    check('generation_converts_ui_labels_to_native_speaker_tokens',
      joint.recipe.text.includes('<|speaker:0|> 你准备好了吗？')
      && joint.recipe.text.includes('<|speaker:1|> 准备好了，我们开始吧。')
      && !joint.recipe.text.includes('S1：') && !joint.recipe.text.includes('S2：'));
    check('only_used_speaker_references_are_submitted', joint.recipe.reference_snapshot.length === 2
      && joint.recipe.reference_snapshot.map(clip => clip.speaker).sort().join() === '0,1');
    check('fresh_dialogue_does_not_send_a_user_delivery_direction', !joint.recipe.direction);
    check('joint_output_is_downloadable', await page.locator('.main-result .download-link').isVisible());
    const ids = joint.recipe.line_ids;
    await page.locator('[data-strategy="per_line"]').click();
    check('strategy_switch_keeps_script_and_bindings', await page.inputValue('#dialogue-script') === script
      && await card(0).locator('.speaker-audio-name').textContent() === voices[0].name
      && await card(1).locator('.speaker-audio-name').textContent() === 'reference.wav');
    await page.locator('#submit-scene').click(); const assembled = await finish(joint.id);
    check('per_turn_generation_still_assembles_the_complete_output', assembled.recipe.scene_strategy === 'per_line'
      && assembled.result.segments.length === 3 && assembled.output && await page.locator('.main-result .download-link').isVisible());
    check('strategy_switch_preserves_turn_ids', JSON.stringify(assembled.recipe.line_ids) === JSON.stringify(ids));
    check('assembled_audio_preserves_turn_speaker_order', assembled.result.segments.map(segment => segment.speaker).join() === '0,1,0');
    await page.evaluate(() => StudioFlow.save());
    const saved = (await api('/api/studio')).drafts.scene.value;
    check('draft_stores_unified_source_and_structured_turns', saved.script === script && saved.scriptVersion === 1
      && saved.lines.length === 3 && saved.strategy === 'per_line');
    await page.reload({ waitUntil: 'networkidle' }); await ready(); await page.locator('.nav[data-view="scene"]').click();
    check('reload_restores_unified_script_and_strategy', await page.inputValue('#dialogue-script') === script
      && await page.locator('[data-strategy="per_line"]').getAttribute('aria-pressed') === 'true');
    check('reload_restores_uploaded_and_library_voices', await card(0).locator('.speaker-audio-name').textContent() === voices[0].name
      && await card(1).locator('.speaker-audio-name').textContent() === 'reference.wav');
    const legacy = structuredClone(saved); delete legacy.script; delete legacy.scriptVersion;
    legacy.roles = legacy.roles.slice(0, 2);
    legacy.roles[0].name = '甲'; legacy.roles[1].name = '乙';
    legacy.lines = [{ id: 'legacy-one', characterId: legacy.roles[0].id, text: '原来的第一句。', direction: 'calm' },
      { id: 'legacy-two', characterId: legacy.roles[1].id, text: '原来的第二句。', direction: '' }];
    legacy.scene = 'Legacy scene.'; legacy.direction = 'Legacy direction.';
    await api('/api/studio/draft', { key: 'scene', value: legacy });
    await page.reload({ waitUntil: 'networkidle' }); await ready(); await page.locator('.nav[data-view="scene"]').click();
    check('old_row_drafts_migrate_without_losing_text_or_delivery_tags',
      await page.inputValue('#dialogue-script') === 'S1：[calm] 原来的第一句。\nS2：原来的第二句。');
    check('old_drafts_retain_stable_role_and_turn_ids', await page.evaluate(() =>
      StudioFlow.flow.scene.lines.map(line => line.id).join() === 'legacy-one,legacy-two'
      && StudioFlow.flow.scene.roles.map(role => role.name).join() === '甲,乙'));
    await page.evaluate(() => StudioFlow.save());
    check('migrated_draft_saves_the_new_editor_source', (await api('/api/studio')).drafts.scene.value.scriptVersion === 1);
    for (const [name, invalid] of [['unassigned', '没有说话人标签。'], ['zero', 'S0：错误编号。'], ['too_many', 'S17：错误编号。']]) {
      await editor().fill(invalid); const count = jobCalls;
      await page.locator('#submit-scene').click(); await settle();
      check(`invalid_dialogue_${name}_does_not_submit`, jobCalls === count && await page.locator('#studio-error').isVisible());
    }
    check('invalid_labels_do_not_create_invalid_speaker_cards', !await card(-1).count() && !await card(16).count());
    await editor().fill('S1：第一句话。\n这是同一个人的下一句。\nS2：换另一个人。');
    check('unlabelled_continuation_lines_remain_in_the_same_turn', await page.evaluate(() =>
      StudioFlow.flow.scene.lines.length === 2 && StudioFlow.flow.scene.lines[0].text === '第一句话。\n这是同一个人的下一句。'));
    await editor().fill('S1：你准备好了吗？\nS2：准备好了，我们开始吧。\nS1：好的。');
    for (const locale of ['zh-CN', 'en', 'ja', 'ko']) {
      await choose('ui-language', locale);
      for (const size of [{ width: 1440, height: 900 }, { width: 1271, height: 712 }, { width: 390, height: 844 }]) {
        await page.setViewportSize(size); await settle();
        check(`dialogue_title_is_localized_${locale}_${size.width}`, await page.locator('#view-title').textContent()
          === await page.evaluate(() => t('Dialogue')));
        check(`dialogue_fits_without_horizontal_overflow_${locale}_${size.width}`, await page.evaluate(() =>
          document.documentElement.scrollWidth <= innerWidth + 1));
        check(`generation_action_stays_in_view_${locale}_${size.width}`, await page.locator('#submit-scene').evaluate(node => {
          const r = node.getBoundingClientRect(); return r.left >= 0 && r.right <= innerWidth + 1 && r.top >= 0 && r.bottom <= innerHeight + 1;
        }));
        check(`strategy_labels_stay_on_one_line_${locale}_${size.width}`, await page.locator('.scene-strategies button').evaluateAll(nodes =>
          nodes.every(node => getComputedStyle(node).whiteSpace === 'nowrap' && node.scrollWidth <= node.clientWidth + 1)));
        check(`editor_and_speakers_do_not_overlap_${locale}_${size.width}`, await page.evaluate(() => {
          const e = document.querySelector('.dialogue-editor-area').getBoundingClientRect();
          const s = document.querySelector('.dialogue-speakers').getBoundingClientRect();
          return e.width > 0 && e.height >= 239 && (innerWidth > 900 ? e.right < s.left : s.bottom < e.top);
        }));
        check(`speaker_capsules_do_not_overlap_adjacent_turns_${locale}_${size.width}`,
          await host().locator('.script-speaker').evaluateAll(nodes => nodes.every((node, index) =>
            !index || node.getBoundingClientRect().top >= nodes[index - 1].getBoundingClientRect().bottom + 1)));
        await openVoice(0);
        check(`voice_dialog_and_apply_fit_${locale}_${size.width}`, await page.locator('#dialogue-voice-dialog').evaluate(node => {
          const r = node.getBoundingClientRect(), apply = node.querySelector('#dialogue-voice-apply').getBoundingClientRect();
          return r.left >= 0 && r.top >= 0 && r.right <= innerWidth + 1 && r.bottom <= innerHeight + 1
            && apply.top >= r.top && apply.bottom <= r.bottom;
        }));
        await closeVoice();
        if (locale === 'zh-CN') await shot('dialogue-workspace-' + size.width);
      }
    }
    check('no_browser_exceptions', !report.errors.length);
  } catch (error) {
    report.errors.push(error.stack || String(error));
    if (page) {
      const file = path.join(folder, 'dialogue-ui-failure.png');
      await page.screenshot({ path: file, fullPage: true }).catch(() => {}); report.screenshots.push(file);
    }
  } finally {
    if (browser) await browser.close();
    report.passed = !report.errors.length && Object.values(report.checks).every(Boolean);
    fs.writeFileSync(path.join(root, 'research/2026-10-20-dialogue-ui-acceptance.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report, null, 2)); if (!report.passed) process.exitCode = 1;
  }
})();
