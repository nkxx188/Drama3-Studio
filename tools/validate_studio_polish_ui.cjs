/* Visual and interaction contracts across the actual workbench; isolated synthetic inference. */
const fs = require('node:fs'), path = require('node:path'), { chromium } = require('playwright');
const root = path.resolve(__dirname, '..'), base = process.env.DRAMA_UI_URL, folder = process.env.DRAMA_STUDIO_FOLDER;
const report = { scope: 'Shared visual tokens, all workspaces, four locales, narrow/scaled viewports, keyboard focus and real waveform rendering. Synthetic generation only.',
  checks: {}, errors: [], screenshots: [] };
(async () => {
  let browser, page;
  try {
    browser = await chromium.launch({ channel: 'chrome', headless: true });
    const context = await browser.newContext({ viewport: { width: 1440, height: 960 }, locale: 'zh-CN' });
    await context.addInitScript(() => localStorage.setItem('drama3.ui.language', 'zh-CN'));
    page = await context.newPage();
    page.on('pageerror', error => report.errors.push(error.stack || error.message));
    const check = (name, value) => { report.checks[name] = Boolean(value); if (!value) throw new Error(name); };
    const api = async (url, data) => {
      const response = data === undefined ? await context.request.get(base + url)
        : await context.request.post(base + url, { data });
      if (!response.ok()) throw new Error(url + ': ' + await response.text());
      return response.json();
    };
    const ready = () => page.waitForFunction(() => document.documentElement.dataset.studioReady === 'true');
    const settle = () => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    const nav = async name => { await page.locator(`.nav[data-view="${name}"]`).click(); await settle(); };
    const shot = async name => {
      await page.mouse.move(5, 5); await settle();
      const file = path.join(folder, name + '.png');
      await page.screenshot({ path: file, animations: 'disabled' }); report.screenshots.push(file);
    };
    const inView = locator => locator.evaluate(node => {
      const r = node.getBoundingClientRect();
      return r.width > 0 && r.left >= -1 && r.right <= innerWidth + 1 && r.top >= 0 && r.bottom <= innerHeight + 1;
    });
    await page.goto(base, { waitUntil: 'networkidle' }); await ready();
    check('neutral_text_and_shared_type_scale', await page.evaluate(() => {
      const root = getComputedStyle(document.documentElement);
      return root.getPropertyValue('--text').trim() === '#222831'
        && getComputedStyle(document.querySelector('.cm-editor')).fontSize === '18px'
        && getComputedStyle(document.querySelector('#view-title')).fontSize === '24px';
    }));
    check('input_boundaries_are_stronger_than_panel_boundaries', await page.locator('#direction').evaluate(node =>
      getComputedStyle(node).borderTopColor !== getComputedStyle(document.querySelector('.compose-panel')).borderTopColor));
    check('model_picker_has_a_visible_input_boundary', await page.locator('#active-model-trigger').evaluate(node =>
      getComputedStyle(node).borderTopColor === 'rgb(185, 194, 206)'));
    check('framed_tools_use_the_shared_eight_pixel_radius', await page.locator('.compose-panel').evaluate(node =>
      getComputedStyle(node).borderRadius === '8px'));
    await page.locator('#direction').focus();
    await page.waitForFunction(() => getComputedStyle(document.querySelector('#direction')).borderTopColor === 'rgb(198, 74, 46)');
    check('focused_input_has_a_distinct_nonempty_focus_ring', await page.locator('#direction').evaluate(node =>
      getComputedStyle(node).boxShadow !== 'none' && getComputedStyle(node).borderTopColor === 'rgb(198, 74, 46)'));
    await page.locator('#direction').fill('自然、清晰，语速适中。');
    const editor = page.locator('.compose-panel > .editor-area .cm-content');
    await editor.fill('把今天的想法，变成听得见的声音。\n[轻声说]先从这一句开始。');
    await page.locator('#saved-voice-options > summary').click();
    await page.locator('#voice-picks .voice-pick').first().click();
    await page.locator('#submit-job').click();
    await page.waitForFunction(() => document.querySelector('#job-badge').dataset.status === 'complete');
    await page.waitForFunction(() => document.querySelector('.main-result .player-wave rect'));
    const job = await api('/api/jobs/' + await page.locator('#result-box').getAttribute('data-job-id'));
    check('complete_result_hides_task_progress_but_keeps_generation_time',
      !await page.locator('.progress-track').isVisible() && (await page.locator('#progress-count').textContent()).includes('生成耗时'));
    check('download_stays_direct_and_uses_a_secondary_button', await page.locator('.main-result .download-link').isVisible()
      && !await page.locator('.main-result .download-link').evaluate(node => node.classList.contains('primary-btn')));
    check('logo_and_real_waveform_assets_render', await page.locator('.brand-mark img').evaluate(node => node.naturalWidth > 0)
      && await page.locator('.main-result .player-wave rect').count() === 80);
    const buttonStyle = locator => locator.evaluate(node => {
      const r = node.getBoundingClientRect(), s = getComputedStyle(node);
      return [r.width, r.height, s.borderRadius, s.fontSize];
    });
    check('saved_voice_and_result_use_identical_play_control_geometry',
      JSON.stringify(await buttonStyle(page.locator('#selected-reference-details .reference-compact-play')))
        === JSON.stringify(await buttonStyle(page.locator('.main-result .player-play'))));
    await page.locator('#selected-reference-details .reference-compact-play').focus();
    await page.keyboard.press('Tab');
    check('keyboard_focus_is_visible_on_icon_controls', await page.locator('#selected-reference-details .selected-voice-clear').evaluate(node =>
      document.activeElement === node && getComputedStyle(node).outlineStyle !== 'none'));
    await page.locator('#direction').focus(); await page.locator('#active-model-trigger').focus();
    await page.keyboard.press('Enter'); await page.locator('#active-model-listbox').waitFor();
    check('model_menu_keeps_selection_and_keyboard_focus', await page.locator('#active-model-listbox [aria-selected="true"]').count() === 1
      && await page.locator('#active-model-trigger').getAttribute('aria-expanded') === 'true');
    await page.keyboard.press('Escape');
    check('escape_restores_focus_to_model_control', await page.locator('#active-model-trigger').evaluate(node => document.activeElement === node));
    await page.locator('#direction').focus(); await page.keyboard.press('Tab');
    await shot('polished-generation-desktop');
    await page.evaluate(job => renderJob({ ...job, status: 'running', timing: { schema: 1, phase: 'model_load', generation_seconds: 0 } }), job);
    check('loading_still_has_a_visible_progress_indicator', await page.locator('.progress-track').isVisible()
      && (await page.locator('#progress-stage').textContent()).includes('加载'));
    await page.evaluate(job => renderJob({ ...job, status: 'error', error: '用于验收的错误提示。' }), job);
    check('error_feedback_remains_visible_without_retired_details', await page.locator('#result-error').isVisible());
    await page.evaluate(job => renderJob(job), job);
    await page.locator('#advanced-settings > summary').click();
    await shot('polished-generation-settings');
    await page.keyboard.press('Escape');
    await editor.click(); await editor.press('End'); await editor.press('Shift+ArrowLeft');
    await page.locator('.compose-panel > .editor-area [data-label="Tags"]').click();
    check('tag_panel_retains_categories_and_custom_input', await page.locator('.tag-panel:visible .tag-category').count() >= 5
      && await page.locator('.tag-panel:visible .tag-custom input').isVisible());
    await shot('polished-tag-panel'); await page.keyboard.press('Escape');
    await page.locator('#describe-voice').click();
    await page.locator('#voice-description').fill('温和沉稳的成年女声，发音清晰，带一点沙哑。');
    await page.locator('.compose-panel > .editor-area .cm-content').fill('你好，很高兴在这里与你相遇。');
    await shot('polished-voice-design');
    await page.locator('#back-to-generation').click();
    await page.locator('.main-result .creative-edit').click();
    await page.locator('#audio-edit-workspace').waitFor();
    await page.locator('[data-edit-mode="local"]').click();
    await page.waitForFunction(() => document.querySelector('#audio-edit-source').duration > 0);
    await page.evaluate(() => StudioAudio.setSelection(document.querySelector('#audio-edit-source'), .5, 2, { emit: true }));
    await page.locator('#audio-edit-quote').fill('今天的想法');
    await page.locator('#audio-edit-workspace').scrollIntoViewIfNeeded(); await shot('polished-inline-editor');
    await page.locator('#audio-edit-close').click();
    await nav('scene');
    await page.locator('.dialogue-editor-area .cm-content').fill('S1: 我们准备好了吗？\nS2: 准备好了，开始吧。\nS1: 好的。');
    const voices = (await api('/api/voices')).voices;
    for (let index = 0; index < 2; index++) {
      await page.locator('.dialogue-speaker').nth(index).locator('.speaker-change').click();
      await page.locator(`.dialogue-voice-option[data-voice-id="${voices[index].id}"] .dialogue-voice-pick`).click();
      await page.locator('#dialogue-voice-apply').click();
    }
    await shot('polished-dialogue-desktop');
    for (const name of ['清晰旁白', '落ち着いたナレーション', '차분한 설명', 'A voice with a long name for layout validation']) {
      await api('/api/voices/from-references', { name, references: [{ voice_id: voices[0].id, reference_index: 0, speaker: 0, text: '参考音频的台词。' }] });
    }
    await page.evaluate(() => loadVoices()); await nav('voices');
    check('voice_tiles_align_without_nested_players', await page.locator('.voice-card').evaluateAll(nodes => {
      const heights = nodes.map(node => node.getBoundingClientRect().height);
      return Math.max(...heights) - Math.min(...heights) < 1 && !document.querySelector('#voice-list .studio-player');
    }));
    check('voice_titles_metadata_and_transcripts_do_not_overlap', await page.locator('.voice-card').evaluateAll(nodes =>
      nodes.every(node => node.querySelector('.voice-card-copy').getBoundingClientRect().bottom
        <= node.querySelector(':scope > .voice-transcript').getBoundingClientRect().top + 1)));
    await shot('polished-voice-library-desktop');
    await page.locator('#add-voice').click();
    check('add_voice_dialog_uses_the_shared_control_radius', await page.locator('#voice-name').evaluate(node => getComputedStyle(node).borderRadius === '6px'));
    await shot('polished-add-voice-dialog'); await page.keyboard.press('Escape');
    await nav('takes');
    check('history_has_no_duplicate_visible_page_heading', !await page.locator('.history-panel .panel-heading').evaluate(node => {
      const r = node.getBoundingClientRect(); return r.width > 2 && r.height > 2;
    }));
    check('history_keeps_direct_play_download_and_more', await page.locator('.job-card').first().locator('.history-play').isVisible()
      && await page.locator('.job-card').first().locator('.history-download').isVisible()
      && await page.locator('.job-card').first().locator('.history-actions').isVisible());
    await shot('polished-history-desktop');
    await nav('settings'); await shot('polished-settings-desktop');
    check('machine_details_are_readable_unboxed_rows', await page.locator('#hardware-card .metric').first().evaluate(node =>
      getComputedStyle(node).borderRadius === '0px' && getComputedStyle(node).backgroundColor === 'rgba(0, 0, 0, 0)'));
    for (const locale of ['zh-CN', 'en', 'ja', 'ko']) {
      await page.evaluate(locale => I18n.setLocale(locale), locale);
      for (const size of [{ width: 1440, height: 960 }, { width: 1024, height: 640 }, { width: 853, height: 480 }, { width: 390, height: 844 }]) {
        await page.setViewportSize(size);
        for (const name of ['generate', 'scene', 'voices', 'takes', 'settings']) {
          await nav(name);
          check(`no_horizontal_overflow_${locale}_${size.width}_${name}`,
            await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
          if (name === 'generate' || name === 'scene') {
            check(`primary_action_stays_visible_${locale}_${size.width}_${name}`,
              await inView(page.locator(name === 'generate' ? '#submit-job' : '#submit-scene')));
          }
          if (name === 'generate') {
            check(`voice_actions_fit_${locale}_${size.width}`, await page.locator('.voice-source-bar').evaluate(node =>
              [...node.children].filter(item => getComputedStyle(item).display !== 'none')
                .every(item => item.getBoundingClientRect().right <= node.getBoundingClientRect().right + 1)));
          }
          if (name === 'voices') {
            check(`long_voice_names_keep_play_controls_inside_tiles_${locale}_${size.width}`,
              await page.locator('.voice-card').evaluateAll(nodes => nodes.every(node => {
                const card = node.getBoundingClientRect(), play = node.querySelector('.voice-tile-preview').getBoundingClientRect();
                return play.right <= card.right && play.left >= card.left;
              })));
          }
        }
      }
    }
    await page.evaluate(() => I18n.setLocale('zh-CN'));
    await page.setViewportSize({ width: 390, height: 844 });
    await nav('generate'); await shot('polished-generation-mobile');
    await nav('scene'); await shot('polished-dialogue-mobile');
    await nav('voices'); await shot('polished-library-mobile');
    await page.emulateMedia({ reducedMotion: 'reduce' });
    check('reduced_motion_removes_control_transitions', await page.locator('#add-voice').evaluate(node =>
      getComputedStyle(node).transitionDuration.split(',').every(value => parseFloat(value) === 0)));
    check('no_script_errors', report.errors.length === 0);
  } catch (error) {
    report.errors.push(error.stack || String(error));
    if (page) {
      const file = path.join(folder, 'polish-failure.png');
      await page.screenshot({ path: file }).catch(() => {}); report.screenshots.push(file);
    }
  } finally {
    if (browser) await browser.close();
    report.passed = !report.errors.length && Object.values(report.checks).every(Boolean);
    fs.writeFileSync(path.join(root, 'research/2026-10-21-webui-polish-acceptance.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report, null, 2));
    if (!report.passed) process.exitCode = 1;
  }
})();
