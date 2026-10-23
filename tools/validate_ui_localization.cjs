/* Real Chrome checks: locale persistence, user-input retention and model selection. */
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');
const root = path.resolve(__dirname, '..');
const base = process.env.DRAMA_UI_URL;
if (!base) throw new Error('Run through validate_ui_localization.py to use isolated settings.');
const report = { scope: 'Real local model APIs and browser UI; no inference. Busy record is synthetic.',
  checks: {}, errors: [], screenshots: [] };
const out = path.join(root, process.env.DRAMA_UI_REPORT || 'research/phase38-ui-localization.json');
const screenshotFolder = path.join(root, process.env.DRAMA_UI_SCREENSHOTS || 'research/phase38-ui-screenshots');
fs.mkdirSync(screenshotFolder, { recursive: true });

(async () => {
  let browser;
  try {
    browser = await chromium.launch({ channel: 'chrome', headless: true });
    const context = await browser.newContext({ locale: 'zh-CN', viewport: { width: 1440, height: 1000 } });
    const page = await context.newPage();
    page.on('pageerror', error => report.errors.push(error.message));
    const choose = async (selector, value) => {
      const id = selector.replace(/^#/, '');
      const trigger = page.locator(`#${id}-trigger`);
      if (await trigger.count()) {
        await trigger.click();
        const options = page.locator(`#${id}-listbox .custom-option`);
        const at = await options.evaluateAll((items, target) => items.findIndex(item => item.dataset.value === target), value);
        if (at < 0) throw new Error(`No menu option for ${id}: ${value}`);
        await options.nth(at).click();
      } else {
        await page.selectOption(selector, value);
      }
    };

    const api = async (pathname, value) => {
      const response = await context.request[value == null ? 'get' : 'post'](base + pathname,
        value == null ? {} : { data: value });
      return { status: response.status(), value: await response.json() };
    };
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.locator('#active-model:not([disabled])').waitFor();
    report.checks.browser_language_detected = await page.locator('html').getAttribute('lang') === 'zh-CN';
    const catalog = (await api('/api/model')).value;
    report.checks.catalog_covers_both_backends = catalog.available_models.some(row => row.runtime === 'python')
      && catalog.available_models.some(row => row.runtime === 'gguf');
    const requestText = '保留这段文本 Keep this text この文を保持 이 문장 유지';
    await page.fill('#text', requestText);
    await page.locator('#advanced-settings').evaluate(node => { node.open = true; });
    await page.fill('#language', 'ja');
    await page.fill('#seed', '456');
    await page.click('.nav[data-view="scene"]');
    const sceneInput = page.locator('#characters input').first();
    await sceneInput.fill('Keep this character');
    await page.click('.nav[data-view="generate"]');
    const expected = {
      'zh-CN': ['生成语音', '切换模型', '管理模型'],
      en: ['Generate speech', 'Switch model', 'Manage models'],
      ja: ['音声を生成', 'モデルを切り替え', 'モデル管理'],
      ko: ['음성 생성', '모델 전환', '모델 관리'],
    };
    for (const [locale, texts] of Object.entries(expected)) {
      await choose('#ui-language', locale);
      report.checks[`locale_${locale}`] = await page.locator('html').getAttribute('lang') === locale
        && await page.locator('#view-title').textContent() === texts[0]
        && await page.locator('#apply-active-model').textContent() === texts[1]
        && await page.locator('#manage-models').textContent() === texts[2];
      report.checks[`inputs_preserved_${locale}`] = await page.inputValue('#text') === requestText
        && await page.inputValue('#language') === 'ja' && await page.inputValue('#seed') === '456'
        && await sceneInput.inputValue() === 'Keep this character';
      report.checks[`catalog_translations_complete_${locale}`] = await page.evaluate(() => {
        const entries = [...document.querySelectorAll('[data-i18n]')];
        const missing = entries.filter(node => !I18n.has(node.dataset.i18n));
        return !missing.length && entries.every(node => node.textContent === I18n.t(node.dataset.i18n,
          JSON.parse(node.dataset.i18nArgs || '{}')));
      });
      report.checks[`navigation_labels_fit_${locale}`] = await page.evaluate(() =>
        [...document.querySelectorAll('.nav > [data-i18n]')].every(node => node.getBoundingClientRect().height < 30));
      report.checks[`dynamic_status_not_overwritten_${locale}`] = await page.evaluate(() =>
        (document.querySelector('#engine-status').textContent.includes(state.hardware.gpu)
          || document.querySelector('#engine-status').title.includes(state.hardware.gpu))
        && document.querySelector('#model-badge').textContent === I18n.t('READY'));
      const image = path.join(screenshotFolder, `desktop-${locale}.png`);
      await page.screenshot({ path: image, fullPage: true, animations: 'disabled' });
      report.screenshots.push(image);
    }
    await page.reload({ waitUntil: 'networkidle' });
    await page.locator('#active-model:not([disabled])').waitFor();
    report.checks.locale_survives_reload = await page.inputValue('#ui-language') === 'ko'
      && await page.locator('html').getAttribute('lang') === 'ko';
    if (process.env.DRAMA_UI_DESIGN_AUDIT === '1') {
      await page.click('#active-model-trigger');
      await page.locator('.select-popover-active-model input').fill('__no_model_matches__');
      report.checks.model_search_no_results = await page.locator('#active-model-listbox .custom-option:visible').count() === 0
        && await page.locator('.select-popover-active-model .select-no-results').isVisible();
      await page.keyboard.press('Escape');
      report.checks.menu_escape_restores_focus = await page.locator('#active-model-trigger').getAttribute('aria-expanded') === 'false'
        && await page.locator('#active-model-trigger').evaluate(node => node === document.activeElement);
      const ggufOption = catalog.available_models.find(row => row.runtime === 'gguf');
      await page.click('#active-model-trigger');
      await page.locator('.select-popover-active-model input').fill(ggufOption.directory.replaceAll('\\', '/').split('/').at(-1));
      await page.keyboard.press('Enter');
      report.checks.model_search_enter_selects = await page.inputValue('#active-model') === ggufOption.directory
        && await page.locator('#active-model-trigger').getAttribute('aria-expanded') === 'false';
      await page.click('#ui-language-trigger');
      await page.click('#view-title');
      report.checks.menu_outside_click_closes = await page.locator('#ui-language-trigger').getAttribute('aria-expanded') === 'false';
      report.checks.native_selects_removed_from_tab_order = await page.evaluate(() =>
        [...document.querySelectorAll('select')].every(node => node.tabIndex === -1 && node.getAttribute('aria-hidden') === 'true'));
    }
    const ready = [];
    for (const row of catalog.available_models) {
      const result = await api('/api/model/inspect', { directory: row.directory });
      if (result.status === 200 && result.value.inspection.ready) ready.push(row);
    }
    const python = ready.find(row => row.runtime === 'python');
    const native = ready.find(row => row.runtime === 'gguf');
    if (!python || !native) throw new Error('This local acceptance requires one ready Python and GGUF model.');
    for (const row of [native, python]) {
      await choose('#active-model', row.directory);
      const before = (await api('/api/model')).value.selected_directory;
      if (before !== row.directory) {
        await page.click('#apply-active-model');
        await page.waitForFunction(directory => state.model?.selected_directory === directory
          && !state.switchingModel, row.directory);
      }
      const current = (await api('/api/model')).value;
      report.checks[`switch_${row.runtime}`] = current.selected_directory === row.directory
        && await page.inputValue('#active-model') === row.directory;
      report.checks[`capabilities_${row.runtime}`] = await page.locator('#mode option[value="infill"]').isEnabled();
      if (row.runtime === 'gguf') {
        await page.click('#mode-trigger');
        report.checks.gguf_all_task_modes_enabled = await page.locator('#mode-listbox .custom-option:disabled').count() === 0;
        await page.keyboard.press('Escape');
        report.checks.gguf_controls_enabled = await page.locator('#direction').isEnabled()
          && await page.locator('#language').isEnabled()
          && await page.locator('#duration').isEnabled()
          && await page.locator('#submit-scene').isEnabled();
        for (const mode of ['tts', 'revision', 'edit', 'infill', 'stream', 'audio_stream']) {
          await choose('#mode', mode);
          report.checks[`gguf_mode_fields_${mode}`] = await page.evaluate(current => {
            const disabled = id => document.getElementById(id).disabled;
            return document.getElementById('mode').value === current
              && disabled('text') === (current === 'stream')
              && disabled('direction') === ['stream', 'infill'].includes(current)
              && disabled('language') === ['stream', 'infill'].includes(current)
              && disabled('duration') === (current === 'stream')
              && disabled('previous-take') === ['stream', 'infill'].includes(current);
          }, mode);
        }
        await choose('#mode', 'tts');
      }
    }
    report.selected_model = python.directory;
    report.checks.selection_did_not_load_weights = (await api('/api/health')).value.engine_loaded === false;
    await api('/__test_busy', { busy: true });
    await page.click('#refresh-btn');
    await page.waitForFunction(() => state.model?.busy && state.jobs.some(job => job.status === 'running'));
    await choose('#active-model', native.directory);
    const rejected = await api('/api/model/select', { directory: native.directory });
    report.checks.active_job_blocks_switch = await page.locator('#apply-active-model').isDisabled()
      && await page.locator('#select-model').isDisabled()
      && rejected.status === 400 && (await api('/api/model')).value.selected_directory === python.directory;
    await api('/__test_busy', { busy: false });
    await page.click('#refresh-btn');
    await page.waitForFunction(() => !state.model?.busy && !state.jobs.some(job => job.status === 'running'));
    report.checks.idle_queue_reenables_switch = await page.locator('#apply-active-model').isEnabled();
    await choose('#ui-language', 'ja');
    await choose('#active-model', '__manage__');
    report.checks.add_model_opens_settings = await page.locator('#view-settings').isVisible()
      && await page.locator('#installed-model').evaluate(node => node === document.activeElement);
    // A failing model switch keeps the last valid selection and shows a translated error.
    const missingModel = path.join(root, '.cache/nonexistent-ui-model');
    await page.evaluate(directory => switchModel(directory), missingModel);
    await page.waitForFunction(() => Boolean(state.modelError) && !state.switchingModel);
    report.checks.failed_switch_preserves_active_model = (await api('/api/model')).value.selected_directory === python.directory
      && (await page.locator('#active-model-status').textContent()).startsWith('モデルを切り替えられませんでした');
    await page.click('.nav[data-view="generate"]');
    for (const locale of Object.keys(expected)) {
      await choose('#ui-language', locale);
      await page.setViewportSize({ width: 390, height: 844 });
      report.checks[`mobile_no_overflow_${locale}`] = await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1);
      report.checks[`mobile_language_visible_${locale}`] = await page.locator('#ui-language').isVisible();
      const image = path.join(screenshotFolder, `mobile-${locale}.png`);
      await page.screenshot({ path: image, fullPage: true, animations: 'disabled' });
      report.screenshots.push(image);
    }
    await page.setViewportSize({ width: 1440, height: 1000 });
    await choose('#ui-language', 'en');
    report.missing_translation_keys = await page.evaluate(() => [...document.querySelectorAll('[data-i18n]')]
      .map(node => node.dataset.i18n).filter(key => !I18n.has(key)));
    report.checks.no_browser_errors = report.errors.length === 0;
    if (process.env.DRAMA_UI_DESIGN_AUDIT === '1') {
      await choose('#active-model', python.directory);
      await page.click('#manage-models');
      await choose('#installed-model', python.directory);
      await page.waitForFunction(directory => state.model?.inspection?.directory === directory,
        python.directory);
      await page.click('.nav[data-view="generate"]');
      report.checks.light_theme = await page.evaluate(() => getComputedStyle(document.documentElement).colorScheme === 'light'
        && getComputedStyle(document.querySelector('.panel')).backgroundColor === 'rgb(255, 255, 255)');
      report.checks.advanced_settings_collapsed_by_default = !await page.locator('#advanced-settings').evaluate(node => node.open);
      await page.locator('#mode-trigger').focus();
      await page.keyboard.press('ArrowDown');
      await page.keyboard.press('ArrowDown');
      await page.keyboard.press('Enter');
      report.checks.mode_menu_keyboard_selects = await page.inputValue('#mode') === 'revision'
        && await page.locator('#previous-take').isVisible();
      await choose('#mode', 'tts');
      await page.locator('#advanced-settings').evaluate(node => { node.open = false; });
      report.typography = await page.evaluate(() => Object.fromEntries([
        ['title', '#view-title'], ['navigation', '.nav'], ['editor', '#text'], ['field', '.field-label'],
      ].map(([key, selector]) => [key, parseFloat(getComputedStyle(document.querySelector(selector)).fontSize)])));
      report.checks.readable_primary_type_scale = report.typography.title >= 28 && report.typography.navigation >= 15
        && report.typography.editor >= 17 && report.typography.field >= 14;
      for (const mode of ['tts', 'revision', 'edit', 'infill', 'stream', 'audio_stream']) {
        await choose('#mode', mode);
        const payload = await page.evaluate(() => payloadFromForm());
        report.checks[`workflow_${mode}`] = payload.mode === mode
          && (mode !== 'infill' || await page.locator('#source-take').isVisible())
          && (mode !== 'revision' || await page.locator('#previous-take').isVisible())
          && (mode !== 'edit' || await page.locator('#previous-take').isVisible())
          && (mode !== 'stream' || (await page.locator('#chunks').isVisible() && await page.locator('#text').isDisabled()));
      }
      await choose('#mode', 'tts');
      await page.locator('#advanced-settings').evaluate(node => { node.open = false; });
      await page.click('.nav[data-view="voices"]');
      const samples = 8000;
      const wav = Buffer.alloc(44 + samples * 2);
      wav.write('RIFF', 0); wav.writeUInt32LE(wav.length - 8, 4); wav.write('WAVEfmt ', 8);
      wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20); wav.writeUInt16LE(1, 22);
      wav.writeUInt32LE(8000, 24); wav.writeUInt32LE(16000, 28); wav.writeUInt16LE(2, 32);
      wav.writeUInt16LE(16, 34); wav.write('data', 36); wav.writeUInt32LE(samples * 2, 40);
      for (let i = 0; i < samples; i++) wav.writeInt16LE(Math.round(1000 * Math.sin(i * Math.PI / 20)), 44 + i * 2);
      await page.fill('#voice-name', 'Studio reference');
      await page.fill('#voice-transcript', 'A synthetic audio fixture for interface validation.');
      await page.locator('#voice-audio').setInputFiles({ name: 'studio-reference.wav', mimeType: 'audio/wav', buffer: wav });
      await page.waitForFunction(() => document.querySelector('#voice-file-audio').duration === 1);
      report.checks.reference_upload_preview = await page.locator('#voice-upload-preview').isVisible()
        && await page.locator('#voice-file-name').textContent() === 'studio-reference.wav';
      await page.locator('#voice-upload-preview .player-play').click();
      await page.waitForFunction(() => document.querySelector('#voice-file-audio').currentTime > .04);
      await page.locator('#voice-upload-preview .player-play').click();
      report.checks.custom_player_plays_and_pauses_real_audio = await page.locator('#voice-file-audio').evaluate(audio =>
        audio.paused && audio.currentTime > 0);
      const seek = page.locator('#voice-upload-preview .player-seek');
      const seekBox = await seek.boundingBox();
      await seek.click({ position: { x: seekBox.width * .55, y: seekBox.height / 2 } });
      await page.locator('#voice-upload-preview .player-rate').click();
      report.checks.custom_player_seek_and_speed = await page.locator('#voice-file-audio').evaluate(audio =>
        audio.currentTime > .4 && audio.currentTime < .7 && audio.playbackRate === 1.25);
      await page.locator('#voice-upload-preview .player-mute').click();
      report.checks.custom_player_mute = await page.locator('#voice-file-audio').evaluate(audio => audio.muted);
      await page.locator('#voice-upload-preview .player-mute').click();
      await page.locator('#voice-upload-preview .player-volume').focus();
      await page.keyboard.press('End');
      await page.keyboard.press('ArrowLeft');
      report.checks.custom_player_volume_level = await page.locator('#voice-file-audio').evaluate(audio =>
        !audio.muted && audio.volume < 1 && audio.volume > .9);
      const [savedResponse] = await Promise.all([
        page.waitForResponse(response => response.url().endsWith('/api/voices') && response.request().method() === 'POST'),
        page.click('#save-voice'),
      ]);
      if (savedResponse.status() !== 201) throw new Error('Reference upload failed');
      const savedVoice = await savedResponse.json();
      await page.waitForFunction(id => state.voices.some(voice => voice.id === id), savedVoice.id);
      const reference = await page.evaluate(id => state.voices.find(voice => voice.id === id), savedVoice.id);
      report.checks.reference_save = await page.locator('#voice-list').textContent().then(text => text.includes('Studio reference'));
      await page.fill('#voice-search', '__no_voice_matches__');
      report.checks.voice_library_search = await page.locator('#voice-list .voice-card:visible').count() === 0;
      await page.fill('#voice-search', '');
      await page.locator(`.use-voice[data-id="${reference.id}"]`).click();
      report.checks.library_voice_applies_to_generation = await page.locator('#view-generate').isVisible()
        && await page.evaluate(id => selectedVoices().includes(id), reference.id);
      await page.click('#default-voice');
      await page.click('.nav[data-view="generate"]');
      await page.locator(`#voice-picks [data-voice="${reference.id}"]`).click();
      await choose('#ui-language', 'ja');
      report.checks.cloning_reference_survives_locale_switch = await page.evaluate(identifier =>
        payloadFromForm().voice_ids.includes(identifier)
        && document.querySelector('#default-voice').getAttribute('aria-pressed') === 'false', reference.id);
      await page.click('#default-voice');
      report.checks.default_voice_clears_references = await page.evaluate(() => !payloadFromForm().voice_ids.length
        && document.querySelector('#default-voice').getAttribute('aria-pressed') === 'true');
      await page.evaluate(() => showInfillSource({ frames: 128, samples_per_frame: 2048, sample_rate: 44100, duration_seconds: 128 * 2048 / 44100 }));
      await choose('#mode', 'infill');
      await page.fill('#start-time', '0.101');
      await page.fill('#end-time', '0.101');
      report.checks.infill_time_frame_controls = await page.inputValue('#start-frame') === await page.inputValue('#end-frame');
      await choose('#mode', 'stream');
      await page.evaluate(() => {
        state.activeJob = { id: 'design-stream', mode: 'stream', status: 'running', events: [],
          stream_input: { live: true, finished: false } };
        renderJob(state.activeJob);
      });
      report.checks.live_stream_controls_retained = await page.locator('#live-stream-controls').isVisible()
        && await page.locator('#push-stream').isEnabled() && await page.locator('#finish-stream').isEnabled()
        && await page.locator('#cancel-job').isVisible();
      await page.evaluate(audio => {
        state.activeJob = { id: 'design-result', mode: 'tts', status: 'complete', events: [], output: audio };
        renderJob(state.activeJob);
        window.designAudio = document.querySelector('#result-box .main-result audio');
      }, reference.audio_url);
      await choose('#ui-language', 'en');
      report.checks.result_audio_and_download_retained = await page.evaluate(() =>
        designAudio === document.querySelector('#result-box .main-result audio')
        && document.querySelector('#result-box a.download-link').hasAttribute('download'));
      await page.waitForFunction(() => document.querySelector('#result-box .player-wave')?.childElementCount === 80);
      report.checks.result_waveform_decoded_from_audio = true;
      await page.evaluate(() => {
        state.activeJob.events = [{ stage: 'decode', frames: 21, message: 'Generating' }];
        renderJob(state.activeJob);
      });
      report.checks.result_details_collapsed_by_default = !await page.locator('#job-log').evaluate(node => node.open);
      await page.locator('#job-log summary').click();
      report.checks.result_details_remain_accessible = await page.locator('#job-events .event').isVisible();
      await page.locator('#job-log summary').click();
      const playerImage = path.join(screenshotFolder, 'result-player.png');
      await page.screenshot({ path: playerImage, fullPage: true, animations: 'disabled' });
      report.screenshots.push(playerImage);
      await choose('#mode', 'tts');
      await page.click('.nav[data-view="scene"]');
      const characters = await page.locator('#characters .repeat-row').count();
      await page.click('#add-character');
      await page.locator('#characters .repeat-row').last().locator('button').click();
      const lines = await page.locator('#scene-lines .repeat-row').count();
      await page.click('#add-line');
      await page.locator('#scene-lines .repeat-row').last().locator('input').last().fill('A new line.');
      report.checks.scene_editor_add_remove = await page.locator('#characters .repeat-row').count() === characters
        && await page.locator('#scene-lines .repeat-row').count() === lines + 1
        && await page.evaluate(() => scenePayload().lines.at(-1).text === 'A new line.');
      await page.locator('#scene-lines .repeat-row').last().locator('button').click();
      await page.click('.nav[data-view="settings"]');
      const downloadCatalog = (await api('/api/model/catalog')).value.models;
      report.checks.local_model_setup_with_downloader = await page.locator('#installed-model').isVisible()
        && await page.locator('#model-root-list').isVisible()
        && await page.locator('#browse-model-directory').isVisible()
        && await page.locator('#select-model').isVisible()
        && await page.locator('#model-directory, #inspect-model, #model-active-folder').count() === 0
        && await page.locator('#model-download-choice').isVisible()
        && await page.locator('#model-download-sources').isVisible()
        && await page.locator('#download-model').isVisible()
        && ['fp8', 'int4', 'int8', 'w4a8', 'w6a8'].every(id => downloadCatalog.some(model => model.id === id))
        && downloadCatalog.some(model => model.id.startsWith('gguf-'))
        && await page.locator('a[href="/api/diagnostics/export"]').isVisible();
      // Clear only the synthetic monitor state before capturing the normal empty workspace.
      await page.evaluate(() => {
        state.activeJob = null;
        document.querySelector('#result-box').classList.add('hidden');
        document.querySelector('#live-stream-controls').classList.add('hidden');
        document.querySelector('#cancel-job').classList.add('hidden');
        document.querySelector('#job-events').innerHTML = '';
        document.querySelector('#job-log').classList.add('hidden');
        document.querySelector('#output-placeholder').hidden = false;
        document.querySelector('#output-placeholder').innerHTML = emptyState('headphones', 'No audio yet', '', 'output-empty');
        document.querySelector('#job-badge').innerHTML = translated('IDLE');
        document.querySelector('#job-badge').removeAttribute('data-status');
        document.querySelector('#job-title').innerHTML = translated('Ready');
        document.querySelector('#progress-stage').innerHTML = translated('Waiting for a request');
        document.querySelector('#progress-bar').style.width = '0%';
        document.querySelector('#advanced-settings').open = false;
      });
      await choose('#ui-language', 'zh-CN');
      for (const width of [1440, 1024, 390]) {
        await page.setViewportSize({ width, height: width > 700 ? 1000 : 844 });
        for (const name of ['generate', 'scene', 'voices', 'takes', 'settings']) {
          await page.click(`.nav[data-view="${name}"]`);
          report.checks[`layout_${width}_${name}`] = await page.locator(`#view-${name}`).isVisible()
            && await page.locator(`.nav[data-view="${name}"]`).getAttribute('aria-current') === 'page'
            && await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1
              && [...document.querySelectorAll('.nav')].every(node => node.getBoundingClientRect().right <= innerWidth + 1));
          const image = path.join(screenshotFolder, `${width}-${name}.png`);
          await page.screenshot({ path: image, fullPage: true, animations: 'disabled' });
          report.screenshots.push(image);
        }
      }
      report.checks.no_browser_errors = report.errors.length === 0;
    }
  } catch (error) {
    report.errors.push(error.stack || String(error));
  } finally {
    if (browser) await browser.close();
    report.passed = !report.errors.length && Object.values(report.checks).every(Boolean);
    fs.writeFileSync(out, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify(report, null, 2));
    if (!report.passed) process.exitCode = 1;
  }
})();
