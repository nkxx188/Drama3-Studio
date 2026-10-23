/* Observe actual startup geometry with delayed local APIs; no model inference. */
const fs = require('node:fs'), path = require('node:path'), { chromium } = require('playwright');
const root = path.resolve(__dirname, '..'), base = process.env.DRAMA_UI_URL, folder = process.env.DRAMA_STUDIO_FOLDER;
const baseline = process.env.DRAMA_UI_STARTUP_BASELINE === '1';
const report = { scope: 'Startup paints, hydration, delayed APIs and post-ready geometry; real HTTP/store and an explicit synthetic scroll fixture, no model weights.',
  baseline, checks: {}, cases: [], errors: [], screenshots: [] };
(async () => {
  let browser, page;
  try {
    browser = await chromium.launch({ channel: 'chrome', headless: true });
    const context = await browser.newContext({ locale: 'zh-CN' });
    const check = (name, value) => { report.checks[name] = Boolean(value); if (!value) throw new Error(name); };
    const sizes = baseline ? [1440] : [1440, 390];
    const views = baseline ? ['generate', 'scene', 'settings'] : ['generate', 'scene', 'voices', 'takes', 'settings'];
    for (const width of sizes) for (const view of views) {
      page = await context.newPage(); await page.setViewportSize({ width, height: width === 390 ? 844 : 960 });
      page.on('pageerror', error => report.errors.push(error.stack || error.message));
      await page.addInitScript(({ view }) => {
        sessionStorage.setItem('drama3.ui.view', view); localStorage.setItem('drama3.ui.language', 'zh-CN');
        const trace = window.startupTrace = { frames: [], shifts: [] };
        const visible = node => node && getComputedStyle(node).visibility === 'visible'
          && getComputedStyle(node).display !== 'none' && getComputedStyle(node).opacity !== '0'
          && node.getClientRects().length > 0;
        const rect = selector => {
          const node = document.querySelector(selector); if (!node) return null;
          const r = node.getBoundingClientRect();
          return [r.x, r.y, r.width, r.height].map(value => Math.round(value * 10) / 10);
        };
        new PerformanceObserver(list => {
          for (const shift of list.getEntries()) if (!shift.hadRecentInput) trace.shifts.push({ value: shift.value, at: shift.startTime });
        }).observe({ type: 'layout-shift', buffered: true });
        const record = () => {
          const html = document.documentElement;
          if (html && [...document.styleSheets].some(sheet => sheet.href?.endsWith('/assets/studio.css'))) {
            trace.frames.push({ at: performance.now(), ready: html.dataset.studioReady === 'true',
              phase: html.dataset.startup || '', compact: html.classList.contains('compact-studio'),
              views: [...document.querySelectorAll('.view')].filter(visible).map(node => node.id),
              nativeSelects: [...document.querySelectorAll('select')].filter(visible).length,
              sidebar: rect('.sidebar'), device: rect('.local-card'), title: rect('#view-title'), editor: rect(view === 'scene' ? '#dialogue-editor-area' : '.compose-panel > .editor-area'),
              dock: rect(view === 'scene' ? '#submit-scene' : '#submit-job'),
              layoutReady: Boolean(document.querySelector('.action-dock')),
            });
          }
          if (!trace.stop) requestAnimationFrame(record);
        };
        requestAnimationFrame(record);
      }, { view });
      const calls = {};
      await page.route('**/api/**', async route => {
        const name = new URL(route.request().url()).pathname;
        if (route.request().method() === 'GET') {
          calls[name] = (calls[name] || 0) + 1;
          await new Promise(resolve => setTimeout(resolve, name === '/api/studio' ? 350 : 180));
        }
        await route.continue();
      });
      await page.goto(base, { waitUntil: 'commit' });
      await page.waitForFunction(() => window.startupTrace?.frames.some(frame => !frame.ready));
      if (view === 'generate') {
        const file = path.join(folder, `startup-${baseline ? 'before' : 'pending'}-${width}.png`);
        await page.screenshot({ path: file }); report.screenshots.push(file);
      }
      await page.waitForFunction(() => document.documentElement.dataset.studioReady === 'true');
      await page.waitForTimeout(250);
      const trace = await page.evaluate(() => { startupTrace.stop = true; return startupTrace; });
      const frames = trace.frames, pending = frames.filter(frame => !frame.ready), settled = frames.filter(frame => frame.ready);
      const item = { view, width, calls, pendingVisibleFrames: pending.filter(frame => frame.views.length).length,
        themeStates: [...new Set(frames.map(frame => frame.compact))],
        sidebarWidths: [...new Set(frames.map(frame => frame.sidebar?.[2]).filter(value => value !== undefined))],
        nativeControlFrames: frames.filter(frame => frame.nativeSelects).length, shifts: trace.shifts,
        pendingFrames: pending.length, readyFrames: settled.length, first: frames[0], final: frames.at(-1),
        readyGeometryStates: [...new Set(settled.map(frame => JSON.stringify([frame.sidebar, frame.editor, frame.dock])))].length };
      item.deviceGeometryStates = [...new Set(frames.map(frame => JSON.stringify(frame.device)))].length;
      report.cases.push(item);
      if (!baseline) {
        const suffix = `${view}_${width}`;
        check(`theme_is_final_from_first_paint_${suffix}`, item.themeStates.length === 1 && item.themeStates[0] === true);
        check(`unhydrated_workspace_is_never_exposed_${suffix}`, item.pendingVisibleFrames === 0);
        check(`no_native_selects_flash_${suffix}`, item.nativeControlFrames === 0);
        check(`ready_workspace_is_the_requested_page_${suffix}`, settled.length > 1
          && settled.every(frame => frame.views.join() === `view-${view}`));
        check(`sidebar_does_not_resize_${suffix}`, item.sidebarWidths.length === 1);
        check(`hardware_summary_does_not_jump_${suffix}`, item.deviceGeometryStates === 1);
        check(`layout_is_stable_after_ready_${suffix}`, item.readyGeometryStates === 1);
        for (const name of ['/api/studio', '/api/health', '/api/voices', '/api/model', '/api/capabilities', '/api/jobs']) {
          check(`one_initial_request_${suffix}_${name.split('/').at(-1)}`, calls[name] === 1);
        }
        await page.evaluate(() => {
          window.startupNavigationEvents = 0;
          document.addEventListener('drama:viewchange', () => { window.startupNavigationEvents++; });
        });
        await page.locator('#ui-language-trigger').click();
        check(`ordinary_control_clicks_do_not_reenter_navigation_${suffix}`,
          await page.locator('#ui-language-trigger').getAttribute('aria-expanded') === 'true'
          && await page.evaluate(() => startupNavigationEvents === 0));
        await page.keyboard.press('Escape');
      }
      if (view === 'generate') {
        const file = path.join(folder, `startup-ready-${width}.png`);
        await page.screenshot({ path: file }); report.screenshots.push(file);
      }
      await page.close(); page = null;
    }
    if (!baseline) {
      const ready = () => page.waitForFunction(() => document.documentElement.dataset.studioReady === 'true');
      const makePage = async view => {
        page = await context.newPage(); await page.setViewportSize({ width: 1440, height: 960 });
        page.on('pageerror', error => report.errors.push(error.stack || error.message));
        await page.addInitScript(view => {
          if (!sessionStorage.getItem('drama3.ui.view')) sessionStorage.setItem('drama3.ui.view', view);
          localStorage.setItem('drama3.ui.language', 'zh-CN');
        }, view);
      };
      await makePage('settings');
      let rejectApi = true;
      await page.route('**/api/studio', route => rejectApi
        ? route.fulfill({ status: 503, json: { error: 'Fixture workspace storage is unavailable.' } }) : route.continue());
      await page.goto(base, { waitUntil: 'networkidle' });
      await page.waitForFunction(() => document.documentElement.dataset.startup === 'error');
      check('failed_initial_data_shows_a_recoverable_error', await page.locator('.startup-failure').isVisible()
        && (await page.locator('#startup-error').textContent()).includes('Fixture workspace storage')
        && await page.locator('#workspace-content').evaluate(node => node.inert));
      rejectApi = false;
      await Promise.all([page.waitForNavigation({ waitUntil: 'networkidle' }), page.locator('.startup-failure a').click()]); await ready();
      check('retrying_initial_data_keeps_the_requested_page', await page.locator('#view-settings').isVisible());
      await page.close(); page = null;
      await makePage('voices');
      let rejectScript = true;
      await page.route('**/assets/studio.js', route => rejectScript ? route.abort('failed') : route.continue());
      await page.goto(base, { waitUntil: 'networkidle' });
      await page.waitForFunction(() => document.documentElement.dataset.startup === 'error');
      check('missing_script_does_not_leave_an_endless_loading_screen', await page.locator('.startup-failure').isVisible()
        && (await page.locator('#startup-error').textContent()).includes('studio.js'));
      rejectScript = false;
      await Promise.all([page.waitForNavigation({ waitUntil: 'networkidle' }), page.locator('.startup-failure a').click()]); await ready();
      check('retry_after_script_failure_restores_the_voice_library', await page.locator('#view-voices').isVisible());
      await page.close(); page = null;
      await makePage('generate'); await page.goto(base, { waitUntil: 'networkidle' }); await ready();
      await page.locator('#clone-audio').setInputFiles(process.env.DRAMA_STUDIO_REFERENCE);
      await page.locator('#clone-transcript').fill('启动恢复测试用的参考逐字稿。');
      await page.locator('.compose-panel > .editor-area .cm-content').fill('刷新后仍保留的创作台词。');
      await page.evaluate(() => StudioFlow.save());
      await page.addInitScript(() => {
        const decode = OfflineAudioContext.prototype.decodeAudioData;
        const gate = new Promise(resolve => { window.releaseStartupWave = resolve; });
        OfflineAudioContext.prototype.decodeAudioData = async function (...args) {
          await gate;
          return decode.apply(this, args);
        };
      });
      await page.reload({ waitUntil: 'domcontentloaded' }); await ready();
      const beforeWave = await page.locator('#clone-upload-preview').boundingBox();
      check('waveform_space_is_reserved_while_decoding', await page.locator('#clone-upload-preview .player-wave-surface').evaluate(node =>
        node.getBoundingClientRect().height === 50 && !node.querySelector('.player-wave').childElementCount));
      await page.evaluate(() => releaseStartupWave());
      await page.waitForFunction(() => document.querySelector('#clone-upload-preview .player-wave')?.childElementCount === 80);
      const afterWave = await page.locator('#clone-upload-preview').boundingBox();
      check('late_waveform_decode_does_not_resize_the_reference_editor', Math.abs(beforeWave.height - afterWave.height) < 1);
      check('reference_and_writing_are_hydrated_before_the_workspace_is_enabled',
        await page.inputValue('#clone-transcript') === '启动恢复测试用的参考逐字稿。'
        && await page.inputValue('#text') === '刷新后仍保留的创作台词。'
        && await page.locator('#workspace-content').getAttribute('aria-busy') === 'false');
      const response = await context.request.post(base + '/api/jobs', { data: { text: 'Scroll fixture.', max_new_tokens: 32 } });
      const job = await response.json();
      await page.waitForFunction(async id => (await (await fetch('/api/jobs/' + id)).json()).status === 'complete', job.id);
      await context.request.post(base + '/__test__/archive', { data: { source: job.id } });
      await page.evaluate(() => loadJobs()); await page.locator('.nav[data-view="takes"]').click();
      await page.evaluate(() => window.scrollTo({ top: 750, behavior: 'instant' }));
      const scroll = await page.evaluate(() => scrollY);
      check('scroll_restore_fixture_has_real_overflow', scroll > 700);
      await page.reload({ waitUntil: 'networkidle' }); await ready();
      check('refresh_restores_scroll_after_the_final_layout_is_measured', await page.evaluate(y =>
        state.view === 'takes' && Math.abs(scrollY - y) < 1, scroll));
      await page.close(); page = null;
    }
    check('no_browser_exceptions', !report.errors.length);
  } catch (error) {
    report.errors.push(error.stack || String(error));
    if (page) {
      const file = path.join(folder, 'startup-failure.png');
      await page.screenshot({ path: file, fullPage: true }).catch(() => {}); report.screenshots.push(file);
    }
  } finally {
    if (browser) await browser.close();
    report.passed = !report.errors.length && Object.values(report.checks).every(Boolean);
    fs.writeFileSync(path.join(root, `research/2026-10-20-ui-startup-${baseline ? 'baseline' : 'acceptance'}.json`), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report, null, 2)); if (!report.passed) process.exitCode = 1;
  }
})();
