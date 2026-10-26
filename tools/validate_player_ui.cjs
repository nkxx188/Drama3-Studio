/* Real audio and pointer regression for the shared player; no model or user data is used. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const { chromium } = require('playwright');

const root = path.resolve(__dirname, '..');
const folder = path.join(root, '.cache', 'player-ui-acceptance');
const report = { checks: [], errors: [], screenshots: [] };
const near = (a, b, epsilon = .002) => Math.abs(a - b) < epsilon;
const waves = new Map();

function tone(seconds) {
  if (waves.has(seconds)) return waves.get(seconds);
  const rate = 8000, count = seconds * rate, data = Buffer.alloc(44 + count * 2);
  data.write('RIFF'); data.writeUInt32LE(data.length - 8, 4); data.write('WAVEfmt ', 8);
  data.writeUInt32LE(16, 16); data.writeUInt16LE(1, 20); data.writeUInt16LE(1, 22);
  data.writeUInt32LE(rate, 24); data.writeUInt32LE(rate * 2, 28);
  data.writeUInt16LE(2, 32); data.writeUInt16LE(16, 34);
  data.write('data', 36); data.writeUInt32LE(count * 2, 40);
  for (let i = 0; i < count; i++) {
    const at = i / rate, envelope = .08 + .45 * Math.sin(at * 2.3) ** 2;
    data.writeInt16LE(Math.round(Math.sin(at * 2 * Math.PI * 220) * envelope * 32767), 44 + i * 2);
  }
  waves.set(seconds, data); return data;
}

const html = fs.readFileSync(path.join(root, 'ui', 'index.html'), 'utf8');
const iconStart = html.indexOf('<svg class="icon-sprite"');
const icons = html.slice(iconStart, html.indexOf('</svg>', iconStart) + 6);
const fixture = `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="/app.css"><link rel="stylesheet" href="/studio.css">
<style>.fixture { width: min(960px, calc(100% - 40px)); margin: 32px auto; }
.fixture section { margin-bottom: 40px; } .fixture h2 { font-size: 20px; }
.fixture .audio-import-file-heading { margin-bottom: 16px; }</style></head>
<body class="compact-studio">${icons}<main class="fixture audio-editor-workspace audio-import-page">
<section class="audio-edit-source"><h2>原音频</h2>
<div id="audio-import-preview"><div class="audio-import-file-heading"><strong>result (8).wav</strong></div>
<audio id="edit" src="/tone/32.wav" preload="auto" data-wave-window="15"
 data-selection="true" data-frame-seconds="0.046439909297052155"></audio></div></section>
<section><h2>生成结果</h2><audio id="normal" src="/tone/60.wav" preload="auto" data-selection="true"></audio></section>
<section><h2>短音频</h2><audio id="short" src="/tone/8.wav" preload="auto" data-wave-window="15"></audio></section>
<section><h2>长音频</h2><audio id="long" src="/tone/180.wav" preload="auto" data-wave-window="15"></audio></section>
</main><script src="/i18n.js"></script><script src="/player.js"></script>
<script>StudioAudio.enhance();</script></body></html>`;

const server = http.createServer((request, response) => {
  const route = new URL(request.url, 'http://localhost').pathname;
  if (route === '/') {
    response.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' }); response.end(fixture);
  } else if (/^\/tone\/(8|32|60|180)\.wav$/.test(route)) {
    const data = tone(Number(route.match(/\d+/)[0]));
    const range = request.headers.range?.match(/^bytes=(\d+)-(\d*)$/);
    if (range) {
      const start = Number(range[1]), end = Math.min(Number(range[2] || data.length - 1), data.length - 1);
      response.writeHead(206, { 'Content-Type': 'audio/wav', 'Accept-Ranges': 'bytes',
        'Content-Range': `bytes ${start}-${end}/${data.length}`, 'Content-Length': end - start + 1 });
      response.end(data.subarray(start, end + 1));
    } else {
      response.writeHead(200, { 'Content-Type': 'audio/wav', 'Accept-Ranges': 'bytes', 'Content-Length': data.length });
      response.end(data);
    }
  } else if (['/app.css', '/studio.css', '/i18n.js', '/player.js'].includes(route)) {
    response.writeHead(200, { 'Content-Type': route.endsWith('.css') ? 'text/css' : 'text/javascript' });
    response.end(fs.readFileSync(path.join(root, 'ui', route.slice(1))));
  } else { response.writeHead(404); response.end(); }
});

(async () => {
  let browser;
  fs.mkdirSync(folder, { recursive: true });
  try {
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--mute-audio'] });
    const page = await browser.newPage({ viewport: { width: 1440, height: 960 }, locale: 'zh-CN' });
    page.on('pageerror', error => report.errors.push(error.stack || error.message));
    page.setDefaultTimeout(10000);
    const check = (name, value) => { assert.ok(value, name); report.checks.push(name); console.log('PASS', name); };
    const audio = id => page.locator(`#${id}`);
    const player = id => audio(id).locator('..');
    const button = (id, name) => player(id).locator(`.player-${name}`);
    const state = id => audio(id).evaluate(node => {
      const box = node.closest('.studio-player'), pan = box.querySelector('.player-pan');
      return {
        duration: node.duration, time: node.currentTime, paused: node.paused,
        offset: Number(pan.value), span: node.duration - Number(pan.max), panHidden: pan.hidden,
        selection: StudioAudio.selection(node),
        ticks: [...box.querySelectorAll('.player-wave-timeline span')].map(tick => tick.textContent),
        thumb: Number.parseFloat(pan.style.getPropertyValue('--pan-size')),
        width: box.querySelector('.player-wave-surface').clientWidth,
      };
    });
    const panTo = (id, offset) => button(id, 'pan').evaluate((node, value) => {
      node.value = value; node.dispatchEvent(new Event('input', { bubbles: true }));
    }, offset);
    const select = (id, start, end) => audio(id).evaluate((node, range) =>
      StudioAudio.setSelection(node, ...range), [start, end]);
    const seekTo = (id, time) => button(id, 'seek').evaluate((node, value) => {
      node.value = value; node.dispatchEvent(new Event('input', { bubbles: true }));
    }, time);
    const screenshot = async name => {
      const file = path.join(folder, `${name}.png`);
      await page.screenshot({ path: file, fullPage: true }); report.screenshots.push(file);
    };

    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    await page.waitForFunction(() => [...document.querySelectorAll('audio')].every(node =>
      node.duration > 0 && node.closest('.studio-player').querySelectorAll('.player-wave rect').length === 80));
    check('editor_initial_window_is_15_seconds', near((await state('edit')).span, 15));
    check('long_audio_is_not_limited_to_eight_times_zoom', near((await state('long')).span, 15));
    check('short_audio_stays_fully_visible', near((await state('short')).span, 8) && (await state('short')).panHidden);
    check('normal_result_stays_fully_visible', near((await state('normal')).span, 60) && (await state('normal')).panHidden);
    const initial = await state('edit');
    check('scrollbar_thumb_represents_visible_fraction', near(initial.thumb / initial.width, 15 / 32));
    check('visible_window_has_absolute_time_ticks', initial.ticks.join(',') === '0:00,0:07,0:15');
    const scrollbar = await button('edit', 'pan').boundingBox();
    await page.mouse.move(scrollbar.x + initial.thumb / 2, scrollbar.y + scrollbar.height / 2);
    await page.mouse.down();
    await page.mouse.move(scrollbar.x + initial.thumb / 2 + scrollbar.width * .2,
      scrollbar.y + scrollbar.height / 2, { steps: 8 });
    await page.mouse.up();
    check('scrollbar_thumb_drags_without_seeking', (await state('edit')).offset > 0 && near((await state('edit')).time, 0));
    await panTo('edit', 0);

    await page.evaluate(() => { window.originalPlayIcon = document.querySelector('#edit').closest('.studio-player').querySelector('.player-play svg'); });
    for (let i = 0; i < 3; i++) {
      await button('edit', 'play').click({ delay: 120 });
      await page.waitForFunction(() => !document.querySelector('#edit').paused);
      await button('edit', 'play').click({ delay: 160 });
      await page.waitForFunction(() => document.querySelector('#edit').paused);
    }
    check('main_play_button_reliably_pauses_on_repeated_real_clicks', (await state('edit')).paused);
    check('playback_keeps_the_same_icon_node', await page.evaluate(() =>
      originalPlayIcon === document.querySelector('#edit').closest('.studio-player').querySelector('.player-play svg')));
    await button('edit', 'play').focus(); await page.keyboard.press('Space');
    await page.waitForFunction(() => !document.querySelector('#edit').paused);
    await page.keyboard.press('Space'); await page.waitForFunction(() => document.querySelector('#edit').paused);
    check('keyboard_play_and_pause_work', (await state('edit')).paused);

    await select('edit', 2, 6);
    await button('edit', 'preview-selection').click({ delay: 100 });
    await page.waitForFunction(() => document.querySelector('#edit').currentTime > 2.3);
    await button('edit', 'preview-selection').click({ delay: 160 });
    const paused = await state('edit');
    check('selection_preview_can_pause', paused.paused);
    await button('edit', 'preview-selection').click();
    const resumed = await state('edit');
    check('selection_preview_resumes_without_restarting', !resumed.paused && resumed.time >= paused.time
      && resumed.time < paused.time + .5);
    await audio('edit').evaluate(node => { node.currentTime = StudioAudio.selection(node).end - .05; });
    await page.waitForFunction(() => document.querySelector('#edit').paused);
    const stopped = await state('edit');
    check('selection_preview_stops_exactly_at_selection_end',
      near(stopped.time, stopped.selection.end));

    await seekTo('edit', 8); await panTo('edit', 17);
    const beforeZoom = await state('edit');
    check('browsing_does_not_seek_or_change_selection', near(beforeZoom.time, 8)
      && near(beforeZoom.offset, 17) && JSON.stringify(beforeZoom.selection) === JSON.stringify(paused.selection));
    await button('edit', 'zoom-in').click();
    const zoomed = await state('edit');
    check('zoom_does_not_change_audio_time_or_frame_boundaries', near(zoomed.time, 8) && near(zoomed.span, 7.5)
      && JSON.stringify(zoomed.selection) === JSON.stringify(beforeZoom.selection));
    await button('edit', 'zoom-out').click(); await panTo('edit', 17);
    await button('edit', 'play').click({ delay: 100 });
    const playing = await state('edit');
    check('play_reveals_actual_playhead_instead_of_seeking_to_window_start',
      playing.time >= 8 && playing.time < 9 && playing.offset <= playing.time
      && playing.offset + playing.span > playing.time);
    await panTo('edit', 17);
    await page.waitForTimeout(200);
    check('manual_browsing_is_not_overridden_during_playback', near((await state('edit')).offset, 17));
    await button('edit', 'play').click({ delay: 100 });
    await seekTo('edit', 14.9); await button('edit', 'play').click();
    await page.waitForFunction(() => Number(document.querySelector('#edit').closest('.studio-player').querySelector('.player-pan').value) >= 15);
    check('playback_follows_into_the_next_window', (await state('edit')).time > 15);
    await button('edit', 'play').click();

    await audio('edit').evaluate(node => StudioAudio.clearSelection(node));
    await panTo('edit', 0);
    const surface = button('edit', 'wave-surface');
    const box = await surface.boundingBox();
    await page.mouse.move(box.x + box.width * .2, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width - 2, box.y + box.height / 2, { steps: 12 });
    await page.waitForFunction(() => Number(document.querySelector('#edit').closest('.studio-player').querySelector('.player-pan').value) > 1);
    await page.mouse.up();
    const extended = await state('edit');
    check('dragging_at_the_edge_scrolls_and_extends_selection_beyond_one_window',
      extended.offset > 1 && extended.selection.end > 15 && near(extended.selection.start, Math.floor(3 * 44100 / 2048) * 2048 / 44100));
    const selectedBeforePan = JSON.stringify(extended.selection);
    await panTo('edit', 17);
    check('selection_survives_scrolling_out_of_view', JSON.stringify((await state('edit')).selection) === selectedBeforePan);
    await audio('edit').evaluate(node => { node.dataset.selection = 'false'; });
    await page.waitForTimeout(50);
    check('switching_edit_scope_does_not_reset_view', near((await state('edit')).offset, 17));
    await audio('edit').evaluate(node => { node.dataset.selection = 'true'; });
    await select('edit', 19, 22); await panTo('edit', 17);
    const handle = button('edit', 'region-handle.region-start');
    const beforeKey = await state('edit');
    await handle.focus(); await handle.press('ArrowRight');
    const afterKey = await state('edit');
    check('keyboard_selection_still_uses_original_codec_frame_step',
      near(afterKey.selection.start - beforeKey.selection.start, 2048 / 44100)
      && near(afterKey.selection.end, beforeKey.selection.end));

    for (const viewport of [{ width: 1440, height: 960 }, { width: 760, height: 900 }, { width: 390, height: 844 }]) {
      await page.setViewportSize(viewport); await page.waitForTimeout(100);
      check(`responsive_player_has_no_horizontal_overflow_${viewport.width}`, await page.evaluate(() =>
        document.documentElement.scrollWidth <= innerWidth + 1
        && [...document.querySelectorAll('.player-controls button:not([hidden])')].every(node => {
          const rect = node.getBoundingClientRect(); return rect.left >= 0 && rect.right <= innerWidth;
        })));
      await screenshot(`player-${viewport.width}`);
    }
    await page.setViewportSize({ width: 1440, height: 960 });
    await audio('edit').evaluate(node => { node.src = '/tone/60.wav'; });
    await page.waitForFunction(() => document.querySelector('#edit').duration === 60);
    check('replacement_source_initializes_a_fresh_view_without_old_selection',
      near((await state('edit')).span, 15) && near((await state('edit')).offset, 0) && !(await state('edit')).selection);
    check('no_browser_errors', report.errors.length === 0);
  } catch (error) {
    report.errors.push(error.stack || error.message); console.error(error); process.exitCode = 1;
  } finally {
    await browser?.close();
    await new Promise(resolve => server.close(resolve));
    fs.writeFileSync(path.join(folder, 'report.json'), JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify(report, null, 2));
  }
})();
