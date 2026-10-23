/* Real CodeMirror selection, localized prompts and responsive palette; no inference. */
const fs = require('node:fs'), path = require('node:path'), { chromium } = require('playwright');
const root = path.resolve(__dirname, '..'), base = process.env.DRAMA_UI_URL;
const folder = process.env.DRAMA_STUDIO_FOLDER;
const report = { scope: 'Actual bundled editor, native/keyboard/pointer selections, continuous category scrolling and localized tags; no inference.',
  checks: {}, errors: [], screenshots: [], scroll_traces: {} };
const categories = ['emotions', 'tone', 'sounds', 'effects', 'other'];
const expected = [
  'happy sad angry excited calm nervous confident surprised satisfied delighted scared worried upset frustrated depressed empathetic embarrassed disgusted moved proud relaxed grateful curious sarcastic',
  'disdainful unhappy anxious hysterical indifferent uncertain doubtful confused disappointed regretful guilty ashamed jealous envious hopeful optimistic pessimistic nostalgic lonely bored contemptuous sympathetic compassionate determined resigned',
  'in a hurry tone|shouting|screaming|whispering|soft tone|emphasis',
  'laughing|chuckling|sobbing|crying loudly|sighing|groaning|panting|gasping|yawning|snoring|clear throat',
  'audience laughing|background laughter|crowd laughing|break|long-break',
].flatMap((value, index) => value.split(index < 2 ? ' ' : '|'));
(async () => {
  let browser;
  try {
    browser = await chromium.launch({ channel: 'chrome', headless: true });
    const context = await browser.newContext({ viewport: { width: 1440, height: 960 }, locale: 'zh-CN' });
    await context.addInitScript(() => {
      localStorage.setItem('drama3.ui.language', 'zh-CN');
      localStorage.setItem('drama3.recentTags', '[{"value":"old-recent","kind":"event"}]');
    });
    const page = await context.newPage();
    page.on('pageerror', error => report.errors.push(error.stack || error.message));
    const check = (name, value) => {
      report.checks[name] = Boolean(value); if (!value) throw new Error(name);
    };
    const editor = () => page.locator('.compose-panel > .editor-area .cm-content');
    const host = () => page.locator('.compose-panel > .editor-area .script-editor');
    const open = async () => host().locator('[data-label="Tags"]').click();
    const close = async () => {
      if (await host().locator('.tag-panel').isVisible()) await host().locator('.tag-search [data-label="Close"]').click();
    };
    const search = value => host().locator('.tag-search input').fill(value);
    const settle = () => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    const choices = () => host().locator('.tag-choices');
    const tab = category => host().locator(`.tag-categories > [role="tab"][data-category="${category}"]`);
    const scrollState = () => choices().evaluate(node => ({
      top: node.scrollTop, max: node.scrollHeight - node.clientHeight,
      active: node.closest('.tag-panel').querySelector('.tag-category[aria-selected="true"]')?.dataset.category,
      selected: node.closest('.tag-panel').querySelectorAll('.tag-category[aria-selected="true"]').length,
    }));
    const targetIsVisible = category => host().locator(`.tag-category-section[data-category="${category}"]`).evaluate(node => {
      const target = node.getBoundingClientRect(), viewport = node.closest('.tag-choices').getBoundingClientRect();
      return target.bottom > viewport.top + 1 && target.top < viewport.bottom - 1;
    });
    const jump = async category => {
      await tab(category).click(); await settle();
      await page.waitForFunction(category => document.querySelector(
        '.compose-panel > .editor-area .tag-categories .tag-category[aria-selected="true"]'
      )?.dataset.category === category, category);
    };
    const wheelThroughCategories = async name => {
      await jump('emotions');
      const box = await choices().boundingBox();
      await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
      const seen = [], trace = [];
      for (let step = 0; step < 100; step++) {
        const state = await scrollState(); trace.push(state);
        if (seen.at(-1) !== state.active) seen.push(state.active);
        if (state.top >= state.max - 1) break;
        await page.mouse.wheel(0, 44); await settle(); await settle();
      }
      report.scroll_traces[name] = trace;
      check(`wheel_passes_each_category_in_order_${name}`, JSON.stringify(seen) === JSON.stringify(categories));
      const bottom = await scrollState(), pageTop = await page.evaluate(() => scrollY);
      await page.mouse.wheel(0, 1600); await settle(); await settle();
      const after = await scrollState();
      check(`last_category_stops_without_wrapping_${name}`, bottom.active === 'other' && after.active === 'other'
        && after.selected === 1 && Math.abs(after.top - after.max) <= 1 && Math.abs(after.top - bottom.top) <= 1);
      check(`wheel_stays_inside_palette_${name}`, await page.evaluate(() => scrollY) === pageTop);
    };
    const nativeSelect = async (from, to, activate = false) => page.evaluate(({ from, to, activate }) => {
      const source = document.getElementById('text');
      const view = ScriptEditor.get(source).view;
      view.focus();
      const anchor = view.domAtPos(from), head = view.domAtPos(to);
      document.getSelection().setBaseAndExtent(anchor.node, anchor.offset, head.node, head.offset);
      const pending = { nativeText: document.getSelection().toString(),
        codeMirrorSelectionEmpty: view.state.selection.main.empty };
      if (activate) source.previousElementSibling.querySelector('[data-label="Tags"]').click();
      return pending;
    }, { from, to, activate });
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.waitForFunction(() => document.documentElement.dataset.studioReady === 'true');
    await editor().fill('你好，这是一次本地语音测试。');
    await editor().press('End');
    // Same-turn native selection reproduces a browser selection pending before CM's observer flush.
    const pending = await nativeSelect(9, 13, true);
    check('regression_reproduces_native_selection_ahead_of_codemirror',
      pending.codeMirrorSelectionEmpty && pending.nativeText === '语音测试');
    check('first_open_uses_pending_native_selection',
      await host().locator('[data-tag="whisper"][data-kind="scope"]').isEnabled());
    await host().locator('[data-tag="whisper"][data-kind="scope"]').click();
    check('first_open_wraps_only_native_selected_words',
      await page.inputValue('#text') === '你好，这是一次本地<whisper>语音测试</whisper>。');
    await host().locator('[data-label="Undo"]').click();
    check('tag_change_is_one_undo_step', await page.inputValue('#text') === '你好，这是一次本地语音测试。');

    await editor().press('End'); await editor().press('ArrowLeft');
    await editor().press('Shift+ArrowLeft'); await editor().press('Shift+ArrowLeft');
    await open();
    await search('whisper');
    await host().locator('[data-tag="whisper"][data-kind="scope"]').click();
    check('keyboard_selection_survives_search_focus',
      await page.inputValue('#text') === '你好，这是一次本地语音<whisper>测试</whisper>。');
    await host().locator('[data-label="Undo"]').click();

    await editor().click(); await editor().press('End'); await open();
    check('collapsed_selection_does_not_reuse_old_words',
      await host().locator('[data-tag="whisper"][data-kind="scope"]').isDisabled());
    check('recent_section_is_removed_even_with_legacy_storage',
      !/最近使用|old-recent/.test(await host().locator('.tag-panel').textContent()));
    await search('[');
    const tags = await host().locator('.tag-option[data-kind="event"]').evaluateAll(nodes => nodes.map(node => node.dataset.tag));
    check('all_71_requested_tags_are_searchable', expected.length === 71 && expected.every(value => tags.includes(value)));
    check('catalog_has_no_duplicate_prompt_values', new Set(tags).size === tags.length);
    const chinese = await host().locator('.tag-option[data-kind="event"]').allTextContents();
    check('chinese_palette_has_no_english_only_labels', chinese.every(label => /[\u3400-\u9fff]/.test(label)));
    await search('frustrated');
    check('search_crosses_inactive_categories_with_localized_labels',
      await host().locator('[data-tag="frustrated"]').textContent() === '[挫败]');
    await host().locator('[data-tag="frustrated"]').click();
    check('chinese_ui_inserts_chinese_prompt', (await page.inputValue('#text')).endsWith('[挫败]'));
    await host().locator('[data-label="Undo"]').click();

    await open(); await search('no-such-emotion-zz');
    check('empty_search_has_a_clear_state', await host().locator('.tag-empty').textContent() === '没有匹配的标签');
    await search('');
    check('all_categories_render_in_one_continuous_scroll_container', await choices().locator(':scope > .tag-category-section').evaluateAll(nodes =>
      JSON.stringify(nodes.map(node => node.dataset.category)) === JSON.stringify(['emotions', 'tone', 'sounds', 'effects', 'other'])));
    check('continuous_sections_keep_all_presets_without_filtering', await choices().locator('.tag-option[data-kind="event"]').count() === 74);
    check('category_counts_remain_complete', await choices().locator('.tag-category-section').evaluateAll(nodes => {
      const counts = [49, 6, 11, 5, 3];
      return nodes.every((node, index) => node.querySelectorAll('.tag-option[data-kind="event"]').length === counts[index]);
    }));
    await wheelThroughCategories('desktop');
    for (const category of categories) {
      await jump(category);
      const state = await scrollState();
      check(`tab_jumps_to_section_without_filtering_${category}`, state.active === category && state.selected === 1
        && await targetIsVisible(category) && await choices().locator('.tag-option[data-kind="event"]').count() === 74);
      check(`tab_focus_tracks_active_category_${category}`, await tab(category).getAttribute('tabindex') === '0'
        && await host().locator('.tag-categories > [role="tab"][tabindex="0"]').count() === 1);
    }
    await jump('emotions'); await tab('emotions').press('ArrowRight'); await settle();
    check('keyboard_arrow_jumps_to_next_section', (await scrollState()).active === 'tone'
      && await targetIsVisible('tone') && await tab('tone').evaluate(node => document.activeElement === node));
    await tab('tone').press('End'); await settle();
    check('keyboard_end_jumps_to_last_section', (await scrollState()).active === 'other'
      && await targetIsVisible('other') && await tab('other').evaluate(node => document.activeElement === node));
    await tab('other').press('Home'); await settle();
    check('keyboard_home_returns_to_first_section', (await scrollState()).active === 'emotions'
      && (await scrollState()).top <= 1 && await tab('emotions').evaluate(node => document.activeElement === node));
    await jump('sounds');
    check('sounds_section_is_visible_with_all_categories_retained',
      await choices().locator('.tag-category-section[data-category="sounds"] .tag-option[data-kind="event"]').count() === 11
      && await targetIsVisible('sounds') && await choices().locator('.tag-option[data-kind="event"]').count() === 74);
    await search('laughs');
    check('legacy_drama_sound_names_are_search_aliases', await host().locator('[data-tag="laughing"]').count() === 1);
    check('search_still_matches_across_all_scroll_sections', await host().locator('[data-tag="laughing"]').textContent() === '[大笑]');
    await search('');
    check('clearing_search_restores_all_continuous_sections', await choices().locator('.tag-category-section').count() === categories.length
      && await choices().locator('.tag-option[data-kind="event"]').count() === 74
      && await host().locator('.tag-categories').isVisible());
    await close();
    await editor().fill('选中这里，其他不变。'); await nativeSelect(0, 4, true); await jump('effects');
    await host().locator('[data-tag="fast"][data-kind="scope"]').click();
    check('category_jump_keeps_original_selected_words_for_scope_controls',
      await page.inputValue('#text') === '<fast>选中这里</fast>，其他不变。');
    await host().locator('[data-label="Undo"]').click();
    await editor().fill('自定义标签插入测试。'); await editor().press('Home'); await open(); await jump('other');
    await host().locator('.tag-custom input').fill('轻声微笑');
    await host().locator('.tag-custom button[type="submit"]').click();
    check('custom_prompt_insertion_still_works_after_category_scrolling',
      await page.inputValue('#text') === '[轻声微笑]自定义标签插入测试。');
    await host().locator('[data-label="Undo"]').click();

    const locales = [
      ['zh-CN', 'Happy', '[开心]'], ['en', 'happy', '[happy]'],
      ['ja', 'happy', '[嬉しい]'], ['ko', 'angry', '[화나서]'],
    ];
    for (const [locale, query, inserted] of locales) {
      await page.evaluate(locale => I18n.setLocale(locale), locale);
      await editor().fill('Hello.'); await editor().press('Home'); await open(); await search(query);
      await host().locator(`[data-tag="${query.toLowerCase()}"][data-kind="event"]`).click();
      check(`localized_prompt_${locale}`, await page.inputValue('#text') === inserted + 'Hello.');
      const before = await page.inputValue('#text');
      await page.evaluate(() => I18n.setLocale('en'));
      check(`locale_switch_keeps_existing_prompt_${locale}`, await page.inputValue('#text') === before);
    }

    await page.evaluate(() => I18n.setLocale('zh-CN'));
    await editor().fill('原文选中这一段。');
    await host().locator('[data-label="Source text"]').click();
    await page.locator('#text').evaluate(node => { node.focus(); node.setSelectionRange(2, 7); });
    await open();
    check('raw_text_selection_is_available_on_open',
      await host().locator('[data-tag="whisper"][data-kind="scope"]').isEnabled());
    await host().locator('[data-tag="whisper"][data-kind="scope"]').click();
    check('raw_text_selection_wraps_exact_words', await page.inputValue('#text') === '原文<whisper>选中这一段</whisper>。');

    await editor().fill('[happy]原话。'); await editor().press('End');
    await page.locator('.compose-panel .script-event').click();
    await host().locator('.tag-panel').getByRole('button', { name: '移除标签', exact: true }).click();
    check('legacy_english_tag_removal_keeps_spoken_text', await page.inputValue('#text') === '原话。');
    await editor().fill('只修改我的演绎。'); await nativeSelect(0, 2, true);
    await host().locator('[data-tag="soft"][data-kind="scope"]').click();
    await host().locator('.script-tag-edge').first().click();
    await host().locator('[data-tag="whisper"][data-kind="scope"]').click();
    check('replacing_scope_does_not_nest_or_duplicate_words', await page.inputValue('#text') === '<whisper>只修</whisper>改我的演绎。');

    for (const locale of ['zh-CN', 'en', 'ja', 'ko']) {
      await page.evaluate(locale => I18n.setLocale(locale), locale);
      for (const size of [{ width: 1440, height: 900 }, { width: 390, height: 844 }, { width: 1271, height: 712 }]) {
        await page.setViewportSize(size); await close(); await editor().click(); await open();
        check(`palette_fits_${locale}_${size.width}`, await host().locator('.tag-panel').evaluate(panel => {
          const r = panel.getBoundingClientRect();
          return r.left >= 0 && r.top >= 0 && r.right <= innerWidth + 1 && r.bottom <= innerHeight + 1;
        }));
        check(`palette_controls_fit_${locale}_${size.width}`, await host().locator('.tag-panel').evaluate(panel =>
          Array.from(panel.querySelectorAll('.tag-search,.tag-scopes,.tag-custom')).every(node =>
            node.scrollWidth <= node.clientWidth + 1)));
        check(`category_labels_stay_on_one_line_${locale}_${size.width}`, await host().locator('.tag-categories > [role="tab"]').evaluateAll(nodes =>
          nodes.every(node => getComputedStyle(node).whiteSpace === 'nowrap' && node.scrollWidth <= node.clientWidth + 1)));
        check(`continuous_categories_remain_complete_${locale}_${size.width}`,
          await choices().locator('.tag-category-section').count() === categories.length
          && await choices().locator('.tag-option[data-kind="event"]').count() === 74);
        await jump('other');
        check(`localized_last_tab_jumps_in_view_${locale}_${size.width}`, (await scrollState()).active === 'other'
          && await targetIsVisible('other') && await tab('other').getAttribute('aria-selected') === 'true');
        check(`active_category_is_visible_in_horizontal_navigation_${locale}_${size.width}`, await tab('other').evaluate(node => {
          const r = node.getBoundingClientRect(), nav = node.closest('.tag-categories').getBoundingClientRect();
          return r.left >= nav.left - 1 && r.right <= nav.right + 1;
        }));
        if (locale === 'zh-CN' && size.width === 390) await wheelThroughCategories('mobile');
      }
    }
    await page.setViewportSize({ width: 1271, height: 900 });
    await close(); await page.evaluate(() => I18n.setLocale('zh-CN'));
    await editor().fill('你好，这是一次本地语音测试。'); await nativeSelect(9, 13, true);
    const shot = path.join(folder, 'tag-palette-zh.png');
    await page.screenshot({ path: shot }); report.screenshots.push(shot);
    const pointerPage = await context.newPage();
    pointerPage.on('pageerror', error => report.errors.push(error.stack || error.message));
    await pointerPage.goto(base, { waitUntil: 'networkidle' });
    await pointerPage.waitForFunction(() => document.documentElement.dataset.studioReady === 'true');
    await pointerPage.locator('.compose-panel > .editor-area .cm-content').fill('你好，这是一次本地语音测试。');
    const points = await pointerPage.evaluate(() => {
      const view = ScriptEditor.get(document.getElementById('text')).view;
      const from = view.coordsAtPos(9), to = view.coordsAtPos(13);
      return { x1: from.left + 1, x2: to.left, y: (from.top + from.bottom) / 2 };
    });
    await pointerPage.mouse.move(points.x1, points.y); await pointerPage.mouse.down();
    await pointerPage.mouse.move(points.x2, points.y, { steps: 8 }); await pointerPage.mouse.up();
    await pointerPage.locator('.compose-panel > .editor-area [data-label="Tags"]').click();
    check('first_pointer_selection_keeps_scope_enabled',
      await pointerPage.locator('.compose-panel > .editor-area [data-tag="whisper"][data-kind="scope"]').isEnabled());
    await pointerPage.locator('.compose-panel > .editor-area [data-tag="whisper"][data-kind="scope"]').click();
    check('first_pointer_selection_wraps_exact_words',
      await pointerPage.inputValue('#text') === '你好，这是一次本地<whisper>语音测试</whisper>。');
    await pointerPage.close();
    check('no_browser_exceptions', report.errors.length === 0);
    report.passed = true;
  } catch (error) {
    report.passed = false; report.errors.push(error.stack || String(error)); process.exitCode = 1;
  } finally {
    await browser?.close();
    const target = path.join(root, process.env.DRAMA_TAG_REPORT || 'research/2026-10-20-tag-palette-ui-acceptance.json');
    fs.writeFileSync(target, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify(report, null, 2));
  }
})();
