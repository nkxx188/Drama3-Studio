import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const source = fs.readFileSync(path.join(root, 'ui/editor-source/tag-catalog.js'), 'utf8');
const { tagSections, tagCategories } = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));
const checks = {};
const check = (name, value) => { checks[name] = Boolean(value); assert.ok(value, name); };
const expected = [
  ['happy', 'sad', 'angry', 'excited', 'calm', 'nervous', 'confident', 'surprised', 'satisfied', 'delighted', 'scared', 'worried', 'upset', 'frustrated', 'depressed', 'empathetic', 'embarrassed', 'disgusted', 'moved', 'proud', 'relaxed', 'grateful', 'curious', 'sarcastic'],
  ['disdainful', 'unhappy', 'anxious', 'hysterical', 'indifferent', 'uncertain', 'doubtful', 'confused', 'disappointed', 'regretful', 'guilty', 'ashamed', 'jealous', 'envious', 'hopeful', 'optimistic', 'pessimistic', 'nostalgic', 'lonely', 'bored', 'contemptuous', 'sympathetic', 'compassionate', 'determined', 'resigned'],
  ['in a hurry tone', 'shouting', 'screaming', 'whispering', 'soft tone', 'emphasis'],
  ['laughing', 'chuckling', 'sobbing', 'crying loudly', 'sighing', 'groaning', 'panting', 'gasping', 'yawning', 'snoring', 'clear throat'],
  ['audience laughing', 'background laughter', 'crowd laughing', 'break', 'long-break'],
];
for (let index = 0; index < expected.length; index++) {
  assert.deepEqual(tagSections[index].tags.map(([value]) => value), expected[index]);
  check(`complete_official_section_${index + 1}`, true);
}
check('official_prompt_count_is_71', expected.flat().length === 71);
const entries = tagSections.flatMap(section => section.tags);
check('no_duplicate_canonical_prompt_values', new Set(entries.map(([value]) => value)).size === entries.length);
check('three_existing_extra_drama_cues_are_preserved', entries.length === 74);
check('all_sections_have_a_navigation_category', tagSections.every(section => tagCategories.some(([id]) => id === section.category)));

const context = vm.createContext({
  window: {}, localStorage: { getItem() { return 'zh-CN'; }, setItem() {} },
  navigator: { languages: ['zh-CN'] }, CustomEvent: class { constructor(name) { this.type = name; } },
  document: { documentElement: {}, getElementById() { return null; }, querySelectorAll() { return []; }, dispatchEvent() {} },
});
vm.runInContext(fs.readFileSync(path.join(root, 'ui/i18n.js'), 'utf8'), context);
const i18n = context.window.I18n;
for (const locale of ['en', 'zh-CN', 'ja', 'ko']) {
  i18n.setLocale(locale, false);
  for (const [value, caption] of entries) {
    check(`translated_${locale}_${value}`, i18n.has(caption) && i18n.translations(caption).length === 4);
    const prompt = locale === 'en' ? value : i18n.t(caption);
    check(`valid_prompt_${locale}_${value}`, prompt.length <= 160 && !/[\[\]<>\r\n]/.test(prompt));
    if (locale !== 'en') check(`localized_${locale}_${value}`, prompt !== caption);
  }
  for (const [, caption] of tagCategories) check(`category_translation_${locale}_${caption}`, i18n.has(caption));
}
const bundle = fs.readFileSync(path.join(root, 'ui/script-editor.js'), 'utf8');
check('recent_section_is_not_in_shipped_editor', !/drama3\.recentTags|Recently used/.test(bundle));
const report = { scope: 'Actual structured tag catalog and I18n runtime; no browser or inference.',
  checks, passed: true, official_tags: 71, additional_drama_cues: 3 };
fs.writeFileSync(path.join(root, 'research/2026-10-20-tag-catalog-validation.json'), JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ passed: true, checks: Object.keys(checks).length, official_tags: 71, additional_drama_cues: 3 }));
