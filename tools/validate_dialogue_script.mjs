import assert from 'node:assert/strict';
import { writeFileSync } from 'node:fs';
import { parseDialogue, formatDialogue, reconcileDialogue, maxSpeakers } from '../ui/editor-source/dialogue.js';

const checks = {};
const check = (name, value) => { checks[name] = Boolean(value); assert.ok(value, name); };
const roles = [{ id: 'a', speaker: 0 }, { id: 'b', speaker: 1 }];
let next = 0;
const id = () => `new-${++next}`;
const parsed = parseDialogue('S1：你好。\nS2: Hello.\ns1 ：再见。');
check('ascii_fullwidth_and_lowercase_markers', JSON.stringify(parsed.turns.map(turn => turn.speaker)) === '[0,1,0]');
check('labels_are_not_spoken_text', JSON.stringify(parsed.turns.map(turn => turn.text)) === '["你好。","Hello.","再见。"]');
check('valid_labels_have_no_issues', !parsed.issues.length);
check('speaker_spans_include_only_label', parsed.turns.every(turn =>
  ['S1：', 'S2:', 's1 ：'].includes('S1：你好。\nS2: Hello.\ns1 ：再见。'.slice(turn.from, turn.to))));
const continued = parseDialogue('\n  S1: first\nsecond\n\nS2：last');
check('multiline_utterances_stay_with_their_speaker', continued.turns[0].text === 'first\nsecond');
check('leading_blank_lines_are_safe', !continued.issues.length);
check('unassigned_text_is_an_error', parseDialogue('No speaker.\nS1: Hello.').issues[0].kind === 'unassigned');
check('empty_script_is_valid_draft', !parseDialogue('').issues.length && !parseDialogue('').turns.length);
check('unfinished_turns_are_preserved', parseDialogue('S1:\nS2：').turns.length === 2);
check('invalid_speaker_zero_rejected', parseDialogue('S0: hi').issues[0].kind === 'speaker');
check('invalid_speaker_seventeen_rejected', parseDialogue('S17: hi').issues[0].kind === 'speaker');
check('all_sixteen_speakers_supported', maxSpeakers === 16 && !parseDialogue('S16: hi').issues.length);
check('native_protocol_can_be_pasted', parseDialogue('<|speaker:1|> Hi.').turns[0].speaker === 1);
check('colon_in_dialogue_is_not_a_new_turn', parseDialogue('S1: We call this S2: voice.').turns.length === 1);
check('crlf_is_parsed_without_carriage_returns', parseDialogue('S1: one\r\nS2: two\r\n').turns[0].text === 'one');
check('inline_tags_survive_parsing', parseDialogue('S1: [happy]<whisper>Hello.</whisper>').turns[0].text === '[happy]<whisper>Hello.</whisper>');
const original = [
  { id: 'one', characterId: 'a', text: 'First.', direction: '' },
  { id: 'two', characterId: 'b', text: 'Second.', direction: '' },
  { id: 'three', characterId: 'a', text: 'Last.', direction: '' },
];
const reconcile = script => reconcileDialogue(parseDialogue(script).turns, roles, original, id);
check('unchanged_turn_ids_stay_stable', JSON.stringify(reconcile('S1: First.\nS2: Second.\nS1: Last.').map(row => row.id)) === '["one","two","three"]');
check('editing_one_turn_keeps_its_id', reconcile('S1: Changed.\nS2: Second.\nS1: Last.')[0].id === 'one');
const inserted = reconcile('S2: Added.\nS1: First.\nS2: Second.\nS1: Last.');
check('insertion_does_not_steal_other_turn_ids', inserted[0].id.startsWith('new-') && inserted.slice(1).map(row => row.id).join() === 'one,two,three');
check('reordered_utterances_retain_ids', reconcile('S1: Last.\nS1: First.\nS2: Second.').map(row => row.id).join() === 'three,one,two');
check('deletion_keeps_surviving_ids', reconcile('S2: Second.\nS1: Last.').map(row => row.id).join() === 'two,three');
check('unknown_speaker_is_not_assigned_to_another_voice', reconcileDialogue(parseDialogue('S3: Hi.').turns, roles, [], id).length === 0);
const formatted = formatDialogue([{ id: 'old', characterId: 'b', text: '<|speaker:1|> Hello.', direction: 'soft' }], roles);
check('legacy_line_direction_is_migrated_into_visible_text', formatted === 'S2：[soft] Hello.');
check('legacy_native_marker_is_not_duplicated', parseDialogue(formatted).turns[0].text === '[soft] Hello.');
const roundTrip = formatDialogue(reconcile('S1: First.\nS2: Second.\nS1: Last.'), roles);
check('structured_script_round_trips', JSON.stringify(parseDialogue(roundTrip).turns.map(row => row.text)) === '["First.","Second.","Last."]');
const duplicate = reconcileDialogue(parseDialogue('S1: Changed.\nS1: Same.').turns, roles, [
  { id: 'd1', characterId: 'a', text: 'Same.' }, { id: 'd2', characterId: 'a', text: 'Same.' },
], id);
check('duplicate_utterances_prefer_their_original_position', duplicate.map(row => row.id).join() === 'd1,d2');
const report = { passed: true, checks, scope: 'Dialogue text parsing, migration and stable utterance identities; no inference.' };
writeFileSync(new URL('../research/2026-10-20-dialogue-parser-acceptance.json', import.meta.url), JSON.stringify(report, null, 2));
console.log(JSON.stringify({ passed: true, checks: Object.keys(checks).length }));
