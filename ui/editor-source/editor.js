import { EditorState, Compartment } from '@codemirror/state';
import { EditorView, Decoration, ViewPlugin, keymap, placeholder, drawSelection } from '@codemirror/view';
import { history, historyKeymap, defaultKeymap, undo, redo } from '@codemirror/commands';
import { createElement, Tag, Undo2, Redo2, Code2, X, Search, Languages,
  ArrowLeftRight, Plus, Play, Pause, Trash2 } from 'lucide';
import { tagCategories, tagSections } from './tag-catalog.js';
import { parseDialogue, formatDialogue, reconcileDialogue, maxSpeakers } from './dialogue.js';

const editors = new WeakMap(), mounted = new Set();
const t = (source, args) => window.I18n.t(source, args);
const scopes = [
  ['whisper', 'Whisper'], ['emphasis', 'Emphasis'], ['soft', 'Soft'],
  ['fast', 'Faster'], ['slow', 'Slower'], ['stress', 'Stress'], ['forceful', 'Forceful'],
];
const button = (icon, label) => {
  const node = document.createElement('button'); node.type = 'button';
  node.className = 'editor-tool'; node.dataset.label = label;
  if (icon) node.append(createElement(icon, { width: 18, height: 18, 'aria-hidden': 'true' }));
  return node;
};

function tokens(text) {
  const found = [], stack = [];
  const syntax = /<(\/?)(whisper|emphasis|soft|fast|slow|stress|forceful)>|\[([^\]\n]{1,160})\]/g;
  const phonemes = /<\|phoneme_start\|>([^<>]*?)<\|phoneme_end\|>/g;
  for (const match of text.matchAll(phonemes)) {
    const from = match.index, contentFrom = from + '<|phoneme_start|>'.length;
    const contentTo = contentFrom + match[1].length, to = contentTo + '<|phoneme_end|>'.length;
    found.push({ from, to, contentFrom, contentTo, kind: 'phoneme', value: match[1] });
  }
  for (const match of text.matchAll(syntax)) {
    const from = match.index, to = from + match[0].length;
    if (match[3]) found.push({ from, to, kind: 'event', value: match[3] });
    else if (!match[1]) stack.push({ from, contentFrom: to, value: match[2] });
    else {
      const opening = stack.pop();
      if (opening?.value === match[2]) found.push({
        ...opening, to, contentTo: from, kind: 'scope',
      });
    }
  }
  return found.sort((a, b) => a.from - b.from || b.to - a.to);
}

const tagged = ViewPlugin.fromClass(class {
  constructor(view) { this.decorations = this.make(view); }
  update(value) { if (value.docChanged) this.decorations = this.make(value.view); }
  make(view) {
    return Decoration.set(tokens(view.state.doc.toString()).flatMap(token => {
      const whole = Decoration.mark({ class: `script-capsule script-${token.kind}`, attributes: {
        'data-tag': token.value, title: token.value,
      } }).range(token.from, token.to);
      return token.kind === 'event' ? [whole] : [
        whole, Decoration.mark({ class: 'script-tag-edge' }).range(token.from, token.contentFrom),
        Decoration.mark({ class: 'script-tag-edge' }).range(token.contentTo, token.to),
      ];
    }), true);
  }
}, { decorations: value => value.decorations });

const speakers = ViewPlugin.fromClass(class {
  constructor(view) { this.decorations = this.make(view); }
  update(value) { if (value.docChanged) this.decorations = this.make(value.view); }
  make(view) {
    return Decoration.set(parseDialogue(view.state.doc.toString()).turns.map(turn => Decoration.mark({
      class: 'script-speaker',
      attributes: { 'data-speaker': String(turn.speaker) },
    }).range(turn.from, turn.to)), true);
  }
}, { decorations: value => value.decorations });

function mount(textarea) {
  for (const item of mounted) if (!item.host.isConnected) {
    item.view.destroy(); mounted.delete(item);
  }
  if (editors.has(textarea)) return editors.get(textarea);
  const host = document.createElement('div'); host.className = 'script-editor';
  textarea.before(host); textarea.classList.add('script-raw'); textarea.hidden = true;
  const canvas = document.createElement('div'); canvas.className = 'script-canvas';
  const toolbar = document.createElement('div'); toolbar.className = 'script-toolbar';
  const tags = button(Tag, 'Tags'), label = document.createElement('span');
  label.dataset.tagLabel = 'true'; tags.append(label);
  const back = button(Undo2, 'Undo'), forward = button(Redo2, 'Redo'), raw = button(Code2, 'Source text');
  toolbar.append(tags, back, forward, raw); host.append(canvas, toolbar);
  const panel = document.createElement('div'); panel.className = 'tag-panel'; panel.hidden = true;
  panel.setAttribute('role', 'dialog'); panel.setAttribute('aria-label', t('Tags'));
  const searchRow = document.createElement('div'); searchRow.className = 'tag-search';
  searchRow.append(createElement(Search, { width: 18, height: 18, 'aria-hidden': 'true' }));
  const search = document.createElement('input'); search.type = 'search';
  search.autocomplete = 'off'; search.setAttribute('aria-label', t('Search tags'));
  const close = button(X, 'Close'); searchRow.append(search, close);
  const scopeChoices = document.createElement('div'); scopeChoices.className = 'tag-scopes';
  const pronunciationButton = document.createElement('button');
  pronunciationButton.type = 'button'; pronunciationButton.className = 'tag-option tag-pronunciation-action';
  pronunciationButton.dataset.action = 'phoneme';
  pronunciationButton.append(createElement(Languages, { width: 16, height: 16, 'aria-hidden': 'true' }));
  const pronunciationLabel = document.createElement('span'); pronunciationLabel.textContent = t('Control pronunciation');
  pronunciationButton.append(pronunciationLabel);
  const pronunciationForm = document.createElement('form');
  pronunciationForm.className = 'tag-pronunciation-form'; pronunciationForm.hidden = true;
  const pronunciationSource = document.createElement('div'); pronunciationSource.className = 'tag-pronunciation-source';
  const pronunciationField = document.createElement('label');
  pronunciationField.className = 'tag-pronunciation-field';
  const pronunciationFieldLabel = document.createElement('span');
  pronunciationFieldLabel.textContent = t('Pinyin for selected Chinese characters');
  const pronunciationInput = document.createElement('input');
  pronunciationInput.type = 'text'; pronunciationInput.autocomplete = 'off';
  pronunciationInput.spellcheck = false; pronunciationInput.placeholder = 'chong2 qing4';
  pronunciationInput.setAttribute('aria-label', t('Pinyin for selected Chinese characters'));
  pronunciationField.append(pronunciationFieldLabel, pronunciationInput);
  const pronunciationHelp = document.createElement('p');
  pronunciationHelp.className = 'tag-pronunciation-help';
  pronunciationHelp.textContent = t('Enter one tone-number pinyin syllable per Chinese character, separated by spaces.');
  const pronunciationError = document.createElement('p');
  pronunciationError.className = 'tag-pronunciation-error'; pronunciationError.hidden = true;
  pronunciationError.setAttribute('role', 'status'); pronunciationError.setAttribute('aria-live', 'polite');
  const pronunciationActions = document.createElement('div'); pronunciationActions.className = 'tag-pronunciation-actions';
  const pronunciationSubmit = document.createElement('button');
  pronunciationSubmit.type = 'submit'; pronunciationSubmit.className = 'tag-option';
  pronunciationSubmit.textContent = t('Insert pronunciation');
  const pronunciationCancel = document.createElement('button');
  pronunciationCancel.type = 'button'; pronunciationCancel.className = 'text-btn';
  pronunciationCancel.textContent = t('Cancel');
  pronunciationActions.append(pronunciationSubmit, pronunciationCancel);
  pronunciationForm.append(pronunciationSource, pronunciationField, pronunciationHelp,
    pronunciationError, pronunciationActions);
  const navigation = document.createElement('div'); navigation.className = 'tag-categories';
  navigation.setAttribute('role', 'tablist'); navigation.setAttribute('aria-label', t('Tag categories'));
  const choices = document.createElement('div'); choices.className = 'tag-choices';
  choices.setAttribute('role', 'tabpanel'); choices.id = `tag-choices-${mounted.size}-${textarea.id || 'script'}`;
  const custom = document.createElement('form'); custom.className = 'tag-custom';
  const customInput = document.createElement('input');
  customInput.maxLength = 160; customInput.setAttribute('aria-label', t('Custom direction'));
  const insert = document.createElement('button'); insert.type = 'submit'; insert.className = 'text-btn';
  insert.textContent = t('Insert tag');
  custom.append(customInput, insert);
  panel.append(searchRow, scopeChoices, navigation, choices, custom); host.append(panel);
  const editable = new Compartment(), prompt = new Compartment();
  let syncing = false, isRaw = false, savedSelection = null, activeToken = null, pendingSelection = null;
  let pronunciationRange = null;
  let category = 'emotions', scrollFrame = 0;
  let disabled = textarea.disabled, readOnly = textarea.readOnly, promptText = textarea.placeholder;
  const access = () => [
    EditorState.readOnly.of(disabled || readOnly),
    EditorView.editable.of(!disabled && !readOnly),
    EditorView.contentAttributes.of({
      'aria-readonly': String(readOnly), 'aria-disabled': String(disabled),
      tabindex: disabled ? '-1' : '0',
    }),
  ];
  const stateConfig = {
      extensions: [
        history(), keymap.of([...historyKeymap, ...defaultKeymap]), EditorView.lineWrapping,
        editable.of(access()),
        prompt.of(placeholder(textarea.placeholder)),
        EditorView.contentAttributes.of({ 'aria-label': textarea.getAttribute('aria-label') || t('Text to speak') }),
        tagged, ...(textarea.dataset.dialogue === 'true' ? [speakers, drawSelection()] : []),
        EditorView.updateListener.of(update => {
          if (update.docChanged && !syncing) {
            textarea.value = update.state.doc.toString();
            textarea.dispatchEvent(new Event('input', { bubbles: true }));
          }
        }),
      ],
  };
  const view = new EditorView({
    parent: canvas, state: EditorState.create({ ...stateConfig, doc: textarea.value }),
  });
  const tell = label => {
    host.dataset.notice = t(label);
    host.dispatchEvent(new CustomEvent('drama:editor-notice', { bubbles: true, detail: t(label) }));
  };
  function sync() {
    if (textarea.value !== view.state.doc.toString()) {
      syncing = true;
      view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: textarea.value } });
      syncing = false;
    }
    const effects = [];
    if (disabled !== textarea.disabled || readOnly !== textarea.readOnly) {
      disabled = textarea.disabled; readOnly = textarea.readOnly;
      effects.push(editable.reconfigure(access()));
    }
    if (promptText !== textarea.placeholder) {
      promptText = textarea.placeholder; effects.push(prompt.reconfigure(placeholder(promptText)));
    }
    if (effects.length) view.dispatch({ effects });
    tags.disabled = back.disabled = forward.disabled = disabled || readOnly;
    raw.disabled = disabled;
    host.dataset.readonly = String(readOnly);
    if ((disabled || readOnly) && !panel.hidden) closePanel(false);
  }
  function reset() {
    closePanel(false); syncing = true;
    view.setState(EditorState.create({ ...stateConfig, doc: textarea.value }));
    view.dispatch({ effects: [
      editable.reconfigure(access()), prompt.reconfigure(placeholder(textarea.placeholder)),
    ] });
    syncing = false; textarea.setSelectionRange(0, 0); sync(); translate();
  }
  function append(text) {
    if (textarea.disabled || textarea.readOnly) return;
    sync();
    const from = view.state.doc.length, addition = (from && !view.state.doc.toString().endsWith('\n') ? '\n' : '') + text;
    const cursor = from + addition.length;
    view.dispatch({ changes: { from, insert: addition }, selection: { anchor: cursor },
      scrollIntoView: true, userEvent: 'input' });
    if (isRaw) { textarea.focus(); textarea.setSelectionRange(cursor, cursor); }
    else view.focus();
  }
  function closePanel(focus = true) {
    panel.hidden = true; tags.setAttribute('aria-expanded', 'false');
    pendingSelection = savedSelection = activeToken = null;
    pronunciationRange = null; pronunciationForm.hidden = true; pronunciationError.hidden = true;
    if (focus) view.focus();
  }
  function captureSelection() {
    if (isRaw) {
      const from = textarea.selectionStart, to = textarea.selectionEnd;
      return { from, to, empty: from === to };
    }
    // Browser/extension selections can arrive before CodeMirror's selection observer runs.
    const native = view.dom.ownerDocument.getSelection();
    if (native?.anchorNode && native.focusNode
        && view.contentDOM.contains(native.anchorNode) && view.contentDOM.contains(native.focusNode)) {
      const anchor = view.posAtDOM(native.anchorNode, native.anchorOffset);
      const head = view.posAtDOM(native.focusNode, native.focusOffset);
      return { from: Math.min(anchor, head), to: Math.max(anchor, head), empty: anchor === head };
    }
    return view.state.selection.main;
  }
  function crossesDialogueTurn(from, to) {
    if (textarea.dataset.dialogue !== 'true') return false;
    const selected = view.state.doc.sliceString(from, to);
    return selected.includes('\n')
      || /(?:^|\n)\s*(?:[Ss]\d+\s*[:：]|<\|speaker:\d+\|>)/.test(selected);
  }
  function apply(value, kind) {
    if (view.composing || textarea.disabled || textarea.readOnly) return;
    const doc = view.state.doc.toString();
    const selection = savedSelection || view.state.selection.main;
    let from = selection.from, to = selection.to;
    const current = activeToken || tokens(doc).find(token => token.kind === 'scope'
      && ((token.from === from && token.to === to) || (token.contentFrom === from && token.contentTo === to)));
    let changes, cursor;
    if (kind === 'remove' && activeToken) {
      const token = activeToken;
      const content = token.kind === 'scope' ? doc.slice(token.contentFrom, token.contentTo) : '';
      changes = { from: token.from, to: token.to, insert: content }; cursor = token.from + content.length;
    } else if (kind === 'scope') {
      if (current?.kind === 'scope') { from = current.from; to = current.to; }
      if (from === to) { tell('Select words first'); return; }
      if (crossesDialogueTurn(from, to)) {
        tell('Select words within one speaker’s turn.'); return;
      }
      const overlapping = tokens(doc).filter(token => token.kind === 'scope' && token.from < to && token.to > from);
      if (overlapping.some(token => token.from !== current?.from || token.to !== current?.to)) {
        tell('Select one marked range'); return;
      }
      const content = current?.kind === 'scope'
        ? doc.slice(current.contentFrom, current.contentTo) : doc.slice(from, to);
      const opening = `<${value}>`, closing = `</${value}>`;
      changes = { from, to, insert: opening + content + closing }; cursor = from + changes.insert.length;
    } else {
      if (activeToken?.kind === 'event') {
        from = activeToken.from; to = activeToken.to;
      } else to = from;
      changes = { from, to, insert: `[${value}]` }; cursor = from + changes.insert.length;
    }
    view.dispatch({ changes, selection: { anchor: cursor }, userEvent: 'input.tag' });
    closePanel(); activeToken = null; savedSelection = null;
  }
  function openPronunciation() {
    if (!/^zh(?:-|$)/i.test(window.I18n.locale) || view.composing) return;
    const doc = view.state.doc.toString(), selection = savedSelection || view.state.selection.main;
    const token = activeToken?.kind === 'phoneme' ? activeToken : null;
    const from = token?.from ?? selection.from, to = token?.to ?? selection.to;
    const selectedText = doc.slice(from, to);
    if (!token && selection.empty) { tell('Select Chinese text first.'); return; }
    if (crossesDialogueTurn(from, to)) {
      tell('Select Chinese text within one speaker’s turn.'); return;
    }
    pronunciationRange = { from, to, token };
    pronunciationSource.textContent = token ? token.value : selectedText;
    pronunciationInput.value = token ? token.value : '';
    pronunciationError.hidden = true;
    pronunciationForm.hidden = false;
    pronunciationButton.after(pronunciationForm);
    pronunciationInput.focus();
  }
  function applyPronunciation(event) {
    event.preventDefault();
    if (view.composing || textarea.disabled || textarea.readOnly || !pronunciationRange) return;
    const { from, to, token } = pronunciationRange;
    const doc = view.state.doc.toString(), selectedText = doc.slice(from, to);
    const syllables = pronunciationInput.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
    const hanzi = Array.from(selectedText).filter(character => /\p{Script=Han}/u.test(character));
    const fail = message => {
      pronunciationError.textContent = t(message);
      pronunciationError.hidden = false;
    };
    if (!syllables.length || syllables.some(value => !/^[a-züvê]+[1-5]$/u.test(value))) {
      fail('Use lowercase pinyin syllables with tone numbers from 1 to 5.');
      return;
    }
    if (token) {
      if (syllables.length !== 1) {
        fail('Enter one pinyin syllable for the selected character.');
        return;
      }
    } else {
      if (!hanzi.length) { fail('Select Chinese text first.'); return; }
      if (syllables.length !== hanzi.length) {
        fail('Enter one pinyin syllable for each selected Chinese character.');
        return;
      }
      if (tokens(doc).some(item => item.kind === 'phoneme' && item.from < to && item.to > from)) {
        fail('The selection contains an existing pronunciation tag. Edit that tag separately.');
        return;
      }
    }
    let replacement;
    if (token) {
      replacement = `<|phoneme_start|>${syllables[0]}<|phoneme_end|>`;
    } else {
      let index = 0;
      replacement = Array.from(selectedText, character => /\p{Script=Han}/u.test(character)
        ? `<|phoneme_start|>${syllables[index++]}<|phoneme_end|>` : character).join('');
    }
    view.dispatch({ changes: { from, to, insert: replacement },
      selection: { anchor: from + replacement.length }, userEvent: 'input.phoneme' });
    closePanel();
  }
  function positionPanel(realign = false) {
    if (panel.hidden) return;
    const viewport = window.visualViewport;
    const width = viewport?.width || innerWidth, height = viewport?.height || innerHeight;
    const originX = viewport?.offsetLeft || 0, originY = viewport?.offsetTop || 0;
    const margin = 12, panelWidth = Math.min(570, width - margin * 2);
    const anchor = tags.getBoundingClientRect();
    const below = originY + height - anchor.bottom - margin - 6;
    const above = anchor.top - originY - margin - 6;
    const useBelow = below >= 460 || below >= above;
    const space = Math.max(below, above);
    const compact = width <= 700;
    const maxHeight = Math.min(580, compact || space < 350 ? height - margin * 2 : (useBelow ? below : above));
    panel.style.width = `${panelWidth}px`;
    panel.style.maxHeight = `${maxHeight}px`;
    const panelHeight = panel.getBoundingClientRect().height;
    const top = compact ? originY + height - panelHeight - margin : space < 350 ? originY + margin
      : useBelow ? anchor.bottom + 6 : anchor.top - 6 - panelHeight;
    panel.style.left = `${Math.max(originX + margin, Math.min(anchor.left, originX + width - panelWidth - margin))}px`;
    panel.style.top = `${Math.max(originY + margin, top)}px`;
    const listHeight = `${choices.clientHeight}px`;
    if (choices.style.getPropertyValue('--tag-list-height') !== listHeight) {
      choices.style.setProperty('--tag-list-height', listHeight);
    }
    if (realign) jumpCategory(category);
  }
  function selectCategory(value) {
    category = value;
    for (const tab of navigation.children) {
      const selected = tab.dataset.category === category;
      tab.setAttribute('aria-selected', String(selected)); tab.tabIndex = selected ? 0 : -1;
      if (selected) {
        const bounds = tab.getBoundingClientRect(), frame = navigation.getBoundingClientRect();
        if (bounds.left < frame.left) navigation.scrollLeft -= frame.left - bounds.left;
        else if (bounds.right > frame.right) navigation.scrollLeft += bounds.right - frame.right;
      }
    }
  }
  function jumpCategory(value) {
    const section = choices.querySelector(`[data-category="${value}"]`);
    if (!section || search.value.trim()) return;
    const inset = parseFloat(getComputedStyle(choices).paddingTop) || 0;
    const top = choices.scrollTop + section.getBoundingClientRect().top - choices.getBoundingClientRect().top - inset;
    choices.scrollTo({ top: Math.max(0, top), behavior: 'instant' });
    selectCategory(value);
  }
  function followCategory() {
    scrollFrame = 0;
    if (panel.hidden || search.value.trim()) return;
    const edge = choices.getBoundingClientRect().top + (parseFloat(getComputedStyle(choices).paddingTop) || 0) + 4;
    let current = tagCategories[0][0];
    for (const section of choices.querySelectorAll('.tag-category-section')) {
      if (section.getBoundingClientRect().top > edge) break;
      current = section.dataset.category;
    }
    selectCategory(current);
  }
  function drawChoices() {
    choices.replaceChildren(); scopeChoices.replaceChildren();
    const selection = savedSelection || view.state.selection.main;
    const chineseUI = /^zh(?:-|$)/i.test(window.I18n.locale);
    const query = search.value.trim().normalize('NFKC').toLowerCase();
    const append = (container, title, items, kind) => {
      const matches = items.filter(([value, caption, aliases = []]) =>
        [value, `[${value}]`, ...window.I18n.translations(caption), ...aliases].join(' ')
          .normalize('NFKC').toLowerCase().includes(query));
      if (!matches.length) return 0;
      const heading = document.createElement('h3'); heading.textContent = t(title); container.append(heading);
      const row = document.createElement('div'); row.className = 'tag-group';
      for (const [value, caption] of matches) {
        const option = document.createElement('button'); option.type = 'button';
        option.className = kind === 'scope' ? 'tag-option' : 'tag-option tag-event-option';
        option.dataset.tag = value; option.dataset.kind = kind;
        const promptValue = kind === 'scope' || window.I18n.locale === 'en' ? value : t(caption);
        option.textContent = kind === 'scope' ? t(caption) : `[${promptValue}]`;
        option.dataset.prompt = promptValue;
        option.title = kind === 'scope' ? `<${value}>${t('Selected words')}</${value}>` : `[${value}]`;
        option.disabled = kind === 'scope' && selection.empty && activeToken?.kind !== 'scope';
        if (option.disabled) option.title = t('Select words first');
        option.onclick = () => apply(promptValue, kind); row.append(option);
      }
      container.append(row);
      return matches.length;
    };
    const scopeCount = append(scopeChoices, 'Control selected words', scopes, 'scope');
    const selectionText = selection.empty ? '' : view.state.doc.sliceString(selection.from, selection.to);
    const hasChineseText = /\p{Script=Han}/u.test(selectionText);
    const showPronunciation = chineseUI && (hasChineseText || activeToken?.kind === 'phoneme');
    pronunciationButton.hidden = !showPronunciation;
    if (!chineseUI) {
      pronunciationForm.hidden = true;
      pronunciationRange = null;
    }
    scopeChoices.append(pronunciationButton);
    if (!pronunciationForm.hidden && showPronunciation) scopeChoices.append(pronunciationForm);
    scopeChoices.hidden = !scopeCount && !showPronunciation;
    navigation.hidden = Boolean(query);
    choices.classList.toggle('tag-search-results', Boolean(query));
    selectCategory(category);
    let eventCount = 0;
    for (const [value] of tagCategories) {
      const group = document.createElement('section');
      group.className = 'tag-category-section'; group.dataset.category = value;
      let count = 0;
      for (const section of tagSections.filter(item => item.category === value)) {
        count += append(group, section.title, section.tags, 'event');
      }
      if (count) choices.append(group);
      eventCount += count;
    }
    if (!scopeCount && !eventCount) {
      const empty = document.createElement('p'); empty.className = 'tag-empty';
      empty.textContent = t('No matching tags'); choices.append(empty);
    }
    if (activeToken && activeToken.kind !== 'phoneme') {
      const remove = document.createElement('button'); remove.type = 'button'; remove.className = 'text-btn';
      remove.textContent = t('Remove tag'); remove.onclick = () => apply('', 'remove'); scopeChoices.append(remove);
      scopeChoices.hidden = false;
    }
    choices.scrollTop = 0;
    positionPanel();
    if (!query) jumpCategory(category);
  }
  function open(token = null) {
    if (textarea.disabled || textarea.readOnly || view.composing) return;
    const selection = pendingSelection || captureSelection(); pendingSelection = null;
    if (isRaw) {
      isRaw = false; textarea.hidden = true; canvas.hidden = false; raw.setAttribute('aria-pressed', 'false'); sync();
    }
    activeToken = token; savedSelection = selection;
    if (token) savedSelection = { from: token.from, to: token.to, empty: false };
    if (token?.kind === 'event') {
      const section = tagSections.find(item => item.tags.some(([value, caption, aliases = []]) =>
        [value, ...window.I18n.translations(caption), ...aliases].includes(token.value)));
      if (section) category = section.category;
    }
    search.value = ''; panel.hidden = false; tags.setAttribute('aria-expanded', 'true'); drawChoices();
    search.focus();
  }
  for (const [value, caption] of tagCategories) {
    const tab = document.createElement('button'); tab.type = 'button'; tab.className = 'tag-category';
    tab.dataset.category = value; tab.dataset.caption = caption;
    tab.setAttribute('role', 'tab'); tab.setAttribute('aria-controls', choices.id);
    tab.onclick = () => jumpCategory(value);
    tab.onkeydown = event => {
      const directions = { ArrowLeft: -1, ArrowRight: 1, Home: -Infinity, End: Infinity };
      if (!(event.key in directions)) return;
      event.preventDefault();
      const offset = directions[event.key], index = tagCategories.findIndex(([item]) => item === value);
      const next = offset === -Infinity ? 0 : offset === Infinity ? tagCategories.length - 1
        : (index + offset + tagCategories.length) % tagCategories.length;
      jumpCategory(tagCategories[next][0]); navigation.children[next].focus({ preventScroll: true });
    };
    navigation.append(tab);
  }
  tags.addEventListener('pointerdown', event => {
    if (event.button !== 0 || textarea.disabled || textarea.readOnly || view.composing) return;
    pendingSelection = captureSelection();
    if (event.pointerType !== 'touch') event.preventDefault();
  });
  tags.addEventListener('keydown', event => {
    if (event.key === 'Enter' || event.key === ' ') pendingSelection = captureSelection();
  });
  tags.onclick = () => panel.hidden ? open() : closePanel();
  close.onclick = () => closePanel();
  back.onclick = () => { if (!textarea.disabled && !textarea.readOnly) undo(view); view.focus(); };
  forward.onclick = () => { if (!textarea.disabled && !textarea.readOnly) redo(view); view.focus(); };
  raw.onclick = () => {
    closePanel(false); isRaw = !isRaw; textarea.hidden = !isRaw; canvas.hidden = isRaw;
    raw.setAttribute('aria-pressed', String(isRaw));
    if (isRaw) {
      textarea.focus(); const range = view.state.selection.main; textarea.setSelectionRange(range.from, range.to);
    } else {
      const from = textarea.selectionStart, to = textarea.selectionEnd; sync();
      view.dispatch({ selection: { anchor: from, head: to } }); view.focus();
    }
  };
  textarea.addEventListener('input', () => { if (isRaw) sync(); });
  choices.addEventListener('scroll', () => {
    if (!scrollFrame) scrollFrame = requestAnimationFrame(followCategory);
  });
  search.oninput = drawChoices;
  pronunciationButton.onclick = openPronunciation;
  pronunciationForm.onsubmit = applyPronunciation;
  pronunciationCancel.onclick = () => {
    pronunciationRange = null; pronunciationForm.hidden = true; pronunciationError.hidden = true;
    pronunciationButton.focus();
  };
  custom.onsubmit = event => {
    event.preventDefault();
    const value = customInput.value.trim();
    if (!value || /[\[\]\r\n<>]/.test(value)) return;
    apply(value, 'event'); customInput.value = '';
  };
  panel.addEventListener('keydown', event => {
    if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); closePanel(); }
  });
  document.addEventListener('pointerdown', event => {
    if (!panel.hidden && !panel.contains(event.target) && !tags.contains(event.target)) closePanel(false);
  });
  canvas.addEventListener('dblclick', event => {
    const target = event.target.closest('.script-capsule');
    if (!target || view.composing) return;
    const offset = view.posAtDOM(target);
    const token = tokens(view.state.doc.toString()).find(item => item.from <= offset && item.to > offset);
    if (token) { event.preventDefault(); open(token); }
  });
  canvas.addEventListener('click', event => {
    const target = event.target.closest('.script-tag-edge,.script-event,.script-phoneme');
    if (!target || view.composing) return;
    const offset = view.posAtDOM(target);
    const token = tokens(view.state.doc.toString()).find(item => item.from <= offset && item.to > offset);
    if (token) { event.preventDefault(); open(token); }
  });
  function translate() {
    for (const node of toolbar.querySelectorAll('[data-label]')) {
      node.title = t(node.dataset.label); node.setAttribute('aria-label', t(node.dataset.label));
    }
    label.textContent = t('Tags'); search.placeholder = t('Search tags');
    search.setAttribute('aria-label', t('Search tags'));
    view.contentDOM.setAttribute('aria-label', textarea.getAttribute('aria-label') || t('Text to speak'));
    customInput.placeholder = t('Custom direction'); insert.textContent = t('Insert tag');
    pronunciationLabel.textContent = t('Control pronunciation');
    pronunciationFieldLabel.textContent = t('Pinyin for selected Chinese characters');
    pronunciationInput.setAttribute('aria-label', t('Pinyin for selected Chinese characters'));
    pronunciationHelp.textContent = t('Enter one tone-number pinyin syllable per Chinese character, separated by spaces.');
    pronunciationSubmit.textContent = t('Insert pronunciation'); pronunciationCancel.textContent = t('Cancel');
    customInput.setAttribute('aria-label', t('Custom direction')); close.setAttribute('aria-label', t('Close'));
    panel.setAttribute('aria-label', t('Tags')); close.title = t('Close');
    navigation.setAttribute('aria-label', t('Tag categories'));
    for (const node of navigation.children) node.textContent = t(node.dataset.caption);
    if (!panel.hidden) drawChoices();
  }
  const editor = { view, sync, reset, append, host, translate, open, positionPanel,
    tokens: () => tokens(view.state.doc.toString()) };
  editors.set(textarea, editor); mounted.add(editor); sync(); translate(); return editor;
}

window.ScriptEditor = Object.freeze({
  mount, get: textarea => editors.get(textarea),
  sync: textarea => editors.get(textarea)?.sync(),
  reset: textarea => editors.get(textarea)?.reset(),
  append: (textarea, text) => editors.get(textarea)?.append(text),
  dialogue: Object.freeze({ parse: parseDialogue, format: formatDialogue, reconcile: reconcileDialogue, maxSpeakers }),
  icon: name => createElement({ switch: ArrowLeftRight, plus: Plus, play: Play, pause: Pause, remove: Trash2 }[name],
    { width: 18, height: 18, 'aria-hidden': 'true' }),
  translate: () => { for (const editor of mounted) editor.translate(); },
});
const repositionPanels = event => {
  for (const editor of mounted) if (editor.host.isConnected) editor.positionPanel(event?.type === 'resize');
};
window.addEventListener('resize', repositionPanels);
window.addEventListener('scroll', repositionPanels, true);
window.visualViewport?.addEventListener('resize', repositionPanels);
