/* Edit an immutable take on the page; local inputs describe only the changed words. */
(() => {
  'use strict';
  const copy = value => JSON.parse(JSON.stringify(value));
  let hooks, active, opening = false, saved = { entries: {} }, saving = Promise.resolve(), timer;
  const svg = name => `<svg class="icon" aria-hidden="true"><use href="#i-${name}"></use></svg>`;
  const el = (tag, className = '') => {
    const value = document.createElement(tag); value.className = className; return value;
  };
  const label = (tag, source, className = '') => {
    const value = el(tag, className); value.dataset.i18n = source; value.textContent = t(source); return value;
  };
  const button = (source, action, className = 'secondary-btn') => {
    const value = label('button', source, className); value.type = 'button'; value.onclick = action; return value;
  };
  const field = (source, input) => {
    const box = el('div'), caption = label('label', source, 'field-label');
    caption.htmlFor = input.id; box.append(caption, input); return box;
  };
  const control = (id, value = '', type = 'text') => {
    const input = el(type === 'textarea' ? 'textarea' : 'input', type === 'textarea' ? 'compact-input' : 'input');
    input.id = id; if (type !== 'textarea') input.type = type;
    if (type === 'number') UiNumber.write(input, value); else input.value = value;
    return input;
  };
  const singleText = value => value.replace(/^\s*<\|speaker:\d+\|>\s*/, '')
    .replace(/^Direction:[^\n]*\nLine:\s*/i, '');
  function persist() {
    clearTimeout(timer);
    const body = JSON.stringify({ key: 'audio_editor', value: copy(saved) });
    saving = saving.catch(() => {}).then(() => api('/api/studio/draft', {
      method: 'POST', keepalive: new Blob([body]).size < 60000, body,
    }));
    return saving;
  }
  const saveSoon = () => { clearTimeout(timer); timer = setTimeout(() => persist().catch(() => {}), 400); };
  function difference(before, after) {
    let first = 0, last = 0;
    while (first < Math.min(before.length, after.length) && before[first] === after[first]) first++;
    while (last < Math.min(before.length, after.length) - first
      && before[before.length - last - 1] === after[after.length - last - 1]) last++;
    return { first, last, removed: before.slice(first, before.length - last),
      added: after.slice(first, after.length - last) };
  }
  function drawDiff(box, before, after) {
    const { first, last, removed, added } = difference(before, after);
    box.replaceChildren(document.createTextNode(after.slice(0, first)));
    if (removed) { const old = el('del'); old.textContent = removed; box.append(old); }
    if (added) { const next = el('ins'); next.textContent = added; box.append(next); }
    if (last) box.append(document.createTextNode(after.slice(after.length - last)));
    box.hidden = before === after;
  }
  function safeRange(text, start, end, replacement) {
    if (/<\|[^>]*\|>/.test(replacement)) return false;
    // Do not cut a protocol token or leave a scope open across the replacement.
    for (const match of text.matchAll(/<\|[^>]*\|>|<\/?[A-Za-z][^>]*>|\[[^\]\n]*\]/g)) {
      if ([start, end].some(at => at > match.index && at < match.index + match[0].length)) return false;
    }
    const balanced = value => {
      const stack = [];
      for (const match of value.matchAll(/<(\/?)(whisper|emphasis|soft|fast|slow|stress|forceful)>/g)) {
        if (!match[1]) stack.push(match[2]);
        else if (stack.pop() !== match[2]) return false;
      }
      return !stack.length;
    };
    return balanced(text.slice(start, end)) && balanced(replacement);
  }
  async function open(source, { selection = null, mode = null, returnView: requestedReturnView = null } = {}) {
    if (opening) return;
    const requestedPage = ['generate', 'scene'].includes(requestedReturnView) ? requestedReturnView
      : ['generate', 'scene'].includes(hooks.currentView?.()) ? hooks.currentView()
        : ['generate', 'scene'].includes(active?.returnView) ? active.returnView : 'generate';
    if (active) {
      if (JSON.stringify(active.source) === JSON.stringify(source) || active.busy) {
        hooks.rememberSource?.(active.source, active.returnView);
        hooks.enterEditor?.(active.returnView);
        active.element.scrollIntoView({ block: 'start', behavior: 'smooth' }); active.element.focus(); return;
      }
      if (!await active.close()) return;
    }
    opening = true;
    let version, returnView;
    try {
      await hooks.beforeOpen();
      version = await hooks.version(source);
      returnView = requestedPage;
    } finally { opening = false; }
    const recipe = version.recipe || {};
    if (!version.take_file || version.take_missing) throw new Error(t('No editable take'));
    const joint = recipe.scene_strategy === 'joint' && recipe.characters?.length && recipe.lines?.length
      && source.part_index === undefined;
    const roles = joint ? recipe.characters : [];
    const sourceSpeaker = version.text.match(/^\s*<\|speaker:(\d+)\|>/)?.[1];
    const originalLines = joint ? recipe.lines.map((line, index) => ({
      ...copy(line), key: recipe.line_ids?.[index] || crypto.randomUUID(),
    })) : [{ text: singleText(version.text) }];
    const identity = `${source.job_id}:${source.part_index ?? 'all'}`;
    const draft = saved.entries[identity] || {
      source: copy(source), mode: mode || (selection ? 'local' : 'whole'), range: null,
      local: { schema: 2, quote: '', replacement: '', target: null },
      whole: { lines: copy(originalLines), direction: recipe.direction || '', references: {}, checked: false },
      parameters: {
        maximum: String((recipe.max_new_tokens || Number(document.querySelector('#max-frames').value) || 768) * 2048 / 44100),
        duration: '', temperature: String(recipe.temperature ?? document.querySelector('#temperature').value),
        top_p: String(recipe.top_p ?? document.querySelector('#top-p').value),
        top_k: String(recipe.top_k ?? document.querySelector('#top-k').value),
        seed: String(recipe.seed ?? 123), random: true,
      },
    };
    // Migrate the actual text difference, never a guessed word/audio alignment.
    if (draft.local.lines) {
      draft.previous_local = copy(draft.local);
      const changed = draft.local.lines.map((line, index) => ({ line, index }))
        .filter(({ line, index }) => originalLines[index]?.text !== line.text);
      const local = { schema: 2, quote: '', replacement: '', target: null };
      if (changed.length === 1 && originalLines[changed[0].index]) {
        const { line, index } = changed[0], before = originalLines[index].text;
        const diff = difference(before, line.text);
        if (diff.first || diff.last) Object.assign(local, { quote: diff.removed, replacement: diff.added,
          target: { line: index, start: diff.first, end: before.length - diff.last } });
      }
      draft.local = local;
    }
    if (draft.local.schema !== 2) {
      // Earlier UI revisions treated a snippet typed into the full-text box
      // as a deletion of the entire transcript. Do not revive that mapping.
      if (originalLines.some(line => line.text === draft.local.quote)
        && !difference(draft.local.quote, draft.local.replacement).first
        && !difference(draft.local.quote, draft.local.replacement).last) {
        draft.previous_local = copy(draft.local);
        draft.local = { quote: '', replacement: '', target: null };
      }
      draft.local.schema = 2;
    }
    draft.whole ||= draft.record || { lines: copy(originalLines), direction: recipe.direction || '' };
    draft.whole.references ||= {};
    if (draft.continue) { draft.retired_continuation = draft.continue; delete draft.continue; }
    if (!['local', 'whole'].includes(draft.mode)) draft.mode = selection ? 'local' : 'whole';
    if (mode && ['local', 'whole'].includes(mode)) draft.mode = mode;
    if (joint) draft.whole.lines.forEach((line, index) => { line.key ||= originalLines[index]?.key || crypto.randomUUID(); });
    saved.entries[identity] = draft;
    for (const key of Object.keys(saved.entries).filter(key => key !== identity).slice(0, -5)) delete saved.entries[key];
    if (selection) { draft.range = copy(selection); draft.mode = 'local'; }
    const workspace = el('section', 'panel audio-editor-workspace');
    workspace.id = 'audio-edit-workspace'; workspace.tabIndex = -1;
    workspace.setAttribute('aria-labelledby', 'audio-edit-title');
    const heading = el('div', 'audio-edit-heading'), title = label('h2', 'Source audio');
    title.id = 'audio-edit-title';
    const tools = el('div', 'audio-edit-heading-actions');
    const dismiss = button('Back', () => closeEditor(), 'text-btn'); dismiss.id = 'audio-edit-back';
    delete dismiss.dataset.i18n;
    dismiss.innerHTML = `${svg('arrow')}<span data-i18n="Back">${t('Back')}</span>`;
    dismiss.querySelector('.icon').classList.add('audio-edit-back-icon');
    tools.append(dismiss); heading.append(title, tools);
    const context = { element: workspace, source, version, draft, returnView, busy: false };
    active = context;
    hooks.rememberSource?.(source, returnView);
    const body = el('div', 'audio-edit-body');
    const tabs = el('div', 'audio-edit-tabs'); tabs.setAttribute('role', 'tablist');
    tabs.setAttribute('aria-label', t('Modification scope')); tabs.dataset.i18nAriaLabel = 'Modification scope';
    const media = el('section', 'audio-edit-source');
    const audio = el('audio'); audio.id = 'audio-edit-source'; audio.controls = true;
    audio.preload = 'metadata'; audio.src = fileUrl(version.output);
    audio.dataset.selection = 'true'; audio.dataset.selectionRequired = 'true';
    const secondsPerFrame = version.take.samples_per_frame / version.take.sample_rate;
    audio.dataset.frameSeconds = secondsPerFrame;
    const localActions = el('div', 'audio-edit-range-actions');
    const insert = button('Insert at playhead', () => {
      StudioAudio.setSelection(audio, audio.currentTime, audio.currentTime, { emit: true });
      draft.local.target = null; locator.open = true; refresh(); saveSoon();
    }, 'text-btn'); insert.id = 'audio-edit-insert';
    const rangeDetails = el('details', 'audio-edit-exact'), bounds = el('div', 'field-grid');
    const start = control('audio-edit-start', '', 'number'), end = control('audio-edit-end', '', 'number');
    for (const input of [start, end]) { input.min = '0'; input.step = 'any'; }
    bounds.append(field('Start seconds', start), field('End seconds', end));
    rangeDetails.append(label('summary', 'Exact selection'), bounds);
    localActions.append(insert, rangeDetails); media.append(audio, localActions);
    const localFields = el('section', 'audio-edit-local-fields');
    const quote = control('audio-edit-quote', draft.local.quote, 'textarea'); quote.rows = 2;
    quote.setAttribute('aria-describedby', 'audio-edit-mapping-help');
    const matches = el('div', 'audio-edit-matches'); matches.id = 'audio-edit-matches';
    const mappingHelp = label('p', 'Audio and words are matched manually. Choose the words heard in the selection.',
      'field-help'); mappingHelp.id = 'audio-edit-mapping-help';
    const locator = el('details', 'audio-edit-locator'); locator.id = 'audio-edit-locator';
    locator.append(label('summary', 'Locate in source text'));
    originalLines.forEach((line, index) => {
      const row = el('div', 'audio-edit-original-line');
      const sourceText = control(`audio-edit-original-${index}`, line.text, 'textarea');
      // Keep native caret navigation for insertion and keyboard selection.
      // A native readonly textarea may navigate the page instead of its caret.
      sourceText.setAttribute('aria-readonly', 'true');
      sourceText.addEventListener('beforeinput', event => event.preventDefault());
      sourceText.addEventListener('input', () => { sourceText.value = line.text; });
      const caption = label('label', 'Original text', 'field-label'); caption.htmlFor = sourceText.id;
      if (joint) { delete caption.dataset.i18n; caption.textContent = line.character; }
      const use = button('Use selected words', () => {
        const a = sourceText.selectionStart, b = sourceText.selectionEnd;
        if (isInsertion()) {
          draft.local.target = { line: index, start: b, end: b };
          draft.local.quote = ''; quote.value = ''; locator.open = false;
        } else {
          if (a === b) return;
          setQuote(line.text.slice(a, b)); draft.local.target = { line: index, start: a, end: b };
          locator.open = false;
        }
        drawMatches(); refresh(); saveSoon();
      }, 'text-btn');
      const update = () => {
        const insertion = isInsertion();
        use.dataset.i18n = insertion ? 'Insert at cursor' : 'Use selected words';
        use.textContent = t(use.dataset.i18n);
        use.disabled = context.busy || (!insertion && sourceText.selectionStart === sourceText.selectionEnd);
      };
      for (const event of ['select', 'keyup', 'click']) sourceText.addEventListener(event, update);
      row.append(caption, sourceText, use); locator.append(row); row.updateSelection = update;
    });
    const replacement = control('audio-edit-text-0', draft.local.replacement, 'textarea');
    replacement.setAttribute('aria-label', t('Replacement words')); replacement.dataset.i18nAriaLabel = 'Replacement words';
    const replacementField = field('Replacement words', replacement); replacementField.className = 'audio-edit-replacement';
    const diff = el('div', 'transcript-diff'); diff.id = 'audio-edit-diff'; diff.hidden = true;
    localFields.append(field('Original words in selection', quote), mappingHelp, matches, locator, replacementField, diff);
    const wholeFields = el('section', 'audio-edit-whole-fields');
    const scripts = el('div', 'audio-edit-script'); scripts.id = 'audio-edit-scripts';
    const addLine = button('Add line', () => {
      draft.whole.lines.push({ character: roles[0].name, text: '', key: crypto.randomUUID() });
      renderScripts(); refresh(); saveSoon();
    }, 'text-btn'); addLine.id = 'audio-edit-add-line';
    const direction = control('audio-edit-direction', draft.whole.direction);
    const sourceVoices = el('section', 'audio-edit-source-voices'); sourceVoices.id = 'audio-edit-source-voices'; sourceVoices.hidden = true;
    const directionField = field('Direction', direction);
    wholeFields.append(label('h3', 'Text to speak', 'audio-edit-script-label'), scripts, addLine, directionField, sourceVoices);
    const error = el('p', 'studio-error'); error.id = 'audio-edit-error'; error.hidden = true; error.setAttribute('role', 'alert');
    const sourceNote = label('p', 'This older take has no saved settings. Current model settings will be used.',
      'field-help audio-edit-source-note'); sourceNote.hidden = Boolean(version.complete_recipe);
    body.append(media, localFields, wholeFields, sourceNote, error);
    const parameters = el('details', 'audio-edit-parameters');
    parameters.append(label('summary', 'Generation settings'));
    const paramFields = el('div', 'field-grid'), paramInputs = {};
    for (const [key, caption, min, max, step] of [
      ['maximum', 'Max duration (seconds)', '.01', '95.109', 'any'],
      ['duration', 'Target duration (seconds)', '.01', '95.109', 'any'],
      ['temperature', 'Temperature', '0', '2', 'any'], ['top_p', 'Top P', '0', '1', 'any'],
      ['top_k', 'Top K', '0', '8193', '1'], ['seed', 'Seed', '0', String(2 ** 32 - 1), '1'],
    ]) {
      const input = control(`audio-edit-${key}`, draft.parameters[key], 'number');
      Object.assign(input, { min, max, step }); paramInputs[key] = input;
      input.oninput = () => {
        draft.parameters[key] = UiNumber.value(input); draft.parameters.edited = true; saveSoon(); refresh();
      };
      paramFields.append(field(caption, input));
    }
    const random = control('audio-edit-random', '', 'checkbox'); random.checked = draft.parameters.random;
    const randomField = el('label', 'check-field'); randomField.append(random, label('span', 'Random seed'));
    random.onchange = () => {
      draft.parameters.random = random.checked; draft.parameters.edited = true; saveSoon(); refresh();
    };
    parameters.append(paramFields, randomField);
    parameters.classList.add('dock-settings');
    const settingsSummary = parameters.querySelector(':scope > summary');
    delete settingsSummary.dataset.i18n;
    settingsSummary.innerHTML = `${svg('settings')}<span data-i18n="Generation settings">${t('Generation settings')}</span>`;
    const settingsPopover = el('div', 'generation-settings-popover');
    const settingsHeading = el('div', 'settings-popover-heading');
    const closeSettings = button('Close', () => {
      parameters.open = false; settingsSummary.focus();
    }, 'icon-btn');
    closeSettings.innerHTML = svg('close'); delete closeSettings.dataset.i18n;
    closeSettings.setAttribute('aria-label', t('Close')); closeSettings.dataset.i18nAriaLabel = 'Close';
    settingsHeading.append(label('strong', 'Generation settings'), closeSettings);
    settingsPopover.append(settingsHeading);
    for (const child of [...parameters.children]) if (child !== settingsSummary) settingsPopover.append(child);
    parameters.append(settingsPopover);
    const footer = el('footer', 'compose-footer action-dock audio-edit-dock'); footer.id = 'audio-edit-dock';
    const dockInner = el('div', 'action-dock-inner'), dockChoices = el('div', 'dock-choices');
    const dockActions = el('div', 'dock-actions'), shortcut = label('span', 'Ctrl + Enter', 'dock-shortcut');
    const submit = button('Generate edited version', () => generate(), 'primary-btn'); submit.id = 'audio-edit-submit';
    delete submit.dataset.i18n;
    submit.replaceChildren(label('span', 'Generate edited version'),
      document.createRange().createContextualFragment(svg('arrow')));
    dockChoices.append(parameters);
    dockActions.append(shortcut, submit);
    dockInner.append(dockChoices, dockActions); footer.append(dockInner);
    const results = el('section', 'audio-edit-results'); results.id = 'audio-edit-results'; results.hidden = true;
    results.setAttribute('aria-labelledby', 'audio-edit-results-title');
    const resultHeading = el('div', 'audio-edit-results-heading'), resultTitle = label('h3', 'Edited result');
    resultTitle.id = 'audio-edit-results-title';
    const resultStatus = el('div', 'audio-edit-results-status'), badge = el('span', 'badge');
    badge.id = 'audio-edit-result-badge';
    const cancel = button('Cancel', async () => {
      const job = context.result;
      if (!job || cancel.disabled) return;
      cancel.disabled = true;
      try {
        await api(`/api/jobs/${encodeURIComponent(job.id)}/cancel`, { method: 'POST' });
        if (workspace.isConnected && context.result?.id === job.id) {
          context.result.cancel_requested = true; drawResult();
        }
      } catch (issue) { context.failure = message(issue.message); cancel.disabled = false; refresh(); }
    }, 'text-btn');
    cancel.id = 'audio-edit-cancel';
    resultStatus.append(badge, cancel); resultHeading.append(resultTitle, resultStatus);
    const progress = el('div', 'progress-wrap'), track = el('div', 'progress-track'), bar = el('div');
    const progressRow = el('div', 'progress-row'), stage = el('span'), count = el('span', 'progress-count');
    progressRow.setAttribute('role', 'status'); progressRow.setAttribute('aria-live', 'polite');
    track.append(bar); progressRow.append(stage, count); progress.append(track, progressRow);
    const resultError = el('p', 'studio-error'); resultError.setAttribute('role', 'alert'); resultError.hidden = true;
    const resultBox = el('div', 'result-box hidden'); resultBox.id = 'audio-edit-result-box';
    results.append(resultHeading, progress, resultError, resultBox);
    const columns = el('div', 'audio-edit-columns'), inputs = el('div', 'audio-edit-inputs');
    inputs.append(body);
    columns.append(inputs, results);
    workspace.append(heading, tabs, columns, footer);
    document.querySelector('#view-audio-edit').replaceChildren(workspace);
    hooks.enterEditor?.(returnView);
    function drawResult() {
      const job = context.result;
      if (!job) return;
      results.hidden = false; results.dataset.state = job.status;
      badge.dataset.status = job.status; badge.textContent = statusLabel(job.status);
      const running = ['queued', 'running'].includes(job.status), timing = jobTiming(job);
      const phase = { model_load: 'Loading model', initialization: 'Initializing',
        waiting: 'Waiting for text', generation: 'Generating' }[timing.phase];
      stage.textContent = running ? job.status === 'queued' ? t('Queued')
        : phase ? t(phase) : stageLabel((job.events || []).at(-1)?.stage) || t('Generating') : '';
      count.textContent = !running || timing.phase === 'generation' ? generationTimeLabel(job) : '';
      bar.style.width = running ? '32%' : job.status === 'complete' ? '100%' : '0%';
      bar.classList.toggle('indeterminate', running);
      cancel.hidden = !running; cancel.disabled = Boolean(job.cancel_requested);
      cancel.dataset.i18n = job.cancel_requested ? 'Cancelling…' : 'Cancel';
      cancel.textContent = t(cancel.dataset.i18n);
      resultError.hidden = !job.error; resultError.textContent = job.error ? message(job.error) : '';
      renderResults(job, resultBox);
    }
    async function pollResult(id) {
      if (!workspace.isConnected || context.result?.id !== id) return;
      try {
        const job = await api(`/api/jobs/${encodeURIComponent(id)}`);
        if (!workspace.isConnected || context.result?.id !== id) return;
        if (resultWorkspaceFor(job) !== 'editor' || job.studio?.edit_source !== identity) {
          delete draft.result_job_id; context.result = null; context.busy = false; results.hidden = true;
          refresh(); saveSoon(); return;
        }
        context.result = job; context.busy = ['queued', 'running'].includes(job.status);
        const index = state.jobs.findIndex(item => item.id === id);
        if (index >= 0) state.jobs[index] = job;
        drawResult(); refresh(); window.StudioFlow?.syncControls();
        if (!context.busy) { loadJobs(); return; }
      } catch (issue) {
        if (!workspace.isConnected || context.result?.id !== id) return;
        context.result = { ...context.result, status: 'interrupted', error: issue.message };
        context.busy = false; drawResult(); refresh(); return;
      }
      context.pollTimer = setTimeout(() => pollResult(id), 900);
    }
    function trackResult(job) {
      clearTimeout(context.pollTimer);
      draft.result_job_id = job.id; context.result = job;
      context.busy = ['queued', 'running'].includes(job.status);
      drawResult(); refresh(); saveSoon(); pollResult(job.id);
    }
    async function closeEditor() {
      if (context.closing || context.busy) return false;
      context.closing = true;
      try {
        await persist(); cleanup(); hooks.clearRememberedSource?.(); hooks.leaveEditor?.(returnView); return true;
      }
      catch (issue) { context.failure = message(issue.message); refresh(); return false; }
      finally { context.closing = false; }
    }
    context.close = closeEditor;
    function cleanup() {
      clearTimeout(context.pollTimer);
      workspace.querySelectorAll('audio').forEach(item => item.pause()); workspace.remove();
      document.removeEventListener('drama:capabilities', onCapabilities);
      document.removeEventListener('drama:resourceschange', onCapabilities);
      document.removeEventListener('drama:languagechange', onLanguage);
      document.removeEventListener('drama:jobupdate', onJob);
      document.removeEventListener('drama:jobsubmitted', onSubmitted);
      if (active === context) active = null;
      for (const reference of Object.values(draft.whole.references)) ReferencePrep.cancel(reference);
      StudioAudio.enhance();
    }
    const isInsertion = () => draft.range && draft.range.start === draft.range.end;
    function candidates() {
      const value = draft.local.quote, result = [];
      if (!value) return result;
      originalLines.forEach((line, index) => {
        let at = line.text.indexOf(value);
        while (at >= 0) {
          result.push({ line: index, start: at, end: at + value.length });
          at = line.text.indexOf(value, at + 1);
        }
      });
      return result;
    }
    function target() {
      const local = draft.local, chosen = local.target, line = originalLines[chosen?.line];
      if (isInsertion()) return line && chosen.start === chosen.end && chosen.start >= 0
        && chosen.end <= line.text.length ? chosen : null;
      const options = candidates();
      if (options.length === 1) return options[0];
      return options.find(option => chosen && option.line === chosen.line && option.start === chosen.start
        && option.end === chosen.end) || null;
    }
    function setQuote(value) {
      const old = draft.local.quote; draft.local.quote = value; quote.value = value; draft.local.target = null;
      if (!draft.local.replacement || draft.local.replacement === old) {
        draft.local.replacement = value; replacement.value = value; ScriptEditor.sync(replacement);
      }
    }
    function drawMatches() {
      matches.replaceChildren();
      const options = candidates(), selected = target();
      if (!isInsertion() && options.length > 1) {
        matches.append(label('p', 'These words appear more than once. Choose the occurrence to edit.', 'field-help'));
        options.forEach((option, index) => {
          const line = originalLines[option.line], choose = el('button', 'audio-edit-match');
          choose.type = 'button'; choose.setAttribute('aria-pressed', String(selected?.line === option.line && selected?.start === option.start));
          const role = el('span', 'field-help'); role.textContent = `${index + 1}${joint ? ` · ${line.character}` : ''}`;
          const contextText = el('span'), mark = el('mark'); mark.textContent = draft.local.quote;
          contextText.append(document.createTextNode((option.start > 18 ? '…' : '')
            + line.text.slice(Math.max(0, option.start - 18), option.start)), mark,
          document.createTextNode(line.text.slice(option.end, option.end + 18) + (line.text.length > option.end + 18 ? '…' : '')));
          choose.append(role, contextText);
          choose.onclick = () => { draft.local.target = copy(option); drawMatches(); refresh(); saveSoon(); };
          matches.append(choose);
        });
      } else if (isInsertion() && selected) {
        const line = originalLines[selected.line], text = el('p', 'audio-edit-insertion-context');
        const mark = label('mark', 'Insert here');
        text.append(document.createTextNode(line.text.slice(Math.max(0, selected.start - 24), selected.start)),
          mark, document.createTextNode(line.text.slice(selected.end, selected.end + 24))); matches.append(text);
      }
    }
    function localLines() {
      const chosen = target();
      if (!chosen) return null;
      const lines = copy(originalLines), line = lines[chosen.line], newWords = draft.local.replacement;
      if (!safeRange(line.text, chosen.start, chosen.end, newWords)) return null;
      line.text = line.text.slice(0, chosen.start) + newWords + line.text.slice(chosen.end);
      return lines;
    }
    function transcript(lines) {
      return joint ? lines.map(line => {
        const role = roles.find(role => role.name === line.character);
        return `<|speaker:${role.speaker}|> ${line.text}`;
      }).join('\n') : sourceSpeaker === undefined ? lines[0].text : `<|speaker:${sourceSpeaker}|> ${lines[0].text}`;
    }
    const textUnchanged = () => draft.whole.lines.length === originalLines.length
      && draft.whole.lines.every((line, index) => line.text === originalLines[index].text
        && line.character === originalLines[index].character);
    const speakers = [...new Set([...version.text.matchAll(/<\|speaker:(\d+)\|>/g)].map(match => Number(match[1])))];
    if (!speakers.length) speakers.push(0);
    const availableSpeakers = new Set((recipe.reference_snapshot || []).map(reference => reference.speaker));
    const missingSpeakers = speakers.filter(speaker => !availableSpeakers.has(speaker));
    async function prepareSourceVoices() {
      if (context.voicesStarted) return;
      context.voicesStarted = true; context.preparing = true;
      submit.disabled = true;
      sourceVoices.append(label('h3', 'Source voice', 'audio-edit-script-label'));
      const rows = el('div');
      const checked = control('audio-edit-reference-checked', '', 'checkbox'); checked.checked = Boolean(draft.whole.checked);
      const check = el('label', 'check-field');
      check.append(checked, label('span', 'Each reference contains only its speaker and the transcript is correct.'));
      checked.onchange = () => { draft.whole.checked = checked.checked; context.failure = ''; saveSoon(); refresh(); };
      sourceVoices.append(rows, check);
      try {
        for (const speaker of missingSpeakers) {
          const references = draft.whole.references;
          references[speaker] ||= ReferencePrep.fresh();
          if (speakers.length === 1 && version.take.duration_seconds <= 120
            && !references[speaker].upload && !references[speaker].voiceId) {
            try {
              const upload = await api('/api/reference-audio/from-version', {
                method: 'POST', body: JSON.stringify({ source_version: source }),
              });
              if (!workspace.isConnected) return;
              references[speaker].upload = upload;
              references[speaker].transcript = singleText(version.text)
                .replace(/<\/?(whisper|emphasis|soft|fast|slow|stress|forceful)>/g, '').replace(/\[[^\]\n]*\]/g, '').trim();
            } catch (issue) { context.failure = message(issue.message); }
          }
          const row = el('div', 'audio-edit-reference-row'), content = el('div');
          if (joint) {
            const name = el('strong'); name.textContent = roles.find(role => role.speaker === speaker)?.name || String(speaker);
            row.append(name);
          }
          const changed = () => {
            draft.whole.checked = false; checked.checked = false; context.failure = ''; saveSoon(); refresh();
          };
          const draw = () => {
            content.replaceChildren(ReferencePrep.mount(references[speaker], {
              prefix: `audio-edit-reference-${speaker}`, changed, allowLibrary: true,
            }));
            StudioControls.init(); StudioAudio.enhance(content);
          };
          if (speakers.length > 1) row.append(button('Use selection for this speaker', async () => {
            try {
              const selection = StudioAudio.selection(audio);
              if (!selection || selection.end - selection.start < .5) throw new Error(t('Select one speaker’s audio first.'));
              const upload = await api('/api/reference-audio/from-version', {
                method: 'POST', body: JSON.stringify({ source_version: source, ...selection }),
              });
              if (!workspace.isConnected) return;
              ReferencePrep.cancel(references[speaker]);
              references[speaker] = { ...ReferencePrep.fresh(), upload };
              changed(); draw();
            } catch (issue) { context.failure = message(issue.message); refresh(); }
          }, 'text-btn'));
          row.append(content); rows.append(row); draw();
        }
      } catch (issue) { context.failure = message(issue.message); }
      finally { context.preparing = false; refresh(); saveSoon(); }
    }
    function refresh() {
      const local = draft.mode === 'local', range = draft.range, chosen = target();
      const lines = local ? localLines() : draft.whole.lines, caps = state.capabilities || {};
      const supported = local ? caps.infill : caps.director_revision && (!joint || caps.joint_dialogue);
      const mismatch = recipe.model && recipe.model.replaceAll('\\', '/').toLowerCase()
        !== state.model?.selected_directory?.replaceAll('\\', '/').toLowerCase();
      const missing = version.audio_missing || version.take_missing;
      const empty = !lines?.length || lines.some(line => !line.text.trim());
      const changedText = !textUnchanged();
      const unchanged = !local && !changedText && !draft.parameters.edited
        && draft.whole.direction === (recipe.direction || '');
      submit.disabled = context.busy || context.preparing || modelBusy() || !supported || mismatch
        || missing || empty || unchanged || local && (!range || isInsertion() && !draft.local.replacement.trim());
      for (const item of [dismiss, insert, addLine, ...tabs.children]) item.disabled = context.busy;
      workspace.setAttribute('aria-busy', String(context.busy));
      for (const input of workspace.querySelectorAll('input, textarea, select, .audio-edit-match')) {
        if (input !== random) input.disabled = context.busy;
      }
      for (const textarea of workspace.querySelectorAll('.audio-edit-whole-fields textarea, #audio-edit-text-0')) {
        textarea.readOnly = context.busy; ScriptEditor.sync(textarea);
      }
      insert.disabled ||= !Number.isFinite(audio.duration);
      random.disabled = context.busy;
      paramInputs.seed.disabled = context.busy || random.checked;
      paramInputs.duration.disabled = context.busy || !caps.duration_hint;
      direction.disabled = context.busy || caps.direction === false;
      sourceVoices.hidden = local || !changedText || !missingSpeakers.length;
      audio.dataset.selection = String(local || (!sourceVoices.hidden && speakers.length > 1));
      if (!sourceVoices.hidden) prepareSourceVoices();
      quote.parentElement.hidden = Boolean(isInsertion());
      let issue = '';
      if (local && range) {
        if (isInsertion() && !chosen) issue = 'Choose the text cursor position in the source.';
        else if (draft.local.quote && !chosen) issue = candidates().length
          ? 'Choose which occurrence to replace.' : 'The original words do not match the source text.';
        else if (chosen && !lines) issue = 'Select complete tags, or words within the same tag.';
      }
      error.hidden = !mismatch && !missing && Boolean(supported) && !context.failure && !issue;
      if (mismatch) error.textContent = t('This source uses another model. Switch models or start a new draft.');
      else if (missing) error.textContent = t('Some saved result files are missing.');
      else if (!supported) error.textContent = t('This task is unavailable in this model or runtime. Your draft was kept.');
      else error.textContent = context.failure || t(issue);
      if (range) { UiNumber.write(start, range.start); UiNumber.write(end, range.end); }
      const caption = local && isInsertion() ? 'Generate insertion' : 'Generate edited version';
      submit.replaceChildren(label('span', caption), document.createRange().createContextualFragment(svg('arrow')));
      if (local) drawDiff(diff, isInsertion() ? '' : draft.local.quote, draft.local.replacement);
      for (const row of locator.querySelectorAll('.audio-edit-original-line')) row.updateSelection();
    }
    function renderScripts() {
      scripts.replaceChildren();
      draft.whole.lines.forEach((line, index) => {
        const row = el('div', 'audio-edit-line'), textarea = control(`audio-edit-whole-${index}`, line.text, 'textarea');
        const caption = label('label', 'Text to speak', 'sr-only'); caption.htmlFor = textarea.id;
        if (joint) {
          const role = el('select'); role.id = `audio-edit-role-${index}`;
          role.setAttribute('aria-label', t('Character')); role.dataset.i18nAriaLabel = 'Character';
          for (const character of roles) role.add(new Option(character.name, character.name));
          role.value = line.character;
          role.onchange = () => { line.character = role.value; saveSoon(); refresh(); };
          row.append(role);
          if (draft.whole.lines.length > 1) row.append(button('Remove', () => {
            draft.whole.lines.splice(index, 1); renderScripts(); refresh(); saveSoon();
          }, 'text-btn'));
        }
        textarea.oninput = () => { line.text = textarea.value; saveSoon(); refresh(); };
        row.append(caption, textarea); scripts.append(row); ScriptEditor.mount(textarea);
      });
      addLine.hidden = !joint; StudioControls.init();
    }
    function changeMode(mode) {
      if (context.busy) return;
      draft.mode = mode; const local = mode === 'local';
      localFields.hidden = !local; wholeFields.hidden = local;
      localActions.hidden = !local; audio.dataset.selection = String(local);
      if (local && draft.range && Number.isFinite(audio.duration)) {
        StudioAudio.setSelection(audio, draft.range.start, draft.range.end);
      } else if (!local) StudioAudio.clearSelection(audio);
      for (const tab of tabs.children) {
        tab.setAttribute('aria-selected', String(tab.dataset.editMode === mode));
        tab.tabIndex = tab.dataset.editMode === mode ? 0 : -1;
      }
      if (!local) renderScripts();
      drawMatches(); refresh(); saveSoon();
    }
    for (const [mode, caption] of [['local', 'Selected passage'], ['whole', 'Whole audio']]) {
      const tab = button(caption, () => changeMode(mode), 'audio-edit-tab');
      tab.dataset.editMode = mode; tab.setAttribute('role', 'tab'); tabs.append(tab);
    }
    tabs.onkeydown = event => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key) || context.busy) return;
      event.preventDefault();
      const choices = [...tabs.children], current = choices.findIndex(tab => tab.dataset.editMode === draft.mode);
      const index = event.key === 'Home' ? 0 : event.key === 'End' ? choices.length - 1
        : (current + (event.key === 'ArrowRight' ? 1 : -1) + choices.length) % choices.length;
      changeMode(choices[index].dataset.editMode); choices[index].focus();
    };
    workspace.addEventListener('keydown', event => {
      if ((event.ctrlKey || event.metaKey) && event.key === 'Enter' && !event.isComposing) {
        event.preventDefault(); event.stopPropagation(); if (!submit.disabled) generate();
      }
    });
    direction.oninput = () => { draft.whole.direction = direction.value; saveSoon(); refresh(); };
    quote.oninput = () => { setQuote(quote.value); drawMatches(); refresh(); saveSoon(); };
    replacement.oninput = () => { draft.local.replacement = replacement.value; refresh(); saveSoon(); };
    audio.addEventListener('drama:selectionchange', event => {
      if (!event.detail || draft.mode !== 'local') return;
      const insertionChanged = Boolean(isInsertion()) !== (event.detail.start === event.detail.end);
      draft.range = copy(event.detail);
      if (insertionChanged) draft.local.target = null;
      drawMatches(); saveSoon(); refresh();
    });
    audio.addEventListener('loadedmetadata', () => {
      if (draft.range) StudioAudio.setSelection(audio, draft.range.start, draft.range.end, { emit: true });
      refresh();
    });
    for (const input of [start, end]) input.onchange = () => {
      const first = UiNumber.read(start), last = UiNumber.read(end);
      if (Number.isFinite(first) && Number.isFinite(last) && first >= 0 && last >= first) {
        StudioAudio.setSelection(audio, first, last, { emit: true });
      } else refresh();
    };
    function sampling() {
      for (const [key, input] of Object.entries(paramInputs)) {
        if (key === 'seed' && random.checked || key === 'duration' && !input.value) continue;
        if (!input.value || !input.checkValidity()) throw new Error(t('Check generation settings.'));
      }
      const read = key => UiNumber.read(paramInputs[key]), maximum = read('maximum'), duration = read('duration');
      if (!(read('temperature') > 0 && read('top_p') > 0)) throw new Error(t('Check generation settings.'));
      if (duration > maximum) throw new Error(t('Target duration exceeds the maximum generation length'));
      return {
        seed: random.checked ? crypto.getRandomValues(new Uint32Array(1))[0] : read('seed'),
        temperature: read('temperature'), top_p: read('top_p'), top_k: read('top_k'),
        max_new_tokens: Math.max(1, Math.round(maximum * 44100 / 2048)),
        duration_frames: duration ? Math.max(1, Math.round(duration * 44100 / 2048)) : null,
      };
    }
    async function generate() {
      if (submit.disabled || context.busy) return;
      context.busy = true; context.failure = ''; refresh();
      try {
        const local = draft.mode === 'local', lines = copy(local ? localLines() : draft.whole.lines);
        const parameters = sampling(), operation = local ? 'infill' : textUnchanged() ? 'revision' : 'reword';
        const range = copy(draft.range);
        const studio = copy(version.studio || { work_id: crypto.randomUUID(), segment_id: crypto.randomUUID(), layout: 'single' });
        studio.layout = joint || version.studio?.layout === 'scene' ? 'scene' : 'single';
        studio.result_scope = 'audio_editor'; studio.edit_source = identity;
        if (version.line_id) studio.line_id = version.line_id;
        const payload = { source_version: copy(source), operation, studio, text: transcript(lines),
          direction: local ? '' : draft.whole.direction || '', ...parameters };
        if (joint) {
          payload.lines = lines.map(({ character, text }) => ({ character, text }));
          payload.line_ids = lines.map(line => line.key || crypto.randomUUID()); payload.scene_order = copy(payload.line_ids);
        }
        if (local) {
          const snap = value => Math.abs(value - Math.round(value)) < .00002 ? Math.round(value) : value;
          payload.start_frame = Math.floor(snap(range.start / secondsPerFrame));
          payload.end_frame = isInsertion() ? payload.start_frame : Math.ceil(snap(range.end / secondsPerFrame));
        }
        let bindings = [];
        if (operation === 'reword' && missingSpeakers.length) {
          if (!draft.whole.checked) throw new Error(t('Check the reference transcript first.'));
          for (const speaker of missingSpeakers) {
            const clips = await ReferencePrep.readAll(draft.whole.references[speaker], speaker);
            if (!clips.length) throw new Error(t('Choose a reference for each missing speaker.'));
            bindings.push(...clips);
          }
          payload.reword_references = bindings; payload.reference_transcripts_checked = true;
        }
        if (!version.complete_recipe) {
          delete payload.source_version; payload.legacy_source = source; delete payload.operation;
          payload.model = state.model?.selected_directory;
          payload.mode = local ? 'infill' : operation === 'revision' ? 'revision' : 'tts';
          if (local) payload.source_take = version.take_file;
          else if (operation === 'revision') payload.previous_take = version.take_file;
          if (operation === 'reword') {
            const prepared = await api('/api/studio/prepare-restore', {
              method: 'POST', body: JSON.stringify({ ...source, prepared_references: bindings, reference_transcripts_checked: true }),
            });
            payload.voice_bindings = prepared.references.map(reference => ({
              id: reference.id, speaker: reference.speaker, text: reference.text,
            }));
          }
          delete payload.reword_references;
          if (joint) Object.assign(payload, { scene_strategy: 'joint', single_chunk: true, characters: recipe.characters,
            character_ids: recipe.character_ids, scene: recipe.scene || 'A conversation.' });
        }
        await persist();
        if (!workspace.isConnected) return;
        await hooks.submit(payload);
      } catch (issue) { context.failure = message(issue.message); }
      finally {
        context.busy = ['queued', 'running'].includes(context.result?.status);
        if (workspace.isConnected) refresh();
      }
    }
    const onCapabilities = () => refresh();
    const onLanguage = () => { drawMatches(); drawResult(); refresh(); };
    const onJob = () => refresh();
    const onSubmitted = event => {
      const job = event.detail;
      if (resultWorkspaceFor(job) === 'editor' && job.studio?.edit_source === identity) trackResult(job);
    };
    document.addEventListener('drama:capabilities', onCapabilities);
    document.addEventListener('drama:resourceschange', onCapabilities);
    document.addEventListener('drama:languagechange', onLanguage);
    document.addEventListener('drama:jobupdate', onJob);
    document.addEventListener('drama:jobsubmitted', onSubmitted);
    for (const other of document.querySelectorAll('audio')) other.pause();
    ScriptEditor.mount(replacement); changeMode(draft.mode); StudioAudio.enhance(body);
    if (draft.result_job_id) trackResult({ id: draft.result_job_id, status: 'queued' });
    workspace.scrollIntoView({ block: 'start', behavior: 'smooth' }); workspace.focus({ preventScroll: true });
  }
  window.AudioEditor = Object.freeze({
    configure(value, draft) { hooks = value; if (draft?.entries) saved = copy(draft); },
    open,
    async restoreActive() {
      let source, returnView;
      try {
        source = JSON.parse(sessionStorage.getItem('drama3.ui.audio-edit-source') || 'null');
        returnView = sessionStorage.getItem('drama3.ui.audio-edit-return');
      } catch (_) { return false; }
      if (!source?.job_id) return false;
      if (!['generate', 'scene'].includes(returnView)) returnView = 'generate';
      await open(source, { returnView });
      return Boolean(active);
    },
  });
})();
