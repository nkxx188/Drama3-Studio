/* Uploaded recordings have their own editor; codec preparation belongs to generation. */
(() => {
  'use strict';
  const $ = id => document.getElementById(`audio-import-${id}`);
  const copy = value => JSON.parse(JSON.stringify(value));
  const plain = value => String(value || '').replace(/^\s*<\|speaker:0\|>\s*/, '');
  const fresh = () => ({
    upload: null, original: '', text: '', direction: '', mode: 'whole', quote: '', replacement: '',
    target: null, range: null, resultJobId: null, workId: crypto.randomUUID(), segmentId: crypto.randomUUID(),
    parameters: { maximum: '35.67', duration: '', max_chars: '160', temperature: '1', top_p: '.9', top_k: '0', seed: '123', random: true },
  });
  let draft = fresh(), ready = false, hydrating = false, uploading = false, submitting = false;
  let job = null, asrBusy = false, asrStatus = '', failure = '', request = 0, originalRevision = 0;
  let pollTimer, saveTimer, saveQueue = Promise.resolve();
  const params = {};
  const active = () => submitting || ['queued', 'running'].includes(job?.status);
  const controlsBusy = () => uploading || hydrating || active();

  function persist() {
    clearTimeout(saveTimer);
    const body = JSON.stringify({ key: 'external_audio_editor', value: { schema: 2, ...copy(draft) } });
    saveQueue = saveQueue.catch(() => {}).then(() => api('/api/studio/draft', { method: 'POST', body }));
    return saveQueue;
  }
  const saveSoon = () => { clearTimeout(saveTimer); saveTimer = setTimeout(() => persist().catch(() => {}), 350); };
  function writeText(id, value, rich = false) {
    if ($(id).value === value) return;
    $(id).value = value;
    if (rich) ScriptEditor.sync($(id));
  }
  function matches() {
    const found = [];
    if (!draft.quote) return found;
    let at = draft.original.indexOf(draft.quote);
    while (at >= 0) {
      found.push({ start: at, end: at + draft.quote.length });
      at = draft.original.indexOf(draft.quote, at + 1);
    }
    return found;
  }
  const insertion = () => draft.range && draft.range.start === draft.range.end;
  function target() {
    if (insertion()) return draft.target && draft.target.start === draft.target.end
      && draft.target.start >= 0 && draft.target.end <= draft.original.length ? draft.target : null;
    const found = matches();
    return found.length === 1 ? found[0]
      : found.find(value => value.start === draft.target?.start && value.end === draft.target?.end);
  }
  function safeRange(start, end) {
    if (/<\|[^>]*\|>/.test(draft.replacement)) return false;
    for (const match of draft.original.matchAll(/<\|[^>]*\|>|<\/?[A-Za-z][^>]*>|\[[^\]\n]*\]/g)) {
      if ([start, end].some(at => at > match.index && at < match.index + match[0].length)) return false;
    }
    const balanced = value => {
      const stack = [];
      for (const tag of value.matchAll(/<(\/?)(whisper|emphasis|soft|fast|slow|stress|forceful)>/g)) {
        if (!tag[1]) stack.push(tag[2]);
        else if (stack.pop() !== tag[2]) return false;
      }
      return !stack.length;
    };
    return balanced(draft.original.slice(start, end)) && balanced(draft.replacement);
  }
  function editedText() {
    if (draft.mode === 'whole') return draft.text;
    const chosen = target();
    if (!chosen || !safeRange(chosen.start, chosen.end)) return null;
    return draft.original.slice(0, chosen.start) + draft.replacement + draft.original.slice(chosen.end);
  }
  function drawMatches() {
    $('matches').replaceChildren();
    if (insertion() || matches().length < 2) return;
    for (const match of matches()) {
      const button = document.createElement('button');
      button.className = 'audio-edit-match'; button.type = 'button';
      button.textContent = draft.original.slice(Math.max(0, match.start - 18), match.end + 18);
      button.setAttribute('aria-pressed', String(target()?.start === match.start));
      button.disabled = controlsBusy();
      button.onclick = () => { draft.target = match; drawMatches(); render(); saveSoon(); };
      $('matches').append(button);
    }
  }
  function renderResult() {
    const running = ['queued', 'running'].includes(job?.status);
    $('results').dataset.state = job?.status || 'idle';
    $('results').setAttribute('aria-busy', String(running));
    $('stage').hidden = !running;
    $('result-badge').hidden = !job;
    $('result-badge').dataset.status = job?.status || 'idle';
    $('result-badge').textContent = job ? statusLabel(job.status) : '';
    $('cancel').hidden = !running;
    $('cancel').disabled = Boolean(job?.cancel_requested);
    $('result-error').hidden = !job?.error;
    $('result-error').textContent = job?.error ? message(job.error) : '';
    $('output-empty').hidden = Boolean(job);
    if (!job) {
      $('result-box').querySelectorAll('audio').forEach(audio => audio.pause());
      $('result-box').replaceChildren(); delete $('result-box').dataset.jobId;
      $('result-box').classList.add('hidden');
      $('stage').textContent = $('progress-count').textContent = '';
      return;
    }
    const event = (job.events || []).at(-1), timing = jobTiming(job);
    const phase = {
      model_load: 'Loading model', initialization: 'Preparing source audio',
      waiting: 'Waiting for text', generation: 'Generating',
    }[timing.phase];
    // Runtime events also describe context changes; the measured phase is authoritative.
    $('stage').textContent = !running ? '' : job.status === 'queued' ? t('Queued')
      : phase ? t(phase) : stageLabel(event?.stage) || t('Generating');
    $('progress-count').textContent = !running || timing.phase === 'generation' ? generationTimeLabel(job) : '';
    $('progress-bar').style.width = running ? '32%' : job.status === 'complete' ? '100%' : '0%';
    $('progress-bar').classList.toggle('indeterminate', running);
    renderResults(job, $('result-box'));
    const retry = $('result-box').querySelector('.main-result .creative-retry');
    if (retry) { retry.disabled = controlsBusy(); retry.onclick = retryResult; }
    const edit = $('result-box').querySelector('.main-result .creative-edit');
    if (edit) { edit.disabled = controlsBusy(); edit.onclick = editResult; }
  }
  function render() {
    const hasAudio = Boolean(draft.upload), busy = controlsBusy(), local = draft.mode === 'local';
    $('dropzone').classList.toggle('hidden', hasAudio);
    $('preview').classList.toggle('hidden', !hasAudio);
    $('filename').textContent = draft.upload?.filename || '';
    const audio = $('player'), src = draft.upload?.audio_url || '';
    if (audio.getAttribute('src') !== src) {
      if (src) audio.src = src; else audio.removeAttribute('src');
    }
    if (audio.dataset.selection !== String(local && !busy)) audio.dataset.selection = String(local && !busy);
    $('local').hidden = !local; $('whole').hidden = local;
    $('original').hidden = !hasAudio;
    $('asr-status').textContent = message(asrStatus);
    $('asr-status').dataset.state = asrBusy ? 'busy' : '';
    for (const [id, key, rich] of [
      ['transcript', 'original', false], ['text', 'text', true], ['direction', 'direction', false],
      ['quote', 'quote', false], ['replacement', 'replacement', true],
    ]) {
      writeText(id, draft[key], rich);
      $(id).disabled = busy;
      if (rich) { $(id).readOnly = busy; ScriptEditor.sync($(id)); }
    }
    $('file').disabled = busy;
    $('choose').disabled = busy;
    $('remove').disabled = busy;
    $('transcribe').disabled = busy || asrBusy;
    $('use-selection').hidden = !local;
    $('use-selection').disabled = busy;
    $('insert').disabled = busy || !Number.isFinite(audio.duration);
    $('start').disabled = $('end').disabled = busy;
    for (const tab of document.querySelectorAll('[data-import-mode]')) {
      tab.setAttribute('aria-selected', String(tab.dataset.importMode === draft.mode));
      tab.disabled = busy;
    }
    if (draft.range) {
      UiNumber.write($('start'), draft.range.start); UiNumber.write($('end'), draft.range.end);
    }
    for (const input of [params.maximum, params.duration, params.temperature, params.top_p, params.top_k]) input.disabled = busy;
    params.duration.disabled ||= !state.capabilities?.duration_hint;
    params.seed.disabled = busy || draft.parameters.random;
    $('random').disabled = busy;
    $('local-note').textContent = local && draft.range && (draft.quote || insertion()) && !target()
      ? t(insertion() ? 'Choose the text cursor position in the source.'
        : matches().length > 1 ? 'Choose which occurrence to replace.' : 'The original words do not match the source text.')
      : local && target() && !editedText() ? t('Select complete tags, or words within the same tag.') : '';
    const supported = state.capabilities?.audio_import && (local ? state.capabilities?.infill : state.capabilities?.director_revision);
    const changed = draft.text !== draft.original || Boolean(draft.direction.trim()) || draft.parameters.edited;
    $('submit').disabled = busy || asrBusy || !hasAudio || !draft.original.trim() || !editedText()?.trim()
      || !supported || state.switchingModel || state.releasingResources
      || !local && !changed || local && (!draft.range || insertion() && !draft.replacement.trim());
    $('error').hidden = !failure && (!hasAudio || Boolean(supported));
    $('error').textContent = failure ? message(failure) : t('The selected model or runtime does not support audio editing.');
    StudioAudio.enhance($('preview'));
    renderResult();
  }
  function resetResult() {
    clearTimeout(pollTimer); draft.resultJobId = null; job = null;
  }
  function setOriginal(value) {
    const previous = draft.original;
    draft.original = value;
    if (!draft.text || draft.text === previous) draft.text = value;
    draft.target = null; originalRevision++;
    drawMatches(); render(); saveSoon();
  }
  async function transcribe(force = false) {
    if (!draft.upload || controlsBusy() || asrBusy) return;
    const revision = originalRevision, token = request;
    asrBusy = true; asrStatus = 'Preparing transcription…'; render();
    try {
      let task = await api('/api/transcriptions', {
        method: 'POST', body: JSON.stringify({ upload_id: draft.upload.id, language: 'auto', local_only: true, force }),
      });
      while (['queued', 'running'].includes(task.status)) {
        if (token !== request) return;
        asrStatus = { loading: 'Loading Qwen3-ASR…', queued: 'Waiting to transcribe…',
          preparing: 'Preparing transcription…', cpu_fallback: 'GPU memory unavailable. Transcribing on CPU…' }[task.stage] || 'Transcribing…';
        render();
        await new Promise(resolve => setTimeout(resolve, 700));
        task = await api(`/api/transcriptions/${encodeURIComponent(task.id)}`);
      }
      if (token !== request) return;
      if (task.status === 'complete') {
        if (revision === originalRevision) setOriginal(plain(task.result.text));
        asrStatus = 'Transcription complete. Review the text before editing.';
        await persist();
      } else {
        asrStatus = task.status === 'skipped'
          ? 'Local transcription model is not installed. Enter the transcript manually or install it in Settings.'
          : task.error || 'Transcription failed. Enter the transcript manually or try again.';
        $('original').open = !draft.original;
      }
    } catch (error) {
      if (token === request) { asrStatus = error.message; $('original').open = !draft.original; }
    } finally {
      if (token === request) { asrBusy = false; render(); }
    }
  }
  async function upload(file) {
    if (!file || controlsBusy()) return;
    const token = ++request;
    asrBusy = false; uploading = true; failure = ''; asrStatus = ''; render();
    try {
      if (file.size > 127 * 1024 ** 2) throw new Error('Audio file must be no larger than 127 MiB.');
      const form = new FormData(); form.append('audio', file, file.name);
      const response = await fetch('/api/reference-audio', { method: 'POST', body: form });
      const uploaded = await response.json();
      if (!response.ok) throw new Error(uploaded.error || `HTTP ${response.status}`);
      const asset = await api(`/api/reference-audio/${encodeURIComponent(uploaded.id)}`);
      if (token !== request) return;
      const parameters = draft.parameters;
      draft = fresh(); draft.parameters = parameters;
      draft.upload = { id: asset.id, filename: asset.filename, audio_url: asset.audio_url };
      originalRevision++; resetResult();
      StudioAudio.clearSelection($('player')); $('original').open = false;
      await persist();
    } catch (error) { if (token === request) failure = error.message; }
    finally {
      if (token === request) { uploading = false; $('file').value = ''; render(); }
    }
    if (token === request && !failure) transcribe();
  }
  async function poll(id) {
    clearTimeout(pollTimer);
    try {
      const result = await api(`/api/jobs/${encodeURIComponent(id)}`);
      if (id !== draft.resultJobId) return;
      job = result; render();
      if (['queued', 'running'].includes(job.status)) pollTimer = setTimeout(() => poll(id), 900);
      else { persist().catch(() => {}); loadJobs(); loadHealth(); }
    } catch (error) {
      if (id === draft.resultJobId) {
        job = { ...job, status: 'interrupted', error: error.message }; render();
      }
    }
  }
  function sampling() {
    for (const [key, input] of Object.entries(params)) {
      if (key === 'seed' && draft.parameters.random || key === 'duration' && !input.value) continue;
      if (!input.value || !input.checkValidity()) throw new Error(t('Check generation settings.'));
    }
    const read = key => UiNumber.read(params[key]), duration = read('duration'), maximum = read('maximum');
    if (!(read('temperature') > 0 && read('top_p') > 0)) throw new Error(t('Check generation settings.'));
    if (duration > maximum) throw new Error(t('Target duration exceeds the maximum generation length'));
    return {
      seed: draft.parameters.random ? crypto.getRandomValues(new Uint32Array(1))[0] : read('seed'),
      max_new_tokens: Math.max(1, Math.round(maximum * 44100 / 2048)),
      max_chunk_chars: Math.round(read('max_chars')),
      duration_frames: duration ? Math.max(1, Math.round(duration * 44100 / 2048)) : null,
      temperature: read('temperature'), top_p: read('top_p'), top_k: read('top_k'),
    };
  }
  async function startRequest(payload) {
    const created = await api('/api/audio-edit/jobs', { method: 'POST', body: JSON.stringify(payload) });
    draft.resultJobId = created.id;
    job = { id: created.id, status: 'queued', events: [], studio: payload.studio };
    await persist(); poll(created.id); loadJobs();
  }
  async function retryResult() {
    if (controlsBusy() || !job?.recipe?.audio_edit) return;
    const recipe = copy(job.recipe);
    submitting = true; failure = ''; render();
    try {
      await startRequest({
        ...recipe, upload_id: recipe.audio_edit.upload_id,
        original_transcript: recipe.audio_edit.original_transcript,
        model: state.model?.selected_directory, seed: crypto.getRandomValues(new Uint32Array(1))[0],
      });
    } catch (error) { failure = error.message; }
    finally { submitting = false; render(); }
  }
  async function editResult() {
    if (controlsBusy() || !job?.output) return;
    const source = job;
    uploading = true; failure = ''; render();
    try {
      const asset = await api('/api/reference-audio/from-version', {
        method: 'POST', body: JSON.stringify({ source_version: { job_id: source.id } }),
      });
      const parameters = draft.parameters;
      draft = fresh(); draft.parameters = parameters;
      draft.upload = { id: asset.id, filename: asset.filename, audio_url: asset.audio_url };
      draft.original = plain(source.recipe.text).replace(/<\/?[\w-]+>/g, '').replace(/\[[^\]\n]*\]/g, '').trim();
      draft.text = draft.original;
      request++; originalRevision++; asrStatus = ''; resetResult();
      StudioAudio.clearSelection($('player')); await persist();
    } catch (error) { failure = error.message; }
    finally { uploading = false; render(); }
  }
  async function generate() {
    if ($('submit').disabled) return;
    submitting = true; failure = ''; render();
    try {
      const local = draft.mode === 'local';
      const payload = {
        upload_id: draft.upload.id, original_transcript: draft.original, text: editedText(),
        operation: local ? 'infill' : draft.text === draft.original ? 'revision' : 'reword',
        direction: local ? '' : draft.direction, model: state.model?.selected_directory, ...sampling(),
        studio: { work_id: draft.workId, segment_id: draft.segmentId, layout: 'single',
          result_scope: 'audio_editor', edit_source: 'upload:' + draft.upload.id },
      };
      if (local) {
        const snap = value => Math.abs(value - Math.round(value)) < .00002 ? Math.round(value) : value;
        payload.start_frame = Math.floor(snap(draft.range.start * 44100 / 2048));
        payload.end_frame = insertion() ? payload.start_frame : Math.ceil(snap(draft.range.end * 44100 / 2048));
      }
      await persist();
      await startRequest(payload);
    } catch (error) { failure = error.message; }
    finally { submitting = false; render(); }
  }
  async function hydrate() {
    if (ready || hydrating) return;
    hydrating = true; render();
    try {
      const studio = await api('/api/studio'), saved = studio.drafts?.external_audio_editor?.value;
      if (saved?.schema === 2) draft = { ...fresh(), ...saved, parameters: { ...fresh().parameters, ...saved.parameters } };
      else if (saved?.upload) Object.assign(draft, { upload: saved.upload, original: plain(saved.transcript), text: plain(saved.transcript) });
      if (draft.upload) {
        const asset = await api(`/api/reference-audio/${encodeURIComponent(draft.upload.id)}`);
        draft.upload = { id: asset.id, filename: asset.filename, audio_url: asset.audio_url };
      }
      if (!saved?.parameters) {
        draft.parameters.temperature = state.model?.runtime === 'gguf' ? '1' : '.7';
        draft.parameters.top_p = state.model?.runtime === 'gguf' ? '.9' : '.7';
      }
      for (const [key, input] of Object.entries(params)) UiNumber.write(input, draft.parameters[key]);
      $('random').checked = draft.parameters.random;
      ready = true;
      if (draft.resultJobId) await poll(draft.resultJobId);
    } catch (error) { failure = error.message; }
    finally {
      hydrating = false; render();
      if (ready && draft.upload && !draft.original) transcribe();
    }
  }
  for (const [key, label, min, max, step] of [
    ['maximum', 'Max duration (seconds)', '.01', '95.109', 'any'],
    ['duration', 'Target duration (seconds)', '.01', '95.109', 'any'],
    ['max_chars', 'Text chunk target', '80', '320', '1'],
    ['temperature', 'Temperature', '0', '2', 'any'], ['top_p', 'Top P', '0', '1', 'any'],
    ['top_k', 'Top K', '0', '8193', '1'], ['seed', 'Seed', '0', String(2 ** 32 - 1), '1'],
  ]) {
    const wrap = document.createElement('label'); wrap.className = 'field-label';
    const caption = document.createElement('span'); caption.dataset.i18n = label; caption.textContent = t(label);
    const input = document.createElement('input');
    Object.assign(input, { id: `audio-import-${key}`, className: 'input', type: 'number', min, max, step });
    UiNumber.write(input, draft.parameters[key]); params[key] = input;
    input.oninput = () => { draft.parameters[key] = UiNumber.value(input); draft.parameters.edited = true; render(); saveSoon(); };
    wrap.append(caption, input); $('parameters').append(wrap);
  }
  for (const id of ['text', 'replacement']) ScriptEditor.mount($(id));
  $('file').onchange = event => upload(event.target.files[0]);
  $('choose').onclick = () => $('file').click();
  for (const name of ['dragenter', 'dragover', 'dragleave', 'drop']) {
    $('dropzone').addEventListener(name, event => {
      event.preventDefault(); $('dropzone').classList.toggle('dragging', ['dragenter', 'dragover'].includes(name));
      if (name === 'drop') upload(event.dataTransfer.files[0]);
    });
  }
  $('transcribe').onclick = () => transcribe(true);
  $('transcript').oninput = () => setOriginal($('transcript').value);
  for (const [id, key] of [['text', 'text'], ['direction', 'direction'], ['replacement', 'replacement']]) {
    $(id).addEventListener('input', () => { draft[key] = $(id).value; render(); saveSoon(); });
  }
  $('quote').oninput = () => {
    const previous = draft.quote; draft.quote = $('quote').value; draft.target = null;
    if (!draft.replacement || draft.replacement === previous) draft.replacement = draft.quote;
    drawMatches(); render(); saveSoon();
  };
  $('use-selection').onclick = () => {
    const start = $('transcript').selectionStart, end = $('transcript').selectionEnd;
    if (!insertion() && start === end) return;
    draft.quote = insertion() ? '' : draft.original.slice(start, end);
    draft.target = { start: insertion() ? end : start, end };
    if (!draft.replacement) draft.replacement = draft.quote;
    drawMatches(); render(); saveSoon();
  };
  $('insert').onclick = () => {
    const audio = $('player');
    StudioAudio.setSelection(audio, audio.currentTime, audio.currentTime, { emit: true });
    draft.target = null; draft.quote = ''; $('original').open = true;
    render(); $('transcript').focus();
  };
  for (const input of [$('start'), $('end')]) input.onchange = () => {
    const start = UiNumber.read($('start')), end = UiNumber.read($('end'));
    if (start >= 0 && end >= start) StudioAudio.setSelection($('player'), start, end, { emit: true });
  };
  $('player').addEventListener('drama:selectionchange', event => {
    draft.range = event.detail ? copy(event.detail) : null; render(); saveSoon();
  });
  $('player').addEventListener('loadedmetadata', () => {
    if (draft.range) StudioAudio.setSelection($('player'), draft.range.start, draft.range.end);
    render();
  });
  for (const tab of document.querySelectorAll('[data-import-mode]')) tab.onclick = () => {
    draft.mode = tab.dataset.importMode; drawMatches(); render(); saveSoon();
  };
  document.querySelector('#view-audio-import [role="tablist"]').onkeydown = event => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key) || controlsBusy()) return;
    event.preventDefault();
    const tabs = [...document.querySelectorAll('[data-import-mode]')];
    const current = tabs.findIndex(tab => tab.dataset.importMode === draft.mode);
    const index = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (current + 1) % tabs.length;
    tabs[index].click(); tabs[index].focus();
  };
  $('remove').onclick = () => {
    if (controlsBusy()) return;
    request++; asrBusy = false; asrStatus = ''; failure = ''; resetResult();
    const parameters = draft.parameters; draft = fresh(); draft.parameters = parameters;
    StudioAudio.clearSelection($('player')); render(); persist().catch(() => {});
  };
  $('random').onchange = () => { draft.parameters.random = $('random').checked; render(); saveSoon(); };
  $('close-settings').onclick = () => { $('settings').open = false; };
  $('submit').onclick = generate;
  $('cancel').onclick = async () => {
    if (!active() || !job?.id) return;
    try {
      await api(`/api/jobs/${encodeURIComponent(job.id)}/cancel`, { method: 'POST', body: '{}' });
      job.cancel_requested = true; render();
    } catch (error) { failure = error.message; render(); }
  };
  document.getElementById('view-audio-import').addEventListener('keydown', event => {
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter' && !event.isComposing) {
      event.preventDefault(); event.stopPropagation(); generate();
    }
  });
  document.addEventListener('drama:viewchange', event => { if (event.detail === 'audio-import') hydrate(); });
  window.addEventListener('drama:startup-ready', () => { if (document.documentElement.dataset.view === 'audio-import') hydrate(); });
  for (const event of ['drama:capabilities', 'drama:resourceschange', 'drama:languagechange']) {
    document.addEventListener(event, () => { if (ready) { drawMatches(); render(); } });
  }
  render();
})();
