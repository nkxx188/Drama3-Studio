/* Shared preparation for character references and additional clips of one voice. */
(() => {
  'use strict';
  const runtime = new WeakMap();
  const post = (url, value) => api(url, { method: 'POST', body: JSON.stringify(value) });
  const fresh = () => ({ key: crypto.randomUUID(), upload: null, voiceId: '', transcript: '' });
  function syncLibrary(select) {
    const reference = select.referenceDraft;
    select.replaceChildren(label(new Option('', ''), 'No reference'));
    for (const voice of state.voices) select.add(new Option(voice.name, voice.id));
    select.value = reference.voiceId || '';
    StudioControls.sync(select.id);
  }
  const element = (tag, className = '') => {
    const value = document.createElement(tag); value.className = className; return value;
  };
  function label(value, source, attribute = '') {
    value.setAttribute(attribute ? `data-i18n-${attribute}` : 'data-i18n', source);
    if (attribute) value.setAttribute(attribute, t(source)); else value.textContent = t(source);
    return value;
  }
  function button(source, action) {
    const value = label(element('button', 'text-btn'), source);
    value.type = 'button'; value.onclick = () => Promise.resolve().then(action).catch(error => {
      document.dispatchEvent(new CustomEvent('drama:editor-notice', { detail: error.message }));
    });
    return value;
  }
  function live(reference) {
    if (!runtime.has(reference)) runtime.set(reference, { request: 0, edits: 0, pending: null, busy: false, status: '' });
    return runtime.get(reference);
  }
  function cancel(reference) {
    const value = live(reference); value.request++; value.pending = null; value.status = '';
    value.recognized = null; value.error = ''; delete reference.asrTask;
    for (const child of reference.groupReferences || []) cancel(child);
    value.render?.();
  }
  const identity = reference => reference.upload?.id
    || `${reference.voiceId || ''}:${reference.referenceIndex ?? 0}`;
  const edited = reference => { reference.editRevision = (reference.editRevision || 0) + 1; };
  function forVoice(voiceId, index = 0, bundle = true) {
    const voice = state.voices.find(item => item.id === voiceId);
    const clips = voice?.references || (voice ? [voice] : []);
    const clip = clips[index];
    if (!clip) return { ...fresh(), voiceId, libraryId: voiceId, referenceIndex: index };
    return { ...fresh(), voiceId, libraryId: voiceId, referenceIndex: index,
      transcript: (clip.transcript || '').replace(/^\s*<\|speaker:\d+\|>\s*/, ''),
      groupReferences: bundle ? clips.slice(1).map((_clip, offset) => forVoice(voiceId, offset + 1, false)) : [] };
  }
  async function uploadFile(file) {
    if (!/\.(wav|mp3|flac|ogg|m4a)$/i.test(file.name)) {
      throw new Error(t('Choose a WAV, MP3, FLAC, OGG or M4A reference'));
    }
    if (file.size < 32 || file.size > 128 * 1024 ** 2) {
      throw new Error(t('Reference audio must be nonempty and no larger than 128 MiB'));
    }
    const body = new FormData(); body.append('audio', file);
    const response = await fetch('/api/reference-audio', { method: 'POST', body });
    const value = await response.json();
    if (!response.ok) throw new Error(value.error || `HTTP ${response.status}`);
    return api(`/api/reference-audio/${value.id}`);
  }
  function source(reference) {
    if (reference.upload) return reference.upload;
    const voice = state.voices.find(item => item.id === reference.voiceId);
    return voice?.references?.[reference.referenceIndex ?? 0] || voice;
  }
  function transcribe(reference, changed, force = false) {
    const value = live(reference);
    if (!source(reference) || value.pending || value.missing) return value.pending;
    const saved = !force && reference.asrTask;
    const request = ++value.request, revision = saved ? saved.editRevision : (reference.editRevision || 0);
    const target = identity(reference);
    const current = () => request === value.request && target === identity(reference);
    value.error = ''; value.status = 'Preparing transcription…'; value.render?.();
    const pending = (async () => {
      try {
        let task = saved && saved.source === target ? await api(`/api/transcriptions/${saved.id}`)
          : await post('/api/transcriptions', {
            ...(reference.upload ? { upload_id: reference.upload.id }
              : { voice_id: reference.voiceId, reference_index: reference.referenceIndex ?? 0 }),
            language: 'auto', force, local_only: true,
          });
        if (!current()) return;
        if (task.status === 'skipped') { value.status = ''; return; }
        reference.asrTask = { id: task.id, source: target, editRevision: revision }; changed();
        while (current() && ['queued', 'running'].includes(task.status)) {
          value.status = 'Transcribing…'; value.render?.();
          await new Promise(resolve => setTimeout(resolve, 650));
          if (!current()) return;
          task = await api(`/api/transcriptions/${task.id}`);
        }
        if (!current()) return;
        if (task.status === 'skipped') { value.status = ''; return; }
        if (task.status !== 'complete') throw new Error(task.error || 'Transcription failed.');
        value.recognized = task.result?.text || '';
        if (!value.recognized.trim()) throw new Error('No speech was recognized. Enter the transcript manually or try another clip.');
        if ((reference.editRevision || 0) === revision) {
          reference.transcript = value.recognized; value.recognized = null;
        }
        value.status = '';
        changed();
      } catch (error) {
        if (current()) {
          value.status = 'Transcription failed. Enter the transcript manually or try again.';
          value.error = message(error.message);
        }
      } finally {
        if (current()) { value.pending = null; delete reference.asrTask; value.render?.(); changed(); }
      }
    })();
    value.pending = pending; value.render?.(); return pending;
  }
  function mount(reference, { prefix, changed, remove, allowLibrary = true } = {}) {
    if (reference.voiceId && reference.referenceIndex === undefined) {
      const selected = forVoice(reference.voiceId);
      Object.assign(reference, { referenceIndex: 0, libraryId: reference.voiceId,
        groupReferences: selected.groupReferences,
        transcript: reference.transcript || selected.transcript });
    }
    if (reference.voiceId && !reference.upload && !reference.transcript) {
      reference.transcript = state.voices.find(voice => voice.id === reference.voiceId)?.transcript
        .replace(/^\s*<\|speaker:\d+\|>\s*/, '') || '';
    }
    delete reference.asrMode; reference.key ||= reference.id || crypto.randomUUID();
    reference.transcript ||= '';
    const value = live(reference), root = element('div', 'reference-preparation');
    root.dataset.referenceKey = reference.key; root.boundReference = reference;
    const material = element('details', 'reference-material');
    const summary = element('summary', 'reference-compact-heading');
    const summaryCopy = element('span', 'reference-compact-copy');
    const summaryName = element('strong'), summaryText = element('span', 'reference-compact-transcript');
    const summaryDuration = element('span', 'field-help'), summaryPlay = element('button', 'reference-compact-play icon-btn');
    summaryPlay.type = 'button';
    summaryCopy.append(summaryName, summaryText); summary.append(summaryCopy, summaryDuration, summaryPlay);
    const heading = element('div', 'reference-preparation-heading');
    heading.hidden = !remove;
    if (remove) heading.append(button('Remove audio', () => { cancel(reference); remove(); }));
    const library = label(element('select'), 'Reference voices', 'aria-label'); library.id = `${prefix}-voice`;
    library.dataset.referenceLibrary = 'true'; library.referenceDraft = reference;
    syncLibrary(library);
    const file = element('input'); file.type = 'file'; file.accept = 'audio/*'; file.id = `${prefix}-upload`; file.hidden = true;
    label(file, 'Reference audio', 'aria-label');
    const upload = label(element('button', 'reference-upload-button secondary-btn'), 'Upload audio to clone a voice');
    upload.type = 'button'; upload.onclick = () => file.click();
    const sources = element('div', 'reference-source-controls');
    if (allowLibrary) sources.append(library);
    sources.append(upload, file);
    const filename = element('span', 'reference-filename field-help');
    const audio = element('audio'); audio.id = `${prefix}-audio`; audio.controls = true;
    audio.preload = 'metadata'; audio.dataset.selection = 'true';
    const transcript = label(element('textarea', 'compact-input'), 'Reference transcript', 'aria-label');
    transcript.id = `${prefix}-transcript`;
    const actions = element('div', 'role-reference-actions');
    const recognize = button('Transcribe again', () => transcribe(reference, changed, true));
    const apply = button('Use recognized text', () => {
      reference.transcript = value.recognized || ''; value.recognized = null; value.edits++; edited(reference);
      value.status = ''; value.render(); changed();
    });
    const crop = button('Use selected audio', async () => {
      const selection = StudioAudio.selection(audio);
      if (!selection || selection.end - selection.start < .5) {
        throw new Error(t('Choose at least 0.5 seconds of reference audio'));
      }
      const before = { upload: reference.upload, voiceId: reference.voiceId, transcript: reference.transcript };
      cancel(reference); const request = value.request;
      value.busy = true; value.status = 'Preparing selected audio…'; value.render();
      try {
        const current = source(reference);
        const filename = current.filename || current.audio || decodeURIComponent(new URL(
          current.audio_url, location.origin).pathname.split('/').at(-1));
        const original = reference.upload || await uploadFile(new File(
          [await (await fetch(current.audio_url)).blob()], filename));
        const cropped = await post('/api/reference-audio/crop', { id: original.id, ...selection });
        if (request !== value.request) return;
        reference.original ||= before; reference.upload = cropped; reference.voiceId = '';
        reference.transcript = ''; value.edits++; edited(reference); changed();
        value.status = 'Audio cropped. Review the transcript.';
      } finally { value.busy = false; value.render(); }
      transcribe(reference, changed);
    });
    const restore = button('Restore original audio', () => {
      cancel(reference); Object.assign(reference, reference.original); reference.original = null;
      value.missing = false;
      value.edits++; edited(reference); value.render(); changed();
      if (!reference.transcript) transcribe(reference, changed);
    });
    const status = element('span', 'field-help reference-status');
    status.setAttribute('role', 'status'); status.setAttribute('aria-live', 'polite');
    status.setAttribute('aria-atomic', 'true');
    const group = element('details', 'reference-group-clips');
    const groupTitle = label(element('summary'), 'More reference clips');
    const groupBody = element('div'); group.append(groupTitle, groupBody);
    actions.append(apply, crop, restore);
    const transcriptLabel = label(element('label', 'field-label'), 'Reference transcript'); transcriptLabel.htmlFor = transcript.id;
    const transcriptHeading = element('div', 'transcription-heading');
    transcriptHeading.append(transcriptLabel, recognize);
    material.append(summary, heading, sources, filename, audio, transcriptHeading, status, transcript, actions, group);
    root.append(material);
    const updateSummary = () => {
      const current = source(reference), valid = Number.isFinite(audio.duration) && audio.duration > 0;
      summaryName.textContent = current?.filename || current?.name || current?.audio || reference.voiceId || '';
      summaryText.textContent = reference.transcript?.trim() || t('Reference transcript needed');
      if (reference.transcript?.trim()) delete summaryText.dataset.i18n;
      else summaryText.dataset.i18n = 'Reference transcript needed';
      summaryDuration.textContent = valid ? `${audio.duration.toFixed(1)}s` : '';
      summaryPlay.disabled = !valid || Boolean(value.missing);
      summaryPlay.innerHTML = icon(audio.paused ? 'play' : 'pause');
      label(summaryPlay, audio.paused ? 'Preview voice' : 'Pause audio', 'aria-label');
    };
    for (const event of ['loadedmetadata', 'emptied', 'play', 'pause']) audio.addEventListener(event, updateSummary);
    audio.addEventListener('error', () => {
      if (!audio.getAttribute('src')) return;
      value.missing = true; value.status = 'Reference audio is missing'; value.render?.();
    });
    summaryPlay.onclick = event => {
      event.preventDefault(); event.stopPropagation();
      if (!audio.paused) audio.pause();
      else {
        document.querySelectorAll('audio').forEach(other => { if (other !== audio) other.pause(); });
        audio.play().catch(error => document.dispatchEvent(new CustomEvent(
          'drama:editor-notice', { detail: error.message })));
      }
    };
    material.addEventListener('toggle', () => {
      if (!source(reference) || Boolean(reference.collapsed) === !material.open) return;
      reference.collapsed = !material.open; changed();
    });
    value.render = () => {
      const current = source(reference);
      const hasReference = Boolean(current || reference.voiceId || reference.upload);
      if (reference.voiceId && !current) value.status = 'Reference audio is missing';
      root.classList.toggle('has-audio', Boolean(current && !value.missing));
      summary.hidden = !hasReference;
      material.open = !hasReference || !reference.collapsed;
      updateSummary();
      filename.textContent = current?.filename || current?.audio || current?.name || '';
      library.value = reference.voiceId || '';
      const url = current?.audio_url || (reference.upload?.audio_path && fileUrl(reference.upload.audio_path));
      if (url && audio.getAttribute('src') !== url) audio.src = url;
      if (reference.upload && !url && value.lookup !== reference.upload.id) {
        const identifier = reference.upload.id;
        value.lookup = identifier;
        api(`/api/reference-audio/${identifier}`).then(uploaded => {
          if (reference.upload?.id === identifier) { reference.upload = uploaded; value.render(); changed(); }
        }).catch(error => {
          if (reference.upload?.id !== identifier) return;
          value.missing = true; value.status = 'Reference audio is missing'; value.error = error.message; value.render();
        });
      }
      for (const item of [filename, audio, transcriptHeading, transcript, actions]) item.hidden = !hasReference;
      audio.hidden = !current || Boolean(value.missing);
      if (transcript.value !== reference.transcript) transcript.value = reference.transcript;
      file.disabled = upload.disabled = library.disabled = value.busy;
      recognize.disabled = Boolean(value.pending || value.busy || value.missing || !current);
      apply.hidden = !value.recognized;
      const selection = StudioAudio.selection(audio);
      crop.disabled = value.busy || !selection || selection.end - selection.start < .5;
      restore.hidden = !reference.original; restore.disabled = value.busy;
      status.textContent = t(value.status); status.title = value.error || '';
      status.hidden = !value.status;
      status.dataset.state = value.pending || value.busy ? 'busy' : value.error || value.missing ? 'error' : 'idle';
      transcript.setAttribute('aria-busy', String(Boolean(value.pending || value.busy)));
      if (I18n.has(value.status)) status.dataset.i18n = value.status; else delete status.dataset.i18n;
      group.hidden = !(reference.groupReferences || []).length;
      const existing = new Map([...groupBody.children].map(child => [child.dataset.referenceKey, child]));
      for (const child of reference.groupReferences || []) {
        if (!existing.has(child.key)) groupBody.append(mount(child, {
          prefix: `${prefix}-clip-${child.key}`, changed, allowLibrary: false,
          remove: () => {
            reference.groupReferences = reference.groupReferences.filter(item => item !== child);
            value.render(); changed();
          },
        }));
        existing.delete(child.key);
      }
      for (const child of existing.values()) child.remove();
      if (root.isConnected) {
        StudioAudio.enhance(root);
        if (allowLibrary) StudioControls.sync(library.id);
      }
    };
    transcript.oninput = () => { reference.transcript = transcript.value; value.edits++; edited(reference); changed(); };
    audio.addEventListener('drama:selectionchange', value.render);
    const acceptFile = selected => {
      if (!selected || value.busy) return;
      cancel(reference); const request = value.request;
      value.busy = true; value.status = 'Uploading…'; value.render();
      uploadFile(selected).then(uploaded => {
        if (request !== value.request) return;
        reference.upload = uploaded; reference.voiceId = ''; reference.transcript = '';
        value.missing = false;
        reference.collapsed = false;
        reference.original = null; value.edits++; edited(reference); value.status = ''; changed();
        transcribe(reference, changed);
      }).catch(error => { value.status = message(error.message); })
        .finally(() => { value.busy = false; file.value = ''; value.render(); });
    };
    file.onchange = () => acceptFile(file.files[0]);
    for (const event of ['dragenter', 'dragover']) sources.addEventListener(event, input => {
      input.preventDefault(); if (!value.busy) sources.classList.add('dragging');
    });
    for (const event of ['dragleave', 'drop']) sources.addEventListener(event, input => {
      input.preventDefault(); sources.classList.remove('dragging');
    });
    sources.addEventListener('drop', event => acceptFile(event.dataTransfer?.files?.[0]));
    library.onchange = () => {
      const selected = library.value;
      cancel(reference);
      const selectedVoice = selected ? forVoice(selected) : fresh();
      Object.assign(reference, {
        voiceId: selectedVoice.voiceId, libraryId: selectedVoice.libraryId || '', referenceIndex: 0,
        upload: null, original: null, transcript: selectedVoice.transcript,
        groupReferences: selectedVoice.groupReferences || [],
        collapsed: false,
      });
      value.missing = false;
      value.edits++; edited(reference); value.render(); changed();
    };
    value.render();
    if (reference.asrTask || source(reference) && !reference.transcript.trim()) {
      Promise.resolve().then(() => transcribe(reference, changed));
    }
    return root;
  }
  async function read(reference, speaker) {
    const value = live(reference), target = identity(reference);
    if (value.busy) throw new Error(t('Wait for reference preparation to finish.'));
    if (!reference.upload?.id && !reference.voiceId) return null;
    if (value.missing || !source(reference)) throw new Error(t('Reference audio is missing'));
    const binding = { ...(reference.upload ? { id: reference.upload.id }
      : { voice_id: reference.voiceId, reference_index: reference.referenceIndex ?? 0 }), speaker };
    let text = reference.transcript.trim();
    if (!text && value.pending) {
      await value.pending; text = reference.transcript.trim();
    }
    if (target !== identity(reference)) {
      throw new Error(t('Reference changed. Click Generate again.'));
    }
    if (!text) throw new Error(t('Enter the exact words spoken in the reference audio.'));
    return { ...binding, text };
  }
  async function readAll(reference, speaker) {
    const [binding, children] = await Promise.all([
      read(reference, speaker), Promise.all((reference.groupReferences || []).map(child => readAll(child, speaker))),
    ]);
    return [...(binding ? [binding] : []), ...children.flat()];
  }
  function previews(reference, speaker) {
    const clip = source(reference);
    const rows = clip ? [{
      audio: clip.audio_path, audio_url: clip.audio_url, text: reference.transcript, speaker,
    }] : [];
    return rows.concat((reference.groupReferences || []).flatMap(child => previews(child, speaker)));
  }
  window.ReferencePrep = Object.freeze({
    fresh, forVoice, mount, read, readAll, previews, cancel,
    resume(reference, changed) {
      if ((reference.asrTask || source(reference) && !reference.transcript?.trim()) && !live(reference).pending) {
        transcribe(reference, changed);
      }
      for (const child of reference.groupReferences || []) this.resume(child, changed);
    },
    refreshLibraries() { for (const select of document.querySelectorAll('select[data-reference-library]')) syncLibrary(select); },
  });
})();
