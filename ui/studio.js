/* Creative state stays separate from immutable jobs and the inference protocol. */
(() => {
  'use strict';
  const uid = () => crypto.randomUUID();
  const randomSeed = () => crypto.getRandomValues(new Uint32Array(1))[0];
  const copy = value => JSON.parse(JSON.stringify(value));
  const post = (url, value) => api(url, { method: 'POST', body: JSON.stringify(value) });
  const node = (tag, className = '', text = '') => {
    const value = document.createElement(tag); value.className = className; value.textContent = text; return value;
  };
  const localized = (tag, className, source, args = {}) => {
    const value = node(tag, className, t(source, args)); value.dataset.i18n = source;
    if (Object.keys(args).length) value.dataset.i18nArgs = JSON.stringify(args); return value;
  };
  const command = (label, action, className = 'secondary-btn') => {
    const button = node('button', className, t(label)); button.type = 'button';
    button.dataset.i18n = label; button.onclick = action; return button;
  };
  const field = (label, input) => {
    const box = node('div'); const caption = node('label', 'field-label', t(label));
    caption.dataset.i18n = label; if (input.id) caption.htmlFor = input.id;
    box.append(caption, input); return box;
  };
  const input = (id, value = '', type = 'text') => {
    const control = node(type === 'textarea' ? 'textarea' : 'input',
      type === 'textarea' ? 'compact-input' : type === 'checkbox' ? '' : 'input');
    control.id = id; if (type !== 'textarea') control.type = type; control.value = value; return control;
  };
  const plainText = text => text.replace(/<\|speaker:\d+\|>/g, '')
    .replace(/<\/?(whisper|emphasis|soft|fast|slow|stress|forceful)>/g, '').replace(/\[[^\]\n]*\]/g, '').trim();
  const fresh = layout => ({ work_id: uid(), segment_id: uid(), layout });
  const segmentsOf = job => job.result?.segments?.length ? job.result.segments
    : (job.events || []).filter(event => ['segment', 'audio'].includes(event.stage) && (event.path || event.output));
  const segmentAt = (job, index) => segmentsOf(job).find((row, offset) => (row.index ?? offset) === index);
  const modelMismatch = expected => expected && state.model?.selected_directory
    && expected.replaceAll('\\', '/').toLowerCase() !== state.model.selected_directory.replaceAll('\\', '/').toLowerCase();
  const flow = { ready: false, saving: Promise.resolve(), data: { drafts: {}, works: {} },
    single: fresh('single'), scene: { ...fresh('scene'), roles: [], lines: [], strategy: 'joint' },
    design: fresh('design'), voiceSource: 'reference', designLegacy: null, writing: null, referenceTask: null,
    operation: null, returnDrafts: [], restoring: false, lastVoice: null,
    watch: null, lastSubmit: null, singleForm: null, singleReferences: [], libraryReferences: [],
    versionFocus: null, workJobs: {}, missingJobs: new Set(), historyLoads: new Map(),
    failedHistories: new Set(), restoredFrom: null, restoredModel: null, designRecipe: null,
    contexts: new Map(), contextLoads: new Map(), streamDraft: null, switchingInput: false };
  flow.voiceDraft = { main: ReferencePrep.fresh(), extraReferences: [] };
  const formIds = ['text', 'direction', 'language', 'seed', 'duration',
    'target-seconds', 'max-chars', 'temperature', 'top-p', 'top-k', 'candidate-count', 'chunks',
    'source-take', 'previous-take', 'start-time', 'end-time', 'start-frame', 'end-frame'];
  let saveTimer;
  function fail(error, source = 'action') {
    const notice = $('studio-error');
    notice.querySelector('.studio-error-message').textContent = message(error?.message || String(error));
    notice.dataset.source = source;
    notice.hidden = false;
  }
  function clearError() {
    $('studio-error').hidden = true; delete $('studio-error').dataset.source;
  }
  async function guarded(action) {
    try { clearError(); await action(); } catch (error) { fail(error); }
  }
  function settings({ checkDuration = false, scene = false } = {}) {
    const control = id => $(scene ? `scene-${id}` : id);
    for (const id of ['max-chars', 'temperature', 'top-p', 'top-k',
      ...(scene ? [] : ['candidate-count'])]) {
      if (!control(id).value || !control(id).checkValidity()) throw new Error(t('Check generation settings.'));
    }
    const values = {
      seed: control('random-seed').checked ? randomSeed() : UiNumber.read(control('seed')),
      automatic_length: true,
      max_chunk_chars: Math.round(UiNumber.read(control('max-chars'))),
      temperature: UiNumber.read(control('temperature')), top_p: UiNumber.read(control('top-p')),
      top_k: UiNumber.read(control('top-k')),
    };
    if (!(values.temperature > 0 && values.top_p > 0)) throw new Error(t('Check generation settings.'));
    const duration = $('target-seconds');
    if (checkDuration && !$('duration').disabled && duration.value) {
      if (!duration.checkValidity()) {
        throw new Error(t('Target duration exceeds the maximum generation length'));
      }
    }
    if (!control('random-seed').checked && (!control('seed').checkValidity() || !control('seed').value)) {
      throw new Error(t('Check generation settings.'));
    }
    return values;
  }
  function capture() {
    rememberWriting();
    const form = Object.fromEntries(formIds.filter(id => $(id)).map(id => [id, UiNumber.value($(id))]));
    Object.assign(form, flow.writing.reference);
    const task = flow.referenceTask;
    if (task) Object.assign(form, { 'previous-take': task.previous, 'source-take': task.source });
    return { context: copy(flow.single), form, mode: task?.mode || $('mode').value,
      random: $('random-seed').checked, progressive: $('progressive-audio').checked,
      autoplay: $('autoplay-audio').checked, streamLive: $('stream-live').checked,
      streamDraft: copy(flow.streamDraft),
      operation: copy(task ? task.operation : flow.operation), returnDrafts: copy(flow.returnDrafts.slice(-4)),
      references: copy(flow.singleReferences),
      libraryReferences: copy(flow.libraryReferences), restoredFrom: copy(task ? task.restoredFrom : flow.restoredFrom),
      restoredModel: task ? task.restoredModel : flow.restoredModel,
      active_job_id: state.resultJobs.generate?.id || null,
      design: { context: copy(flow.design), recipe: copy(flow.designRecipe),
        description: $('voice-description').value, active: voiceDesignActive(), legacy: copy(flow.designLegacy),
        contentVersion: 2, ...copy(flow.writing.design) },
      voices: selectedVoices(), upload: state.cloneUpload, transcript: $('clone-transcript').value,
      referenceCollapsed: !$('clone-upload-preview').open,
      cloneRecognition: { task: copy(state.cloneAsrTask), textRevision: state.cloneTextRevision,
        recognized: state.cloneAsrText, status: state.cloneAsrStatus,
        needsApply: !$('apply-asr-text').classList.contains('hidden') },
      original: state.cloneOriginal ? {
        upload: state.cloneOriginal.upload, transcript: state.cloneOriginal.transcript,
      } : null };
  }
  function sceneSnapshot() {
    return { ...copy(flow.scene), script: $('dialogue-script').value, scriptVersion: 1,
      active_job_id: state.resultJobs.scene?.id || null,
      generation: { random: $('scene-random-seed').checked, form: Object.fromEntries(
        ['seed', 'max-chars', 'temperature', 'top-p', 'top-k']
          .map(id => [id, UiNumber.value($(`scene-${id}`))])) } };
  }
  function saveSoon() {
    if (!flow.ready || flow.restoring || flow.switchingInput) return;
    clearTimeout(saveTimer); saveTimer = setTimeout(() => save(), 450);
  }
  async function save() {
    if (!flow.ready || flow.restoring) return;
    try {
      if (!syncCloneAudioSelection()) return;
      if (state.cloneFile) await uploadCloneAudio();
      const single = capture(), scene = sceneSnapshot(), voice = {
        ...copy(flow.voiceDraft), name: $('voice-name').value,
      };
      flow.saving = flow.saving.catch(() => {}).then(async () => {
        await post('/api/studio/draft', { key: 'single', value: single });
        await post('/api/studio/draft', { key: 'scene', value: scene });
        await post('/api/studio/draft', { key: 'voice', value: voice });
        if ($('studio-error').dataset.source === 'save') clearError();
      });
      await flow.saving;
    } catch (error) {
      fail(error, 'save');
    }
  }
  async function uploadedFile(upload) {
    const value = await api(`/api/reference-audio/${encodeURIComponent(upload.id)}`);
    const response = await fetch(value.audio_url);
    if (!response.ok) throw new Error(t('Reference audio is missing'));
    return { file: new File([await response.blob()], value.filename, { type: response.headers.get('Content-Type') }), upload: value };
  }
  async function restore(value) {
    value = migrateWritingDraft(retireContinuation(value));
    if (value.form?.['top-k'] === '30' || value.form?.['top-k'] === 30) value.form['top-k'] = '0';
    flow.restoring = true;
    try {
      flow.voiceSource = 'reference'; flow.referenceTask = null;
      flow.writing = { reference: writingOf(value.form), design: writingOf(value.design) };
      flow.single = value.context || fresh('single'); flow.operation = value.operation || null;
      if (Object.hasOwn(value, 'streamDraft')) flow.streamDraft = copy(value.streamDraft);
      if (flow.operation && !flow.operation.legacy && !flow.operation.frozenReferences) {
        try {
          const prepared = await post('/api/studio/prepare-restore', {
            ...flow.operation.source, prepared_references: flow.operation.rewordReferences || [],
            reference_transcripts_checked: true,
          });
          flow.operation.frozenReferences = prepared.references;
          delete flow.operation.restoreError;
        } catch (error) { flow.operation.restoreError = 'Source conditions could not be restored.'; fail(error); }
      }
      flow.returnDrafts = value.returnDrafts || [];
      flow.singleReferences.forEach(ReferencePrep.cancel);
      flow.singleReferences = value.references || [];
      flow.libraryReferences.forEach(ReferencePrep.cancel);
      flow.libraryReferences = value.libraryReferences || [];
      flow.restoredFrom = value.restoredFrom || null; flow.restoredModel = value.restoredModel || null;
      flow.design = value.design?.context || fresh('design'); flow.designRecipe = value.design?.recipe || null;
      $('voice-description').value = value.design?.description || '';
      flow.designLegacy = value.design?.migrationBackup || value.design?.legacy || null;
      renderExtraReferences();
      clearCloneAudio();
      if (value.upload) {
        let current;
        try { current = await uploadedFile(value.upload); }
        catch (_error) { showMissingClone(value.upload, value.transcript); }
        if (current) {
          let original = null;
          if (value.original) {
            try { original = { ...await uploadedFile(value.original.upload), transcript: value.original.transcript }; }
            catch (error) { fail(error); }
          }
          selectCloneAudio(current.file, { auto: false, upload: current.upload, original });
          $('clone-transcript').value = value.transcript || '';
        }
        $('clone-upload-preview').open = !value.referenceCollapsed;
      } else {
        state.selectedVoiceIds = new Set(value.voices || []); updateVoiceSelection();
      }
      for (const [id, text] of Object.entries(value.form || {})) if ($(id)) UiNumber.restore($(id), text);
      const recognition = value.cloneRecognition || {};
      state.cloneTextRevision = recognition.textRevision || 0;
      state.cloneAsrTask = recognition.task || null; state.cloneAsrText = recognition.recognized || '';
      state.cloneAsrStatus = recognition.status || '';
      $('apply-asr-text').classList.toggle('hidden', !recognition.needsApply);
      renderTranscriptionStatus();
      $('random-seed').checked = value.random !== false;
      $('progressive-audio').checked = Boolean(value.progressive);
      $('autoplay-audio').checked = Boolean(value.autoplay);
      $('stream-live').checked = Boolean(value.streamLive);
      $('mode').value = value.mode || 'tts'; updateMode(); updateEditorMeta(); enhanceAllSelects();
      if (value.design?.active) switchVoiceSource('design', false);
      else showWriting('reference');
      if (value.mode === 'infill' && $('source-take').value) {
        state.sourceTake = null; $('infill-source-preview').classList.add('hidden');
        try {
          const take = await post('/api/takes/inspect', { path: $('source-take').value });
          showInfillSource(take, {
            start: Number(value.form['start-frame']) * take.samples_per_frame / take.sample_rate,
            end: Number(value.form['end-frame']) * take.samples_per_frame / take.sample_rate,
          });
        } catch (error) {
          if (flow.operation) flow.operation.restoreError = 'Source conditions could not be restored.';
          fail(error);
        }
      }
      renderLibraryReferences(true); renderOperation();
    } finally { flow.restoring = false; }
    if (flow.ready && !voiceDesignActive()) resumeCloneTranscription();
  }
  function writingOf(value = {}) {
    return { text: typeof value.text === 'string' ? value.text : '',
      direction: typeof value.direction === 'string' ? value.direction : '' };
  }
  function migrateWritingDraft(value) {
    const current = copy(value), design = current.design ||= {};
    if (design.contentVersion !== 2) {
      const visible = writingOf(current.form);
      if (design.active) {
        const previous = [...(current.returnDrafts || [])].reverse()
          .find(item => !item.design?.active && item.mode !== 'stream');
        current.form = { ...current.form, ...writingOf(previous?.form) };
        design.text = visible.text; design.direction = visible.direction;
        design.migrationBackup = { ...visible, legacy: design.legacy || null };
      } else Object.assign(design, writingOf(design.legacy || design));
      design.contentVersion = 2;
    }
    if (current.mode === 'stream') {
      current.retiredTextStream = { form: copy(current.form), live: current.streamLive };
      const chunks = String(current.form?.chunks || '').split(/\r?\n/).filter(line => line.trim());
      current.form = { ...current.form,
        text: chunks.length ? chunks.join('\n') : current.form?.text || '',
        'previous-take': '', 'source-take': '' };
      current.mode = 'tts'; current.operation = current.restoredFrom = current.restoredModel = null;
      current.streamLive = false; design.active = false;
    }
    return current;
  }
  function rememberWriting() {
    flow.writing ||= { reference: writingOf({ text: $('text').value, direction: $('direction').value }),
      design: { text: '', direction: '' } };
    flow.writing[voiceDesignActive() ? 'design' : 'reference'] = {
      text: $('text').value, direction: $('direction').value,
    };
  }
  function showWriting(source) {
    const writing = flow.writing[source];
    $('text').value = writing.text; $('direction').value = writing.direction;
    ScriptEditor.reset($('text')); updateEditorMeta();
  }
  function switchVoiceSource(source, remember = true) {
    if (remember) rememberWriting();
    if (source === 'design' && flow.voiceSource !== 'design') {
      flow.referenceTask = { mode: $('mode').value, operation: copy(flow.operation),
        restoredFrom: copy(flow.restoredFrom), restoredModel: flow.restoredModel,
        previous: $('previous-take').value, source: $('source-take').value };
      flow.operation = flow.restoredFrom = flow.restoredModel = null; $('mode').value = 'tts';
    } else if (source === 'reference' && flow.referenceTask) {
      const task = flow.referenceTask;
      flow.operation = task.operation; flow.restoredFrom = task.restoredFrom; flow.restoredModel = task.restoredModel;
      $('mode').value = task.mode; $('previous-take').value = task.previous; $('source-take').value = task.source;
      flow.referenceTask = null;
    }
    flow.voiceSource = source;
    for (const details of document.querySelectorAll('.voice-source-bar > details[open]')) details.open = false;
    $('clone-section').querySelectorAll('audio').forEach(audio => audio.pause()); stopVoicePreview();
    showWriting(source); updateMode(); renderOperation(); saveSoon();
    if (source === 'design') window.ReferencePrewarm?.cancel();
    else if (remember) window.ReferencePrewarm?.schedule();
  }
  function retireContinuation(value) {
    if (value.operation?.action !== 'continue' && !(value.mode === 'edit' && !value.operation)) return value;
    const current = copy(value);
    current.retiredContinuation = { operation: current.operation, previous: current.form?.['previous-take'] };
    current.operation = null; current.mode = 'tts';
    current.restoredFrom = current.restoredModel = null;
    current.context = fresh('single');
    current.form = { ...current.form, 'previous-take': '', 'source-take': '' };
    return current;
  }
  function stash(current = capture()) {
    delete current.returnDrafts;
    delete current.streamDraft;
    if (current.mode === 'stream') flow.streamDraft = copy(current);
    flow.returnDrafts.push(current); flow.returnDrafts = flow.returnDrafts.slice(-4);
  }
  function renderOperation() {
    const box = $('creative-operation'), back = $('return-draft'); box.replaceChildren();
    const restoredFrom = voiceDesignActive() ? flow.designRecipe?.source : flow.restoredFrom;
    const restoredModel = voiceDesignActive() ? flow.designRecipe?.model : flow.restoredModel;
    if (flow.operation) {
      const labels = { revision: 'Adjust performance', reword: 'Rewrite line', infill: 'Edit selected region' };
      box.append(node('strong', '', t(labels[flow.operation.action] || 'Edit')));
      const source = flow.operation.source.job_id.slice(0, 6);
      box.append(node('span', 'field-help', t('Based on {version}', { version: source })));
      if (flow.operation.legacy) box.append(node('span', 'field-help',
        t('Original settings unavailable; using the displayed settings.')));
      if (flow.operation.restoreError) box.append(localized('span', 'field-help', flow.operation.restoreError),
        command('Reload saved source', () => guarded(async () => {
          const saved = capture(); delete saved.operation.frozenReferences;
          await restore(saved); saveSoon();
        }), 'text-btn'));
      if (flow.operation.rewordReferences?.length) box.append(localized('span', 'field-help',
        '{count} reference clips', { count: flow.operation.rewordReferences.length }));
      if (modelMismatch(flow.operation.model)) box.append(localized('span', 'field-help',
        'This source uses another model. Switch models or start a new draft.'));
    }
    box.hidden = !flow.operation;
    if (!flow.operation && restoredFrom) {
      box.hidden = false;
      box.append(localized('strong', '', 'Restored settings'),
        node('span', 'field-help', restoredFrom.job_id.slice(0, 6)));
      const selected = state.model?.selected_directory?.replaceAll('\\', '/').toLowerCase();
      const expected = restoredModel?.replaceAll('\\', '/').toLowerCase();
      if (expected && expected !== selected) {
        box.append(localized('span', 'field-help', 'This draft uses another model. Switch explicitly or use the selected model.'),
          command('Use selected model', () => {
            if (voiceDesignActive()) flow.designRecipe.model = state.model?.selected_directory;
            else flow.restoredModel = state.model?.selected_directory;
            renderOperation(); saveSoon();
          }, 'text-btn'));
      }
    }
    back.hidden = voiceDesignActive() || !flow.returnDrafts.length;
    box.append(back); box.hidden = box.hidden && back.hidden;
    const infill = flow.operation?.action === 'infill';
    $('transcript-diff').hidden = !infill;
    if (infill) renderDiff();
    syncControls();
  }
  function renderDiff() {
    const box = $('transcript-diff'); box.replaceChildren();
    const before = flow.operation?.original || '', after = $('text').value;
    let start = 0, tail = 0;
    while (start < Math.min(before.length, after.length) && before[start] === after[start]) start++;
    while (tail < Math.min(before.length, after.length) - start
      && before[before.length - 1 - tail] === after[after.length - 1 - tail]) tail++;
    box.append(node('span', '', after.slice(0, start)));
    const removed = before.slice(start, before.length - tail), added = after.slice(start, after.length - tail);
    if (removed) box.append(node('del', '', removed));
    if (added) box.append(node('ins', '', added));
    if (tail) box.append(node('span', '', after.slice(after.length - tail)));
  }
  async function derive(source, action, selection = null) {
    if (action === 'continue') throw new Error(t('Not supported by this model'));
    await save();
    const version = await post('/api/studio/version', source);
    if (!version.take_file) throw new Error(t('No editable take'));
    const reword = action === 'reword' ? await prepareMissingVoices(version, source) : { bindings: [], previews: [] };
    if (!reword) return;
    if (version.recipe?.scene_strategy === 'joint' && source.part_index === undefined && action !== 'infill') {
      await deriveScene(version, source, action, reword); return;
    }
    stash();
    const oldReturn = flow.returnDrafts;
    const recipe = version.recipe || {};
    flow.restoring = true;
    try {
      if (voiceDesignActive()) switchVoiceSource('reference');
      clearCloneAudio(); flow.single = { ...(version.studio || fresh('single')), layout: 'single' };
      if (source.part_index !== undefined) flow.single.line_id = version.line_id;
      flow.operation = { action, source, original: version.text, model: recipe.model || null,
        rewordReferences: reword.bindings, legacy: !version.complete_recipe,
        frozenReferences: [...(recipe.reference_snapshot || []), ...reword.previews] };
      flow.versionFocus = source.part_index === undefined ? null : { ...source, line_id: version.line_id };
      $('mode').value = action === 'infill' ? 'infill' : action === 'revision' ? 'revision' : 'edit';
      $('text').value = version.text;
      $('direction').value = action === 'revision' ? '' : recipe.direction || '';
      $('language').value = recipe.language || '';
      if (recipe.max_chunk_chars !== undefined) UiNumber.write($('max-chars'), recipe.max_chunk_chars);
      UiNumber.write($('target-seconds'), recipe.duration_frames ? recipe.duration_frames * 2048 / 44100 : '');
      if (recipe.seed !== undefined) $('seed').value = recipe.seed;
      for (const [key, id] of [['temperature', 'temperature'], ['top_p', 'top-p'], ['top_k', 'top-k']]) {
        if (recipe[key] !== undefined) UiNumber.write($(id), recipe[key]);
      }
      $('previous-take').value = action === 'infill' ? '' : version.take_file;
      if (action === 'infill') {
        $('source-take').value = version.take_file;
        showInfillSource(version.take, selection);
      } else showPreviousSource(version.take);
      flow.returnDrafts = oldReturn; updateMode(); updateEditorMeta(); view('generate');
      $('advanced-settings').open = false; renderOperation(); enhanceAllSelects();
    } finally { flow.restoring = false; }
    saveSoon();
  }
  function restoreScene(value) {
    if (value.operation?.action === 'continue') {
      value = copy(value);
      value.retiredContinuation = value.operation; value.operation = null;
      value.work_id = uid(); value.segment_id = uid();
      delete value.previous_take; delete value.previous_sources;
      delete value.restoredFrom; delete value.restoredMode;
    }
    for (const role of flow.scene.roles) {
      ReferencePrep.cancel(role); (role.extraReferences || []).forEach(ReferencePrep.cancel);
    }
    flow.scene = copy(value);
    if (value.generation) {
      for (const [id, text] of Object.entries(value.generation.form)) if ($(`scene-${id}`)) UiNumber.restore($(`scene-${id}`), text);
      $('scene-random-seed').checked = value.generation.random;
    }
    renderScene(); ScriptEditor.reset($('dialogue-script')); updateStrategy(); renderSceneOperation(); saveSoon();
  }
  async function deriveScene(version, source, action, reword = { bindings: [], previews: [] }) {
    const current = sceneSnapshot(); delete current.returnDrafts;
    const previous = [...(flow.scene.returnDrafts || []), current].slice(-4);
    const recipe = version.recipe;
    const roles = recipe.characters.map((role, index) => ({
      ...sceneRole(role.name, role.speaker), ...role,
      id: recipe.character_ids?.[index] || uid(),
      frozenReferences: [...(recipe.reference_snapshot || []), ...reword.previews]
        .filter(item => item.speaker === role.speaker),
    }));
    const lines = recipe.lines.map((line, index) => ({
      id: recipe.line_ids?.[index] || uid(), characterId: roles.find(role => role.name === line.character).id,
      text: line.text, direction: '',
    }));
    restoreScene({
      ...(version.studio || fresh('scene')), layout: 'scene', strategy: 'joint', roles,
      lines,
      segment_id: version.studio?.segment_id || uid(),
      scene: recipe.scene, direction: action === 'revision' ? '' : recipe.direction || '',
      operation: { action, source, model: recipe.model, rewordReferences: reword.bindings }, returnDrafts: previous,
    });
    for (const [key, id] of [['temperature', 'temperature'], ['top_p', 'top-p'], ['top_k', 'top-k'], ['seed', 'seed']]) {
      if (recipe[key] !== undefined) UiNumber.write($(`scene-${id}`), recipe[key]);
    }
    if (recipe.max_chunk_chars !== undefined) UiNumber.write($('scene-max-chars'), recipe.max_chunk_chars);
    view('scene'); enhanceAllSelects(); saveSoon();
  }
  function findTake(path) {
    for (const job of [...Object.values(state.resultJobs), ...state.jobs, ...Object.values(flow.workJobs)].filter(Boolean)) {
      if (job.take_file === path) return { job_id: job.id };
      const rows = segmentsOf(job), index = rows.findIndex(row => row.take_file === path);
      if (index >= 0) return { job_id: job.id, part_index: rows[index].index ?? index };
    }
    return null;
  }
  async function refresh() {
    flow.data = await api('/api/studio'); await loadJobs();
    if (state.activeJob) renderResults(state.activeJob);
  }
  function knownJob(id) {
    return Object.values(state.resultJobs).find(job => job?.id === id)
      || state.jobs.find(job => job.id === id) || flow.workJobs[id];
  }
  async function ensureWorkHistory(work) {
    if (!work || flow.failedHistories.has(work.id)) return;
    const ids = work.segments.flatMap(part => part.versions);
    if (ids.every(id => knownJob(id) || flow.missingJobs.has(id))) return;
    if (flow.historyLoads.has(work.id)) return flow.historyLoads.get(work.id);
    const pending = (async () => {
      let offset = 0;
      do {
        const page = await api(`/api/studio/work/${encodeURIComponent(work.id)}/jobs?offset=${offset}&limit=100`);
        for (const job of page.jobs) flow.workJobs[job.id] = job;
        for (const id of page.missing) flow.missingJobs.add(id);
        offset = page.next_offset;
      } while (offset !== null);
    })().catch(error => { flow.failedHistories.add(work.id); throw error; })
      .finally(() => {
        flow.historyLoads.delete(work.id);
        if (state.activeJob) renderResults(state.activeJob);
      });
    flow.historyLoads.set(work.id, pending); return pending;
  }
  function watch() {
    clearTimeout(flow.watch);
    flow.watch = setTimeout(async () => {
      try { await refresh(); } catch (error) { fail(error); }
      if (state.jobs.some(job => ['queued', 'running'].includes(job.status))) watch();
    }, 1500);
  }
  async function send(payload) {
    flow.versionFocus = null;
    state.jobSubmitting = true; syncControls();
    try {
      const job = await post('/api/jobs', payload);
      submitted(job.id, payload); state.jobs.unshift({ id: job.id, status: 'queued', studio: payload.studio });
      if (resultWorkspaceFor(payload) !== 'editor') {
        view(payload.studio?.layout === 'scene' ? 'scene' : 'generate'); monitorJob(job.id);
      }
      return job;
    } finally { state.jobSubmitting = false; syncControls(); }
  }
  function submitted(identifier, payload) {
    flow.lastSubmit = { id: identifier, payload: copy(payload) };
    const workspace = resultWorkspaceFor({ mode: payload.mode, studio: payload.studio });
    const job = {
      id: identifier, mode: payload.mode || (workspace === 'scene' ? 'scene' : 'tts'),
      status: 'queued', created_at: Date.now() / 1000, events: [], studio: copy(payload.studio || {}),
    };
    if (workspace !== 'editor') state.resultJobs[workspace] = job;
    document.dispatchEvent(new CustomEvent('drama:jobsubmitted', { detail: job }));
    refresh().catch(fail); saveSoon(); watch();
    const count = Number(payload.candidates || 1);
    for (let index = 1; index < count; index++) {
      post('/api/jobs', { ...payload, candidates: 1, seed: randomSeed() }).then(() => refresh()).catch(fail);
    }
  }
  async function retry(source) {
    const version = await post('/api/studio/version', source);
    await send({ source_version: source, operation: 'retry', seed: randomSeed(),
      studio: version.studio || flow.single });
  }
  function menu(label, items) {
    const details = node('details', 'command-menu');
    const summary = node('summary', 'secondary-btn', t(label)); summary.dataset.i18n = label;
    const list = node('div', 'command-menu-list');
    for (const [caption, action] of items) list.append(command(caption, () => {
      details.open = false; guarded(action);
    }, 'menu-command'));
    details.append(summary, list); return details;
  }
  function actions(job, source, allowInfill = true, resultBox = $('result-box')) {
    const editable = Boolean(job.take_file || source.part_index !== undefined);
    const items = [
      ['Use as a voice', () => saveVoice(source, true, resultBox)],
      ['Save voice', () => saveVoice(source, false, resultBox)],
    ];
    const group = node('div', 'audio-result-tools');
    if (editable) {
      const edit = command('Edit', () => guarded(() => {
        const audio = source.part_index === undefined
          ? resultBox.querySelector('.main-result audio')
          : resultBox.querySelector(`.segment-item[data-part-index="${source.part_index}"] audio`);
        const selection = audio ? StudioAudio.selection(audio) : null;
        return AudioEditor.open(source, { selection, mode: selection ? 'local' : 'whole' });
      }));
      edit.classList.add('creative-edit'); group.append(edit);
    }
    const more = menu('More', items); more.classList.add('result-more'); group.append(more);
    return group;
  }
  function afterResults(job, box, segments) {
    if (job.mode !== 'audio_stream' || !segments.length) return;
    box.querySelector('.segment-list').classList.add('hidden');
    let area = box.querySelector('.progressive-player');
    if (!area) {
      area = node('div', 'progressive-player');
      area.append(localized('span', 'field-help', 'Live audio'));
      const audio = node('audio'); audio.controls = true; audio.preload = 'metadata';
      area.append(audio); box.prepend(area);
      const player = { audio, index: 0, playing: false, waiting: false, rows: [] };
      area.player = player;
      const advance = () => {
        if (player.index >= player.rows.length) { player.waiting = true; return; }
        player.waiting = false;
        audio.src = fileUrl(player.rows[player.index].path || player.rows[player.index].output);
        if (player.playing) audio.play().catch(() => { player.playing = false; });
      };
      player.advance = advance;
      audio.addEventListener('play', () => { player.playing = true; });
      audio.addEventListener('pause', () => { if (!audio.ended) player.playing = false; });
      audio.addEventListener('ended', () => { player.index++; player.playing = true; advance(); });
      player.playing = $('autoplay-audio').checked;
    }
    const player = area.player;
    const first = !player.rows.length; player.rows = segments;
    if (first || player.waiting) player.advance();
    if (job.output && !area.hidden) {
      const final = box.querySelector('.main-result audio');
      const position = segments.slice(0, player.index).reduce((total, row) =>
        total + (row.samples || 0) / (row.sample_rate || 44100), 0) + player.audio.currentTime;
      const playing = !player.audio.paused;
      player.audio.pause(); area.hidden = true;
      const resume = () => {
        final.currentTime = Math.min(position, final.duration || position);
        if (playing) final.play().catch(() => {});
      };
      if (final.readyState >= 1) resume(); else final.addEventListener('loadedmetadata', resume, { once: true });
    }
  }
  function enhanceResult(job, main) {
    const bar = main.querySelector('.result-actions');
    const signature = `${job.id}:${state.capabilities?.director_revision}:${state.capabilities?.infill}`;
    if (bar.dataset.creativeJob === signature) {
      const retry = bar.querySelector('.creative-retry');
      if (retry) { retry.hidden = !completeRecipe(job); retry.disabled = job.status !== 'complete'; }
      return;
    }
    bar.querySelectorAll('.creative-action').forEach(item => item.remove());
    main.querySelectorAll(':scope > .creative-action').forEach(item => item.remove());
    bar.dataset.creativeJob = signature;
    const source = { job_id: job.id };
    const retryButton = command('Regenerate', () => guarded(() => retry(source)));
    retryButton.disabled = !completeRecipe(job) || job.status !== 'complete'; retryButton.classList.add('creative-action', 'creative-retry');
    retryButton.hidden = !completeRecipe(job);
    const resultBox = main.closest('.result-box');
    const edit = actions(job, source, false, resultBox); edit.classList.add('creative-action');
    bar.append(retryButton, edit);
    if (job.mode === 'design') {
      const choose = command('Use this voice', () => guarded(() => saveVoice(source, true, resultBox)));
      choose.classList.add('creative-action'); bar.prepend(choose);
    }
  }
  function enhanceSegment(job, segment, row, index) {
    if (!segment.take_file) return;
    if (job.mode === 'stream' || job.mode === 'audio_stream') return;
    index = segment.index ?? index;
    row.dataset.partIndex = index;
    row.querySelector('audio').dataset.selection = 'true';
    row.querySelector('audio').dataset.frameSeconds = 2048 / 44100;
    const existingBar = row.querySelector('.segment-creative-actions');
    const signature = `${state.capabilities?.director_revision}:${state.capabilities?.infill}`;
    if (existingBar?.dataset.capabilities === signature) return;
    existingBar?.remove();
    const source = { job_id: job.id, part_index: index };
    const bar = node('div', 'segment-creative-actions');
    bar.dataset.capabilities = signature;
    bar.append(command('Regenerate', () => guarded(() => retry(source))),
      actions(job, source, true, row.closest('.result-box')));
    row.append(bar);
  }
  function renderWorks() {
    const more = $('load-older-results'); if (!more || !flow.ready) return;
    more.hidden = !Object.values(flow.data.works || {}).some(work =>
      work.segments.some(part => part.versions.some(id => !knownJob(id) && !flow.missingJobs.has(id))));
  }
  function completeRecipe(job) {
    return job.recipe?.schema === 1 && ['model', 'seed', 'max_new_tokens', 'temperature', 'top_p',
      'top_k', 'reference_snapshot', 'runtime', 'codec'].every(key => Object.hasOwn(job.recipe, key));
  }
  function renderRecovery(job) {
    const area = $('request-recovery'); if (!area) return;
    area.hidden = !completeRecipe(job) || !['interrupted', 'error', 'failed', 'cancelled', 'incomplete'].includes(job.status);
    if (area.hidden || area.dataset.jobId === job.id) return;
    area.dataset.jobId = job.id; area.replaceChildren();
    const source = { job_id: job.id };
    const repeat = command('Retry request', () => guarded(() => retry(source)));
    repeat.disabled = job.mode === 'stream' && !(job.stream_input?.chunks || job.recipe.chunks || []).length;
    area.append(repeat);
  }
  async function showConditions(source) {
    const version = await post('/api/studio/version', source), recipe = version.recipe || {};
    const dialog = makeDialog('Generation settings'); dialog.body.classList.add('conditions-body');
    const summary = node('dl', 'conditions-grid');
    const add = (label, value) => summary.append(localized('dt', '', label),
      node('dd', '', value === undefined ? t('Not recorded') : typeof value === 'number' ? UiNumber.format(value) : String(value)));
    add('Model', recipe.model?.replaceAll('\\', '/').split('/').at(-1));
    add('Generation mode', recipe.mode ? t(modeLabel(recipe.mode)) : undefined);
    add('Seed', recipe.seed); add('Temperature', recipe.temperature);
    add('Top P', recipe.top_p); add('Top K', recipe.top_k);
    add('Text chunk target', recipe.max_chunk_chars);
    add('Max duration (seconds)', recipe.max_new_tokens === undefined ? undefined
      : UiNumber.format(recipe.max_new_tokens * 2048 / 44100));
    add('Target duration (seconds)', recipe.duration_frames
      ? UiNumber.format(recipe.duration_frames * 2048 / 44100) : '-');
    add('Language', recipe.language || '-');
    add('Runtime', recipe.runtime ? `${recipe.runtime.backend}${recipe.runtime.protocol ? ` · ${recipe.runtime.protocol}` : ''}` : undefined);
    dialog.body.append(summary);
    if (!version.complete_recipe) dialog.body.append(localized('p', 'field-help',
      'Original settings unavailable; using the displayed settings.'));
    if (version.audio_missing || version.take_missing) dialog.body.append(localized('p', 'field-help', 'Some saved result files are missing.'));
    dialog.body.append(localized('h3', '', 'Text to speak'), node('pre', 'conditions-script', version.text));
    if (recipe.direction) dialog.body.append(localized('h3', '', 'Direction'),
      node('pre', 'conditions-script', recipe.direction));
    for (const [index, reference] of (recipe.reference_snapshot || []).entries()) {
      const section = node('div', 'condition-reference');
      const audio = node('audio'); audio.controls = true; audio.preload = 'none'; audio.src = fileUrl(reference.audio);
      section.append(localized('h3', '', 'Reference {number}', { number: index + 1 }),
        node('span', 'field-help', t('Speaker') + ' ' + reference.speaker), audio,
        node('p', 'reference-transcript', reference.text)); dialog.body.append(section);
    }
    const details = node('details', 'source-details');
    details.append(localized('summary', '', 'Source details'), node('pre', 'conditions-script',
      JSON.stringify({ model: recipe.model, runtime: recipe.runtime, codec: recipe.codec,
        source_version: recipe.source_version, condition_source: recipe.condition_source,
        previous_take: recipe.previous_take, previous_sources: recipe.previous_sources,
        source_take: recipe.source_take, start_frame: recipe.start_frame, end_frame: recipe.end_frame,
        conditioning: recipe.conditioning }, (_key, value) => typeof value === 'number' ? Number(UiNumber.format(value)) : value, 2)));
    if (version.take?.prompt_text) {
      const prompt = node('details', 'source-details');
      prompt.append(localized('summary', '', 'Final model prompt'),
        node('pre', 'conditions-script', version.take.prompt_text)); dialog.body.append(prompt);
    }
    dialog.body.append(details);
    const fixed = command('Retry with same seed', () => guarded(async () => {
      await send({ source_version: source, operation: 'retry', seed: recipe.seed, studio: version.studio || flow.single });
      dialog.close();
    }));
    fixed.disabled = recipe.seed === undefined || !version.complete_recipe || ['queued', 'running'].includes(version.status)
      || (recipe.mode === 'stream' && !(version.stream_input?.chunks || recipe.chunks || []).length);
    const restoreButton = command('Restore as draft', () => guarded(async () => {
      await restoreVersion(source); dialog.close();
    }), 'primary-btn');
    dialog.footer.append(fixed, restoreButton); dialog.show(); StudioAudio.enhance(dialog.body);
  }
  function recipeSettings(recipe, { scene = false } = {}) {
    const control = id => $(scene ? `scene-${id}` : id);
    for (const [key, id] of [['seed', 'seed'], ['temperature', 'temperature'], ['top_p', 'top-p'], ['top_k', 'top-k']]) {
      if (recipe[key] !== undefined) UiNumber.write(control(id), recipe[key]);
    }
    if (recipe.seed !== undefined) control('random-seed').checked = false;
    if (scene) return;
    $('duration').value = recipe.duration_frames || '';
    UiNumber.write($('target-seconds'), recipe.duration_frames ? recipe.duration_frames * 2048 / 44100 : '');
    $('language').value = recipe.language || ''; $('candidate-count').value = 1;
  }
  const preparedReference = reference => ({
    ...ReferencePrep.fresh(), upload: reference, transcript: reference.text,
  });
  async function restoreVersion(source) {
    await save();
    const previousSingle = capture(), previousScene = sceneSnapshot();
    const prepared = await post('/api/studio/prepare-restore', source);
    const { version, references } = prepared, recipe = version.recipe || {};
    const draftStudio = copy(version.studio || {});
    delete draftStudio.result_scope; delete draftStudio.edit_source;
    if (recipe.operation === 'continue' || recipe.mode === 'edit' && !recipe.operation) {
      recipe.mode = 'tts'; delete recipe.previous_take; delete recipe.previous_sources;
    }
    recipeSettings(recipe, { scene: Boolean(recipe.characters?.length && recipe.lines?.length && recipe.mode !== 'infill') });
    if (recipe.mode === 'design') {
      stash(previousSingle);
      if (!voiceDesignActive()) switchVoiceSource('design');
      flow.design = Object.keys(draftStudio).length ? draftStudio : fresh('design');
      flow.designRecipe = { source, model: recipe.model, character_name: recipe.character_name, scene: recipe.scene };
      $('voice-description').value = recipe.voice_description || '';
      flow.writing.design = { text: version.text, direction: recipe.direction || '' };
      showWriting('design'); updateMode();
      view('generate'); renderOperation(); saveSoon(); return;
    }
    if (recipe.characters?.length && recipe.lines?.length && recipe.mode !== 'infill') {
      const current = previousScene; delete current.returnDrafts;
      const roles = recipe.characters.map((role, index) => {
        const clips = references.filter(reference => reference.speaker === role.speaker);
        return { ...sceneRole(role.name, role.speaker), ...role, id: recipe.character_ids?.[index] || uid(),
          ...(clips.length ? preparedReference(clips[0]) : {}), extraReferences: clips.slice(1).map(preparedReference) };
      });
      const lines = recipe.lines.map((line, index) => ({
        id: recipe.line_ids?.[index] || uid(), characterId: roles.find(role => role.name === line.character).id,
        text: line.text, direction: '',
      }));
      restoreScene({ ...(Object.keys(draftStudio).length ? draftStudio : fresh('scene')), layout: 'scene', roles, lines,
        strategy: recipe.scene_strategy || 'per_line', scene: recipe.scene || '', direction: recipe.direction || '',
        restoredFrom: source, restoredModel: recipe.model, restoredMode: recipe.mode,
        previous_take: recipe.previous_take, previous_sources: recipe.previous_sources,
        returnDrafts: [...(flow.scene.returnDrafts || []), current].slice(-4) });
      view('scene'); saveSoon(); return;
    }
    stash(previousSingle); flow.restoring = true;
    try {
      if (voiceDesignActive()) switchVoiceSource('reference');
      clearCloneAudio(); state.selectedVoiceIds.clear();
      flow.libraryReferences.forEach(ReferencePrep.cancel); flow.libraryReferences = [];
      flow.singleReferences.forEach(ReferencePrep.cancel); flow.singleReferences = [];
      if (references.length) {
        const current = await uploadedFile(references[0]);
        selectCloneAudio(current.file, { auto: false, upload: current.upload });
        $('clone-transcript').value = references[0].text;
        flow.singleReferences = references.slice(1).map(preparedReference);
      }
      flow.single = { ...(Object.keys(draftStudio).length ? draftStudio : fresh('single')), layout: 'single' }; flow.operation = null;
      flow.restoredFrom = source; flow.restoredModel = recipe.model || null;
      $('mode').value = ['audio_stream', 'stream'].includes(recipe.mode) ? 'tts' : recipe.mode || 'tts';
      $('progressive-audio').checked = recipe.mode === 'audio_stream';
      $('text').value = version.text; $('direction').value = recipe.direction || '';
      $('previous-take').value = recipe.previous_take || ''; $('source-take').value = recipe.source_take || '';
      if (recipe.mode === 'infill') {
        const take = await post('/api/takes/inspect', { path: recipe.source_take });
        showInfillSource(take, { start: recipe.start_frame * 2048 / 44100, end: recipe.end_frame * 2048 / 44100 });
      } else if (recipe.previous_take) {
        showPreviousSource(await post('/api/takes/inspect', { path: recipe.previous_take }));
      }
      if (recipe.mode === 'stream') {
        const chunks = version.stream_input?.chunks || recipe.chunks || [];
        if (chunks.length) $('text').value = chunks.join('\n');
        $('stream-live').checked = false;
        $('previous-take').value = $('source-take').value = '';
      }
      rememberWriting(); showWriting('reference');
      updateMode(); updateEditorMeta(); renderLibraryReferences(true); renderExtraReferences();
      renderOperation(); enhanceAllSelects(); view('generate');
      if (!version.complete_recipe) fail(t('Original settings unavailable; using the displayed settings.'));
    } finally { flow.restoring = false; }
    saveSoon();
  }
  async function saveVoice(source, use, resultBox = $('result-box')) {
    const version = await post('/api/studio/version', source);
    const sourceAudio = source.part_index === undefined ? resultBox.querySelector('.main-result audio')
      : resultBox.querySelector(`.segment-item[data-part-index="${source.part_index}"] audio`);
    const selection = StudioAudio.selection(sourceAudio);
    const multiSpeaker = source.part_index === undefined && (version.recipe.characters?.length > 1
      || new Set([...version.text.matchAll(/<\|speaker:(\d+)\|>/g)].map(match => match[1])).size > 1);
    if (multiSpeaker && !selection) {
      throw new Error(t('Select one speaker’s audio first.'));
    }
    const upload = await post('/api/reference-audio/from-version', {
      source_version: source, ...(selection ? { start: selection.start, end: selection.end } : {}),
    });
    const dialog = makeDialog(use ? 'Use this voice' : 'Save voice');
    const audio = node('audio'); audio.controls = true; audio.src = upload.audio_url;
    const transcript = input('save-reference-transcript', selection ? '' : plainText(upload.suggested_text), 'textarea');
    const name = input('save-reference-name');
    const check = node('label', 'check-field'), checked = input('reference-checked', '', 'checkbox');
    check.append(checked, localized('span', '', 'Transcript checked'));
    const singleSpeaker = input('single-speaker-checked', '', 'checkbox');
    const speakerCheck = node('label', 'check-field');
    const speakerLabel = node('span', '', t('This selection contains only one speaker.'));
    speakerLabel.dataset.i18n = 'This selection contains only one speaker.';
    speakerCheck.append(singleSpeaker, speakerLabel); speakerCheck.hidden = !multiSpeaker;
    dialog.body.append(audio);
    if (!use) dialog.body.append(field('Name', name));
    dialog.body.append(field('Reference transcript', transcript), check, speakerCheck);
    const submit = command(use ? 'Use this voice' : 'Save voice', () => guarded(async () => {
      if (!checked.checked || !transcript.value.trim()) throw new Error(t('Check the reference transcript first.'));
      if (multiSpeaker && !singleSpeaker.checked) throw new Error(t('Confirm that the selection contains only one speaker.'));
      if (use) {
        if (voiceDesignActive()) switchVoiceSource('reference');
        const loaded = await uploadedFile(upload);
        selectCloneAudio(loaded.file, { auto: false, upload: loaded.upload });
        $('clone-transcript').value = transcript.value; flow.operation = null;
        flow.restoredFrom = flow.restoredModel = null;
        clearExtraReferences();
        $('previous-take').value = $('source-take').value = ''; $('mode').value = 'tts'; updateMode(); renderOperation();
      } else {
        await post('/api/voices/from-upload', { upload_id: upload.id, name: name.value, transcript: transcript.value });
        await loadVoices();
      }
      dialog.close(); saveSoon();
    }), 'primary-btn');
    dialog.footer.append(submit); dialog.show(); StudioAudio.enhance(dialog.body);
  }
  function makeDialog(title, args = {}) {
    const dialog = node('dialog', 'studio-dialog');
    const heading = node('div', 'studio-dialog-heading');
    const body = node('div', 'studio-dialog-body'), footer = node('div', 'studio-dialog-footer');
    const close = () => { dialog.close(); dialog.remove(); };
    const cancel = command('Close', close, 'icon-btn'); cancel.innerHTML = icon('close'); delete cancel.dataset.i18n;
    cancel.setAttribute('aria-label', t('Close')); cancel.dataset.i18nAriaLabel = 'Close';
    cancel.title = t('Close'); cancel.dataset.i18nTitle = 'Close';
    heading.append(localized('h2', '', title, args), cancel); dialog.append(heading, body, footer); document.body.append(dialog);
    dialog.addEventListener('cancel', event => { event.preventDefault(); close(); });
    return { body, footer, close, element: dialog, show: () => dialog.showModal() };
  }
  async function prepareMissingVoices(version, source) {
    const speakers = [...new Set([...version.text.matchAll(/<\|speaker:(\d+)\|>/g)].map(match => Number(match[1])))];
    if (!speakers.length) speakers.push(0);
    const available = new Set((version.recipe.reference_snapshot || []).map(reference => reference.speaker));
    const missing = speakers.filter(speaker => !available.has(speaker));
    if (!missing.length) return { bindings: [], previews: [] };
    const references = new Map(missing.map(speaker => [speaker, ReferencePrep.fresh()]));
    if (speakers.length === 1 && version.take?.duration_seconds <= 120) {
      const upload = await post('/api/reference-audio/from-version', { source_version: source });
      references.set(missing[0], { ...ReferencePrep.fresh(), upload,
        transcript: plainText(version.text) });
    }
    return new Promise(resolve => {
      const dialog = makeDialog('Prepare source voices');
      const audio = node('audio'); audio.controls = true; audio.preload = 'metadata';
      audio.src = fileUrl(version.output); audio.dataset.selection = 'true'; audio.id = 'reword-source-audio';
      dialog.body.append(audio);
      const rows = new Map(), notice = node('p', 'form-note'); notice.setAttribute('role', 'alert');
      for (const speaker of missing) {
        const row = node('section', 'missing-voice-preparation'); row.dataset.speaker = speaker;
        const name = version.recipe.characters?.find(role => role.speaker === speaker)?.name || `${t('Speaker')} ${speaker}`;
        const content = node('div');
        const draw = () => {
          content.replaceChildren(ReferencePrep.mount(references.get(speaker), {
            prefix: `reword-${speaker}`, changed: () => {}, allowLibrary: true,
          }));
          enhanceAllSelects(); StudioAudio.enhance(content);
        };
        row.append(node('h3', '', name), command('Use selection for this speaker', async () => {
          try {
            const selection = StudioAudio.selection(audio);
            if (!selection || selection.end - selection.start < .5) throw new Error(t('Select one speaker’s audio first.'));
            const upload = await post('/api/reference-audio/from-version', { source_version: source, ...selection });
            ReferencePrep.cancel(references.get(speaker));
            references.set(speaker, { ...ReferencePrep.fresh(), upload, transcript: '' }); draw();
          } catch (error) { notice.textContent = message(error.message); }
        }, 'text-btn'), content);
        dialog.body.append(row); rows.set(speaker, draw);
      }
      const check = node('label', 'check-field'), checked = input('reword-reference-checked', '', 'checkbox');
      check.append(checked, localized('span', '', 'Each reference contains only its speaker and the transcript is correct.'));
      dialog.body.append(check, notice);
      dialog.element.addEventListener('close', () => {
        for (const reference of references.values()) ReferencePrep.cancel(reference);
        resolve(null);
      }, { once: true });
      const next = command('Keep editing', async () => {
        try {
          if (!checked.checked) throw new Error(t('Check the reference transcript first.'));
          const bindings = [], previews = [];
          for (const [speaker, reference] of references) {
            const clips = await ReferencePrep.readAll(reference, speaker);
            if (!clips.length) throw new Error(t('Choose a reference for each missing speaker.'));
            bindings.push(...clips); previews.push(...ReferencePrep.previews(reference, speaker));
          }
          if (!dialog.element.open) return;
          resolve({ bindings, previews }); dialog.close();
        } catch (error) { if (dialog.element.open) notice.textContent = message(error.message); }
      }, 'primary-btn');
      next.id = 'confirm-reword-voices'; dialog.footer.append(next); dialog.show();
      for (const draw of rows.values()) draw(); StudioAudio.enhance(dialog.body);
    });
  }
  function configureControls() {
    const top = node('div', 'creation-toolbar');
    top.append(node('span', 'field-help', ''));
    const buttons = node('div', 'creation-toolbar-actions');
    const back = command('Return to draft', async () => {
      if (flow.switchingInput) return;
      const previous = flow.returnDrafts.pop(); const rest = copy(flow.returnDrafts);
      if (previous) { await restore({ ...previous, returnDrafts: rest }); saveSoon(); }
    }, 'text-btn'); back.id = 'return-draft'; back.hidden = true;
    const designBack = command('Back', () => switchVoiceSource('reference'));
    designBack.id = 'back-to-generation'; designBack.hidden = true; delete designBack.dataset.i18n;
    designBack.innerHTML = icon('arrow'); designBack.append(localized('span', '', 'Back'));
    buttons.append(designBack);
    top.append(buttons); $('view-generate').prepend(top);
    const operation = node('div', 'creative-operation'); operation.id = 'creative-operation'; operation.hidden = true;
    operation.append(back);
    document.querySelector('.compose-panel').prepend(operation);
    const frozen = node('section', 'frozen-voice'); frozen.id = 'frozen-voice'; frozen.hidden = true;
    $('clone-section').before(frozen);
    const recovery = node('div', 'recovery-actions'); recovery.id = 'request-recovery'; recovery.hidden = true;
    $('result-box').after(recovery);
    const supportNote = node('p', 'field-help'); supportNote.id = 'task-capability-notice'; supportNote.hidden = true;
    $('submit-job').closest('.compose-footer').before(supportNote);
    const notice = node('div', 'studio-error'); notice.id = 'studio-error'; notice.hidden = true; notice.setAttribute('role', 'alert');
    const noticeMessage = node('span', 'studio-error-message');
    const dismissNotice = command('Close', clearError, 'icon-btn');
    delete dismissNotice.dataset.i18n;
    dismissNotice.innerHTML = icon('close');
    dismissNotice.setAttribute('aria-label', t('Close')); dismissNotice.dataset.i18nAriaLabel = 'Close';
    dismissNotice.title = t('Close'); dismissNotice.dataset.i18nTitle = 'Close';
    notice.append(noticeMessage, dismissNotice);
    $('view-generate').before(notice);
    const older = command('Load older results', () => guarded(async () => {
      older.disabled = true;
      try {
        const pending = Object.values(flow.data.works || {}).sort((a, b) => b.updated_at - a.updated_at)
          .filter(work => work.segments.some(part => part.versions.some(id => !knownJob(id) && !flow.missingJobs.has(id))));
        for (const work of pending.slice(0, 8)) {
          flow.failedHistories.delete(work.id); await ensureWorkHistory(work);
        }
        await loadJobs();
      } finally { older.disabled = false; }
    }), 'secondary-btn'); older.id = 'load-older-results'; older.hidden = true;
    $('job-list').after(older);
    const historyTitle = document.querySelector('.history-panel .panel-heading h3');
    historyTitle.dataset.i18n = 'History'; historyTitle.textContent = t('History');
    // Keep protocol fields for request/draft compatibility, outside the user controls.
    const internal = node('div'); internal.id = 'internal-task-state'; internal.hidden = true; internal.inert = true;
    internal.append($('mode').closest('.custom-select') || $('mode'));
    $('advanced-settings').append(internal);
    const task = node('details', 'task-details'); task.id = 'task-details';
    const taskInfo = node('dl', 'task-context');
    const taskRow = node('div'), taskValue = node('dd'); taskValue.id = 'current-task-value';
    taskRow.append(localized('dt', '', 'Current task'), taskValue);
    const sourceRow = node('div'); sourceRow.id = 'task-source-row'; sourceRow.hidden = true;
    const sourceValue = node('dd'), sourceName = node('span'); sourceName.id = 'current-source-value';
    const sourceSettings = command('View settings', () => guarded(async () => {
      const source = currentTaskSource().version;
      if (source) await showConditions(source);
    }), 'text-btn'); sourceSettings.id = 'view-task-source';
    sourceValue.append(sourceName, sourceSettings);
    sourceRow.append(localized('dt', '', 'Source version'), sourceValue);
    taskInfo.append(taskRow, sourceRow);
    task.append(localized('summary', '', 'Task details'), taskInfo);
    internal.append(task);
    const performance = node('div', 'performance-fields');
    const direction = $('direction').parentElement; const language = $('language').parentElement;
    performance.append(direction); $('advanced-settings').before(performance);
    $('direction').placeholder = ''; $('direction').removeAttribute('data-i18n-placeholder');
    const context = $('previous-take').closest('.source-path-row');
    const label = document.querySelector('label[for="previous-take"]');
    internal.append(label, context, language, $('infill-source-location'));
    document.querySelector('#infill-source-preview .edit-source-heading').append($('inspect-infill-source'));
    const controls = node('div', 'field-grid');
    for (const [id, label, value, step, min, max] of [
      ['target-seconds', 'Target duration (seconds)', '', 'any', '.01', '95.109'],
      ['max-chars', 'Text chunk target', '160', '1', '80', '320'],
      ['temperature', 'Temperature', '0.8', 'any', '0', '2'],
      ['top-p', 'Top P', '.8', 'any', '0', '1'],
      ['top-k', 'Top K', '0', '1', '0', '8193'],
      ['candidate-count', 'Candidates', '1', '1', '1', '3'],
    ]) {
      const control = input(id, value, 'number'); Object.assign(control, { step, min, max });
      controls.append(field(label, control));
    }
    $('advanced-settings').querySelector('.advanced-content').prepend(controls);
    const exact = node('details', 'source-details');
    const exactTitle = node('summary', '', t('Frame limits')); exactTitle.dataset.i18n = 'Frame limits';
    const frameFields = node('div', 'field-grid');
    const previousGrid = $('duration').parentElement.parentElement;
    frameFields.append($('duration').parentElement);
    internal.append($('max-frames').parentElement);
    previousGrid.classList.remove('three');
    exact.append(exactTitle, frameFields); $('advanced-settings').append(exact);
    $('target-seconds').addEventListener('input', () => {
      $('duration').value = $('target-seconds').value ? Math.round(Number($('target-seconds').value) * 44100 / 2048) : '';
    });
    $('duration').addEventListener('input', () => {
      UiNumber.write($('target-seconds'), $('duration').value ? Number($('duration').value) * 2048 / 44100 : '');
    });
    const random = input('random-seed', '', 'checkbox'); random.checked = true;
    const randomLabel = node('label', 'check-field'); randomLabel.append(random, node('span', '', t('Random seed')));
    randomLabel.lastChild.dataset.i18n = 'Random seed';
    $('seed').parentElement.append(randomLabel);
    const progressive = input('progressive-audio', '', 'checkbox');
    const progressiveLabel = node('label', 'check-field'); progressiveLabel.append(progressive, node('span', '', t('Progressive playback')));
    progressiveLabel.lastChild.dataset.i18n = 'Progressive playback';
    const autoplay = input('autoplay-audio', '', 'checkbox');
    const autoplayLabel = node('label', 'check-field'); autoplayLabel.append(autoplay, node('span', '', t('Autoplay generated audio')));
    autoplayLabel.lastChild.dataset.i18n = 'Autoplay generated audio';
    $('advanced-settings').querySelector('.advanced-content').append(progressiveLabel, autoplayLabel);
    const diff = node('div', 'transcript-diff'); diff.id = 'transcript-diff'; diff.hidden = true;
    $('text').closest('.editor-area').after(diff);
    ScriptEditor.mount($('text')); ScriptEditor.mount($('chunks'));
    const saveReference = command('Save voice', () => guarded(savePreparedVoice), 'text-btn');
    $('clone-upload-preview').append(saveReference);
    const libraryDetails = node('div', 'selected-reference-details'); libraryDetails.id = 'selected-reference-details';
    $('saved-voice-options').after(libraryDetails);
    const extra = node('details', 'additional-references'); extra.id = 'additional-references';
    extra.hidden = true;
    const extraTitle = node('summary', '', t('Additional references · same voice'));
    extraTitle.dataset.i18n = 'Additional references · same voice';
    const extraList = node('div'); extraList.id = 'additional-reference-list';
    const addReference = command('Add reference', () => {
      flow.singleReferences.push(ReferencePrep.fresh()); renderExtraReferences(); saveSoon();
    }, 'text-btn'); addReference.id = 'add-reference';
    extra.append(extraTitle, extraList, addReference); internal.append(extra, $('default-voice'));
    const describe = command('Voice design', () => {
      switchVoiceSource('design'); $('voice-description').focus({ preventScroll: true });
    }, 'text-btn'); describe.id = 'describe-voice';
    describe.setAttribute('aria-controls', 'voice-design');
    $('saved-voice-options').before(describe);
    const design = node('div', 'voice-design'); design.id = 'voice-design'; design.hidden = true;
    const description = input('voice-description', '', 'textarea');
    description.required = true;
    design.append(field('Voice description', description)); describe.after(design);
    for (const element of document.querySelectorAll('.nav[data-view="generate"] [data-i18n]')) {
      element.dataset.i18n = 'Create'; element.textContent = t('Create');
    }
    for (const element of document.querySelectorAll('.nav[data-view="takes"] [data-i18n]')) {
      element.dataset.i18n = 'History'; element.textContent = t('History');
    }
    document.querySelector('.generation-hint').hidden = true;
    document.querySelector('#stream-fields .field-help').hidden = true;
    document.addEventListener('drama:editor-notice', event => fail(event.detail));
  }
  async function useLibraryVoice(identifier) {
    await guarded(async () => {
      const designing = voiceDesignActive();
      if (designing) switchVoiceSource('reference');
      const fixed = designing || flow.operation || flow.restoredFrom || ['revision', 'edit', 'infill', 'stream'].includes($('mode').value);
      if (fixed) {
        await save(); stash();
        flow.single = fresh('single'); flow.operation = null;
        flow.restoredFrom = flow.restoredModel = null;
        $('previous-take').value = $('source-take').value = ''; $('mode').value = 'tts';
        clearMonitoredResult();
      }
      clearCloneAudio(); clearExtraReferences();
      state.selectedVoiceIds.clear(); state.selectedVoiceIds.add(identifier);
      updateVoiceSelection(); updateMode(); updateEditorMeta(); renderOperation();
      view('generate'); $('saved-voice-options').open = false; saveSoon();
      window.ReferencePrewarm?.schedule();
    });
  }
  function clearMonitoredResult() {
    state.resultJobs.generate = null; showWorkspaceResult(state.view);
    $('submit-job').disabled = false;
  }
  async function prepareRequest(payload) {
    const design = voiceDesignActive();
    if (!design && !flow.operation && !syncCloneAudioSelection()) {
      throw new Error($('clone-note').textContent);
    }
    const common = settings({ checkDuration: !design });
    delete payload.max_new_tokens;
    if (design) {
      if (!state.capabilities?.voice_design) throw new Error(t('Not supported by this model'));
      const description = $('voice-description').value.trim();
      if (!description) {
        $('voice-description').focus(); throw new Error(t('Enter a voice description.'));
      }
      return { mode: 'design', text: payload.text, direction: payload.direction,
        voice_description: description, character_name: flow.designRecipe?.character_name || 'Voice',
        scene: flow.designRecipe?.scene || '', studio: copy(flow.design),
        condition_source: flow.designRecipe?.source,
        model: flow.designRecipe?.model || state.model?.selected_directory,
        candidates: Number($('candidate-count').value || 1), ...common };
    }
    const operation = copy(flow.operation), studio = copy(flow.single);
    const restoredFrom = copy(flow.restoredFrom), model = flow.restoredModel || state.model?.selected_directory;
    const duration = UiNumber.value($('target-seconds')), progressive = $('progressive-audio').checked;
    const candidates = Number($('candidate-count').value || 1);
    if (!$('duration').disabled && payload.mode !== 'stream') {
      common.duration_frames = $('target-seconds').value
        ? Math.max(1, Math.round(UiNumber.read($('target-seconds')) * 44100 / 2048)) : null;
    }
    if (operation && !operation.legacy) {
      if (operation.restoreError) throw new Error(t(operation.restoreError));
      return {
        source_version: operation.source, operation: operation.action, studio,
        ...(operation.rewordReferences?.length ? {
          reword_references: operation.rewordReferences, reference_transcripts_checked: true,
        } : {}),
        text: payload.text, direction: payload.direction, ...common,
        ...(operation.action === 'infill' ? {
          start_frame: payload.start_frame, end_frame: payload.end_frame,
        } : {}),
      };
    }
    payload = { ...payload, ...common, studio, candidates, model };
    if (restoredFrom) {
      payload.condition_source = restoredFrom; payload.operation = 'from_settings';
    }
    if (operation?.legacy) {
      payload.legacy_source = operation.source; payload.operation = operation.action;
    }
    renderLibraryReferences();
    const references = [...flow.libraryReferences, ...flow.singleReferences];
    payload.voice_bindings = payload.mode === 'infill' ? [] : await singleBindings(
      payload.uploaded_references ?? null, references);
    if (!['revision', 'edit'].includes(payload.mode)) delete payload.previous_take;
    if (payload.mode !== 'stream' && duration) {
      payload.duration_frames = Math.max(1, Math.round(Number(duration) * 44100 / 2048));
      payload.single_chunk = true;
    }
    if (payload.mode === 'tts' && progressive) payload.mode = 'audio_stream';
    return payload;
  }
  async function referencePrewarmPayload() {
    if (voiceDesignActive() || flow.operation || state.cloneMissing) {
      return { voice_bindings: [] };
    }
    if (flow.restoring) return null;
    if (!syncCloneAudioSelection()) return null;
    if (state.cloneFile && !$('clone-transcript').value.trim()) return null;
    try {
      const primary = state.cloneFile ? await prepareCloneReference() : [];
      const references = [...flow.libraryReferences, ...flow.singleReferences];
      const groups = await Promise.all(references.map(reference =>
        ReferencePrep.readAll(reference, 0)));
      return { voice_bindings: [...primary, ...groups.flat()] };
    } catch (_) {
      return null;
    }
  }
  function sceneRole(name, speaker = 0) {
    return { ...ReferencePrep.fresh(), id: uid(), name, description: '', speaker, extraReferences: [] };
  }
  function sceneLine(role) { return { id: uid(), characterId: role.id, text: '', direction: '' }; }
  function clearExtraReferences() {
    flow.singleReferences.forEach(ReferencePrep.cancel); flow.singleReferences = [];
    renderExtraReferences(); saveSoon(); window.ReferencePrewarm?.schedule();
  }
  function renderLibraryReferences(force = false) {
    const box = $('selected-reference-details'); if (!box || (flow.restoring && !force)) return;
    const selected = new Set(selectedVoices());
    for (const reference of flow.libraryReferences) if (!selected.has(reference.libraryId)) ReferencePrep.cancel(reference);
    flow.libraryReferences = flow.libraryReferences.filter(reference => selected.has(reference.libraryId));
    for (const id of selected) if (!flow.libraryReferences.some(reference => reference.libraryId === id)) {
      flow.libraryReferences.push(ReferencePrep.forVoice(id));
    }
    const existing = new Map([...box.children].map(row => [row.dataset.libraryId, row]));
    for (const reference of flow.libraryReferences) {
      const voice = state.voices.find(voice => voice.id === reference.libraryId);
      let row = existing.get(reference.libraryId);
      if (row?.boundReference !== reference) {
        row?.remove(); row = node('div', 'selected-library-voice'); row.dataset.libraryId = reference.libraryId;
        row.boundReference = reference;
        const name = node('strong');
        const play = command('Preview voice', () => previewVoice(
          state.voices.find(voice => voice.id === reference.libraryId), play), 'icon-btn reference-compact-play');
        delete play.dataset.i18n; play.innerHTML = icon('play');
        play.dataset.i18nAriaLabel = play.dataset.i18nTitle = 'Preview voice';
        play.setAttribute('aria-label', t('Preview voice')); play.title = t('Preview voice');
        play.disabled = !voice?.audio_url;
        const clear = command('Remove audio', () => {
          stopVoicePreview(); state.selectedVoiceIds.delete(reference.libraryId);
          updateVoiceSelection(); saveSoon(); window.ReferencePrewarm?.schedule(0);
        }, 'icon-btn selected-voice-clear');
        delete clear.dataset.i18n; clear.innerHTML = icon('close');
        clear.dataset.i18nAriaLabel = clear.dataset.i18nTitle = 'Remove audio';
        clear.setAttribute('aria-label', t('Remove audio')); clear.title = t('Remove audio');
        row.append(name, play, clear); box.append(row);
      }
      const name = row.querySelector('strong');
      name.textContent = voice?.name || reference.libraryId; name.title = name.textContent;
      row.querySelector('.reference-compact-play').disabled = !voice?.audio_url;
      existing.delete(reference.libraryId);
    }
    for (const row of existing.values()) row.remove();
    box.hidden = !flow.libraryReferences.length;
    $('clone-upload-zone').classList.toggle('hidden',
      Boolean(state.cloneFile || state.cloneMissing || flow.libraryReferences.length));
  }
  async function singleBindings(uploaded = null, references = null) {
    if (!syncCloneAudioSelection()) throw new Error($('clone-note').textContent);
    const revision = state.cloneRevision;
    if (!references) { renderLibraryReferences(); references = [...flow.libraryReferences, ...flow.singleReferences]; }
    const [primary, groups] = await Promise.all([
      uploaded === null ? prepareCloneReference() : uploaded,
      Promise.all(references.map(reference => ReferencePrep.readAll(reference, 0))),
    ]);
    if (!syncCloneAudioSelection() || revision !== state.cloneRevision) {
      throw new Error(t('Reference changed. Click Generate again.'));
    }
    return [...primary, ...groups.flat()];
  }
  async function savePreparedVoice() {
    const references = await singleBindings();
    if (!references.length) throw new Error(t('Choose an audio file.'));
    const dialog = makeDialog('Save voice'), name = input('prepared-voice-name');
    dialog.body.append(field('Name', name), localized('p', 'field-help', '{count} reference clips', { count: references.length }));
    dialog.footer.append(command('Save voice', () => guarded(async () => {
      await post('/api/voices/from-references', { name: name.value, references });
      await loadVoices(); dialog.close();
    }), 'primary-btn')); dialog.show();
  }
  function configureVoiceLibrary() {
    const form = $('voice-form'), legacy = node('div', 'hidden');
    const dialog = $('voice-library-dialog'), note = $('voice-form-note');
    const nameLabel = form.querySelector('label[for="voice-name"]'), name = $('voice-name');
    const footer = form.querySelector('.compose-footer');
    const preparation = node('fieldset', 'voice-library-preparation'); preparation.id = 'voice-library-preparation';
    const clips = node('div'); clips.id = 'voice-library-clips';
    const additional = node('details', 'additional-references');
    additional.append(localized('summary', '', 'Additional references · same voice'));
    const extras = node('div'); extras.id = 'voice-library-extras';
    additional.append(extras, command('Add reference', () => {
      flow.voiceDraft.extraReferences.push(ReferencePrep.fresh()); renderVoiceDraft(); saveSoon();
    }, 'text-btn'));
    preparation.append(clips, additional);
    for (const child of [...form.children]) {
      if (![nameLabel, name, footer].includes(child)) legacy.append(child);
    }
    for (const control of legacy.querySelectorAll('input,textarea')) {
      control.required = false; control.disabled = true;
    }
    const body = node('div', 'studio-dialog-body');
    body.append(nameLabel, name, preparation, legacy); form.replaceChildren(body, footer);
    form.noValidate = true;
    name.value = flow.voiceDraft.name || '';
    const clearNote = () => {
      note.textContent = ''; delete note.dataset.i18n; delete note.dataset.error;
      note.setAttribute('role', 'status');
    };
    const showNote = error => {
      delete note.dataset.i18n; note.textContent = message(error?.message || String(error));
      note.dataset.error = 'true'; note.setAttribute('role', 'alert');
    };
    $('add-voice').disabled = false;
    $('add-voice').onclick = () => {
      if (dialog.open) return;
      stopVoicePreview(); clearNote(); dialog.showModal(); dialog.scrollTop = 0;
    };
    $('close-voice-library').onclick = () => dialog.close();
    dialog.addEventListener('close', () => {
      dialog.querySelectorAll('audio').forEach(audio => audio.pause());
      closeCustomSelects(); saveSoon();
    });
    form.addEventListener('input', clearNote);
    document.addEventListener('drama:editor-notice', event => {
      if (dialog.open) showNote(event.detail);
    });
    const draw = (box, references, primary = false) => {
      const old = new Map([...box.children].map(row => [row.dataset.referenceKey, row]));
      for (const reference of references) {
        let row = old.get(reference.key);
        if (row?.boundReference !== reference) { row?.remove(); row = null; }
        if (!row) box.append(ReferencePrep.mount(reference, {
          prefix: primary ? 'library-new' : `library-new-${reference.key}`, changed: saveSoon,
          allowLibrary: false, ...(!primary ? { remove: () => {
            ReferencePrep.cancel(reference);
            flow.voiceDraft.extraReferences = flow.voiceDraft.extraReferences.filter(item => item !== reference);
            renderVoiceDraft(); saveSoon();
          } } : {}),
        }));
        old.delete(reference.key);
      }
      for (const row of old.values()) row.remove();
    };
    function renderVoiceDraft() {
      draw(clips, [flow.voiceDraft.main], true); draw(extras, flow.voiceDraft.extraReferences || []);
      enhanceAllSelects(); StudioAudio.enhance(preparation);
    }
    form.onsubmit = async event => {
      event.preventDefault();
      if (preparation.disabled) return;
      const original = flow.voiceDraft;
      clearNote(); preparation.disabled = true; name.disabled = true; $('save-voice').disabled = true;
      try {
        if (!name.value.trim()) throw new Error(t('Voice name must not be empty'));
        const references = [];
        for (const clip of [original.main, ...(original.extraReferences || [])]) {
          references.push(...await ReferencePrep.readAll(clip, 0));
        }
        if (!references.length) throw new Error(t('Choose an audio file.'));
        await post('/api/voices/from-references', { name: name.value, references });
        for (const clip of [original.main, ...(original.extraReferences || [])]) ReferencePrep.cancel(clip);
        flow.voiceDraft = { main: ReferencePrep.fresh(), extraReferences: [] };
        name.value = ''; renderVoiceDraft();
        await loadVoices(); saveSoon(); dialog.close();
      } catch (error) {
        showNote(error);
      } finally {
        preparation.disabled = false; name.disabled = false; $('save-voice').disabled = false;
      }
    };
    renderVoiceDraft();
  }
  function renderExtraReferences() {
    const list = $('additional-reference-list'); if (!list) return;
    const existing = new Map([...list.children].map(row => [row.dataset.referenceKey, row]));
    for (const reference of flow.singleReferences) {
      if (existing.has(reference.key) && existing.get(reference.key).boundReference !== reference) {
        existing.get(reference.key).remove(); existing.delete(reference.key);
      }
      if (!existing.has(reference.key)) list.append(ReferencePrep.mount(reference, {
        prefix: `extra-${reference.key}`, changed: () => {
          saveSoon(); window.ReferencePrewarm?.schedule();
        }, remove: () => {
          flow.singleReferences = flow.singleReferences.filter(item => item !== reference);
          renderExtraReferences(); saveSoon(); window.ReferencePrewarm?.schedule();
        },
      }));
      existing.delete(reference.key);
    }
    for (const row of existing.values()) row.remove();
    $('additional-references').hidden = !flow.singleReferences.length;
    renderLibraryReferences();
    enhanceAllSelects();
  }
  const speakerLabel = role => `S${role.speaker + 1}`;
  function speakerTool(label, symbol, action) {
    const button = command(label, () => guarded(action), 'icon-btn');
    delete button.dataset.i18n;
    if (symbol === 'play' || symbol === 'pause') button.innerHTML = icon(symbol);
    else {
      const image = ScriptEditor.icon(symbol);
      image.classList.add('icon');
      button.replaceChildren(image);
    }
    button.title = t(label); button.dataset.i18nTitle = label;
    button.setAttribute('aria-label', t(label)); button.dataset.i18nAriaLabel = label;
    return button;
  }
  function speakerAudio(role) {
    let reference = role, clip = role.frozenReferences?.[0];
    if (!clip) for (const candidate of [role, ...(role.extraReferences || [])]) {
      clip = ReferencePrep.previews(candidate, role.speaker)[0];
      if (clip) { reference = candidate; break; }
    }
    const voice = state.voices.find(item => item.id === reference.voiceId);
    return {
      reference,
      name: reference.upload?.filename || voice?.name || clip?.audio?.replaceAll('\\', '/').split('/').at(-1) || '',
      url: clip?.audio_url || (clip?.audio && fileUrl(clip.audio)),
    };
  }
  function syncDialogue() {
    const parsed = ScriptEditor.dialogue.parse($('dialogue-script').value);
    let added = false;
    if (!flow.scene.operation) for (const turn of parsed.turns) {
      if (turn.speaker < 0 || turn.speaker >= ScriptEditor.dialogue.maxSpeakers
          || flow.scene.roles.some(role => role.speaker === turn.speaker)) continue;
      flow.scene.roles.push(sceneRole(`S${turn.speaker + 1}`, turn.speaker)); added = true;
    }
    flow.scene.lines = ScriptEditor.dialogue.reconcile(parsed.turns, flow.scene.roles, flow.scene.lines, uid);
    flow.scene.script = $('dialogue-script').value; flow.scene.scriptVersion = 1;
    if (added) renderSpeakers();
    return parsed;
  }
  function renderSpeakers() {
    const list = $('characters'); if (!list) return;
    const old = new Map([...list.children].map(row => [row.dataset.roleId, row]));
    for (const [index, role] of flow.scene.roles.entries()) {
      let row = old.get(role.id);
      if (row?.boundRole !== role) { row?.remove(); row = renderRole(role); }
      if (list.children[index] !== row) list.insertBefore(row, list.children[index] || null);
      old.delete(role.id);
      const material = speakerAudio(role), insert = row.querySelector('.speaker-insert');
      const name = row.querySelector('.speaker-audio-name');
      name.textContent = material.name || t('No voice selected'); name.title = material.name || '';
      if (!material.name) name.dataset.i18n = 'No voice selected'; else delete name.dataset.i18n;
      insert.setAttribute('aria-label', t('Add a turn for {speaker}', { speaker: speakerLabel(role) }));
      insert.title = t('Add a turn for {speaker}', { speaker: speakerLabel(role) });
      insert.disabled = flow.scene.operation?.action === 'revision';
      row.querySelector('.speaker-play').disabled = !material.url && !material.reference.upload?.id;
      row.querySelector('.speaker-change').disabled = Boolean(flow.scene.operation) || state.jobSubmitting;
      const remove = row.querySelector('.speaker-remove');
      const removeTitle = flow.scene.roles.length > 1 ? 'Remove speaker' : 'At least one speaker is required.';
      remove.disabled = Boolean(flow.scene.operation) || state.jobSubmitting || flow.scene.roles.length <= 1;
      remove.title = t(removeTitle); remove.dataset.i18nTitle = removeTitle;
      ReferencePrep.resume(role, () => { renderSpeakers(); saveSoon(); });
    }
    for (const row of old.values()) row.remove();
    $('add-character').disabled = Boolean(flow.scene.operation) || state.jobSubmitting
      || flow.scene.roles.length >= ScriptEditor.dialogue.maxSpeakers;
  }
  function renderRole(role) {
    const row = node('section', 'dialogue-speaker'); row.dataset.roleId = role.id;
    row.dataset.speaker = role.speaker; row.boundRole = role;
    const insert = command('', () => ScriptEditor.append($('dialogue-script'), `${speakerLabel(role)}:`), 'speaker-insert');
    delete insert.dataset.i18n;
    insert.append(node('span', 'speaker-badge', speakerLabel(role)), node('span', 'speaker-audio-name'));
    const play = speakerTool('Preview voice', 'play', async () => {
      let material = speakerAudio(role);
      if (!material.url && material.reference.upload?.id) {
        const reference = material.reference, original = reference.upload.id;
        const upload = await api(`/api/reference-audio/${encodeURIComponent(original)}`);
        if (reference.upload?.id !== original || !flow.scene.roles.includes(role)) return;
        reference.upload = upload; material = speakerAudio(role); renderSpeakers(); saveSoon();
      }
      if (material.url) previewVoice({ audio_url: material.url }, play);
    });
    play.classList.add('speaker-play'); play.setAttribute('aria-pressed', 'false');
    const change = speakerTool('Change voice', 'switch', () => chooseSpeakerVoice(role));
    change.classList.add('speaker-change');
    const remove = speakerTool('Remove speaker', 'remove', () => removeSceneSpeaker(role));
    remove.classList.add('speaker-remove');
    row.append(insert, play, change, remove);
    return row;
  }
  function removeSceneSpeaker(role) {
    if (flow.scene.operation || state.jobSubmitting) return;
    if (flow.scene.roles.length <= 1) throw new Error(t('At least one speaker is required.'));
    const textarea = $('dialogue-script'), source = textarea.value;
    const turns = ScriptEditor.dialogue.parse(source).turns.filter(turn => turn.speaker === role.speaker);
    if (turns.some(turn => turn.text.trim())) {
      throw new Error(t('This speaker still has dialogue. Reassign its turns before removing.'));
    }
    const emptyTurnLines = new Set(turns.map(turn => turn.line - 1));
    const nextScript = source.split('\n').filter((_, index) => !emptyTurnLines.has(index)).join('\n');
    ReferencePrep.cancel(role); (role.extraReferences || []).forEach(ReferencePrep.cancel);
    flow.scene.roles = flow.scene.roles.filter(item => item !== role);
    stopVoicePreview();
    textarea.value = nextScript; ScriptEditor.sync(textarea);
    textarea.dispatchEvent(new Event('input', { bubbles: true }));
    renderSpeakers(); saveSoon();
  }
  function chooseSpeakerVoice(role) {
    if (flow.scene.operation || state.jobSubmitting) return;
    stopVoicePreview();
    let draft = copy(role);
    if (draft.extraReferences?.length) {
      const primary = draft.upload || draft.voiceId ? {} : draft.extraReferences.shift();
      draft = { ...draft, ...primary, id: role.id, name: role.name, speaker: role.speaker,
        groupReferences: [...(draft.groupReferences || []), ...(primary.groupReferences || []),
          ...draft.extraReferences], extraReferences: [] };
    }
    let source = draft.upload || !state.voices.length ? 'upload' : 'library';
    const dialog = makeDialog('Voice for {speaker}', { speaker: speakerLabel(role) });
    dialog.element.id = 'dialogue-voice-dialog'; dialog.element.classList.add('dialogue-voice-dialog');
    const tabs = node('div', 'voice-source-tabs'); tabs.setAttribute('role', 'tablist');
    const libraryPanel = node('div', 'dialogue-voice-library'), uploadPanel = node('div', 'dialogue-voice-upload');
    libraryPanel.id = 'dialogue-voice-library'; uploadPanel.id = 'dialogue-voice-upload';
    libraryPanel.setAttribute('role', 'tabpanel'); uploadPanel.setAttribute('role', 'tabpanel');
    const search = input('dialogue-voice-search', '', 'search');
    search.placeholder = t('Search your voices…'); search.dataset.i18nPlaceholder = 'Search your voices…';
    search.setAttribute('aria-label', t('Search your voices…')); search.dataset.i18nAriaLabel = 'Search your voices…';
    const list = node('div', 'dialogue-voice-options'); libraryPanel.append(search, list);
    const fields = node('fieldset', 'dialogue-voice-fields'); fields.append(tabs, libraryPanel, uploadPanel);
    const notice = node('p', 'form-note'); notice.id = 'dialogue-voice-error'; notice.setAttribute('role', 'alert');
    dialog.body.append(fields, notice);
    const redrawPreparation = () => {
      uploadPanel.querySelectorAll('audio').forEach(audio => audio.pause());
      uploadPanel.replaceChildren(ReferencePrep.mount(draft, {
        prefix: `dialogue-voice-${role.id}`, allowLibrary: false, changed: () => { notice.textContent = ''; },
      }));
      enhanceAllSelects(); StudioAudio.enhance(uploadPanel);
    };
    const choose = identifier => {
      ReferencePrep.cancel(draft); (draft.extraReferences || []).forEach(ReferencePrep.cancel);
      draft = { ...ReferencePrep.fresh(), id: role.id, name: role.name, speaker: role.speaker,
        description: role.description, extraReferences: [], ...(identifier ? ReferencePrep.forVoice(identifier) : {}) };
      drawLibrary(); redrawPreparation(); notice.textContent = '';
    };
    function drawLibrary() {
      stopVoicePreview(); list.replaceChildren();
      const query = search.value.trim().normalize('NFKC').toLowerCase();
      if (!query) {
        const none = command('No reference', () => choose(''), 'dialogue-no-reference');
        none.setAttribute('aria-pressed', String(!draft.voiceId && !draft.upload)); list.append(none);
      }
      for (const voice of state.voices.filter(item => [item.name, item.transcript,
        ...(item.references || []).map(clip => clip.transcript)].join(' ').normalize('NFKC').toLowerCase().includes(query))) {
        const row = node('div', 'dialogue-voice-option'); row.dataset.voiceId = voice.id;
        const selected = draft.voiceId === voice.id && !draft.upload;
        const pick = command('', () => choose(voice.id), 'dialogue-voice-pick');
        delete pick.dataset.i18n;
        pick.append(node('span', '', voice.name)); pick.setAttribute('aria-pressed', String(selected));
        if (selected) { const check = node('span', 'voice-choice-check'); check.innerHTML = icon('check'); pick.append(check); }
        const preview = speakerTool('Preview voice', 'play', () => previewVoice(voice, preview));
        preview.classList.add('dialogue-voice-preview'); preview.setAttribute('aria-pressed', 'false');
        row.append(pick, preview); list.append(row);
      }
      if (!list.children.length) list.append(localized('p', 'field-help', 'No matches found'));
    }
    const updateSource = () => {
      libraryPanel.hidden = source !== 'library'; uploadPanel.hidden = source !== 'upload';
      for (const tab of tabs.children) {
        const selected = tab.dataset.voiceSource === source;
        tab.setAttribute('aria-selected', String(selected)); tab.tabIndex = selected ? 0 : -1;
      }
    };
    for (const [value, caption] of [['library', 'Voice library'], ['upload', 'Upload audio']]) {
      const tab = command(caption, () => { source = value; stopVoicePreview(); updateSource(); }, 'voice-source-tab');
      tab.dataset.voiceSource = value; tab.setAttribute('role', 'tab');
      tab.id = `dialogue-voice-tab-${value}`; tab.setAttribute('aria-controls', `dialogue-voice-${value}`);
      (value === 'library' ? libraryPanel : uploadPanel).setAttribute('aria-labelledby', tab.id);
      tab.onkeydown = event => {
        if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
        event.preventDefault(); source = event.key === 'Home' ? 'library' : event.key === 'End' ? 'upload'
          : source === 'library' ? 'upload' : 'library'; updateSource();
        tabs.querySelector(`[data-voice-source="${source}"]`).focus();
      };
      tabs.append(tab);
    }
    search.oninput = drawLibrary;
    const use = command('Use this voice', async () => {
      fields.disabled = true; use.disabled = true; notice.textContent = '';
      try {
        await ReferencePrep.readAll(draft, role.speaker);
        for (const reference of draft.extraReferences || []) await ReferencePrep.readAll(reference, role.speaker);
        if (!flow.scene.roles.includes(role)) return;
        ReferencePrep.cancel(role); (role.extraReferences || []).forEach(ReferencePrep.cancel);
        flow.scene.roles[flow.scene.roles.indexOf(role)] = copy(draft);
        stopVoicePreview(); dialog.close(); renderSpeakers(); saveSoon();
      } catch (error) { notice.textContent = message(error.message); }
      finally { fields.disabled = false; use.disabled = false; }
    }, 'primary-btn'); use.id = 'dialogue-voice-apply';
    dialog.footer.append(use);
    dialog.element.addEventListener('close', () => {
      stopVoicePreview(); dialog.body.querySelectorAll('audio').forEach(audio => audio.pause());
      ReferencePrep.cancel(draft); (draft.extraReferences || []).forEach(ReferencePrep.cancel);
    });
    drawLibrary(); redrawPreparation(); updateSource(); dialog.show();
  }
  function renderScene() {
    const text = $('dialogue-script');
    if (typeof flow.scene.script !== 'string') {
      flow.scene.script = flow.scene.lines.some(line => line.text?.trim())
        ? ScriptEditor.dialogue.format(flow.scene.lines, flow.scene.roles) : '';
    }
    text.value = flow.scene.script; text.readOnly = flow.scene.operation?.action === 'revision';
    ScriptEditor.sync(text); syncDialogue(); renderSpeakers();
  }
  async function sceneRequest(selected = null) {
    const parsed = syncDialogue(), issue = parsed.issues[0];
    if (issue) throw new Error(t(issue.kind === 'speaker' ? 'Use a speaker from S1 to S16 on line {line}.'
      : 'Add a speaker label on line {line}.', { line: issue.line }));
    const missing = parsed.turns.find(turn => !flow.scene.roles.some(role => role.speaker === turn.speaker));
    if (missing) throw new Error(t('Speaker {speaker} is not in the source dialogue.', { speaker: `S${missing.speaker + 1}` }));
    const lines = (selected || flow.scene.lines).filter(line => line.text.trim());
    if (!lines.length) { ScriptEditor.get($('dialogue-script')).view.focus(); throw new Error(t('Enter dialogue text.')); }
    const used = new Set(lines.map(line => line.characterId));
    const roles = flow.scene.roles.filter(role => used.has(role.id));
    const common = {
      scene_order: flow.scene.lines.filter(line => line.text.trim()).map(line => line.id), lines: lines.map(line => ({
        character: roles.find(role => role.id === line.characterId)?.name || '',
        text: line.direction ? `[${line.direction}] ${line.text}` : line.text,
      })), line_ids: lines.map(line => line.id),
      direction: flow.scene.direction || '', ...settings({ scene: true }),
      studio: { work_id: flow.scene.work_id, segment_id: flow.scene.segment_id, layout: 'scene' },
    };
    if (flow.scene.operation) {
      return { ...common, source_version: flow.scene.operation.source, operation: flow.scene.operation.action,
        ...(flow.scene.operation.rewordReferences?.length ? {
          reword_references: flow.scene.operation.rewordReferences, reference_transcripts_checked: true,
        } : {}) };
    }
    const snapshot = {
      mode: flow.scene.strategy === 'joint' ? flow.scene.restoredMode || 'scene' : 'scene',
      scene_strategy: flow.scene.strategy, scene: flow.scene.scene || 'A conversation.',
      characters: roles.map(role => ({
        name: role.name, description: role.description || 'Natural conversational voice.', speaker: role.speaker,
      })), character_ids: roles.map(role => role.id),
      model: flow.scene.restoredModel || state.model?.selected_directory,
      ...(flow.scene.restoredFrom ? {
        condition_source: copy(flow.scene.restoredFrom), operation: 'from_settings',
        single_chunk: flow.scene.strategy === 'joint', previous_take: flow.scene.previous_take,
        previous_sources: copy(flow.scene.previous_sources || null),
      } : {}), ...common,
    };
    const prepared = await Promise.all(roles.flatMap(role =>
      [role, ...(role.extraReferences || [])].map(reference => ReferencePrep.readAll(reference, role.speaker))));
    return { ...snapshot, voice_bindings: prepared.flat() };
  }
  async function submitScene(lines = null, lineId = null) {
    if (state.jobSubmitting) return;
    state.jobSubmitting = true; renderModelPicker(); syncControls();
    try {
      const payload = await sceneRequest(lines);
      if (lineId) { payload.scene_strategy = 'per_line'; payload.studio.line_id = lineId; }
      await save(); await send(payload);
    } finally { state.jobSubmitting = false; renderModelPicker(); syncControls(); }
  }
  function renderSceneOperation() {
    const box = $('scene-operation'); if (!box) return;
    const back = $('return-scene-draft'); box.replaceChildren();
    const operation = flow.scene.operation;
    box.hidden = !operation;
    if (operation) {
      const title = node('strong', '', t({
        revision: 'Adjust performance', reword: 'Rewrite line',
      }[operation.action]));
      box.append(title, node('span', 'field-help',
        t('Based on {version}', { version: operation.source.job_id.slice(0, 6) })));
      if (modelMismatch(operation.model)) box.append(localized('span', 'field-help',
        'This source uses another model. Switch models or start a new draft.'));
    }
    if (!operation && flow.scene.restoredFrom) {
      box.hidden = false; box.append(localized('strong', '', 'Restored settings'),
        node('span', 'field-help', flow.scene.restoredFrom.job_id.slice(0, 6)));
      const selected = state.model?.selected_directory?.replaceAll('\\', '/').toLowerCase();
      const expected = flow.scene.restoredModel?.replaceAll('\\', '/').toLowerCase();
      if (expected && expected !== selected) {
        box.append(localized('span', 'field-help', 'This draft uses another model. Switch explicitly or use the selected model.'),
          command('Use selected model', () => {
            flow.scene.restoredModel = state.model?.selected_directory; renderSceneOperation(); saveSoon();
          }, 'text-btn'));
      }
    }
    back.hidden = !(flow.scene.returnDrafts || []).length;
    box.append(back); box.hidden = box.hidden && back.hidden;
  }
  function configureScene() {
    const parameters = node('details', 'advanced-settings scene-generation-settings');
    parameters.id = 'scene-generation-settings';
    parameters.append(localized('summary', '', 'Generation settings'));
    const fields = node('div', 'field-grid');
    for (const [id, label, value, step, min, max] of [
      ['max-chars', 'Text chunk target', '160', '1', '80', '320'],
      ['temperature', 'Temperature', state.model?.runtime === 'gguf' ? '1' : '.7', 'any', '0', '2'],
      ['top-p', 'Top P', state.model?.runtime === 'gguf' ? '.9' : '.7', 'any', '0', '1'],
      ['top-k', 'Top K', '0', '1', '0', '8193'],
      ['seed', 'Seed', '123', '1', '0', String(2 ** 32 - 1)],
    ]) {
      const control = input(`scene-${id}`, value, 'number'); Object.assign(control, { step, min, max });
      fields.append(field(label, control));
    }
    const random = input('scene-random-seed', '', 'checkbox'); random.checked = true;
    const randomLabel = node('label', 'check-field');
    randomLabel.append(random, localized('span', '', 'Random seed'));
    fields.append(randomLabel); parameters.append(fields);
    $('submit-scene').closest('.compose-footer').before(parameters);
    const supportNote = node('p', 'field-help'); supportNote.id = 'scene-capability-notice'; supportNote.hidden = true;
    $('submit-scene').closest('.compose-footer').before(supportNote);
    if (flow.scene.generation) {
      for (const [id, value] of Object.entries(flow.scene.generation.form || {})) {
        if ($(`scene-${id}`)) $(`scene-${id}`).value = value;
      }
      random.checked = flow.scene.generation.random !== false;
    }
    const back = command('Return to draft', () => {
      const previous = flow.scene.returnDrafts?.pop(), rest = copy(flow.scene.returnDrafts || []);
      if (previous) restoreScene({ ...previous, returnDrafts: rest });
    }, 'text-btn'); back.id = 'return-scene-draft'; back.hidden = true;
    const operation = node('div', 'creative-operation'); operation.id = 'scene-operation'; operation.hidden = true;
    operation.append(back); $('view-scene').prepend(operation);
    const options = node('div', 'scene-strategies'); options.setAttribute('role', 'group');
    options.setAttribute('aria-label', t('Generation strategy'));
    options.dataset.i18nAriaLabel = 'Generation strategy';
    for (const [value, label] of [['joint', 'Joint dialogue'], ['per_line', 'Per-turn & assemble']]) {
      const choice = command(label, () => {
        if (flow.scene.operation) return;
        if (flow.scene.strategy !== value) {
          delete flow.scene.restoredMode; delete flow.scene.previous_take; delete flow.scene.previous_sources;
        }
        flow.scene.strategy = value; updateStrategy(); saveSoon();
      }); choice.dataset.strategy = value; options.append(choice);
    }
    document.querySelector('.dialogue-composer').prepend(options);
    ScriptEditor.mount($('dialogue-script'));
    $('dialogue-script').addEventListener('input', () => { syncDialogue(); saveSoon(); });
    $('add-character').replaceChildren(ScriptEditor.icon('plus'));
    $('add-character').onclick = () => {
      const used = new Set(flow.scene.roles.map(role => role.speaker));
      const speaker = Array.from({ length: 16 }, (_, i) => i).find(i => !used.has(i));
      if (speaker === undefined) return;
      const role = sceneRole(`S${speaker + 1}`, speaker);
      flow.scene.roles.push(role); renderSpeakers(); saveSoon(); chooseSpeakerVoice(role);
    };
    $('submit-scene').onclick = () => guarded(async () => {
      if (flow.scene.strategy === 'joint' && !state.capabilities?.joint_dialogue) {
        throw new Error(t('Not supported by this model'));
      }
      await submitScene();
    });
    updateStrategy();
    renderSceneOperation();
  }
  function updateStrategy() {
    for (const item of document.querySelectorAll('[data-strategy]')) {
      item.setAttribute('aria-pressed', String(item.dataset.strategy === flow.scene.strategy));
      item.disabled = Boolean(flow.scene.operation)
        || !(item.dataset.strategy === 'joint' ? state.capabilities?.joint_dialogue : state.capabilities?.scene_scheduler);
    }
    syncControls();
  }
  function currentTaskSource() {
    if (voiceDesignActive()) return { path: '', version: flow.designRecipe?.source || null };
    const mode = $('mode').value;
    const path = mode === 'infill' ? $('source-take').value : mode === 'stream' ? '' : $('previous-take').value;
    return { path, version: flow.operation?.source || (path ? findTake(path) : null) || flow.restoredFrom };
  }
  function renderTaskDetails() {
    const task = $('current-task-value'); if (!task) return;
    const action = flow.operation?.action;
    const label = { revision: 'Adjust performance', reword: 'Rewrite line',
      infill: 'Edit selected region' }[action]
      || (voiceDesignActive() ? 'Voice design' : $('mode').value === 'tts' ? 'Generate speech' : modeLabel($('mode').value));
    task.dataset.i18n = label; task.textContent = t(label);
    const source = currentTaskSource();
    $('task-source-row').hidden = !source.version && !source.path;
    $('current-source-value').textContent = source.version
      ? t('Version {number}', { number: source.version.job_id.slice(0, 6) })
        + (source.version.part_index === undefined ? '' : ` · ${t('Part {number}', { number: source.version.part_index + 1 })}`)
      : source.path.replaceAll('\\', '/').split('/').at(-1);
    $('view-task-source').hidden = !source.version;
  }
  function voiceDesignActive() {
    return flow.voiceSource === 'design' && !flow.operation && $('mode').value === 'tts';
  }
  function renderVoiceSource() {
    const design = voiceDesignActive(), describe = $('describe-voice');
    $('voice-design').hidden = !design;
    const references = $('reference-voice-source'); if (references) references.hidden = design;
    $('saved-voice-options').hidden = design;
    describe.hidden = design;
    describe.setAttribute('aria-pressed', String(design)); describe.setAttribute('aria-expanded', String(design));
    const voiceBar = document.querySelector('.voice-source-bar'); if (voiceBar) voiceBar.hidden = design;
    const title = document.querySelector('.compose-panel > .panel-head h2');
    title.dataset.i18n = design ? 'Voice design' : 'Create'; title.textContent = t(title.dataset.i18n);
    title.closest('.panel-head').hidden = !design;
    $('back-to-generation').hidden = !design;
  }
  function syncControls() {
    if (!$('target-seconds')) return;
    const caps = state.capabilities || {}, mode = $('mode').value;
    renderCloneSummary();
    const monitored = knownJob(state.monitoredJobId);
    const busy = state.jobSubmitting || state.switchingModel || state.releasingResources || flow.switchingInput
      || ['queued', 'running'].includes(monitored?.status)
      || state.jobs.some(job => ['queued', 'running'].includes(job.status));
    const design = voiceDesignActive();
    const supported = Boolean(caps[design ? 'voice_design' : modeCapabilities[mode]]);
    const singleMismatch = modelMismatch(design ? flow.designRecipe?.model : flow.operation?.model || flow.restoredModel);
    $('submit-job').disabled = busy || !supported || Boolean(singleMismatch)
      || Boolean(flow.operation?.restoreError)
      || (!design && !flow.operation && mode !== 'infill' && state.cloneMissing);
    if (state.cloneMissing) $('transcribe-clone').disabled = true;
    const singleNote = $('task-capability-notice');
    if (singleNote) {
      singleNote.hidden = !state.capabilities || supported || Boolean(singleMismatch);
      singleNote.dataset.i18n = 'This task is unavailable in this model or runtime. Your draft was kept.';
      singleNote.textContent = t(singleNote.dataset.i18n);
    }
    $('mode').disabled = Boolean(flow.operation && !flow.operation.legacy);
    $('previous-take').readOnly = $('source-take').readOnly = true;
    renderTaskDetails();
    renderFrozenVoice();
    renderVoiceSource();
    $('target-seconds').disabled = $('duration').disabled;
    $('target-seconds').parentElement.hidden = $('duration').disabled;
    $('language').parentElement.hidden = $('language').disabled;
    document.querySelector('.performance-fields').hidden = $('direction').disabled;
    const targetLabel = document.querySelector('label[for="target-seconds"]');
    targetLabel.dataset.i18n = mode === 'infill' ? 'Replacement duration (seconds)' : 'Target duration (seconds)';
    targetLabel.textContent = t(targetLabel.dataset.i18n);
    $('progressive-audio').disabled = !caps.streaming_audio || mode !== 'tts' || Boolean(flow.operation) || design;
    $('progressive-audio').parentElement.hidden = $('progressive-audio').disabled;
    $('describe-voice').disabled = busy || mode !== 'tts' || Boolean(flow.operation) || (!design && !caps.voice_design);
    $('back-to-generation').disabled = busy;
    $('return-draft').disabled = flow.switchingInput;
    const available = flow.scene.operation ? caps.director_revision && caps.joint_dialogue
      : flow.scene.strategy === 'joint' ? caps.joint_dialogue : caps.scene_scheduler;
    $('submit-scene').disabled = busy || !available || Boolean(flow.scene.restoreError)
      || Boolean(modelMismatch(flow.scene.operation?.model || flow.scene.restoredModel));
    const sceneNote = $('scene-capability-notice');
    if (sceneNote) {
      sceneNote.hidden = !state.capabilities || Boolean(available);
      sceneNote.dataset.i18n = 'This task is unavailable in this model or runtime. Your draft was kept.';
      sceneNote.textContent = t(sceneNote.dataset.i18n);
    }
    if ($('dialogue-script')) {
      $('dialogue-script').readOnly = flow.scene.operation?.action === 'revision';
      ScriptEditor.sync($('dialogue-script'));
      $('add-character').disabled = busy || Boolean(flow.scene.operation)
        || flow.scene.roles.length >= ScriptEditor.dialogue.maxSpeakers;
      for (const button of document.querySelectorAll('.speaker-change,.speaker-remove')) {
        button.disabled = busy || Boolean(flow.scene.operation)
          || button.classList.contains('speaker-remove') && flow.scene.roles.length <= 1;
      }
    }
  }
  function renderFrozenVoice() {
    const section = $('frozen-voice'); if (!section) return;
    const frozen = Boolean(flow.operation && !flow.operation.legacy && flow.operation.action !== 'infill');
    section.hidden = !frozen;
    if (!frozen) {
      section.querySelectorAll('audio').forEach(audio => audio.pause()); return;
    }
    const references = flow.operation.frozenReferences || [];
    const key = JSON.stringify([flow.operation.source, references]);
    if (section.dataset.source === key) return;
    section.dataset.source = key; section.replaceChildren();
    const header = node('div', 'frozen-voice-heading');
    header.append(localized('span', 'field-label', 'Voice from source version'),
      command('View settings', () => guarded(() => showConditions(flow.operation.source)), 'text-btn'));
    section.append(header);
    if (!references.length) section.append(localized('span', 'field-help', 'Source audio'));
    else {
      const details = node('details');
      details.append(localized('summary', '', '{count} reference clips', { count: references.length }));
      for (const reference of references) {
        const audio = node('audio'); audio.controls = true; audio.preload = 'none';
        audio.src = reference.audio_url || fileUrl(reference.audio_path || reference.audio);
        details.append(audio, node('p', 'reference-transcript', reference.text));
      }
      section.append(details); StudioAudio.enhance(section);
    }
  }
  function translate() {
    ScriptEditor.translate(); renderOperation(); renderSceneOperation(); renderWorks();
    renderSpeakers();
    for (const element of document.querySelectorAll('.studio-dialog [data-i18n]')) element.textContent = t(element.dataset.i18n);
  }
  function configureWorkspaceLayout() {
    const compose = document.querySelector('.compose-panel');
    const head = compose.querySelector(':scope > .panel-head');
    const title = head.querySelector('h2');
    title.id = 'creation-title';
    title.dataset.i18n = 'Create'; title.textContent = t('Create');
    head.append(document.querySelector('.creation-toolbar'));
    // One row for voice sources; preparation tools stay with the selected audio.
    const voiceBar = document.querySelector('#clone-section > .reference-heading');
    voiceBar.classList.add('voice-source-bar');
    const voiceTitle = voiceBar.querySelector('label');
    voiceTitle.dataset.i18n = 'Voice'; voiceTitle.textContent = t('Voice');
    const library = $('saved-voice-options');
    const libraryTitle = library.querySelector('summary [data-i18n]');
    libraryTitle.dataset.i18n = 'Choose from voice library'; libraryTitle.textContent = t('Choose from voice library');
    const libraryBody = node('div', 'voice-library-popover');
    for (const child of [...library.children]) if (child.tagName !== 'SUMMARY') libraryBody.append(child);
    library.append(libraryBody);
    voiceBar.append(library, $('describe-voice'));
    const referenceSource = node('div'); referenceSource.id = 'reference-voice-source';
    for (const child of [...$('clone-section').children]) {
      if (child !== voiceBar && child !== $('voice-design')) referenceSource.append(child);
    }
    voiceBar.after(referenceSource, $('voice-design'));
    const saveUpload = [...$('clone-upload-preview').children]
      .find(child => child.tagName === 'BUTTON' && child.dataset.i18n === 'Save voice');
    if (saveUpload) document.querySelector('#clone-upload-preview .clone-file-actions').append(saveUpload);
    // Reuse the editor's single toolbar instead of a second metadata/action row.
    const editorArea = $('text').closest('.editor-area');
    const editorTools = editorArea.querySelector('.script-toolbar');
    editorTools.insertBefore($('text-count'), editorTools.children[1]);
    const clear = $('clear-text'); clear.setAttribute('aria-label', t('Clear text'));
    clear.dataset.i18nAriaLabel = 'Clear text'; clear.title = t('Clear text');
    clear.dataset.i18nTitle = 'Clear text'; delete clear.dataset.i18n;
    clear.innerHTML = icon('close'); clear.className = 'editor-tool editor-clear';
    editorTools.append(clear); editorArea.querySelector('.editor-meta').hidden = true;
    const scriptLabel = document.querySelector('label[for="text"]');
    scriptLabel.classList.add('script-label');
    const performance = document.querySelector('.performance-fields');
    performance.classList.add('inline-direction');
    scriptLabel.before(performance);
    $('direction').dataset.i18nPlaceholder = 'optional'; $('direction').placeholder = t('optional');
    const seedField = $('seed').parentElement, formerSeedRow = seedField.parentElement;
    $('temperature').parentElement.parentElement.append(seedField);
    formerSeedRow.hidden = true;
    const makeDock = (buttonId, settingsId, noteId) => {
      const primary = $(buttonId), footer = primary.closest('.compose-footer');
      footer.classList.add('action-dock');
      const inner = node('div', 'action-dock-inner'), choices = node('div', 'dock-choices');
      const actions = node('div', 'dock-actions');
      const hint = localized('span', 'dock-shortcut', 'Ctrl + Enter');
      actions.append(hint, primary);
      if (settingsId) {
        const settings = $(settingsId), summary = settings.querySelector(':scope > summary');
        settings.classList.add('dock-settings');
        const panel = node('div', 'generation-settings-popover');
        const heading = node('div', 'settings-popover-heading');
        const close = command('Close', () => { settings.open = false; summary.focus(); }, 'icon-btn');
        close.innerHTML = icon('close'); delete close.dataset.i18n;
        close.setAttribute('aria-label', t('Close')); close.dataset.i18nAriaLabel = 'Close';
        heading.append(localized('strong', '', 'Generation settings'), close); panel.append(heading);
        for (const child of [...settings.children]) if (child !== summary) panel.append(child);
        settings.append(panel); choices.append(settings);
      }
      if (noteId) choices.append($(noteId));
      inner.append(choices, actions); footer.replaceChildren(inner);
    };
    makeDock('submit-job', 'advanced-settings', 'task-capability-notice');
    makeDock('submit-scene', 'scene-generation-settings', 'scene-capability-notice');
    const runtime = node('details', 'settings-section'); runtime.id = 'runtime-details'; runtime.open = true;
    runtime.append(localized('summary', '', 'Local machine'), document.querySelector('.settings-grid'));
    $('view-settings').append(runtime);
    $('view-settings').prepend($('studio-about'));
    document.querySelectorAll('.dock-settings, .voice-source-bar > details').forEach(details => {
      details.addEventListener('toggle', () => {
        if (!details.open) return;
        for (const other of document.querySelectorAll('.dock-settings[open], .voice-source-bar > details[open]')) {
          if (other !== details) other.open = false;
        }
      });
    });
    document.addEventListener('pointerdown', event => {
      for (const details of document.querySelectorAll('.dock-settings[open], .voice-source-bar > details[open], .history-actions[open], .result-more[open]')) {
        if (!details.contains(event.target) && !event.target.closest('.select-popover, dialog')) details.open = false;
      }
    });
    document.addEventListener('keydown', event => {
      if (event.key !== 'Escape' || document.querySelector('dialog[open], .select-popover:not([hidden])')) return;
      const details = document.querySelector('.dock-settings[open], .voice-source-bar > details[open], .result-more[open]');
      if (details) { event.preventDefault(); details.open = false; details.querySelector('summary').focus(); }
    });
    document.addEventListener('drama:viewchange', () => {
      for (const details of document.querySelectorAll('.dock-settings[open], .voice-source-bar > details[open], .history-actions[open], .result-more[open]')) details.open = false;
    });
    fitWritingArea(compose, editorArea);
    fitWritingArea(document.querySelector('.dialogue-composer'), $('dialogue-editor-area'),
      { viewName: 'scene', submitId: 'submit-scene' });
    UiNumber.refresh();
  }
  function fitWritingArea(compose, editorArea, { viewName = 'generate', submitId = 'submit-job' } = {}) {
    const dock = $(submitId).closest('.action-dock'), main = $('main-content');
    let scheduled = false;
    const fit = () => {
      scheduled = false;
      if (!$(`view-${viewName}`).classList.contains('active') || !editorArea.getClientRects().length) return;
      const editor = editorArea.getBoundingClientRect(), panel = compose.getBoundingClientRect();
      const dockHeight = dock.getBoundingClientRect().height;
      const trailing = Math.max(0, panel.bottom - editor.bottom);
      // Use document coordinates so scrolling never changes the writing area's height.
      const height = Math.max(viewName === 'scene' ? 240 : voiceDesignActive() ? 160 : 180,
        Math.floor(innerHeight - (editor.top + scrollY) - trailing - dockHeight - 16));
      if (editorArea.style.height !== `${height}px`) editorArea.style.height = `${height}px`;
      main.style.setProperty('--composer-bottom-space', `${Math.ceil(dockHeight) + 16}px`);
    };
    const schedule = () => {
      if (!scheduled) { scheduled = true; requestAnimationFrame(fit); }
    };
    const observer = new ResizeObserver(schedule);
    for (const element of [compose, dock, document.querySelector('.topbar'), document.querySelector('.model-switcher')]) {
      observer.observe(element);
    }
    for (const event of ['resize', 'drama:studio-ready']) window.addEventListener(event, schedule);
    for (const event of ['drama:viewchange', 'drama:languagechange']) document.addEventListener(event, schedule);
    document.fonts.ready.then(schedule); schedule();
  }
  async function initialize() {
    configureControls(); configureScene(); configureWorkspaceLayout();
    const [data] = await Promise.all([
      api('/api/studio'), loadHealth(), loadVoices(), loadModel(), loadCapabilities(), loadJobs(),
    ]);
    flow.data = data;
    AudioEditor.configure({
      beforeOpen: save,
      currentView: () => state.view,
      rememberSource: (source, returnView) => {
        try {
          sessionStorage.setItem('drama3.ui.audio-edit-source', JSON.stringify(source));
          sessionStorage.setItem('drama3.ui.audio-edit-return', returnView);
        } catch (_) { /* keep editing available without session storage */ }
      },
      clearRememberedSource: () => {
        try { sessionStorage.removeItem('drama3.ui.audio-edit-source'); } catch (_) { /* optional storage */ }
      },
      enterEditor: returnView => view('audio-edit', { returnView }),
      leaveEditor: returnView => view(returnView),
      version: source => post('/api/studio/version', source),
      submit: send,
    }, flow.data.drafts.audio_editor?.value);
    const savedVoice = flow.data.drafts.voice?.value;
    if (savedVoice?.main) flow.voiceDraft = savedVoice;
    configureVoiceLibrary();
    const savedScene = flow.data.drafts.scene?.value;
    if (savedScene) {
      if (savedScene.generation?.form?.['top-k'] === '30'
          || savedScene.generation?.form?.['top-k'] === 30) {
        savedScene.generation.form['top-k'] = '0';
      }
      if (savedScene.operation?.action === 'continue') {
        const retired = copy(savedScene.operation);
        savedScene.operation = null; savedScene.retiredContinuation = retired;
        savedScene.work_id = uid(); savedScene.segment_id = uid();
        delete savedScene.restoredFrom; delete savedScene.previous_take; delete savedScene.previous_sources;
        try {
          const prepared = await post('/api/studio/prepare-restore', retired.source);
          for (const role of savedScene.roles) {
            const clips = prepared.references.filter(reference => reference.speaker === role.speaker);
            if (clips.length && !role.upload && !role.voiceId) {
              Object.assign(role, preparedReference(clips[0]), { extraReferences: clips.slice(1).map(preparedReference) });
            }
          }
        } catch (issue) { savedScene.restoreError = message(issue.message); }
      }
      flow.scene = savedScene;
      if (savedScene.restoreError) fail(new Error(savedScene.restoreError));
    }
    if (!flow.scene.roles.length) {
      flow.scene.roles = [sceneRole('S1', 0), sceneRole('S2', 1)];
      flow.scene.lines = flow.scene.roles.map(sceneLine);
    }
    if (savedScene?.generation) {
      for (const [id, value] of Object.entries(savedScene.generation.form || {})) {
        if ($(`scene-${id}`)) UiNumber.restore($(`scene-${id}`), value);
      }
      $('scene-random-seed').checked = savedScene.generation.random !== false;
    } else {
      $('scene-temperature').value = state.model?.runtime === 'gguf' ? '1' : '.7';
      $('scene-top-p').value = state.model?.runtime === 'gguf' ? '.9' : '.7';
    }
    renderScene(); updateStrategy(); renderSceneOperation();
    const draft = flow.data.drafts.single?.value;
    if (draft) await restore(draft);
    else {
      $('temperature').value = state.model?.runtime === 'gguf' ? '1' : '.7';
      $('top-p').value = state.model?.runtime === 'gguf' ? '.9' : '.7';
    }
    await restoreWorkspaceResults(draft?.active_job_id, savedScene?.active_job_id);
    flow.ready = true; renderWorks(); syncControls();
    if (!voiceDesignActive()) resumeCloneTranscription();
    document.addEventListener('input', saveSoon);
    document.addEventListener('change', saveSoon);
    document.addEventListener('click', event => {
      if (event.target.closest('.voice-pick,#default-voice,#remove-clone-audio,#restore-clone-audio,#crop-clone-audio')) {
        setTimeout(saveSoon, 100);
      }
      if (event.target.closest('.voice-pick,#default-voice,#remove-clone-audio')) clearExtraReferences();
    });
    $('clone-audio').addEventListener('change', clearExtraReferences);
    $('clone-upload-preview').addEventListener('toggle', saveSoon);
    $('text').addEventListener('input', () => { if (flow.operation?.action === 'infill') renderDiff(); });
    document.addEventListener('drama:languagechange', translate);
    document.addEventListener('drama:jobupdate', event => {
      renderRecovery(event.detail);
      syncControls();
      if (flow.lastMonitored !== event.detail.id) { flow.lastMonitored = event.detail.id; saveSoon(); }
    });
    document.addEventListener('drama:capabilities', () => {
      updateStrategy(); syncControls(); renderOperation(); renderSceneOperation();
    });
    document.addEventListener('drama:viewchange', event => {
      if (['audio-edit', 'audio-import'].includes(event.detail)) return;
      const monitor = document.querySelector('.monitor-panel');
      const container = event.detail === 'scene'
        ? document.querySelector('.scene-grid') : document.querySelector('.generation-grid');
      if (monitor.parentElement !== container) container.append(monitor);
      showWorkspaceResult(event.detail);
    });
    if (state.jobs.some(job => ['queued', 'running'].includes(job.status))) watch();
    view(state.view);
    if (state.view === 'audio-edit') {
      try {
        const restored = await AudioEditor.restoreActive();
        if (!restored) view(state.editorReturnView);
      } catch (error) {
        view(state.editorReturnView);
        fail(error);
      }
    }
    await StudioStartup.commit();
  }
  window.StudioFlow = {
    voiceDesignActive,
    sourceActive: () => Boolean(flow.operation && !flow.operation.legacy),
    lockText: () => flow.operation?.action === 'revision' && !flow.operation.legacy,
    lockVoice: () => Boolean(flow.operation && !flow.operation.legacy),
    prepareRequest, submitted, enhanceResult, enhanceSegment, afterResults, renderWorks,
    useLibraryVoice, referenceChanged: saveSoon, referencePrewarmPayload,
    voicesChanged: () => {
      ReferencePrep.refreshLibraries();
      if ($('selected-reference-details')) renderLibraryReferences();
      renderSpeakers();
      syncControls();
    },
    syncControls,
    historyJobs: () => [...Object.values(flow.workJobs), ...Object.values(state.resultJobs).filter(Boolean)],
    restoreVersion: source => guarded(() => restoreVersion(source)),
    async openTake(path, mode, selection) {
      const source = findTake(path); if (!source) return false;
      const job = knownJob(source.job_id) || {};
      const returnView = (job.studio || job.recipe?.studio)?.layout === 'scene' ? 'scene' : 'generate';
      view(returnView);
      await guarded(() => AudioEditor.open(source, { mode: mode === 'infill' ? 'local' : 'whole', selection, returnView }));
      return true;
    },
    flow, refresh, save, derive,
  };
  initialize().catch(error => {
    StudioStartup.fail(error);
    if ($('studio-error')) fail(error); else console.error(error);
  });
})();
