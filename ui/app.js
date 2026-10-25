const state = {
  voices: [], jobs: [], activeJob: null, selectedVoiceIds: new Set(),
  resultWorkspace: 'generate', resultJobs: { generate: null, scene: null },
  monitoredJobId: null, pollTimer: null, streamSubmitting: false, jobSubmitting: false,
  sourceTake: null, previousTake: null, takeRequest: 0,
  pendingAudioRegions: new WeakMap(),
  view: 'generate', hardware: null, capabilities: null, capabilityError: null, model: null,
  switchingModel: false, releasingResources: false, engineLoaded: false,
  modelError: null, modelRequest: 0, modelPickerAvailable: false,
  cloneFile: null, cloneUpload: null, clonePreviewUrl: null, cloneRevision: 0, cloneMissing: false,
  cloneUploadPromise: null, cloneAsrPromise: null, cloneAsrRequest: 0,
  cloneTextRevision: 0, cloneAsrText: '', cloneAsrStatus: '', cloneAsrTask: null,
  cloneOriginal: null, cloneCropping: false,
  referencePrewarmTimer: null, referencePrewarmPollTimer: null,
  referencePrewarmRequest: 0, referencePrewarmId: null,
  referencePrewarmUpdate: Promise.resolve(), referencePrewarmTask: null,
  modelCatalog: [], modelDownloadAvailable: true, modelDownloadSource: null,
  modelTask: null, modelDownloadPollTimer: null,
};
const $ = id => document.getElementById(id);
const { t } = I18n;
const message = value => I18n.message(value);
// Round presentation only. Unchanged fields retain their exact value for
// sampling, codec frames and saved drafts.
const UiNumber = (() => {
  const exact = new WeakMap();
  const format = value => value === '' || value == null ? '' : Number.isFinite(Number(value))
    ? String(Number(Number(value).toFixed(2))) : String(value);
  const write = (input, value) => {
    input.value = format(value); exact.set(input, { raw: value, display: input.value });
  };
  const value = input => {
    const previous = exact.get(input);
    return previous && previous.display === input.value ? String(previous.raw ?? '') : input.value;
  };
  const read = input => Number(value(input));
  const restore = (input, text) => input.type === 'number' ? write(input, text) : (input.value = text);
  const refresh = (root = document) => {
    for (const input of root.querySelectorAll('input[type="number"]')) write(input, value(input));
  };
  document.addEventListener('input', event => exact.delete(event.target), true);
  document.addEventListener('focusout', event => {
    if (event.target.matches('input[type="number"]')) write(event.target, value(event.target));
  });
  return Object.freeze({ format, write, value, read, restore, refresh });
})();
// Historical native reports already separate inference from worker loading.
// Never use total_seconds as an unqualified generation time.
const jobTiming = job => {
  const measured = job.timing;
  if (measured?.schema === 1 && Number.isFinite(measured.generation_seconds)) return measured;
  const result = job.result || {}, chunks = Array.isArray(result.chunks) ? result.chunks : [];
  const number = value => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null;
  let generation = null, loading = null;
  if (chunks.length <= 1 && (result.backend?.includes('GGUF') || job.recipe?.runtime?.backend === 'gguf')) {
    generation = number(result.client_request_seconds) ?? number(result.request_seconds);
    loading = result.session_reused === true ? 0 : result.session_reused === false ? number(result.load_seconds) : null;
  }
  if (generation === null && chunks.length && chunks.every(row => number(row.native_request_seconds) !== null)) {
    generation = chunks.reduce((total, row) => total + row.native_request_seconds, 0);
  }
  if (generation === null && number(result.cold_model_load_seconds) !== null && number(result.total_seconds) !== null) {
    loading = result.cold_model_load_seconds;
    generation = Math.max(0, result.total_seconds - loading);
  }
  return { generation_seconds: generation, model_load_seconds: loading,
    total_seconds: number(result.total_seconds), phase: 'finished' };
};
const generationTimeLabel = job => {
  const value = jobTiming(job).generation_seconds;
  return Number.isFinite(value) ? t('Generation · {seconds}s', { seconds: UiNumber.format(value) }) : '';
};
function audioDurationLabel(job) {
  const audio = job.result?.audio || {};
  const sampleDuration = Number.isFinite(audio.samples) && Number.isFinite(audio.sample_rate) && audio.sample_rate > 0
    ? audio.samples / audio.sample_rate : null;
  const duration = [audio.duration_seconds, job.result?.audio_seconds, job.result?.duration_seconds, sampleDuration]
    .find(value => typeof value === 'number' && Number.isFinite(value) && value >= 0);
  return duration === undefined ? '' : t('Audio · {seconds}s', { seconds: UiNumber.format(duration) });
}
const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, character => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
}[character]));
const translated = (source, args = {}) => `<span data-i18n="${escapeHtml(source)}" data-i18n-args="${escapeHtml(JSON.stringify(args))}">${escapeHtml(t(source, args))}</span>`;
const icon = name => `<svg class="icon" aria-hidden="true"><use href="#i-${name}"></use></svg>`;
const emptyState = (name, title, description = '', extra = '') => `<div class="empty-state ${extra}"><div class="empty-illustration">${icon(name)}</div><b>${translated(title)}</b>${description ? `<span>${translated(description)}</span>` : ''}</div>`;
const refreshCustomSelect = id => StudioControls.sync(id);
const enhanceAllSelects = () => StudioControls.init();
const closeCustomSelects = () => StudioControls.close();
const modeCapabilities = {
  tts: 'ordinary_tts', revision: 'director_revision', edit: 'director_revision',
  infill: 'infill', stream: 'streaming_text', audio_stream: 'streaming_audio',
};
function modelLocation(model) {
  if (model.location) return model.location;
  if (model.runtime !== 'gguf') return model.directory;
  return String(model.directory || '').replace(/[\\/][^\\/]*$/, '') || model.directory;
}
function modelOption(model) {
  const pathParts = String(model.directory || '').split(/[\\/]/);
  const ggufFilename = model.runtime === 'gguf' && /\.gguf$/i.test(model.name || '');
  const name = ggufFilename ? pathParts.at(-2)
    : model.name || pathParts.at(-1) || model.label;
  const option = new Option(name, model.directory);
  option.dataset.runtime = model.runtime || '';
  option.dataset.title = name;
  option.dataset.location = modelLocation(model);
  option.dataset.description = modelLocation(model);
  return option;
}
function statusLabel(status) {
  const labels = { queued: 'Queued', running: 'Running', complete: 'Complete', incomplete: 'Incomplete',
    cancelled: 'Cancelled', cancelling: 'Cancelling…', interrupted: 'Interrupted', failed: 'Failed',
    error: 'Error', idle: 'Idle', unknown: 'Unknown' };
  return t(labels[status] || status || 'Unknown');
}
function stageLabel(stage) {
  const labels = { runtime: 'Loading model', prefill: 'Prefill', decode: 'Generating', generate: 'Generating',
    prepare: 'Initializing', compute_policy: 'Initializing', compute_calibration: 'Initializing',
    generation: 'Generating', codec: 'Audio decoding', reference: 'Reference audio',
    audio: 'Audio segment', segment: 'Audio segment', waiting_for_text: 'Waiting for text',
    text_input: 'Text input', cancel: 'Cancel', cancelled: 'Cancelled', complete: 'Complete',
    error: 'Error', update: 'Update' };
  return stage ? t(labels[stage] || stage.replaceAll('_', ' ')) : '';
}
const fileUrl = path => {
  const value = String(path || '').replaceAll('\\', '/');
  if (value.startsWith('/files/')) return value;
  for (const marker of ['ui_data/', 'outputs/']) {
    const index = value.indexOf(marker);
    if (index >= 0) return `/files/${value.slice(index).split('/').map(encodeURIComponent).join('/')}`;
  }
  return '#';
};
const api = async (path, options = {}) => {
  const response = await fetch(path, {
    ...options, headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
};

const viewTitles = {
  generate: 'Generate speech', 'audio-import': 'Edit existing audio', scene: 'Dialogue',
  voices: 'Voice library', takes: 'History', settings: 'Engine settings', 'audio-edit': 'Edit audio',
};
const viewStorageKey = 'drama3.ui.view';
const audioEditSourceKey = 'drama3.ui.audio-edit-source';
const audioEditReturnKey = 'drama3.ui.audio-edit-return';
function restoreView() {
  const name = document.documentElement.dataset.view || 'generate';
  let returnView;
  if (name === 'audio-edit') {
    try { returnView = sessionStorage.getItem(audioEditReturnKey); } catch (_) { /* optional storage */ }
  }
  view(name, { returnView });
}
function view(name, options = {}) {
  if (!Object.hasOwn(viewTitles, name)) name = 'generate';
  const changed = state.view !== name;
  if (name === 'audio-edit') {
    state.editorReturnView = ['generate', 'scene', 'audio-import'].includes(options.returnView)
      ? options.returnView
      : ['generate', 'scene', 'audio-import'].includes(state.view) ? state.view
        : ['generate', 'scene', 'audio-import'].includes(state.editorReturnView) ? state.editorReturnView
          : (() => {
            try {
              const saved = sessionStorage.getItem(audioEditReturnKey);
              return ['generate', 'scene', 'audio-import'].includes(saved) ? saved : 'generate';
            } catch (_) { return 'generate'; }
          })();
    try {
      sessionStorage.setItem(audioEditReturnKey, state.editorReturnView);
      sessionStorage.setItem(viewStorageKey, name);
    } catch (_) { /* keep navigation available */ }
  } else {
    if (state.view === 'audio-edit') {
      try { sessionStorage.removeItem(audioEditSourceKey); } catch (_) { /* optional storage */ }
    }
    if (['generate', 'scene', 'audio-import'].includes(name)) state.editorReturnView = name;
  }
  state.view = name;
  document.documentElement.dataset.view = name;
  if (name !== 'audio-edit') {
    try { sessionStorage.setItem(viewStorageKey, name); } catch (_) { /* keep navigation available */ }
  }
  closeCustomSelects();
  document.querySelectorAll('.view').forEach(node => node.classList.toggle('active', node.id === `view-${name}`));
  document.querySelectorAll('.nav').forEach(node => {
    const active = node.dataset.view === (name === 'audio-edit' ? state.editorReturnView : name);
    node.classList.toggle('active', active);
    if (active) node.setAttribute('aria-current', 'page');
    else node.removeAttribute('aria-current');
  });
  $('view-title').textContent = t(viewTitles[name]);
  document.querySelector('.model-switcher').hidden = !['generate', 'scene', 'audio-import', 'audio-edit'].includes(name);
  document.dispatchEvent(new CustomEvent('drama:viewchange', { detail: name }));
  if (changed) window.scrollTo({ top: 0, behavior: 'instant' });
}
document.querySelectorAll('.nav, button[data-view="voices"]').forEach(node => {
  node.addEventListener('click', () => view(node.dataset.view));
});

async function useTake(path, mode = 'revision', selection = null) {
  if (!path) return;
  try {
    if (!window.StudioFlow || !await StudioFlow.openTake(path, mode, selection)) {
      throw new Error(t('No editable take'));
    }
  } catch (error) { addEvent({ stage: 'error', message: error.message }); }
}

function setAudioRegion(audio, start, end) {
  const source = audio.getAttribute('src');
  if (!source) return;
  const request = { source, start, end };
  state.pendingAudioRegions.set(audio, request);
  const apply = () => {
    if (state.pendingAudioRegions.get(audio) !== request) return;
    if (audio.getAttribute('src') === source) StudioAudio.setSelection(audio, start, end);
  };
  if (Number.isFinite(audio.duration) && audio.readyState >= 1) apply();
  else audio.addEventListener('loadedmetadata', apply, { once: true });
}
function showPreviousSource(take) {
  state.previousTake = take;
  const audio = $('previous-source-audio');
  const url = take.audio ? fileUrl(take.audio) : null;
  if (url && audio.getAttribute('src') !== url) audio.src = url;
  else if (!url) audio.removeAttribute('src');
  audio.dataset.frameSeconds = take.samples_per_frame / take.sample_rate;
  $('previous-source-preview').classList.toggle('hidden', !['revision', 'edit'].includes($('mode').value));
  $('edit-previous-selection').disabled = true;
  StudioAudio.enhance($('previous-source-preview'));
}
async function inspectPreviousSource() {
  const path = $('previous-take').value;
  if (!path) return;
  const request = ++state.takeRequest;
  try {
    const take = await api('/api/takes/inspect', { method: 'POST', body: JSON.stringify({ path }) });
    if (request !== state.takeRequest || $('previous-take').value !== path) return;
    showPreviousSource(take);
  } catch (error) { addEvent({ stage: 'error', message: error.message }); }
}
$('inspect-previous-source').onclick = inspectPreviousSource;
$('reload-previous-source').onclick = inspectPreviousSource;
$('previous-take').addEventListener('change', inspectPreviousSource);
$('previous-source-audio').addEventListener('drama:selectionchange', event => {
  $('edit-previous-selection').disabled = !event.detail || !state.capabilities?.infill;
});
$('edit-previous-selection').onclick = () => {
  if (!state.previousTake) return;
  const selection = StudioAudio.selection($('previous-source-audio'));
  if (selection) useTake(state.previousTake.path, 'infill', selection);
};
function showInfillSource(take, selection = null) {
  state.sourceTake = take;
  const audio = $('infill-source-audio');
  const url = take.audio ? fileUrl(take.audio) : null;
  if (url && audio.getAttribute('src') !== url) audio.src = url;
  else if (!url) audio.removeAttribute('src');
  audio.dataset.frameSeconds = take.samples_per_frame / take.sample_rate;
  $('infill-source-preview').classList.remove('hidden');
  $('infill-source-location').open = false;
  renderInfillInfo();
  StudioAudio.enhance($('infill-source-preview'));
  if (selection) {
    UiNumber.write($('start-time'), selection.start);
    UiNumber.write($('end-time'), selection.end);
    updateInfillFrames();
  } else {
    $('start-frame').value = Math.floor(take.frames / 3);
    $('end-frame').value = Math.ceil(take.frames * 2 / 3);
    updateInfillTimes();
  }
}
function renderInfillInfo() {
  const take = state.sourceTake;
  if (!take) return;
  $('infill-source-info').textContent = t('{seconds} seconds · {frames} frames. The selection must cover the changed words. Keep the rest of the transcript unchanged. Equal start/end inserts new audio.', {
    seconds: UiNumber.format(take.duration_seconds), frames: take.frames,
  });
}
function updateInfillTimes() {
  const take = state.sourceTake;
  if (!take) return;
  const seconds = take.samples_per_frame / take.sample_rate;
  for (const id of ['start-frame', 'end-frame']) {
    $(id).max = take.frames;
    const value = Number($(id).value);
    if (!Number.isFinite(value)) return;
    $(id).value = Math.max(0, Math.min(take.frames, Math.round(value)));
  }
  if (Number($('end-frame').value) < Number($('start-frame').value)) {
    $('end-frame').value = $('start-frame').value;
  }
  UiNumber.write($('start-time'), Number($('start-frame').value) * seconds);
  UiNumber.write($('end-time'), Number($('end-frame').value) * seconds);
  setAudioRegion($('infill-source-audio'), Number($('start-frame').value) * seconds,
    Number($('end-frame').value) * seconds);
}
function updateInfillFrames() {
  const take = state.sourceTake;
  if (!take) return;
  const ratio = take.sample_rate / take.samples_per_frame;
  const start = UiNumber.read($('start-time')), end = UiNumber.read($('end-time'));
  if (!Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end < start) return;
  // Keep exact values behind the millisecond display through roundtrips.
  const snap = value => Math.abs(value - Math.round(value)) < 0.00002 ? Math.round(value) : value;
  const first = Math.max(0, Math.min(take.frames, Math.floor(snap(start * ratio))));
  const last = start === end ? first : Math.max(0, Math.min(take.frames, Math.ceil(snap(end * ratio))));
  $('start-frame').value = first;
  $('end-frame').value = last;
  setAudioRegion($('infill-source-audio'), first / ratio, last / ratio);
}
$('infill-source-audio').addEventListener('drama:selectionchange', event => {
  if (!event.detail || !state.sourceTake) return;
  UiNumber.write($('start-time'), event.detail.start);
  UiNumber.write($('end-time'), event.detail.end);
  updateInfillFrames();
  updateInfillTimes();
});
$('inspect-infill-source').onclick = async () => {
  const path = $('source-take').value;
  if (!path) return;
  const request = ++state.takeRequest;
  try {
    const take = await api('/api/takes/inspect', {
      method: 'POST', body: JSON.stringify({ path }),
    });
    if (request !== state.takeRequest || $('source-take').value !== path || $('mode').value !== 'infill') return;
    const selection = state.sourceTake?.path === path ? {
      start: Number($('start-frame').value) * take.samples_per_frame / take.sample_rate,
      end: Number($('end-frame').value) * take.samples_per_frame / take.sample_rate,
    } : null;
    showInfillSource(take, selection);
  } catch (error) { addEvent({ stage: 'error', message: error.message }); }
};
for (const id of ['start-time', 'end-time']) $(id).addEventListener('input', updateInfillFrames);
for (const id of ['start-frame', 'end-frame']) $(id).addEventListener('input', updateInfillTimes);
$('mark-infill-start').onclick = () => {
  UiNumber.write($('start-time'), $('infill-source-audio').currentTime);
  updateInfillFrames();
};
$('mark-infill-end').onclick = () => {
  UiNumber.write($('end-time'), $('infill-source-audio').currentTime);
  updateInfillFrames();
};

function setEngine(hardware) {
  state.hardware = hardware;
  $('engine-dot').classList.toggle('ready', !hardware?.error && !state.capabilityError);
  $('engine-status').textContent = state.capabilityError ? message(state.capabilityError)
    : t(hardware?.error ? 'Hardware unavailable' : 'Local engine ready');
  $('engine-status').title = state.capabilityError || hardware?.error
    || (hardware?.gpu ? `${hardware.gpu} · ${UiNumber.format(hardware.vram_gib)} GiB VRAM` : '');
  delete $('hero-hardware').dataset.i18n;
  delete $('hero-memory').dataset.i18n;
  $('hero-hardware').textContent = hardware?.gpu || t('Local hardware');
  $('hero-memory').textContent = hardware?.ram_available_gib
    ? t('{ram} GiB RAM available · automatic placement', { ram: UiNumber.format(hardware.ram_available_gib) }) : t('Automatic placement');
  $('hero-hardware').title = $('hero-hardware').textContent;
  $('hero-memory').title = $('hero-memory').textContent;
  const labels = { platform: 'Platform', device: 'Device', gpu: 'GPU', vram_gib: 'VRAM (GiB)',
    ram_gib: 'RAM (GiB)', ram_available_gib: 'Available RAM (GiB)', commit_available_gib: 'Available commit (GiB)',
    cpu_logical_cores: 'Logical CPU cores' };
  $('hardware-card').innerHTML = Object.entries(hardware || {}).filter(([key]) => key !== 'error')
    .map(([key, value]) => `<div class="metric"><span>${escapeHtml(t(labels[key] || key.replaceAll('_', ' ')))}</span><b>${escapeHtml(typeof value === 'number' ? UiNumber.format(value) : value)}</b></div>`).join('');
}
async function loadHealth() {
  try {
    const health = await api('/api/health');
    state.engineLoaded = Boolean(health.engine_loaded);
    setEngine(health.hardware);
    renderResourceRelease();
  }
  catch (error) { $('engine-status').textContent = message(error.message); }
}
function renderResourceRelease() {
  const button = $('release-resources');
  button.disabled = state.releasingResources || !state.engineLoaded || modelBusy();
  const label = state.releasingResources ? 'Releasing resources…' : 'Release resources';
  button.dataset.i18nTitle = button.dataset.i18nAriaLabel = label;
  button.title = t(label); button.setAttribute('aria-label', t(label));
  button.setAttribute('aria-busy', String(state.releasingResources));
}
$('release-resources').onclick = async () => {
  if (state.releasingResources || modelBusy()) return;
  state.releasingResources = true;
  renderModelPicker(); window.StudioFlow?.syncControls();
  document.dispatchEvent(new CustomEvent('drama:resourceschange'));
  try {
    await cancelReferencePrewarm();
    const model = await api('/api/model/release', { method: 'POST', body: '{}' });
    state.engineLoaded = false;
    renderModel(model);
    await loadHealth();
  } catch (error) {
    $('engine-status').textContent = message(error.message);
    const notice = $('studio-error');
    if (notice) {
      notice.querySelector('.studio-error-message').textContent = message(error.message);
      notice.dataset.source = 'resource_release'; notice.hidden = false;
    }
  } finally {
    state.releasingResources = false;
    renderModelPicker(); window.StudioFlow?.syncControls();
    document.dispatchEvent(new CustomEvent('drama:resourceschange'));
  }
};
async function loadCapabilities() {
  try {
    const capabilities = await api('/api/capabilities');
    state.capabilities = capabilities;
    for (const option of $('mode').options) {
      option.disabled = !capabilities[modeCapabilities[option.value]];
    }
    $('submit-job').disabled = Boolean($('mode').selectedOptions[0]?.disabled);
    $('submit-scene').disabled = !capabilities.scene_scheduler;
    refreshCustomSelect('mode');
    updateMode();
    document.querySelectorAll('.use-take').forEach(button => {
      button.hidden = !capabilities[button.dataset.mode === 'revision' ? 'director_revision' : 'infill'];
    });
    if (state.activeJob) renderResults(state.activeJob);
    document.dispatchEvent(new CustomEvent('drama:capabilities', { detail: capabilities }));
    if (state.capabilityError) { state.capabilityError = null; setEngine(state.hardware); }
  } catch (error) { state.capabilityError = error.message; setEngine(state.hardware); }
}
const gib = bytes => `${(Number(bytes || 0) / 1024 ** 3).toFixed(2)} GiB`;
const modelDownloadActive = task => task?.operation === 'download'
  && ['queued', 'running', 'cancelling'].includes(task?.status);
function modelDownloadSourceLabel(source) {
  return source === 'modelscope' ? t('ModelScope')
    : source === 'huggingface' ? t('Hugging Face') : t('Fastest');
}
function selectedDownloadModel() {
  return state.modelCatalog.find(model => model.id === $('model-download-choice')?.value);
}
function renderModelDownloadCatalog() {
  const select = $('model-download-choice');
  if (!select) return;
  const active = modelDownloadActive(state.modelTask);
  const pending = active ? state.modelTask.model_id : select.value;
  select.replaceChildren();
  for (const model of state.modelCatalog) {
    const label = model.id === 'int4' ? 'INT4' : model.label;
    const option = new Option(t(label), model.id);
    option.dataset.i18n = label;
    option.dataset.description = `${model.format} · ${gib(model.total_bytes)}`;
    select.add(option);
  }
  if (state.modelCatalog.some(model => model.id === pending)) select.value = pending;
  else if (state.modelCatalog.length) select.value = state.modelCatalog[0].id;
  else select.add(new Option(
    t(state.modelDownloadAvailable ? 'Loading model list…' : 'Restart Studio to enable model downloads.'), ''));
  const model = selectedDownloadModel();
  const summary = $('model-download-summary');
  $('model-download-format').textContent = model?.format || '';
  $('model-download-size').textContent = model ? gib(model.total_bytes) : '';
  summary.hidden = !model;
  select.disabled = active || !model || !state.modelDownloadAvailable;
  if (model && (!model.sources.includes(state.modelDownloadSource)
      || (state.modelDownloadSource === 'auto' && model.sources.length < 2))) {
    state.modelDownloadSource = model.sources.length > 1 ? 'auto' : model.sources[0];
  }
  for (const button of document.querySelectorAll('.model-source-option')) {
    const source = button.dataset.source;
    const supported = source === 'auto' ? (model?.sources.length > 1)
      : model?.sources.includes(source);
    button.disabled = active || !supported;
    button.setAttribute('aria-pressed', String(source === state.modelDownloadSource));
  }
  $('model-download-source-note').textContent = !state.modelDownloadAvailable
    ? t('Restart Studio to enable model downloads.')
    : model?.sources.length > 1
    ? t('The fastest source is tested with a small sample. Downloads use resumable parallel chunks and SHA256 verification.')
    : model ? t('Only Hugging Face is available for this model.') : '';
  $('download-model').disabled = active || !model || !state.modelDownloadAvailable;
  refreshCustomSelect('model-download-choice');
}
function renderModelTask(task) {
  state.modelTask = task || null;
  const box = $('model-download-task');
  if (!box) return;
  clearTimeout(state.modelDownloadPollTimer);
  if (task?.operation === 'download' && state.modelCatalog.some(model => model.id === task.model_id)) {
    $('model-download-choice').value = task.model_id;
    const model = state.modelCatalog.find(row => row.id === task.model_id);
    if (model?.sources.includes(task.source_requested)
        || (task.source_requested === 'auto' && model?.sources.length > 1)) {
      state.modelDownloadSource = task.source_requested;
    }
  }
  const active = modelDownloadActive(task);
  box.hidden = !task || task.operation !== 'download';
  box.dataset.state = task?.status || '';
  renderModelDownloadCatalog();
  if (box.hidden) return;
  const progress = task.progress || {};
  const model = state.modelCatalog.find(row => row.id === task.model_id);
  const total = Number(progress.total_bytes) || Number(model?.total_bytes) || 0;
  const done = Math.min(total, Number(progress.done_bytes) || 0);
  const percent = total > 0 ? Math.min(100, done * 100 / total) : 0;
  const percentText = UiNumber.format(percent);
  const stage = progress.stage;
  const waiting = active && (task.status === 'queued' || !stage
    || ['preparing', 'source_probe', 'source_selected', 'connecting'].includes(stage));
  const source = modelDownloadSourceLabel(progress.source || task.source || task.source_requested);
  const verifying = stage === 'verifying' && Number(progress.file_total_bytes) > 0;
  const filePercent = verifying
    ? Math.min(100, Number(progress.file_bytes || 0) * 100 / Number(progress.file_total_bytes)) : 0;
  let text = t('Model download is running…');
  if (task.status === 'queued') text = t('Queued');
  else if (task.status === 'cancelling') text = t('Cancelling download…');
  else if (task.status === 'cancelled') text = t('Download cancelled. Partial files are saved; select Download model to resume.');
  else if (task.status === 'interrupted') text = t('Download interrupted. Select Download model to resume.');
  else if (task.status === 'error') text = stage === 'source_error' && progress.error_kind === 'network'
    ? t('Cannot connect to {source}. Check your network or proxy settings. ({error})', {
      source, error: progress.error || '',
    })
    : stage === 'source_error' && progress.http_status === 404
    ? t('The download source could not find {file} (HTTP 404).', { file: progress.file || '' })
    : t('Model download failed: {error}', { error: message(task.error) });
  else if (task.status === 'complete') text = t('Download complete and verified. Select it under Available models.');
  else if (stage === 'preparing') text = t('Preparing download…');
  else if (stage === 'source_probe') text = t('Testing download sources…');
  else if (stage === 'source_selected') text = t('Using {source}', { source: modelDownloadSourceLabel(progress.source) });
  else if (stage === 'connecting') text = t('Connecting to {source}…', { source });
  else if (stage === 'downloading') text = t('Downloading model…');
  else if (stage === 'verifying') text = t('Verifying {file}…', { file: progress.file || '' });
  $('model-download-message').textContent = text;
  $('model-download-percent').textContent = waiting ? ''
    : verifying ? `${UiNumber.format(filePercent)}%` : total ? `${percentText}%` : '';
  const bar = $('model-download-progress');
  if (waiting || !total) bar.removeAttribute('value');
  else bar.value = verifying ? filePercent : percent;
  bar.hidden = task.status === 'cancelled' || task.status === 'error' || task.status === 'interrupted';
  const speed = Number(progress.speed_mib_s) || 0;
  const size = value => value < 1024 ** 3 ? `${UiNumber.format(value / 1024 ** 2)} MiB` : gib(value);
  $('model-download-detail').textContent = verifying
    ? t('Verified {done} / {total}', {
      done: size(Number(progress.file_bytes || 0)), total: size(Number(progress.file_total_bytes)),
    })
    : waiting && total ? t('Download size: {size} · {source}', { size: size(total), source })
    : total
    ? t('Downloaded {done} / {total} · {speed} MiB/s · {source}', {
      done: size(done), total: size(total), speed: UiNumber.format(speed), source,
    })
    : task.directory || '';
  $('cancel-model-download').hidden = !active;
  if (active) {
    state.modelDownloadPollTimer = setTimeout(pollModelTask, 1000);
  }
}
async function pollModelTask() {
  try {
    const previous = state.modelTask;
    const result = await api('/api/model/task');
    renderModelTask(result.task);
    if (result.task?.status === 'complete' && previous?.status !== 'complete') await loadModel();
  } catch (_) {
    state.modelDownloadPollTimer = setTimeout(pollModelTask, 2000);
  }
}
async function downloadSelectedModel() {
  const model = selectedDownloadModel();
  if (!model || modelDownloadActive(state.modelTask)) return;
  $('download-model').disabled = true;
  try {
    const result = await api('/api/model/tasks', {
      method: 'POST',
      body: JSON.stringify({
        operation: 'download', model_id: model.id,
        source: state.modelDownloadSource || (model.sources.length > 1 ? 'auto' : model.sources[0]),
      }),
    });
    renderModelTask(result.task);
  } catch (error) {
    renderModelTask({
      operation: 'download', status: 'error', error: error.message, directory: '',
    });
  } finally {
    renderModelDownloadCatalog();
  }
}
function modelBusy() {
  return state.switchingModel || state.releasingResources || state.jobSubmitting || Boolean(state.model?.busy)
    || state.jobs.some(job => ['queued', 'running'].includes(job.status))
    || ['queued', 'running'].includes(state.activeJob?.status);
}
function renderModelPicker(reset = false) {
  renderResourceRelease();
  if (!state.model) return;
  const data = state.model, select = $('active-model');
  const pending = reset ? data.selected_directory : select.value || data.selected_directory;
  select.replaceChildren();
  for (const model of data.available_models || []) {
    select.add(modelOption(model));
  }
  if (![...select.options].some(option => option.value === data.selected_directory)) {
    select.add(modelOption({ name: data.selected_name, directory: data.selected_directory, runtime: data.runtime }));
  }
  select.add(new Option(t('Add a model…'), '__manage__'));
  select.value = [...select.options].some(option => option.value === pending) ? pending : data.selected_directory;
  const busy = modelBusy();
  select.disabled = state.switchingModel;
  $('apply-active-model').disabled = busy || !select.value || select.value === data.selected_directory;
  $('apply-active-model').hidden = !state.switchingModel && select.value === data.selected_directory;
  $('select-model').disabled = busy || state.model?.inspection?.ready === false
    || state.model?.inspection?.directory === data.selected_directory;
  $('browse-model-directory').disabled = busy;
  refreshCustomSelect('active-model');
  $('apply-active-model').textContent = t(state.switchingModel ? 'Switching model…' : 'Switch model');
  $('active-model-status').classList.toggle('error', Boolean(state.modelError));
  $('active-model-status').textContent = state.modelError
    ? t('Could not switch model: {error}', { error: message(state.modelError) })
    : state.switchingModel ? t('Switching model…')
    : busy ? ''
    : select.value !== data.selected_directory ? t('Choose a model, then click Switch model.')
    : data.selected_ready === false ? t('Model setup is required. Open Manage models to choose a local model.')
    : '';
  $('active-model-status').hidden = !$('active-model-status').textContent;
}
function renderModel(data) {
  state.model = data;
  if (typeof data.engine_loaded === 'boolean') state.engineLoaded = data.engine_loaded;
  renderReferencePrewarm(data.reference_prewarm || null);
  const inspected = data.inspection;
  $('model-badge').textContent = t(data.selected_ready ? 'READY' : 'SETUP REQUIRED');
  const choices = $('installed-model');
  choices.replaceChildren(new Option(t('Choose a local model…'), ''));
  for (const model of data.available_models || []) choices.add(modelOption(model));
  const pending = inspected.directory || data.selected_directory;
  choices.value = [...choices.options].some(option => option.value === pending)
    ? pending : data.selected_directory;
  renderModelLocations(data.model_locations || []);
  renderModelTask(data.task || null);
  $('model-selection-status').textContent = inspected.ready ? ''
    : t('Model files are incomplete; choose another model.');
  refreshCustomSelect('installed-model');
  renderModelPicker();
}
function renderModelLocations(locations) {
  const list = $('model-root-list');
  list.replaceChildren();
  const defaultLocation = locations.find(location => !location.removable);
  const visible = [
    ...(defaultLocation ? [defaultLocation] : []),
    ...locations.filter(location => location.removable),
  ];
  for (const location of visible) {
    const row = document.createElement('div');
    row.className = 'model-root-row';
    const path = document.createElement('span');
    path.className = 'model-root-path';
    path.textContent = location.path;
    path.title = location.path;
    row.append(path);
    if (location.removable) {
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.className = 'icon-btn model-root-remove';
      remove.title = t('Remove model folder');
      remove.setAttribute('aria-label', remove.title);
      remove.dataset.removeModelRoot = location.path;
      remove.innerHTML = icon('close');
      remove.onclick = async () => {
        remove.disabled = true;
        try {
          renderModel(await api('/api/model/roots/remove', {
            method: 'POST', body: JSON.stringify({ directory: location.path }),
          }));
        } catch (error) {
          $('model-location-status').textContent = message(error.message);
        }
      };
      row.append(remove);
    }
    list.append(row);
  }
}
async function loadModel() {
  try {
    try { state.modelPickerAvailable = (await api('/api/model/browse')).available === true; }
    catch (_) { state.modelPickerAvailable = false; }
    const model = await api('/api/model');
    try {
      const catalog = await api('/api/model/catalog');
      state.modelCatalog = catalog.models || [];
      state.modelDownloadAvailable = true;
    } catch (_) {
      state.modelCatalog = [];
      state.modelDownloadAvailable = false;
    }
    renderModel(model);
    renderModelDownloadCatalog();
  }
  catch (error) { $('model-selection-status').textContent = message(error.message); }
}
$('model-download-choice').onchange = () => {
  state.modelDownloadSource = null;
  renderModelDownloadCatalog();
};
for (const button of document.querySelectorAll('.model-source-option')) {
  button.onclick = () => {
    if (button.disabled) return;
    state.modelDownloadSource = button.dataset.source;
    renderModelDownloadCatalog();
  };
}
$('download-model').onclick = downloadSelectedModel;
$('cancel-model-download').onclick = async () => {
  $('cancel-model-download').disabled = true;
  try {
    const result = await api('/api/model/cancel', { method: 'POST', body: '{}' });
    renderModelTask(result.task);
  } catch (error) {
    $('model-download-message').textContent = message(error.message);
  } finally { $('cancel-model-download').disabled = false; }
};
document.addEventListener('drama:languagechange', renderModelDownloadCatalog);
async function inspectModel() {
  const request = ++state.modelRequest;
  try {
    const directory = $('installed-model').value || state.model?.selected_directory;
    if (!directory) return;
    const result = await api('/api/model/inspect', {
      method: 'POST', body: JSON.stringify({ directory }),
    });
    if (request === state.modelRequest) {
      $('model-location-status').textContent = '';
      renderModel(result);
    }
  } catch (error) { if (request === state.modelRequest) $('model-selection-status').textContent = message(error.message); }
}
$('installed-model').onchange = inspectModel;
$('browse-model-directory').onclick = async () => {
  const button = $('browse-model-directory');
  button.disabled = true;
  try {
    if (!state.modelPickerAvailable) {
      $('model-location-status').textContent = t('Restart Studio to enable folder selection.');
      return;
    }
    const result = await api('/api/model/browse', {
      method: 'POST',
      body: JSON.stringify({ initial_directory: state.model?.model_locations?.[0]?.path }),
    });
    if (!result.directory) return;
    const model = await api('/api/model/roots/add', {
      method: 'POST', body: JSON.stringify({ directory: result.directory }),
    });
    renderModel(model);
    const rootPath = result.directory.replaceAll('/', '\\').replace(/\\+$/, '').toLocaleLowerCase();
    const found = (model.available_models || []).some(row => {
      const location = String(row.location || row.directory).replaceAll('/', '\\').toLocaleLowerCase();
      return location === rootPath || location.startsWith(`${rootPath}\\`);
    });
    $('model-location-status').textContent = found ? '' : t('No Drama 3 models found in this folder.');
  } catch (error) {
    $('model-location-status').textContent = state.modelPickerAvailable
      ? message(error.message) : t('Restart Studio to enable folder selection.');
  } finally {
    button.disabled = modelBusy();
  }
};
async function switchModel(directory) {
  if (modelBusy()) return;
  ++state.modelRequest;
  state.switchingModel = true;
  state.modelError = null;
  renderModelPicker();
  window.StudioFlow?.syncControls();
  try {
    await cancelReferencePrewarm();
    const result = await api('/api/model/select', {
      method: 'POST', body: JSON.stringify({ directory }),
    });
    renderModel(result);
    renderModelPicker(true);
    await Promise.all([loadCapabilities(), loadHealth()]);
    scheduleReferencePrewarm();
  } catch (error) {
    state.modelError = error.message;
    $('model-selection-status').textContent = message(error.message);
  } finally {
    state.switchingModel = false;
    renderModelPicker();
    window.StudioFlow?.syncControls();
  }
}
$('select-model').onclick = () => switchModel($('installed-model').value);
$('apply-active-model').onclick = () => switchModel($('active-model').value);
$('active-model').onchange = () => {
  state.modelError = null;
  if ($('active-model').value === '__manage__') {
    $('active-model').value = state.model?.selected_directory || '';
    view('settings');
    $('installed-model').focus();
  }
  renderModelPicker();
};
$('manage-models').onclick = () => { view('settings'); $('installed-model').focus(); };
function selectedVoices() { return [...state.selectedVoiceIds]; }
function renderReferencePrewarm(task) {
  state.referencePrewarmTask = task;
  if (state.model) state.model.reference_prewarm = task;
  const box = $('reference-prewarm-status'), text = $('reference-prewarm-message');
  if (!box || !text) return;
  const messages = {
    queued: 'Warming voice reference…', running: 'Warming voice reference…',
    loading_model: 'Loading the model', encoding_reference: 'Encoding reference audio…',
    preparing_runtime: 'Preparing the model runtime…', warming_prefix: 'Warming voice reference…',
    canceling: 'Canceling warm-up…', ready: 'Voice reference ready',
    skipped: 'Warm-up skipped to protect available memory.',
    failed: 'Warm-up failed; generation can still continue.',
    unsupported: 'This model warms references during generation.',
  };
  const status = task?.status || 'idle';
  if (['idle', 'cancelled', 'unsupported'].includes(status)) {
    box.hidden = true;
    box.removeAttribute('data-state');
    return;
  }
  box.hidden = false;
  box.dataset.state = ['queued', 'running'].includes(status) ? 'busy'
    : status === 'ready' ? 'ready' : ['failed', 'skipped'].includes(status) ? 'error' : 'idle';
  text.textContent = t(messages[task.stage] || messages[status] || 'Warming voice reference…');
  $('reference-prewarm-cancel').hidden = !['queued', 'running'].includes(status)
    || task.stage === 'canceling';
}
async function referencePrewarmPayload() {
  if (['scene', 'audio-edit'].includes(state.view)) return { voice_bindings: [] };
  if (window.StudioFlow?.referencePrewarmPayload) {
    return window.StudioFlow.referencePrewarmPayload();
  }
  if (state.cloneMissing) return { uploaded_references: [] };
  if (state.cloneFile) {
    const transcript = $('clone-transcript').value.trim();
    if (!transcript) return null;
    const upload = await uploadCloneAudio();
    return { uploaded_references: [{ id: upload.id, text: transcript, speaker: 0 }] };
  }
  return { voice_ids: selectedVoices() };
}
function pollReferencePrewarm(id, request) {
  clearTimeout(state.referencePrewarmPollTimer);
  state.referencePrewarmPollTimer = setTimeout(async () => {
    if (request !== state.referencePrewarmRequest || state.referencePrewarmId !== id) return;
    try {
      const task = await api(`/api/reference-prewarm/${encodeURIComponent(id)}`);
      if (request !== state.referencePrewarmRequest || state.referencePrewarmId !== id) return;
      renderReferencePrewarm(task);
      if (['queued', 'running'].includes(task.status)) pollReferencePrewarm(id, request);
    } catch (_) {
      if (request === state.referencePrewarmRequest) renderReferencePrewarm({ status: 'failed' });
    }
  }, 900);
}
function scheduleReferencePrewarm(delay = 900) {
  clearTimeout(state.referencePrewarmTimer);
  clearTimeout(state.referencePrewarmPollTimer);
  const request = ++state.referencePrewarmRequest;
  state.referencePrewarmId = null;
  state.referencePrewarmTimer = setTimeout(async () => {
    try {
      if (state.jobSubmitting || state.jobs.some(job => ['queued', 'running'].includes(job.status))) return;
      const payload = await referencePrewarmPayload();
      if (request !== state.referencePrewarmRequest) return;
      const pending = state.referencePrewarmUpdate.catch(() => {}).then(async () => {
        if (request !== state.referencePrewarmRequest) return null;
        return payload ? api('/api/reference-prewarm', {
          method: 'POST', body: JSON.stringify(payload),
        }) : api('/api/reference-prewarm/cancel', { method: 'POST' });
      });
      state.referencePrewarmUpdate = pending;
      const task = await pending;
      if (request !== state.referencePrewarmRequest) return;
      state.referencePrewarmId = task.id || null;
      renderReferencePrewarm(task);
      if (task.id && ['queued', 'running'].includes(task.status)) pollReferencePrewarm(task.id, request);
    } catch (_) {
      if (request === state.referencePrewarmRequest) renderReferencePrewarm({ status: 'failed' });
    }
  }, delay);
}
async function cancelReferencePrewarm() {
  clearTimeout(state.referencePrewarmTimer);
  clearTimeout(state.referencePrewarmPollTimer);
  const request = ++state.referencePrewarmRequest;
  state.referencePrewarmId = null;
  try {
    const pending = state.referencePrewarmUpdate.catch(() => {}).then(() => {
      if (request !== state.referencePrewarmRequest) return null;
      return api('/api/reference-prewarm/cancel', { method: 'POST' });
    });
    state.referencePrewarmUpdate = pending;
    const task = await pending;
    if (request !== state.referencePrewarmRequest) return;
    state.referencePrewarmId = task.id || null;
    renderReferencePrewarm(task);
    if (task.id && ['queued', 'running'].includes(task.status)) pollReferencePrewarm(task.id, request);
  } catch (_) {
    if (request === state.referencePrewarmRequest) renderReferencePrewarm(null);
  }
}
window.ReferencePrewarm = Object.freeze({ schedule: scheduleReferencePrewarm, cancel: cancelReferencePrewarm });
document.addEventListener('drama:languagechange', () => renderReferencePrewarm(state.referencePrewarmTask));
document.addEventListener('drama:viewchange', event => {
  if (['scene', 'audio-edit', 'audio-import'].includes(event.detail)) cancelReferencePrewarm();
  else if (event.detail === 'generate') scheduleReferencePrewarm();
});
function clearCloneAudio(resetInput = true) {
  ++state.cloneRevision;
  ++state.cloneAsrRequest;
  ++state.cloneTextRevision;
  state.cloneUploadPromise = state.cloneAsrPromise = null;
  state.cloneAsrText = state.cloneAsrStatus = '';
  state.cloneAsrTask = null;
  state.cloneOriginal = null;
  state.cloneCropping = false;
  if (state.clonePreviewUrl) URL.revokeObjectURL(state.clonePreviewUrl);
  state.cloneFile = state.cloneUpload = state.clonePreviewUrl = null;
  state.cloneMissing = false;
  if (resetInput) $('clone-audio').value = '';
  $('clone-transcript').value = '';
  $('clone-file-audio').pause();
  $('clone-file-audio').removeAttribute('src');
  $('clone-upload-preview').classList.add('hidden');
  $('clone-upload-preview').open = true;
  $('clone-upload-zone').classList.remove('hidden');
  delete $('clone-note').dataset.i18n;
  $('clone-note').textContent = '';
  $('transcribe-clone').disabled = false;
  $('apply-asr-text').classList.add('hidden');
  $('clone-crop-status').textContent = '';
  $('crop-clone-audio').disabled = true;
  $('restore-clone-audio').classList.add('hidden');
  renderTranscriptionStatus();
  updateVoiceSelection();
}
function showMissingClone(upload, transcript) {
  clearCloneAudio();
  state.cloneUpload = upload; state.cloneMissing = true; state.selectedVoiceIds.clear();
  $('clone-file-name').textContent = upload.filename || upload.id;
  $('clone-transcript').value = transcript || '';
  $('clone-upload-preview').classList.remove('hidden'); $('clone-upload-zone').classList.add('hidden');
  $('clone-note').dataset.i18n = 'Reference audio is missing';
  $('clone-note').textContent = t('Reference audio is missing');
  $('transcribe-clone').disabled = true;
  renderCloneSummary(); updateVoiceSelection();
}
function selectCloneAudio(file, { auto = true, upload = null, original = null } = {}) {
  if (!file) return;
  if (!/\.(wav|mp3|flac|ogg|m4a)$/i.test(file.name)) {
    $('clone-note').textContent = t('Choose a WAV, MP3, FLAC, OGG or M4A reference');
    return;
  }
  if (file.size < 32 || file.size > 128 * 1024 ** 2) {
    $('clone-note').textContent = t('Reference audio must be nonempty and no larger than 128 MiB');
    return;
  }
  cancelReferencePrewarm();
  clearCloneAudio(false);
  state.cloneFile = file;
  state.cloneUpload = upload;
  state.cloneOriginal = original;
  state.clonePreviewUrl = URL.createObjectURL(file);
  state.selectedVoiceIds.clear();
  $('clone-file-name').textContent = file.name;
  $('clone-file-audio').src = state.clonePreviewUrl;
  $('clone-upload-preview').classList.remove('hidden');
  $('clone-upload-zone').classList.add('hidden');
  $('saved-voice-options').open = false;
  StudioAudio.enhance($('clone-upload-preview'));
  $('restore-clone-audio').classList.toggle('hidden', !original);
  updateVoiceSelection();
  if (auto) transcribeClone();
}
function renderCloneSummary() {
  const audio = $('clone-file-audio'), preview = $('clone-summary-play');
  const valid = Number.isFinite(audio.duration) && audio.duration > 0;
  $('clone-file-name').title = $('clone-file-name').textContent;
  $('clone-summary-duration').textContent = valid ? `${audio.duration.toFixed(1)}s` : '';
  const transcript = $('clone-summary-transcript');
  transcript.textContent = $('clone-transcript').value.trim() || t('Reference transcript needed');
  if (!$('clone-transcript').value.trim()) transcript.dataset.i18n = 'Reference transcript needed';
  else delete transcript.dataset.i18n;
  preview.disabled = !valid;
  preview.innerHTML = icon(audio.paused ? 'play' : 'pause');
  preview.setAttribute('aria-label', t(audio.paused ? 'Preview voice' : 'Pause audio'));
  preview.dataset.i18nAriaLabel = audio.paused ? 'Preview voice' : 'Pause audio';
}
for (const event of ['loadedmetadata', 'emptied', 'play', 'pause']) {
  $('clone-file-audio').addEventListener(event, renderCloneSummary);
}
$('clone-summary-play').onclick = event => {
  event.preventDefault(); event.stopPropagation();
  const audio = $('clone-file-audio');
  if (!audio.paused) audio.pause();
  else {
    document.querySelectorAll('audio').forEach(other => { if (other !== audio) other.pause(); });
    audio.play().catch(error => addEvent({ stage: 'error', message: error.message }));
  }
};
$('clone-summary-remove').onclick = event => {
  event.preventDefault(); event.stopPropagation();
  $('remove-clone-audio').click();
};
function updateCloneSelectionActions() {
  const selection = StudioAudio.selection($('clone-file-audio'));
  $('crop-clone-audio').disabled = state.cloneCropping || !selection
    || selection.end - selection.start < 0.5;
}
$('clone-file-audio').addEventListener('drama:selectionchange', updateCloneSelectionActions);
$('crop-clone-audio').onclick = async () => {
  const selection = StudioAudio.selection($('clone-file-audio'));
  if (!selection || state.cloneCropping || !state.cloneFile || selection.end - selection.start < 0.5) return;
  const revision = state.cloneRevision;
  state.cloneCropping = true; updateCloneSelectionActions();
  $('clone-crop-status').textContent = t('Preparing selected audio…');
  try {
    const upload = await uploadCloneAudio();
    if (revision !== state.cloneRevision) return;
    const original = state.cloneOriginal || {
      file: state.cloneFile, upload, transcript: $('clone-transcript').value,
    };
    const cropped = await api('/api/reference-audio/crop', {
      method: 'POST', body: JSON.stringify({ id: upload.id, start: selection.start, end: selection.end }),
    });
    const response = await fetch(cropped.audio_url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const bytes = await response.blob();
    if (revision !== state.cloneRevision) return;
    selectCloneAudio(new File([bytes], cropped.filename, { type: 'audio/wav' }), {
      auto: false, upload: cropped, original,
    });
    $('clone-crop-status').textContent = t('Audio cropped. Review the transcript.');
    transcribeClone();
  } catch (error) {
    if (revision === state.cloneRevision) $('clone-crop-status').textContent = message(error.message);
  } finally {
    if (revision === state.cloneRevision) state.cloneCropping = false;
    updateCloneSelectionActions();
  }
};
$('restore-clone-audio').onclick = () => {
  const original = state.cloneOriginal;
  if (!original) return;
  selectCloneAudio(original.file, { auto: false, upload: original.upload });
  $('clone-transcript').value = original.transcript;
  if (!original.transcript) transcribeClone();
  else scheduleReferencePrewarm();
};
async function uploadCloneAudio() {
  if (state.cloneUpload) return state.cloneUpload;
  if (!state.cloneUploadPromise) {
    const revision = state.cloneRevision;
    const file = state.cloneFile;
    const data = new FormData();
    data.append('audio', file);
    $('clone-note').textContent = t('Uploading…');
    const pending = (async () => {
      try {
        const response = await fetch('/api/reference-audio', { method: 'POST', body: data });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
        if (revision !== state.cloneRevision) {
          throw new Error(t('Reference changed. Click Generate again.'));
        }
        state.cloneUpload = result;
        return result;
      } finally {
        if (revision === state.cloneRevision) $('clone-note').textContent = '';
      }
    })();
    state.cloneUploadPromise = pending;
    pending.catch(() => {
      if (state.cloneUploadPromise === pending) state.cloneUploadPromise = null;
    });
  }
  return state.cloneUploadPromise;
}
function renderTranscriptionStatus() {
  const status = $('clone-asr-status'), source = state.cloneAsrStatus;
  const busy = ['Preparing transcription…', 'Waiting to transcribe…', 'Downloading Qwen3-ASR…',
    'Loading Qwen3-ASR…', 'Transcribing…', 'GPU memory unavailable. Transcribing on CPU…'].includes(source);
  const visible = busy || source === 'Transcription failed. Enter the transcript manually or try again.';
  status.textContent = visible ? t(source) : ''; status.hidden = !visible;
  status.dataset.state = busy ? 'busy'
    : source === 'Transcription failed. Enter the transcript manually or try again.' ? 'error' : 'idle';
  $('clone-transcript').setAttribute('aria-busy', String(busy));
  renderCloneSummary();
}
function resumeCloneTranscription() {
  if (state.cloneFile && (state.cloneAsrTask || !$('clone-transcript').value.trim())) transcribeClone(false);
}
function transcribeClone(force = false) {
  if (!state.cloneFile || state.cloneAsrPromise) return state.cloneAsrPromise;
  const revision = state.cloneRevision, request = ++state.cloneAsrRequest;
  const saved = !force && state.cloneAsrTask;
  const textRevision = saved ? saved.textRevision : state.cloneTextRevision;
  const current = () => revision === state.cloneRevision && request === state.cloneAsrRequest;
  const status = source => {
    if (!current()) return;
    state.cloneAsrStatus = source;
    renderTranscriptionStatus();
  };
  $('transcribe-clone').disabled = true;
  $('clone-asr-status').removeAttribute('title');
  const pending = (async () => {
    try {
      status('Preparing transcription…');
      const upload = await uploadCloneAudio();
      if (!current()) return;
      let task = saved && saved.uploadId === upload.id ? await api(`/api/transcriptions/${encodeURIComponent(saved.id)}`)
        : await api('/api/transcriptions', {
          method: 'POST', body: JSON.stringify({ upload_id: upload.id, language: 'auto', force, local_only: true }),
        });
      if (!current()) return;
      if (task.status === 'skipped') { status(''); return; }
      state.cloneAsrTask = { id: task.id, uploadId: upload.id, textRevision };
      window.StudioFlow?.referenceChanged();
      const stages = {
        queued: 'Waiting to transcribe…', preparing: 'Preparing transcription…',
        downloading: 'Downloading Qwen3-ASR…', loading: 'Loading Qwen3-ASR…',
        transcribing: 'Transcribing…', cpu_fallback: 'GPU memory unavailable. Transcribing on CPU…',
      };
      while (current() && ['queued', 'running'].includes(task.status)) {
        status(stages[task.stage] || 'Transcribing…');
        await new Promise(resolve => setTimeout(resolve, 650));
        if (!current()) return;
        task = await api(`/api/transcriptions/${task.id}`);
      }
      if (!current()) return;
      if (task.status === 'skipped') { status(''); return; }
      if (task.status !== 'complete') throw new Error(task.error || 'Transcription failed.');
      state.cloneAsrText = task.result?.text || '';
      if (!state.cloneAsrText.trim()) throw new Error('No speech was recognized. Enter the transcript manually or try another clip.');
      if (state.cloneTextRevision === textRevision) {
        $('clone-transcript').value = state.cloneAsrText;
        $('apply-asr-text').classList.add('hidden');
      } else {
        $('apply-asr-text').classList.remove('hidden');
      }
      status('');
    } catch (error) {
      if (current()) {
        status('Transcription failed. Enter the transcript manually or try again.');
        $('clone-asr-status').title = message(error.message);
      }
    } finally {
      if (current()) {
        state.cloneAsrPromise = null;
        state.cloneAsrTask = null;
        $('transcribe-clone').disabled = false;
        window.StudioFlow?.referenceChanged();
        if ($('clone-transcript').value.trim()) scheduleReferencePrewarm();
      }
    }
  })();
  state.cloneAsrPromise = pending;
  return pending;
}
async function prepareCloneReference() {
  if (state.cloneMissing) throw new Error(t('Reference audio is missing'));
  if (!state.cloneFile) return [];
  if (state.cloneCropping) throw new Error(t('Wait for audio cropping to finish.'));
  const revision = state.cloneRevision;
  let transcript = $('clone-transcript').value.trim();
  if (!transcript && state.cloneAsrPromise) {
    await state.cloneAsrPromise; transcript = $('clone-transcript').value.trim();
  }
  if (revision !== state.cloneRevision) {
    throw new Error(t('Reference changed. Click Generate again.'));
  }
  if (!transcript) {
    $('clone-transcript').focus();
    throw new Error(t('Enter the exact words spoken in the reference audio.'));
  }
  const upload = await uploadCloneAudio();
  if (revision !== state.cloneRevision) {
    throw new Error(t('Reference changed. Click Generate again.'));
  }
  return [{ id: upload.id, text: transcript, speaker: 0 }];
}
function updateVoiceSelection() {
  document.querySelectorAll('.voice-pick').forEach(node => {
    const selected = state.selectedVoiceIds.has(node.dataset.voice);
    node.classList.toggle('selected', selected);
    node.setAttribute('aria-pressed', selected);
  });
  const empty = !state.cloneFile && !state.cloneMissing && !state.selectedVoiceIds.size;
  $('default-voice').classList.toggle('selected', empty);
  $('default-voice').setAttribute('aria-pressed', empty);
  window.StudioFlow?.voicesChanged();
}
$('default-voice').onclick = () => {
  state.selectedVoiceIds.clear(); clearCloneAudio(); cancelReferencePrewarm();
};
$('clone-audio').addEventListener('change', event => selectCloneAudio(event.target.files[0]));
$('remove-clone-audio').onclick = () => { clearCloneAudio(); cancelReferencePrewarm(); };
$('replace-clone-audio').onclick = () => $('clone-audio').click();
$('transcribe-clone').onclick = () => transcribeClone(true);
$('clone-transcript').addEventListener('input', () => {
  ++state.cloneTextRevision; renderCloneSummary(); scheduleReferencePrewarm(1800);
});
$('apply-asr-text').onclick = () => {
  $('clone-transcript').value = state.cloneAsrText;
  ++state.cloneTextRevision;
  $('apply-asr-text').classList.add('hidden');
  state.cloneAsrStatus = '';
  renderTranscriptionStatus();
  scheduleReferencePrewarm();
};
$('reference-prewarm-cancel').onclick = cancelReferencePrewarm;
for (const type of ['dragover', 'dragenter']) $('clone-upload-zone').addEventListener(type, event => {
  event.preventDefault();
  $('clone-upload-zone').classList.add('dragging');
});
for (const type of ['drop', 'dragleave']) $('clone-upload-zone').addEventListener(type, event => {
  event.preventDefault();
  $('clone-upload-zone').classList.remove('dragging');
});
$('clone-upload-zone').addEventListener('drop', event => selectCloneAudio(event.dataTransfer.files[0]));
let audition = null, auditionButton = null;
function renderPreviewButton(button, playing) {
  const history = button.classList.contains('history-play');
  button.innerHTML = icon(playing ? 'pause' : 'play') + (history ? translated(playing ? 'Pause audio' : 'Play') : '');
  button.setAttribute('aria-pressed', String(playing));
  const label = playing ? 'Pause audio' : history ? 'Play' : 'Preview voice';
  button.dataset.i18nAriaLabel = label; button.setAttribute('aria-label', t(label));
}
function stopVoicePreview() {
  audition?.pause();
  if (auditionButton) renderPreviewButton(auditionButton, false);
  audition = auditionButton = null;
}
document.addEventListener('play', event => { if (event.target instanceof HTMLAudioElement) stopVoicePreview(); }, true);
document.addEventListener('drama:viewchange', stopVoicePreview);
function previewVoice(voice, button) {
  const resume = auditionButton === button && audition && button.classList.contains('history-play') && !audition.ended && !audition.error;
  if (auditionButton === button && audition && !audition.paused && !audition.error) {
    if (resume) audition.pause(); else stopVoicePreview();
    return;
  }
  if (!resume) stopVoicePreview();
  if (!voice?.audio_url) return;
  auditionButton = button;
  const audio = resume ? audition : new Audio(voice.audio_url); audition = audio;
  for (const audio of document.querySelectorAll('audio')) audio.pause();
  button.closest('.job-actions')?.querySelector('.history-play-error')?.remove();
  const reset = () => renderPreviewButton(button, false);
  if (!resume) {
    audio.addEventListener('pause', reset);
    audio.addEventListener('ended', reset);
  }
  audio.play().then(() => {
    if (audition !== audio || audio.paused) return;
    renderPreviewButton(button, true);
  }).catch(error => {
    reset();
    if (audition !== audio || error.name === 'AbortError') return;
    if (button.classList.contains('history-play')) {
      const error = document.createElement('p'); error.className = 'history-play-error';
      error.setAttribute('role', 'alert'); error.dataset.i18n = 'Unable to play this audio.';
      error.textContent = t('Unable to play this audio.'); button.closest('.job-actions').append(error);
    } else addEvent({ stage: 'error', message: t('Unable to play this audio.') });
  });
}
function renderVoices() {
  stopVoicePreview();
  const moreClips = voice => (voice.references?.length || 1) < 2 ? '' : (
    `<details class="voice-reference-clips"><summary>${translated('{count} reference clips', { count: voice.references.length })}</summary>${voice.references.slice(1).map(clip => (
      `<div class="voice-reference-clip"><p class="voice-transcript">${escapeHtml(clip.transcript)}</p><button class="icon-btn" type="button" data-preview-clip="${escapeHtml(clip.audio_url)}" data-i18n-title="Preview voice" title="${escapeHtml(t('Preview voice'))}" data-i18n-aria-label="Preview voice" aria-label="${escapeHtml(t('Preview voice'))}" aria-pressed="false">${icon('play')}</button><a class="voice-download" download href="${escapeHtml(clip.audio_url)}" data-i18n-title="Download" title="${escapeHtml(t('Download'))}" data-i18n-aria-label="Download" aria-label="${escapeHtml(t('Download'))}">${icon('download')}</a></div>`
    )).join('')}</details>`
  );
  const markup = state.voices.length ? state.voices.map(voice => (
    `<div class="voice-choice"><button class="voice-pick ${state.selectedVoiceIds.has(voice.id) ? 'selected' : ''}" data-voice="${escapeHtml(voice.id)}" aria-pressed="${state.selectedVoiceIds.has(voice.id)}"><span class="voice-pick-icon">${icon('mic')}</span><span class="voice-pick-copy"><b>${escapeHtml(voice.name)}</b><small>${translated('{count} reference clips', { count: voice.references?.length || 1 })}</small></span><span class="voice-pick-check">${icon('check')}</span></button><button class="voice-preview" type="button" data-preview="${escapeHtml(voice.id)}" data-i18n-title="Preview voice" title="${escapeHtml(t('Preview voice'))}" data-i18n-aria-label="Preview voice" aria-label="${escapeHtml(t('Preview voice'))}" aria-pressed="false">${icon('play')}</button></div>`
  )).join('') : `<span class="muted">${translated('No saved voices yet')}</span>`;
  for (const id of ['voice-picks']) {
    const container = $(id);
    if (!container) continue;
    container.innerHTML = !state.voices.length && id === 'voice-picks' ? '' : markup;
    container.querySelectorAll('[data-voice]').forEach(button => {
      button.onclick = () => {
        const id = button.dataset.voice;
        if (window.StudioFlow) { StudioFlow.useLibraryVoice(id); return; }
        if (state.cloneFile || state.cloneMissing) clearCloneAudio();
        if (state.selectedVoiceIds.has(id)) state.selectedVoiceIds.delete(id);
        else { state.selectedVoiceIds.clear(); state.selectedVoiceIds.add(id); }
        updateVoiceSelection();
        $('saved-voice-options').open = false;
        scheduleReferencePrewarm();
      };
    });
    container.querySelectorAll('[data-preview]').forEach(button => {
      button.onclick = () => previewVoice(state.voices.find(voice => voice.id === button.dataset.preview), button);
    });
  }
  $('voice-hint').innerHTML = state.voices.length
    ? translated('{count} saved references', { count: state.voices.length }) : translated('No saved voices yet');
  $('voice-list').innerHTML = state.voices.length ? state.voices.map(voice => (
    `<div class="voice-card" data-voice-id="${escapeHtml(voice.id)}" data-filter="${escapeHtml(`${voice.name} ${(voice.references || [voice]).map(clip => clip.transcript).join(' ')}`.toLocaleLowerCase())}"><div class="voice-card-head"><div class="voice-card-copy"><strong title="${escapeHtml(voice.name)}">${escapeHtml(voice.name)}</strong>${(voice.references?.length || 1) < 2 ? `<small>${translated('{count} reference clips', { count: 1 })}</small>` : ''}</div><button type="button" class="icon-btn voice-tile-preview" data-preview="${escapeHtml(voice.id)}" data-i18n-title="Preview voice" title="${escapeHtml(t('Preview voice'))}" data-i18n-aria-label="Preview voice" aria-label="${escapeHtml(t('Preview voice'))}" aria-pressed="false">${icon('play')}</button></div><p class="voice-transcript" title="${escapeHtml(voice.transcript)}">${escapeHtml(voice.transcript)}</p>${moreClips(voice)}<div class="voice-card-actions"><button type="button" class="text-btn use-voice" data-id="${escapeHtml(voice.id)}">${translated('Use this voice')}${icon('arrow')}</button><a class="voice-download" download href="${escapeHtml(voice.audio_url)}" data-i18n-title="Download" title="${escapeHtml(t('Download'))}" data-i18n-aria-label="Download" aria-label="${escapeHtml(t('Download'))}">${icon('download')}</a></div></div>`
  )).join('') : emptyState('mic', 'No voices saved');
  $('voice-count').textContent = state.voices.length;
  updateVoiceSelection();
  $('voice-list').querySelectorAll('.use-voice').forEach(button => {
    button.onclick = () => {
      if (window.StudioFlow) { StudioFlow.useLibraryVoice(button.dataset.id); return; }
      clearCloneAudio();
      state.selectedVoiceIds.clear();
      state.selectedVoiceIds.add(button.dataset.id);
      updateVoiceSelection(); scheduleReferencePrewarm(); view('generate'); $('saved-voice-options').open = true;
    };
  });
  $('voice-list').querySelectorAll('[data-preview]').forEach(button => {
    button.onclick = () => previewVoice(state.voices.find(voice => voice.id === button.dataset.preview), button);
  });
  $('voice-list').querySelectorAll('[data-preview-clip]').forEach(button => {
    button.onclick = () => previewVoice({ audio_url: button.dataset.previewClip }, button);
  });
  $('voice-list').querySelectorAll('.voice-reference-clips').forEach(details => {
    details.addEventListener('toggle', () => {
      if (!details.open && details.contains(auditionButton)) stopVoicePreview();
    });
  });
  filterVoices();
}
function filterVoices() {
  const query = $('voice-search').value.trim().toLocaleLowerCase();
  for (const card of $('voice-list').querySelectorAll('.voice-card')) {
    card.hidden = !card.dataset.filter.includes(query);
    if (card.hidden && card.contains(auditionButton)) stopVoicePreview();
  }
}
$('voice-search').addEventListener('input', filterVoices);
async function loadVoices() {
  try {
    state.voices = (await api('/api/voices')).voices;
    const valid = new Set(state.voices.map(voice => voice.id));
    state.selectedVoiceIds = new Set(selectedVoices().filter(id => valid.has(id)));
    renderVoices();
  } catch (error) { $('voice-hint').textContent = message(error.message); }
}

function payloadFromForm() {
  const mode = $('mode').value;
  const payload = {
    mode, text: $('text').value, voice_ids: mode === 'infill' ? [] : selectedVoices(),
    direction: $('direction').disabled ? '' : $('direction').value,
    language: $('language').disabled ? null : $('language').value || null,
    previous_take: $('previous-take').disabled ? null : $('previous-take').value || null,
    seed: Number($('seed').value || 123), max_new_tokens: Number($('max-frames').value || 768),
  };
  if (!$('duration').disabled && $('duration').value) payload.duration_frames = Number($('duration').value);
  if (mode === 'stream') {
    payload.chunks = $('chunks').value.split(/\r?\n/).map(value => value.trim()).filter(Boolean);
    payload.live_stream = $('stream-live').checked; payload.text = payload.chunks.join('\n');
    delete payload.direction;
    delete payload.language;
    delete payload.duration_frames;
    delete payload.previous_take;
  }
  if (mode === 'infill') {
    payload.source_take = $('source-take').value;
    payload.start_frame = Number($('start-frame').value);
    payload.end_frame = Number($('end-frame').value);
  }
  return payload;
}
function addEvent(event) {
  if (event.stage === 'error') {
    $('result-error').textContent = message(event.message);
    $('result-error').hidden = false;
  } else $('progress-stage').textContent = stageLabel(event.stage);
}

function resultWorkspaceFor(job) {
  if ((job.studio || job.recipe?.studio)?.result_scope === 'audio_editor') return 'editor';
  const layout = job.studio?.layout || job.recipe?.studio?.layout;
  return layout ? layout === 'scene' ? 'scene' : 'generate'
    : job.mode === 'scene' || job.recipe?.mode === 'scene' ? 'scene' : 'generate';
}
async function restoreWorkspaceResults(singleId, sceneId) {
  const jobs = [...state.jobs];
  for (const id of new Set([singleId, sceneId].filter(Boolean))) {
    if (jobs.some(job => job.id === id)) continue;
    try { jobs.push(await api(`/api/jobs/${encodeURIComponent(id)}`)); } catch (_) { /* keep recent results available */ }
  }
  for (const [workspace, id] of [['generate', singleId], ['scene', sceneId]]) {
    state.resultJobs[workspace] = id === null ? null
      : jobs.find(job => job?.id === id && resultWorkspaceFor(job) === workspace)
        || jobs.find(job => job && resultWorkspaceFor(job) === workspace) || null;
  }
}
function showWorkspaceResult(name) {
  const workspace = name === 'scene' ? 'scene' : 'generate';
  const job = state.resultJobs[workspace];
  clearTimeout(state.pollTimer);
  state.resultWorkspace = workspace;
  state.activeJob = job; state.monitoredJobId = job?.id || null;
  if (job) {
    renderJob(job);
    if (['queued', 'running'].includes(job.status)) state.pollTimer = setTimeout(() => poll(job.id), 0);
    return;
  }
  const box = $('result-box');
  box.querySelectorAll('audio').forEach(audio => audio.pause());
  box.replaceChildren(); delete box.dataset.jobId; box.classList.add('hidden');
  StudioAudio.enhance(box);
  $('job-title').textContent = t('Ready'); $('job-title').removeAttribute('title');
  $('job-badge').textContent = statusLabel('idle'); $('job-badge').dataset.status = 'idle';
  $('job-badge').closest('.monitor-panel').dataset.state = 'idle';
  $('cancel-job').classList.add('hidden');
  $('progress-bar').style.width = '0%'; $('progress-bar').classList.remove('indeterminate');
  $('progress-stage').textContent = $('progress-count').textContent = '';
  $('result-error').hidden = true; $('result-error').textContent = '';
  $('live-stream-controls').classList.add('hidden');
  $('request-recovery').hidden = true;
  $('view-latest-take').parentElement.classList.add('hidden');
  $('output-placeholder').hidden = false;
  $('output-placeholder').innerHTML = emptyState('headphones', 'No audio yet', '', 'output-empty');
}
function renderResults(job, box = $('result-box')) {
  if (box === $('result-box') && resultWorkspaceFor(job) !== state.resultWorkspace) return;
  const segments = job.result?.segments?.length ? job.result.segments : (job.events || []).filter(event => (
    ['audio', 'segment'].includes(event.stage) && (event.path || event.output)
  ));
  box.classList.toggle('hidden', !job.output && !segments.length);
  if (box.dataset.jobId !== job.id) {
    box.querySelectorAll('audio').forEach(audio => audio.pause());
    box.dataset.jobId = job.id;
    box.innerHTML = `<div class="main-result hidden"><audio controls preload="metadata"></audio><div class="result-actions"><a class="secondary-btn download-link" download>${icon('download')}${translated('Download WAV')}</a></div></div><div class="result-note hidden"></div><div class="segment-list"></div>`;
  }
  const main = box.querySelector('.main-result');
  main.classList.toggle('hidden', !job.output);
  if (job.output) {
    const audio = main.querySelector('audio');
    const url = fileUrl(job.output);
    if (audio.getAttribute('src') !== url) audio.setAttribute('src', url);
    main.querySelector('a').href = url;
    audio.dataset.selection = String(Boolean(job.output));
    audio.dataset.frameSeconds = 2048 / 44100;
    window.StudioFlow?.enhanceResult(job, main);
  }
  const note = box.querySelector('.result-note');
  note.classList.toggle('hidden', !job.result?.deferred_audio_for_host_budget);
  note.textContent = t('Memory budget delayed audio delivery until EOS; the complete WAV and take are preserved.');
  const list = box.querySelector('.segment-list');
  list.classList.toggle('hidden', !segments.length);
  const existing = new Map([...list.children].map(node => [node.dataset.path, node]));
  // Keep each audio element across polls so playback and seek position survive.
  segments.forEach((segment, index) => {
    const path = segment.path || segment.output;
    if (!path) return;
    let row = existing.get(path);
    if (!row) {
      row = document.createElement('div');
      row.className = 'segment-item';
      row.dataset.path = path;
      row.innerHTML = `<div class="segment-head"><span></span><a download class="voice-download" data-i18n-title="Download" title="${escapeHtml(t('Download'))}" data-i18n-aria-label="Download" aria-label="${escapeHtml(t('Download'))}">${icon('download')}</a></div><audio controls preload="none"></audio>`;
      row.querySelector('audio').src = fileUrl(path);
      row.querySelector('a').href = fileUrl(path);
      list.appendChild(row);
    }
    row.querySelector('span').textContent = segment.character
      ? `${segment.character} · ${segment.text || ''}` : t('Part {index} · {text}', { index: index + 1, text: segment.text || '' });
    window.StudioFlow?.enhanceSegment(job, segment, row, index);
  });
  window.StudioFlow?.afterResults(job, box, segments);
  StudioAudio.enhance(box);
}
function renderJob(job) {
  if (!job || resultWorkspaceFor(job) === 'editor') return;
  state.resultJobs[resultWorkspaceFor(job)] = job;
  if (resultWorkspaceFor(job) !== state.resultWorkspace) return;
  $('view-latest-take').parentElement.classList.add('hidden');
  $('job-title').textContent = t(modeLabel(job.mode || 'tts'));
  $('job-title').title = job.id;
  $('job-badge').textContent = statusLabel(job.status);
  $('job-badge').dataset.status = job.status;
  $('job-badge').closest('.monitor-panel').dataset.state = job.status;
  const active = ['queued', 'running'].includes(job.status);
  $('cancel-job').classList.toggle('hidden', !active);
  $('cancel-job').disabled = Boolean(job.cancel_requested);
  delete $('cancel-job').dataset.i18n;
  $('cancel-job').textContent = t(job.cancel_requested ? 'Cancelling…' : 'Cancel');
  const last = (job.events || []).at(-1) || {};
  $('progress-bar').style.width = job.status === 'complete' ? '100%' : active ? '32%' : '0%';
  $('progress-bar').classList.toggle('indeterminate', active);
  const timing = jobTiming(job);
  const phaseLabel = { model_load: 'Loading model', initialization: 'Initializing',
    waiting: 'Waiting for text', generation: 'Generating' }[timing.phase];
  $('progress-stage').textContent = active
    ? job.status === 'queued' ? t('Queued') : phaseLabel ? t(phaseLabel) : stageLabel(last.stage) || t('Generating')
    : '';
  $('progress-count').textContent = !active || timing.phase === 'generation' ? generationTimeLabel(job) : '';
  $('result-error').hidden = !job.error;
  $('result-error').textContent = job.error ? message(job.error) : '';
  renderResults(job);
  const hasAudio = ! $('result-box').classList.contains('hidden');
  $('output-placeholder').hidden = hasAudio;
  if (!hasAudio) {
    $('output-placeholder').innerHTML = emptyState('headphones',
      active ? 'Generating' : job.status === 'error' ? 'Error' : statusSource(job.status),
      active ? 'Audio will appear here when ready.' : job.error || 'No audio was produced.', 'output-empty');
  }
  const live = active && job.mode === 'stream' && job.stream_input?.live;
  $('live-stream-controls').classList.toggle('hidden', !live);
  const inputOpen = live && !job.stream_input.finished && !job.cancel_requested;
  $('push-stream').disabled = $('finish-stream').disabled = !inputOpen || state.streamSubmitting;
  $('live-stream-status').textContent = job.stream_input?.finished
    ? t('Input closed. Finishing queued speech…')
    : t('Input open · {count} queued', { count: job.stream_input?.pending || 0 });
  renderModelPicker();
  document.dispatchEvent(new CustomEvent('drama:jobupdate', { detail: job }));
}
async function poll(id) {
  try {
    const job = await api(`/api/jobs/${id}`);
    const workspace = resultWorkspaceFor(job);
    if (workspace === 'editor') return;
    if (state.resultJobs[workspace]?.id === id) state.resultJobs[workspace] = job;
    if (state.monitoredJobId !== id) return;
    if (resultWorkspaceFor(job) !== state.resultWorkspace) {
      state.resultJobs[resultWorkspaceFor(job)] = job;
      showWorkspaceResult(state.resultWorkspace);
      return;
    }
    state.activeJob = job;
    const position = state.jobs.findIndex(item => item.id === job.id);
    if (position >= 0) state.jobs[position] = job;
    renderJob(job);
    if (['queued', 'running'].includes(job.status)) {
      state.pollTimer = setTimeout(() => poll(id), 900);
    } else {
      loadJobs();
      if (window.StudioFlow) StudioFlow.syncControls(); else $('submit-job').disabled = false;
    }
  } catch (error) {
    if (state.monitoredJobId !== id) return;
    addEvent({ stage: 'error', message: error.message });
    $('submit-job').disabled = false;
  }
}
function monitorJob(id) {
  clearTimeout(state.pollTimer);
  state.monitoredJobId = id;
  return poll(id);
}
$('submit-job').onclick = async () => {
  try {
    state.jobSubmitting = true;
    $('submit-job').disabled = true;
    renderModelPicker(); window.StudioFlow?.syncControls();
    audition?.pause();
    $('result-error').hidden = true;
    let payload = payloadFromForm();
    if (window.StudioFlow) payload = await StudioFlow.prepareRequest(payload);
    else if (payload.mode !== 'infill') {
      payload.uploaded_references = await prepareCloneReference();
    }
    const job = await api('/api/jobs', { method: 'POST', body: JSON.stringify(payload) });
    window.StudioFlow?.submitted(job.id, payload);
    state.jobs.unshift({ id: job.id, status: 'queued' });
    renderModelPicker();
    monitorJob(job.id);
  } catch (error) {
    addEvent({ stage: 'error', message: error.message });
    $('submit-job').disabled = false;
  } finally {
    state.jobSubmitting = false;
    renderModelPicker(); window.StudioFlow?.syncControls();
  }
};
$('cancel-job').onclick = async () => {
  if (!state.activeJob?.id) return;
  $('cancel-job').disabled = true;
  try {
    await api(`/api/jobs/${state.activeJob.id}/cancel`, { method: 'POST' });
    monitorJob(state.activeJob.id);
  } catch (error) {
    addEvent({ stage: 'error', message: error.message });
    $('cancel-job').disabled = false;
  }
};
$('push-stream').onclick = async () => {
  if (!state.activeJob?.id || state.streamSubmitting) return;
  const id = state.activeJob.id, text = $('live-stream-text').value;
  state.streamSubmitting = true; $('push-stream').disabled = $('finish-stream').disabled = true;
  try {
    await api(`/api/jobs/${id}/input`, {
      method: 'POST', body: JSON.stringify({ text }),
    });
    if ($('live-stream-text').value === text) $('live-stream-text').value = '';
  } catch (error) { addEvent({ stage: 'error', message: error.message }); }
  finally {
    state.streamSubmitting = false;
    if (state.monitoredJobId === id) monitorJob(id);
  }
};
$('finish-stream').onclick = async () => {
  if (!state.activeJob?.id || state.streamSubmitting) return;
  const id = state.activeJob.id;
  state.streamSubmitting = true; $('push-stream').disabled = $('finish-stream').disabled = true;
  try {
    await api(`/api/jobs/${id}/finish`, { method: 'POST' });
  } catch (error) { addEvent({ stage: 'error', message: error.message }); }
  finally {
    state.streamSubmitting = false;
    if (state.monitoredJobId === id) monitorJob(id);
  }
};
function updateMode() {
  const mode = $('mode').value;
  const design = Boolean(window.StudioFlow?.voiceDesignActive());
  if (state.previousMode !== mode && ['revision', 'edit'].includes(mode)) $('advanced-settings').open = true;
  state.previousMode = mode;
  $('text').closest('.editor-area').classList.toggle('hidden', mode === 'stream');
  $('stream-fields').classList.toggle('hidden', mode !== 'stream');
  $('infill-fields').classList.toggle('hidden', mode !== 'infill');
  $('previous-source-preview').classList.toggle('hidden',
    !['revision', 'edit'].includes(mode) || !state.previousTake);
  $('clone-section').classList.toggle('hidden', mode === 'infill' || Boolean(window.StudioFlow?.lockVoice()));
  document.querySelector('label[for="text"]').classList.toggle('hidden', mode === 'stream');
  $('text').disabled = mode === 'stream';
  $('text').readOnly = Boolean(window.StudioFlow?.lockText());
  $('clear-text').disabled = $('text').disabled || $('text').readOnly;
  $('direction').disabled = ['stream', 'infill'].includes(mode) || state.capabilities?.direction === false;
  $('language').disabled = design || ['stream', 'infill'].includes(mode) || state.capabilities?.language_hint === false;
  $('duration').disabled = design || mode === 'stream' || state.capabilities?.duration_hint === false;
  $('previous-take').disabled = design || ['stream', 'infill'].includes(mode) || state.capabilities?.director_revision === false;
  $('inspect-previous-source').disabled = $('previous-take').disabled;
  if (['revision', 'edit'].includes(mode) && $('previous-take').value
      && state.previousTake?.path !== $('previous-take').value) inspectPreviousSource();
  const textPlaceholder = mode === 'infill'
    ? 'The complete revised transcript for the infilled take…' : '';
  if (textPlaceholder) {
    $('text').dataset.i18nPlaceholder = textPlaceholder;
    $('text').placeholder = t(textPlaceholder);
  } else {
    delete $('text').dataset.i18nPlaceholder;
    $('text').removeAttribute('placeholder');
  }
  window.StudioFlow?.syncControls();
  window.ScriptEditor?.sync($('text'));
}
$('mode').onchange = updateMode;
function updateEditorMeta() {
  window.ScriptEditor?.sync($('text'));
  $('text-count').textContent = t('{count} chars', { count: Array.from($('text').value).length });
}
$('text').addEventListener('input', updateEditorMeta);
$('clear-text').onclick = () => {
  if ($('text').disabled || $('text').readOnly) return;
  $('text').value = ''; updateEditorMeta();
  const editor = window.ScriptEditor?.get($('text'));
  if (editor) editor.view.focus(); else $('text').focus();
};
document.addEventListener('keydown', event => {
  const submit = state.view === 'generate' ? $('submit-job') : state.view === 'scene' ? $('submit-scene') : null;
  if ((event.ctrlKey || event.metaKey) && event.key === 'Enter' && !event.isComposing
      && submit && !submit.disabled
      && !document.querySelector('dialog[open]')
      && !document.querySelector('.select-popover:not([hidden])')) {
    event.preventDefault(); submit.click();
  }
});
$('refresh-btn').onclick = () => { loadHealth(); loadVoices(); loadJobs(); loadModel(); };
$('refresh-jobs').onclick = loadJobs;
const historyTranscripts = new Map();
function renderHistoryTitle(job, heading) {
  const lines = job.recipe?.lines || job.result?.segments || job.result?.chunks || [];
  const text = [job.recipe?.text, job.result?.text, job.text, job.payload_summary?.text,
    job.stream_input?.chunks?.join('\n'), lines.map(line => line.text || '').join('\n')]
    .find(value => typeof value === 'string' && value.trim());
  const show = source => {
    const unmarked = (source || '').replace(/<\|speaker:\d+\|>/g, ' ')
      .replace(/<\/?(whisper|emphasis|soft|fast|slow|stress|forceful)>/g, '').replace(/\s+/g, ' ').trim();
    const plain = unmarked.replace(/\[[^\]\n]{1,160}\]/g, '').replace(/\s+/g, ' ').trim() || unmarked;
    const letters = Array.from(plain);
    heading.textContent = plain ? letters.slice(0, 64).join('') + (letters.length > 64 ? '…' : '') : t('Untitled audio');
    heading.title = plain;
    if (plain) delete heading.dataset.i18n; else heading.dataset.i18n = 'Untitled audio';
  };
  if (text) { show(text); return; }
  const path = job.take_file || job.result?.native_take_file || job.result?.take_file;
  let cached = historyTranscripts.get(path);
  show(cached?.text);
  if (!path) return;
  if (!cached) {
    cached = { text: '' };
    cached.pending = api('/api/takes/inspect', { method: 'POST', body: JSON.stringify({ path }) })
      .then(take => { cached.text = take.text || ''; }).catch(() => {});
    historyTranscripts.set(path, cached);
  }
  if (!cached.text) cached.pending.then(() => { if (heading.isConnected) show(cached.text); });
}
async function loadJobs() {
  try {
    const wasGenerating = state.jobs.some(job => ['queued', 'running'].includes(job.status));
    state.jobs = (await api('/api/jobs')).jobs;
    if (wasGenerating && !state.jobs.some(job => ['queued', 'running'].includes(job.status))) loadHealth();
    for (const workspace of ['generate', 'scene']) {
      const current = state.resultJobs[workspace];
      const updated = state.jobs.find(job => job.id === current?.id && resultWorkspaceFor(job) === workspace);
      if (updated) state.resultJobs[workspace] = updated;
    }
    if (state.model) state.model.busy = state.jobs.some(job => ['queued', 'running'].includes(job.status));
    renderModelPicker();
    const latest = state.jobs.find(job => job.output && resultWorkspaceFor(job) === state.resultWorkspace);
    const offerLatest = latest && !state.monitoredJobId;
    $('view-latest-take').classList.toggle('hidden', !offerLatest);
    $('view-latest-take').parentElement.classList.toggle('hidden', !offerLatest);
    $('view-latest-take').onclick = () => latest && monitorJob(latest.id);
    const list = $('job-list');
    const existing = new Map([...list.querySelectorAll('.job-card')].map(row => [row.dataset.jobId, row]));
    if (state.jobs.length && list.querySelector('.empty-state')) list.replaceChildren();
    let position = list.firstChild;
    const jobs = [...new Map([...(window.StudioFlow?.historyJobs() || []), ...state.jobs].map(job => [job.id, job])).values()]
      .filter(job => job.mode !== 'audio_import')
      .sort((a, b) => b.created_at - a.created_at);
    for (const job of jobs) {
      let row = existing.get(job.id);
      if (!row) {
        row = document.createElement('div'); row.className = 'job-card'; row.dataset.jobId = job.id;
        row.innerHTML = '<div class="job-copy"><strong class="job-title"></strong><div class="job-details"><small class="job-date"></small><small class="job-model"></small></div></div><div class="job-meta"><span class="job-status"></span><small class="job-size"></small></div><div class="job-actions"></div>';
      }
      renderHistoryTitle(job, row.querySelector('.job-title'));
      row.querySelector('.job-date').dataset.date = job.created_at;
      row.querySelector('.job-date').textContent = new Date(job.created_at * 1000).toLocaleString(I18n.locale);
      row.querySelector('.job-model').textContent = job.recipe?.model?.replaceAll('\\', '/').split('/').at(-1) || '';
      row.querySelector('.job-model').title = row.querySelector('.job-model').textContent;
      const status = row.querySelector('.job-status'); status.dataset.status = job.status;
      status.textContent = statusLabel(job.status);
      row.querySelector('.job-size').textContent = [
        t('{count} chars', { count: job.payload_summary?.text_length || 0 }),
        audioDurationLabel(job), generationTimeLabel(job),
      ].filter(Boolean).join(' · ');
      const actions = row.querySelector('.job-actions'), shape = JSON.stringify([job.output, job.take_file]);
      if (actions.dataset.shape !== shape) {
        actions.dataset.shape = shape;
        actions.innerHTML = `${job.output ? `<button type="button" class="text-btn history-play" aria-pressed="false">${icon('play')}${translated('Play')}</button><a class="icon-btn download-link history-download" download href="${fileUrl(job.output)}" data-i18n-title="Download" title="${escapeHtml(t('Download'))}" data-i18n-aria-label="Download" aria-label="${escapeHtml(t('Download'))}">${icon('download')}</a>` : ''}${job.output || job.take_file ? `<details class="command-menu history-actions"><summary class="text-btn">${translated('More')}</summary><div class="command-menu-list">${job.take_file ? `<button class="menu-command use-take" data-path="${escapeHtml(job.take_file)}" data-mode="revision">${translated('Edit')}</button>` : ''}<button class="menu-command restore-job" data-id="${escapeHtml(job.id)}">${translated('Restore as draft')}</button></div></details>` : ''}`;
      }
      const play = actions.querySelector('.history-play');
      if (play) play.onclick = () => previewVoice({ audio_url: fileUrl(job.output) }, play);
      if (row !== position) list.insertBefore(row, position);
      position = row.nextSibling; existing.delete(job.id);
    }
    for (const row of existing.values()) row.remove();
    if (auditionButton && !auditionButton.isConnected) stopVoicePreview();
    if (!jobs.length) list.innerHTML = emptyState('history', 'No takes yet');
    document.querySelectorAll('.use-take').forEach(button => {
      button.hidden = !state.capabilities?.director_revision && !state.capabilities?.infill;
      button.onclick = () => useTake(button.dataset.path, button.dataset.mode);
    });
    document.querySelectorAll('.restore-job').forEach(button => {
      button.onclick = () => window.StudioFlow?.restoreVersion({ job_id: button.dataset.id });
    });
    StudioAudio.enhance($('job-list'));
    if (document.documentElement.dataset.startup !== 'ready') {
      await Promise.all([...historyTranscripts.values()].map(value => value.pending));
    }
    window.StudioFlow?.renderWorks();
    window.StudioFlow?.syncControls();
  } catch (error) { $('job-list').textContent = message(error.message); }
}
function updateVoiceUpload() {
  const file = $('voice-audio').files[0];
  if (state.voicePreviewUrl) URL.revokeObjectURL(state.voicePreviewUrl);
  state.voicePreviewUrl = null;
  $('voice-upload-preview').classList.toggle('hidden', !file);
  $('voice-file-name').textContent = file?.name || '';
  if (file) {
    state.voicePreviewUrl = URL.createObjectURL(file);
    $('voice-file-audio').src = state.voicePreviewUrl;
  } else {
    $('voice-file-audio').removeAttribute('src');
  }
  StudioAudio.enhance($('voice-upload-preview'));
}
$('voice-audio').addEventListener('change', updateVoiceUpload);
const uploadZone = $('voice-upload-zone');
for (const type of ['dragover', 'dragenter']) uploadZone.addEventListener(type, event => {
  event.preventDefault();
  uploadZone.classList.add('dragging');
});
for (const type of ['drop', 'dragleave']) uploadZone.addEventListener(type, event => {
  event.preventDefault();
  uploadZone.classList.remove('dragging');
});
uploadZone.addEventListener('drop', event => {
  const file = event.dataTransfer.files[0];
  if (!file) return;
  if (!file.type.startsWith('audio/') && !/\.(wav|mp3|flac|ogg|m4a)$/i.test(file.name)) {
    $('voice-form-note').dataset.i18n = 'Choose an audio file.';
    $('voice-form-note').textContent = t('Choose an audio file.');
    return;
  }
  const transfer = new DataTransfer();
  transfer.items.add(file);
  $('voice-audio').files = transfer.files;
  updateVoiceUpload();
});
$('voice-form').onsubmit = async event => {
  event.preventDefault();
  const note = $('voice-form-note'), file = $('voice-audio').files[0];
  if (!file) return;
  $('save-voice').disabled = true;
  note.dataset.i18n = 'Uploading…';
  note.textContent = t('Uploading…');
  const data = new FormData();
  data.append('name', $('voice-name').value);
  data.append('transcript', $('voice-transcript').value);
  data.append('speaker', $('voice-speaker').value);
  data.append('audio', file);
  try {
    const response = await fetch('/api/voices', { method: 'POST', body: data });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error);
    note.dataset.i18n = 'Saved';
    note.textContent = t('Saved');
    event.target.reset();
    updateVoiceUpload();
    loadVoices();
  } catch (error) { delete note.dataset.i18n; note.textContent = message(error.message); }
  finally { $('save-voice').disabled = false; }
};
function modeLabel(mode) {
  return { tts: 'Text to speech', revision: 'Director revision', edit: 'Text to speech',
    infill: 'Local infill', stream: 'Text stream', audio_stream: 'Frame audio stream',
    scene: 'Dialogue', design: 'Voice design' }[mode] || mode;
}
function statusSource(status) {
  return { queued: 'Queued', running: 'Running', complete: 'Complete', incomplete: 'Incomplete',
    cancelled: 'Cancelled', cancelling: 'Cancelling…', interrupted: 'Interrupted',
    failed: 'Failed', error: 'Error' }[status] || 'Unknown';
}
document.addEventListener('drama:languagechange', () => {
  view(state.view);
  if (state.hardware) setEngine(state.hardware);
  if (state.model) renderModel(state.model);
  renderInfillInfo();
  renderTranscriptionStatus();
  updateEditorMeta();
  updateVoiceSelection();
  updateMode();
  if (state.activeJob) renderJob(state.activeJob);
  for (const node of document.querySelectorAll('[data-date]')) {
    node.textContent = new Date(Number(node.dataset.date) * 1000).toLocaleString(I18n.locale);
  }
  I18n.apply();
  StudioControls.syncAll();
  StudioAudio.translate();
  window.ScriptEditor?.translate();
});
restoreView();
updateMode();
updateEditorMeta();
updateVoiceSelection();
enhanceAllSelects();
StudioAudio.enhance();
