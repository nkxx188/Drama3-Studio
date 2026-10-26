/* Waveforms and non-destructive selections follow the actual audio. */
(() => {
  'use strict';
  const players = new WeakMap(), all = new Set(), waveCache = new Map(), wavePending = new Map();
  const waveQueue = [];
  let waveTasks = 0;
  const t = (source, args = {}) => I18n.t(source, args);
  const svg = name => `<svg class="icon" aria-hidden="true"><use href="#i-${name}"></use></svg>`;
  const clock = (value, precise = false) => {
    if (!Number.isFinite(value) || value < 0) return '0:00';
    if (precise) {
      const centiseconds = Math.round(value * 100);
      return `${Math.floor(centiseconds / 6000)}:${String(Math.floor(centiseconds / 100) % 60).padStart(2, '0')}.${String(centiseconds % 100).padStart(2, '0')}`;
    }
    const ms = Math.round(value * 1000);
    return `${Math.floor(ms / 60000)}:${String(Math.floor(ms / 1000) % 60).padStart(2, '0')}`;
  };
  const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

  function pumpWaves() {
    while (waveTasks < 2 && waveQueue.length) {
      waveTasks++;
      waveQueue.shift()().finally(() => { waveTasks--; pumpWaves(); });
    }
  }
  function loadPeaks(source) {
    if (waveCache.has(source)) return Promise.resolve(waveCache.get(source));
    if (wavePending.has(source)) return wavePending.get(source);
    const pending = new Promise((resolve, reject) => {
      waveQueue.push(async () => {
        try {
          const response = await fetch(source);
          if (!response.ok || Number(response.headers.get('Content-Length')) > 128 * 1024 ** 2)
            throw new Error('Waveform audio is unavailable');
          const bytes = await response.arrayBuffer();
          if (bytes.byteLength > 128 * 1024 ** 2) throw new Error('Waveform audio is too large');
          const context = new OfflineAudioContext(1, 1, 8000);
          const decoded = await context.decodeAudioData(bytes);
          const peaks = new Float32Array(2048);
          for (let channel = 0; channel < decoded.numberOfChannels; channel++) {
            const samples = decoded.getChannelData(channel);
            for (let index = 0; index < peaks.length; index++) {
              const start = Math.floor(index * samples.length / peaks.length);
              const end = Math.floor((index + 1) * samples.length / peaks.length);
              for (let at = start; at < end; at++) peaks[index] = Math.max(peaks[index], Math.abs(samples[at]));
            }
          }
          if (waveCache.size >= 32) waveCache.delete(waveCache.keys().next().value);
          waveCache.set(source, peaks); resolve(peaks);
        } catch (error) { reject(error); }
      });
      pumpWaves();
    });
    wavePending.set(source, pending);
    pending.then(() => wavePending.delete(source), () => wavePending.delete(source));
    return pending;
  }
  function enhance(audio) {
    if (players.has(audio)) return players.get(audio);
    const box = document.createElement('div'); box.className = 'studio-player';
    audio.before(box); box.append(audio); audio.controls = false;
    const button = (className, parent = box) => {
      const node = document.createElement('button'); node.type = 'button';
      node.className = className; parent.append(node); return node;
    };
    const transport = document.createElement('div'); transport.className = 'player-transport';
    const track = document.createElement('div'); track.className = 'player-track';
    const surface = document.createElement('div'); surface.className = 'player-wave-surface';
    const wave = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    wave.setAttribute('viewBox', '0 0 400 60'); wave.setAttribute('preserveAspectRatio', 'none');
    wave.setAttribute('aria-hidden', 'true'); wave.classList.add('player-wave');
    const region = document.createElement('div'); region.className = 'player-region'; region.hidden = true;
    const handles = ['start', 'end'].map(edge => {
      const handle = button(`player-region-handle region-${edge}`, region);
      handle.dataset.edge = edge; handle.setAttribute('role', 'slider');
      return handle;
    });
    const playhead = document.createElement('div'); playhead.className = 'player-playhead';
    surface.append(wave, region, playhead);
    const seek = document.createElement('input');
    Object.assign(seek, { type: 'range', min: '0', max: '0', step: '0.01', value: '0',
      className: 'player-seek', disabled: true });
    const timeline = document.createElement('div'); timeline.className = 'player-wave-timeline'; timeline.hidden = true;
    timeline.setAttribute('aria-hidden', 'true');
    const ticks = Array.from({ length: 3 }, () => document.createElement('span'));
    timeline.append(...ticks); track.append(surface, timeline, seek);
    const time = document.createElement('div'); time.className = 'player-time';
    const current = document.createElement('span'), duration = document.createElement('span');
    time.append(current, duration);
    const pan = document.createElement('input');
    Object.assign(pan, { type: 'range', min: '0', step: '0.001', value: '0',
      className: 'player-pan', hidden: true });
    transport.append(track, time, pan); box.append(transport);
    const controls = document.createElement('div'); controls.className = 'player-controls'; box.append(controls);
    const play = button('player-play', controls), rate = button('player-rate', controls);
    const volumeGroup = document.createElement('div'); volumeGroup.className = 'player-volume-group';
    const mute = button('player-mute', volumeGroup);
    const volume = document.createElement('input');
    Object.assign(volume, { type: 'range', min: '0', max: '1', step: '0.01', value: audio.volume, className: 'player-volume' });
    volumeGroup.append(volume); controls.append(volumeGroup);
    const regionInfo = document.createElement('span'); regionInfo.className = 'player-region-info';
    controls.append(regionInfo);
    const preview = button('player-preview-selection', controls), clear = button('player-clear-selection', controls);
    const zoomOut = button('player-zoom-out', controls), zoomIn = button('player-zoom-in', controls);
    // Replacing icon nodes during playback can swallow a mouse click before pointerup.
    play.innerHTML = svg('play'); mute.innerHTML = svg('volume');
    preview.innerHTML = svg('play'); clear.innerHTML = svg('close');
    const previewLabel = document.createElement('span'); preview.append(previewLabel);
    zoomOut.textContent = '−'; zoomIn.textContent = '+';
    const hint = document.createElement('span'); hint.className = 'player-wave-hint'; controls.append(hint);
    const error = document.createElement('p'); error.className = 'player-error'; error.hidden = true; box.append(error);
    const ui = { audio, box, track, surface, seek, wave, region, handles, pan, play, rate,
      current, duration, error, selection: null, zoom: 1, offset: 0, waveSource: null,
      source: audio.getAttribute('src'), stopAt: null, raf: null, abort: null, request: 0, peaks: null,
      viewReady: false, followPlayback: false };
    let drag = null, edgeRaf = null;
    players.set(audio, ui); all.add(ui);
    const valid = () => Number.isFinite(audio.duration) && audio.duration > 0;
    const selectable = () => audio.dataset.selection === 'true';
    const width = () => audio.duration / ui.zoom;
    const maxZoom = () => Math.max(8, audio.duration);
    const fraction = value => (value - ui.offset) / width();
    const setText = (node, value) => { if (node.textContent !== value) node.textContent = value; };
    const setIcon = (node, name) => {
      const use = node.querySelector('use'), href = `#i-${name}`;
      if (use.getAttribute('href') !== href) use.setAttribute('href', href);
    };
    const notify = () => audio.dispatchEvent(new CustomEvent('drama:selectionchange',
      { bubbles: true, detail: ui.selection ? { ...ui.selection } : null }));
    function setOffset(value) {
      const next = clamp(value, 0, Math.max(0, audio.duration - width()));
      if (Math.abs(next - ui.offset) < 0.000001) return;
      ui.offset = next; drawWave();
    }
    function revealPlayhead() {
      if (!valid() || ui.zoom === 1) return;
      if (fraction(audio.currentTime) < 0 || fraction(audio.currentTime) >= 1)
        setOffset(Math.floor(audio.currentTime / width()) * width());
    }
    function renderRegion() {
      const has = valid() && selectable() && ui.selection !== null;
      region.hidden = !has; box.classList.toggle('has-selection', has);
      preview.hidden = !has;
      clear.hidden = !has || audio.dataset.selectionRequired === 'true';
      if (has) {
        const first = fraction(ui.selection.start), last = fraction(ui.selection.end);
        region.style.left = `${clamp(first, 0, 1) * 100}%`;
        region.style.width = `${Math.max(0, clamp(last, 0, 1) - clamp(first, 0, 1)) * 100}%`;
        handles[0].hidden = first < 0 || first > 1; handles[1].hidden = last < 0 || last > 1;
        handles.forEach((handle, index) => {
          const edge = index ? 'end' : 'start';
          handle.setAttribute('aria-label', t(index ? 'Selection end' : 'Selection start'));
          handle.setAttribute('aria-valuemin', '0'); handle.setAttribute('aria-valuemax', String(Number(audio.duration.toFixed(2))));
          handle.setAttribute('aria-valuenow', String(Number(ui.selection[edge].toFixed(2))));
          handle.setAttribute('aria-valuetext', clock(ui.selection[edge], true));
        });
        regionInfo.textContent = `${clock(ui.selection.start, true)} – ${clock(ui.selection.end, true)}`;
        preview.disabled = ui.selection.end <= ui.selection.start;
      } else regionInfo.textContent = '';
    }
    function setSelection(start, end, { emit = false } = {}) {
      if (!valid() || !selectable()) return;
      const limit = audio.duration;
      let a = clamp(Number(start), 0, limit), b = clamp(Number(end), 0, limit);
      if (!Number.isFinite(a) || !Number.isFinite(b)) return;
      [a, b] = [Math.min(a, b), Math.max(a, b)];
      const step = Number(audio.dataset.frameSeconds || 0);
      if (step > 0) {
        const snap = value => Math.abs(value - Math.round(value)) < 0.00002 ? Math.round(value) : value;
        const equal = a === b;
        a = clamp(Math.floor(snap(a / step)) * step, 0, limit);
        b = equal ? a : clamp(Math.ceil(snap(b / step)) * step, 0, limit);
      }
      const firstSelection = !ui.selection;
      if (ui.selection?.start !== a || ui.selection?.end !== b) ui.stopAt = null;
      ui.selection = { start: a, end: b };
      if (firstSelection && !drag && Number(audio.dataset.waveWindow) > 0
        && (fraction(a) < 0 || fraction(b) > 1)) {
        setOffset((a + b - width()) / 2); update();
      }
      renderRegion(); if (emit) notify();
    }
    function drawWave() {
      wave.replaceChildren(); if (!ui.peaks || !valid()) return;
      const first = ui.offset / audio.duration, span = 1 / ui.zoom;
      const peaks = Array.from({ length: 80 }, (_, index) => {
        const a = Math.floor((first + span * index / 80) * ui.peaks.length);
        const b = Math.max(a + 1, Math.ceil((first + span * (index + 1) / 80) * ui.peaks.length));
        let peak = 0;
        for (let at = a; at < Math.min(b, ui.peaks.length); at++) peak = Math.max(peak, ui.peaks[at]);
        return peak;
      });
      const maximum = Math.max(...ui.peaks, 0.001);
      for (const [index, peak] of peaks.entries()) {
        const bar = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        const height = Math.max(1.5, 50 * peak / maximum);
        for (const [key, value] of Object.entries({ x: index * 5, y: (60 - height) / 2,
          width: 2.7, height, rx: 1.35 })) bar.setAttribute(key, value);
        wave.append(bar);
      }
    }
    function update() {
      if (audio.getAttribute('src') !== ui.source) {
        ui.source = audio.getAttribute('src'); ui.selection = null; ui.zoom = 1; ui.offset = 0; ui.stopAt = null;
        ui.viewReady = false; ui.followPlayback = false;
        ui.waveSource = null; ui.peaks = null; wave.replaceChildren();
        notify();
      }
      if (valid() && !ui.viewReady) {
        const target = Number(audio.dataset.waveWindow);
        ui.viewReady = true;
        ui.zoom = target > 0 ? clamp(audio.duration / target, 1, maxZoom()) : 1;
        ui.offset = clamp(audio.currentTime - width() / 2, 0, audio.duration - width());
        drawWave();
      }
      track.classList.toggle('with-waveform', Boolean(audio.getAttribute('src')));
      if (valid()) seek.max = audio.duration;
      seek.disabled = !valid(); seek.value = audio.currentTime || 0;
      seek.style.setProperty('--played', `${valid() ? 100 * audio.currentTime / audio.duration : 0}%`);
      playhead.hidden = !valid() || fraction(audio.currentTime) < 0 || fraction(audio.currentTime) > 1;
      if (valid()) playhead.style.left = `${clamp(fraction(audio.currentTime), 0, 1) * 100}%`;
      [...wave.children].forEach((bar, index) => bar.classList.toggle('played',
        valid() && index < fraction(audio.currentTime) * wave.childElementCount));
      setText(current, clock(audio.currentTime)); setText(duration, valid() ? clock(audio.duration) : '--:--');
      seek.setAttribute('aria-valuetext', `${clock(audio.currentTime)} / ${clock(audio.duration)}`);
      const previewing = !audio.paused && ui.stopAt !== null;
      setIcon(play, audio.paused ? 'play' : 'pause');
      setIcon(preview, previewing ? 'pause' : 'play');
      for (const [node, label] of [
        [play, audio.paused ? 'Play audio' : 'Pause audio'], [seek, 'Audio position'], [rate, 'Playback speed'],
        [mute, audio.muted || audio.volume === 0 ? 'Unmute audio' : 'Mute audio'], [volume, 'Volume'],
        [preview, previewing ? 'Pause selection' : 'Play selection'], [clear, 'Clear selection'], [zoomOut, 'Zoom out waveform'],
        [zoomIn, 'Zoom in waveform'], [pan, 'Waveform position'],
      ]) { node.setAttribute('aria-label', t(label)); node.title = t(label); }
      setText(previewLabel, t(previewing ? 'Pause selection' : 'Play selection'));
      setText(rate, `${audio.playbackRate}×`);
      setIcon(mute, audio.muted || audio.volume === 0 ? 'muted' : 'volume');
      volume.value = audio.muted ? 0 : audio.volume;
      zoomOut.disabled = !ui.peaks || ui.zoom === 1; zoomIn.disabled = !ui.peaks || ui.zoom >= maxZoom();
      pan.hidden = !valid() || ui.zoom === 1;
      pan.max = valid() ? Math.max(0, audio.duration - width()) : 0; pan.value = ui.offset;
      pan.style.setProperty('--pan-size', `${Math.max(24, surface.clientWidth / ui.zoom)}px`);
      pan.setAttribute('aria-valuetext', valid() ? `${clock(ui.offset, true)} – ${clock(ui.offset + width(), true)}` : '');
      timeline.hidden = pan.hidden;
      if (!timeline.hidden) ticks.forEach((tick, index) =>
        setText(tick, clock(ui.offset + width() * index / (ticks.length - 1), width() < 5)));
      box.classList.toggle('wave-selectable', selectable());
      hint.textContent = ''; hint.hidden = true;
      surface.title = selectable() && !ui.selection ? t('Drag on the waveform to select') : '';
      surface.setAttribute('aria-label', t('Audio waveform')); renderRegion();
    }
    async function startPlayback(end = null) {
      for (const other of all) if (other.audio !== audio && !other.audio.paused) other.audio.pause();
      ui.stopAt = end; ui.followPlayback = true; revealPlayhead(); error.hidden = true;
      try { await audio.play(); }
      catch (failure) {
        if (failure.name !== 'AbortError') { error.textContent = t('Unable to play this audio.'); error.hidden = false; }
      }
    }
    function watchPlayback() {
      if (ui.stopAt !== null && audio.currentTime >= ui.stopAt) {
        const end = ui.stopAt; ui.stopAt = null; audio.pause(); audio.currentTime = end;
      }
      if (ui.followPlayback && !drag) revealPlayhead();
      update();
      if (!audio.paused && box.isConnected) ui.raf = requestAnimationFrame(watchPlayback);
    }
    play.onclick = () => audio.paused ? startPlayback() : audio.pause();
    preview.onclick = () => {
      if (!ui.selection || ui.selection.end <= ui.selection.start) return;
      if (!audio.paused && ui.stopAt !== null) { audio.pause(); return; }
      const resume = ui.stopAt === ui.selection.end && audio.currentTime >= ui.selection.start
        && audio.currentTime < ui.selection.end;
      if (!resume) audio.currentTime = ui.selection.start;
      startPlayback(ui.selection.end);
    };
    clear.onclick = () => { ui.selection = null; ui.stopAt = null; update(); notify(); };
    seek.addEventListener('input', () => {
      if (!seek.disabled) {
        ui.stopAt = null; audio.currentTime = Number(seek.value); ui.followPlayback = true; revealPlayhead();
      }
      update();
    });
    rate.onclick = () => {
      const rates = [1, 1.25, 1.5, 2, 0.75];
      audio.playbackRate = rates[(rates.indexOf(audio.playbackRate) + 1) % rates.length]; update();
    };
    mute.onclick = () => { audio.muted = !audio.muted; update(); };
    volume.addEventListener('input', () => { audio.muted = false; audio.volume = Number(volume.value); update(); });
    function zoom(multiplier) {
      if (!valid() || !ui.peaks) return;
      const center = ui.selection ? (ui.selection.start + ui.selection.end) / 2
        : fraction(audio.currentTime) >= 0 && fraction(audio.currentTime) <= 1
          ? audio.currentTime : ui.offset + width() / 2;
      ui.followPlayback = false;
      ui.zoom = clamp(ui.zoom * multiplier, 1, maxZoom());
      ui.offset = clamp(center - width() / 2, 0, audio.duration - width()); drawWave(); update();
    }
    zoomIn.onclick = () => zoom(2); zoomOut.onclick = () => zoom(0.5);
    pan.addEventListener('input', () => {
      ui.followPlayback = false; setOffset(Number(pan.value)); update();
    });
    const timeAt = event => {
      const bounds = surface.getBoundingClientRect();
      return clamp(ui.offset + clamp((event.clientX - bounds.left) / bounds.width, 0, 1) * width(), 0, audio.duration);
    };
    surface.addEventListener('pointerdown', event => {
      if (!valid() || event.button !== 0 || drag) return;
      event.preventDefault(); audio.pause(); ui.stopAt = null; ui.followPlayback = false;
      const edge = event.target.closest('[data-edge]')?.dataset.edge;
      drag = { id: event.pointerId, x: event.clientX, lastX: event.clientX, offset: ui.offset, anchor: timeAt(event),
        edge, move: !edge && ui.selection?.end > ui.selection?.start && region.contains(event.target),
        secondsPerPixel: width() / surface.getBoundingClientRect().width,
        prior: ui.selection ? { ...ui.selection } : null, moved: false };
      surface.setPointerCapture(event.pointerId);
    });
    function applyDrag(clientX) {
      const value = timeAt({ clientX });
      if (selectable()) {
        if (drag.edge && drag.prior) {
          const other = drag.edge === 'start' ? drag.prior.end : drag.prior.start;
          setSelection(drag.edge === 'start' ? Math.min(value, other) : other,
            drag.edge === 'end' ? Math.max(value, other) : other, { emit: true });
        } else if (drag.move) {
          const { start, end } = drag.prior;
          let delta = (clientX - drag.x) * drag.secondsPerPixel + ui.offset - drag.offset;
          const step = Number(audio.dataset.frameSeconds || 0);
          // Snap one shared displacement so moving never adds a codec frame.
          if (step > 0) {
            delta = clamp(Math.round(delta / step), Math.ceil(-start / step - 0.00002),
              Math.floor((audio.duration - end) / step + 0.00002)) * step;
          } else delta = clamp(delta, -start, audio.duration - end);
          surface.classList.add('moving-region');
          setSelection(start + delta, end + delta, { emit: true });
        } else setSelection(drag.anchor, value, { emit: true });
      } else { audio.currentTime = value; update(); }
    }
    function scrollAtEdge(now) {
      edgeRaf = null;
      if (!drag?.moved || !selectable() || ui.zoom === 1) return;
      const bounds = surface.getBoundingClientRect();
      const direction = drag.lastX < bounds.left + 16 ? -1 : drag.lastX > bounds.right - 16 ? 1 : 0;
      if (!direction) { drag.edgeTime = null; return; }
      const elapsed = drag.edgeTime ? Math.min((now - drag.edgeTime) / 1000, .05) : 1 / 60;
      drag.edgeTime = now;
      const before = ui.offset;
      setOffset(ui.offset + direction * width() * elapsed * .6);
      if (ui.offset !== before) { applyDrag(drag.lastX); update(); }
      edgeRaf = requestAnimationFrame(scrollAtEdge);
    }
    surface.addEventListener('pointermove', event => {
      if (!drag || drag.id !== event.pointerId) return;
      drag.lastX = event.clientX;
      if (Math.abs(event.clientX - drag.x) > 3) drag.moved = true;
      if (!drag.moved) return;
      applyDrag(event.clientX);
      if (!edgeRaf) edgeRaf = requestAnimationFrame(scrollAtEdge);
    });
    function cancelDrag(event) {
      if (!drag || drag.id !== event.pointerId) return;
      cancelAnimationFrame(edgeRaf); edgeRaf = null;
      drag = null; surface.classList.remove('moving-region');
      if (surface.hasPointerCapture(event.pointerId)) surface.releasePointerCapture(event.pointerId);
    }
    function finishDrag(event) {
      if (!drag || drag.id !== event.pointerId) return;
      if (!drag.moved && !drag.edge) { audio.currentTime = timeAt(event); update(); }
      cancelDrag(event);
    }
    surface.addEventListener('pointerup', finishDrag);
    surface.addEventListener('pointercancel', cancelDrag);
    surface.addEventListener('lostpointercapture', cancelDrag);
    for (const handle of handles) handle.addEventListener('keydown', event => {
      if (!ui.selection || !['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      const edge = handle.dataset.edge;
      const step = (Number(audio.dataset.frameSeconds) || 0.01) * (event.shiftKey ? 10 : 1);
      const target = event.key === 'Home' ? 0 : event.key === 'End' ? audio.duration
        : ui.selection[edge] + step * (event.key === 'ArrowLeft' ? -1 : 1);
      setSelection(edge === 'start' ? clamp(target, 0, ui.selection.end) : ui.selection.start,
        edge === 'end' ? clamp(target, ui.selection.start, audio.duration) : ui.selection.end, { emit: true });
    });
    async function loadWave() {
      update();
      if (!audio.getAttribute('src') || ui.waveSource === audio.src) return;
      const source = audio.src, request = ++ui.request;
      ui.waveSource = source; ui.abort?.abort(); ui.abort = new AbortController();
      wave.replaceChildren(); box.classList.add('loading-waveform');
      try {
        const peaks = await loadPeaks(source);
        if (request !== ui.request || !audio.getAttribute('src')) return;
        ui.peaks = peaks; drawWave(); track.classList.add('with-waveform'); update();
      } catch (_) {
        /* Keep the accessible seek bar when the browser cannot decode a format. */
      } finally { if (request === ui.request) box.classList.remove('loading-waveform'); }
    }
    audio.addEventListener('loadstart', loadWave);
    audio.addEventListener('loadedmetadata', () => { drawWave(); loadWave(); });
    audio.addEventListener('emptied', () => { ui.request++; ui.abort?.abort(); ui.waveSource = null; update(); });
    audio.addEventListener('play', () => { cancelAnimationFrame(ui.raf); watchPlayback(); });
    audio.addEventListener('pause', () => { cancelAnimationFrame(ui.raf); update(); });
    for (const name of ['timeupdate', 'loadedmetadata', 'durationchange', 'ended', 'ratechange', 'volumechange']) {
      audio.addEventListener(name, update);
    }
    new MutationObserver(() => { update(); if (audio.preload !== 'none') loadWave(); }).observe(audio, {
      attributes: true, attributeFilter: ['src', 'data-selection', 'data-frame-seconds'],
    });
    const observer = new IntersectionObserver(entries => {
      if (entries.some(entry => entry.isIntersecting)) {
        if (audio.preload === 'none' && audio.getAttribute('src') && !Number.isFinite(audio.duration)) {
          audio.preload = 'metadata'; audio.load();
        }
        loadWave();
      }
    }, { rootMargin: '120px' });
    observer.observe(box);
    const resizeObserver = new ResizeObserver(update); resizeObserver.observe(surface);
    ui.observer = observer; ui.resizeObserver = resizeObserver;
    ui.update = update; ui.setSelection = setSelection; update();
    if (audio.preload !== 'none') loadWave();
    return ui;
  }
  window.StudioAudio = Object.freeze({
    enhance(root = document) {
      for (const ui of all) if (!ui.box.isConnected) {
        ui.audio.pause(); cancelAnimationFrame(ui.raf);
        ui.abort?.abort(); ui.observer.disconnect(); ui.resizeObserver.disconnect(); all.delete(ui);
      }
      for (const audio of root.querySelectorAll('audio')) enhance(audio);
    },
    selection(audio) { const region = players.get(audio)?.selection; return region ? { ...region } : null; },
    setSelection(audio, start, end, options) { enhance(audio).setSelection(start, end, options); },
    clearSelection(audio) { const ui = players.get(audio); if (ui) { ui.selection = null; ui.update(); } },
    translate() {
      for (const ui of all) if (ui.box.isConnected) ui.update();
    },
  });
})();
