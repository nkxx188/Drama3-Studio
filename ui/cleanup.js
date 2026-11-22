/* The selection applies to rows currently shown by the search or filter. */
const StudioCleanup = (() => {
  const lists = {
    voices: { list: $('voice-list'), itemLabel: 'Select voice', titleSelector: '.voice-card-copy strong' },
    jobs: { list: $('job-list'), itemLabel: 'Select result', titleSelector: '.job-title' },
  };
  const dialog = $('delete-dialog'), confirm = $('delete-confirm'), cancel = $('delete-cancel');
  let request = null, deleting = false;
  const rows = list => [...list.querySelectorAll('[data-cleanup-id]:not([hidden])')];
  const checkboxes = list => rows(list).map(row => row.querySelector('.cleanup-check')).filter(box => !box.disabled);
  const setLabel = (node, label, args = {}) => {
    node.dataset.i18n = label;
    node.dataset.i18nArgs = JSON.stringify(args);
    node.textContent = t(label, args);
  };

  function sync(kind) {
    const { list, toggle, bar, all, count, remove, selected, selecting } = lists[kind];
    const shown = rows(list);
    const boxes = shown.map(row => row.querySelector('.cleanup-check'));
    const available = new Set(boxes.filter(box => !box.disabled).map(box => box.value));
    for (const id of selected) if (!available.has(id)) selected.delete(id);
    for (const box of boxes) {
      const checked = selected.has(box.value);
      box.checked = checked;
      box.closest('[data-cleanup-id]').classList.toggle('cleanup-selected', checked);
    }
    list.classList.toggle('cleanup-selecting', selecting);
    toggle.disabled = !selecting && !available.size;
    toggle.setAttribute('aria-pressed', String(selecting));
    setLabel(toggle, selecting ? 'Exit selection' : 'Select items');
    bar.hidden = !selecting;
    all.disabled = !available.size;
    all.checked = available.size > 0 && selected.size === available.size;
    all.indeterminate = selected.size > 0 && !all.checked;
    setLabel(count, '{count} selected', { count: selected.size });
    remove.disabled = !selected.size;
  }

  function checkbox(kind, id) {
    const box = document.createElement('input');
    box.className = 'cleanup-check'; box.type = 'checkbox'; box.value = id;
    box.dataset.i18nAriaLabel = lists[kind].itemLabel;
    box.setAttribute('aria-label', t(lists[kind].itemLabel));
    return box;
  }

  function open(kind, ids, title = '', trigger = document.activeElement) {
    if (dialog.open || !ids.length) return;
    request = { kind, ids, trigger };
    const one = ids.length === 1;
    if (one && !title) {
      const { list, titleSelector } = lists[kind];
      const row = rows(list).find(item => item.dataset.cleanupId === ids[0]);
      title = row?.querySelector(titleSelector)?.textContent || '';
    }
    const label = kind === 'voices' ? (one ? 'Delete voice?' : 'Delete {count} voices?')
      : (one ? 'Delete result?' : 'Delete {count} results?');
    setLabel($('delete-title'), label, { count: ids.length });
    $('delete-item').textContent = title;
    $('delete-item').hidden = !title;
    $('delete-error').hidden = true;
    stopVoicePreview();
    dialog.showModal();
  }

  function notice(kind, deleted, busy) {
    const note = $(`${kind}-cleanup-note`);
    note.innerHTML = [deleted ? translated('Deleted {count} items.', { count: deleted }) : '',
      busy ? translated('{count} active jobs were kept.', { count: busy }) : ''].filter(Boolean).join(' ');
    note.hidden = !note.textContent;
  }

  cancel.onclick = () => dialog.close();
  dialog.oncancel = event => { if (deleting) event.preventDefault(); };
  dialog.onclose = () => {
    if (request) {
      const { trigger, kind } = request;
      const fallback = kind === 'voices' ? $('add-voice') : $('refresh-jobs');
      (trigger?.isConnected && trigger.getClientRects().length && !trigger.disabled ? trigger : fallback).focus();
    }
    request = null;
  };
  confirm.onclick = async () => {
    if (!request || deleting) return;
    const { kind, ids } = request;
    const list = lists[kind];
    deleting = confirm.disabled = cancel.disabled = true;
    setLabel(confirm.firstElementChild, 'Deleting…');
    $('delete-error').hidden = true;
    try {
      await window.StudioFlow?.save();
      await window.AudioEditor?.flushDraft();
      const result = await api(`/api/${kind}/delete`, { method: 'POST', body: JSON.stringify({ ids }) });
      const deleted = new Set(result.deleted);
      for (const id of deleted) list.selected.delete(id);
      if (kind === 'voices') {
        await loadVoices(result);
      } else {
        removeHistoryJobs(deleted);
        await loadJobs();
      }
      if (!checkboxes(list.list).length) list.selecting = false;
      sync(kind);
      notice(kind, deleted.size, result.busy.length);
      dialog.close();
    } catch (error) {
      $('delete-error').textContent = message(error.message);
      $('delete-error').hidden = false;
    } finally {
      deleting = confirm.disabled = cancel.disabled = false;
      setLabel(confirm.firstElementChild, 'Delete');
    }
  };

  for (const [kind, config] of Object.entries(lists)) {
    const bar = $(`${kind}-cleanup`), toggle = $(`${kind}-select`);
    const trash = ScriptEditor.icon('remove'); trash.classList.add('icon');
    bar.innerHTML = `<label class="check-field"><input type="checkbox">${translated('Select all shown')}</label><span class="cleanup-count" role="status"></span><button class="secondary-btn danger-button" type="button">${trash.outerHTML}${translated('Delete selected')}</button>`;
    const all = bar.querySelector('input'), count = bar.querySelector('.cleanup-count'), remove = bar.querySelector('button');
    Object.assign(config, { bar, toggle, all, count, remove, selected: new Set(), selecting: false });
    toggle.onclick = () => {
      config.selecting = !config.selecting;
      config.selected.clear(); $(`${kind}-cleanup-note`).hidden = true;
      sync(kind);
    };
    all.onchange = () => {
      config.selected = new Set(all.checked ? checkboxes(config.list).map(box => box.value) : []);
      sync(kind);
    };
    remove.onclick = () => open(kind, [...config.selected]);
    config.list.addEventListener('change', event => {
      const box = event.target;
      if (!box.matches('.cleanup-check')) return;
      if (box.checked) config.selected.add(box.value);
      else config.selected.delete(box.value);
      sync(kind);
    });
    document.addEventListener('drama:viewchange', event => {
      if (event.detail === (kind === 'jobs' ? 'takes' : 'voices')) return;
      config.selecting = false; config.selected.clear(); $(`${kind}-cleanup-note`).hidden = true;
      sync(kind);
    });
    sync(kind);
  }
  document.addEventListener('drama:languagechange', () => {
    for (const kind of Object.keys(lists)) sync(kind);
  });
  return Object.freeze({ sync, checkbox, open });
})();
