/* Local select components. The source select retains the form/API value. */
(() => {
  'use strict';
  const controls = new Map();
  const t = source => I18n.t(source);
  const label = option => option?.dataset.i18n ? t(option.dataset.i18n) : option?.dataset.title || option?.textContent || '';
  const svg = name => `<svg class="icon" aria-hidden="true"><use href="#i-${name}"></use></svg>`;
  let opened = null;
  let typed = '';
  let typeTimer;

  function close(restoreFocus = false) {
    if (!opened) return;
    const control = opened;
    opened = null;
    control.popup.hidden = true;
    control.trigger.setAttribute('aria-expanded', 'false');
    control.trigger.removeAttribute('aria-activedescendant');
    if (restoreFocus) control.trigger.focus({ preventScroll: true });
  }
  function place(control) {
    const rect = control.trigger.getBoundingClientRect();
    const preferredWidth = control.search ? Math.max(rect.width, 760) : Math.min(360, Math.max(rect.width, 230));
    const width = Math.min(innerWidth - 24, preferredWidth);
    const below = innerHeight - rect.bottom - 20;
    const above = rect.top - 20;
    const upward = below < 240 && above > below;
    control.popup.style.width = `${width}px`;
    control.popup.style.left = `${Math.min(Math.max(12, rect.left), innerWidth - width - 12)}px`;
    control.popup.style.maxHeight = `${Math.min(440, Math.max(160, upward ? above : below))}px`;
    control.popup.style.top = upward ? 'auto' : `${rect.bottom + 8}px`;
    control.popup.style.bottom = upward ? `${innerHeight - rect.top + 8}px` : 'auto';
  }
  function activate(control, index, scroll = true) {
    const available = control.items.filter(item => !item.hidden && !item.disabled);
    for (const item of control.items) item.classList.remove('keyboard-active');
    if (!available.length) {
      control.active = null;
      control.trigger.removeAttribute('aria-activedescendant');
      control.search?.removeAttribute('aria-activedescendant');
      return;
    }
    const item = available[(index + available.length) % available.length];
    control.active = item;
    item.classList.add('keyboard-active');
    control.trigger.setAttribute('aria-activedescendant', item.id);
    control.search?.setAttribute('aria-activedescendant', item.id);
    if (scroll) item.scrollIntoView({ block: 'nearest' });
  }
  function filter(control) {
    const term = (control.search?.value || '').trim().toLocaleLowerCase();
    for (const item of control.items) item.hidden = !item.dataset.label.toLocaleLowerCase().includes(term);
    control.empty.hidden = control.items.some(item => !item.hidden);
    activate(control, 0, false);
  }
  function choose(control, item) {
    if (!item || item.disabled) return;
    control.select.value = item.dataset.value;
    close();
    control.select.dispatchEvent(new Event('change', { bubbles: true }));
    sync(control.select.id);
    // "Add a model" deliberately moves focus to the model-path field.
    if (item.dataset.value !== '__manage__') control.trigger.focus({ preventScroll: true });
  }
  function open(control, last = false) {
    if (control.select.disabled) return;
    close();
    opened = control;
    sync(control.select.id);
    if (control.search) control.search.value = '';
    for (const item of control.items) item.hidden = false;
    control.empty.hidden = true;
    const container = control.select.closest('dialog') || document.body;
    if (control.popup.parentElement !== container) container.append(control.popup);
    control.popup.hidden = false;
    control.trigger.setAttribute('aria-expanded', 'true');
    place(control);
    const enabled = control.items.filter(item => !item.disabled);
    const selected = enabled.findIndex(item => item.dataset.value === control.select.value);
    activate(control, last ? enabled.length - 1 : Math.max(0, selected), false);
    control.search?.focus({ preventScroll: true });
    control.active?.scrollIntoView({ block: 'nearest' });
  }
  function keyboard(control, event) {
    if (event.isComposing) return;
    if (event.key === 'Tab') { close(); return; }
    if (event.key === 'Escape' && opened === control) {
      event.preventDefault(); event.stopPropagation(); close(true); return;
    }
    if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) {
      event.preventDefault();
      if (opened !== control) { open(control, event.key === 'End'); return; }
      const items = control.items.filter(item => !item.hidden && !item.disabled);
      const at = items.indexOf(control.active);
      activate(control, event.key === 'Home' ? 0 : event.key === 'End' ? items.length - 1
        : at + (event.key === 'ArrowDown' ? 1 : -1));
    } else if (event.key === 'Enter' || (event.key === ' ' && event.target !== control.search)) {
      event.preventDefault();
      if (opened === control) choose(control, control.active);
      else open(control);
    } else if (event.key.length === 1 && event.target !== control.search && !event.ctrlKey && !event.metaKey) {
      if (opened !== control) open(control);
      typed += event.key.toLocaleLowerCase();
      clearTimeout(typeTimer); typeTimer = setTimeout(() => { typed = ''; }, 650);
      const items = control.items.filter(item => !item.hidden && !item.disabled);
      const at = items.findIndex(item => item.dataset.label.toLocaleLowerCase().startsWith(typed));
      if (at >= 0) activate(control, at);
    }
  }
  function sync(id) {
    const control = controls.get(id);
    if (!control) return;
    const select = control.select;
    const selected = select.selectedOptions[0] || select.options[0];
    control.value.textContent = label(selected);
    control.trigger.disabled = select.disabled;
    const field = select.getAttribute('aria-label') || document.querySelector(`label[for="${id}"]`)?.textContent.trim() || '';
    control.trigger.setAttribute('aria-label', `${field}: ${label(selected)}`);
    control.trigger.title = label(selected);
    if (control.search) {
      control.search.placeholder = t('Search models…');
      control.search.setAttribute('aria-label', t('Search models…'));
    }
    control.empty.textContent = t('No matches found');
    if (select.disabled && opened === control) close();
    const signature = JSON.stringify([...select.options].map(option =>
      [option.value, label(option), option.disabled, option.dataset.runtime, option.dataset.description]));
    if (signature !== control.signature) {
      control.signature = signature;
      control.list.replaceChildren();
      control.items = [...select.options].map((option, index) => {
        const item = document.createElement('button');
        item.type = 'button';
        item.className = 'custom-option';
        item.id = `${id}-option-${index}`;
        item.tabIndex = -1;
        item.setAttribute('role', 'option');
        item.dataset.value = option.value;
        item.dataset.label = `${label(option)} ${option.textContent} ${option.dataset.description || ''}`;
        item.disabled = option.disabled;
        const copy = document.createElement('span');
        copy.className = 'option-copy';
        const text = document.createElement('span');
        text.className = 'custom-option-main';
        text.textContent = label(option);
        copy.append(text);
        if (option.dataset.description || option.dataset.runtime) {
          const detail = document.createElement('small');
          detail.textContent = option.dataset.description || (option.dataset.runtime === 'gguf' ? 'GGUF' : 'Safetensors');
          copy.append(detail);
        } else if (option.disabled) {
          const detail = document.createElement('small');
          detail.textContent = t('Not supported by this model');
          copy.append(detail);
        }
        const check = document.createElement('span');
        check.className = 'option-check'; check.innerHTML = svg('check');
        item.append(copy, check);
        item.addEventListener('pointerdown', event => event.preventDefault());
        item.addEventListener('click', () => choose(control, item));
        control.list.append(item);
        return item;
      });
      if (opened === control) filter(control);
    }
    for (const item of control.items) item.setAttribute('aria-selected', String(item.dataset.value === select.value));
    if (opened === control) place(control);
  }
  function enhance(id) {
    const select = document.getElementById(id);
    if (!select) return;
    const previous = controls.get(id);
    if (previous?.select === select) return;
    if (previous) {
      if (opened === previous) close();
      previous.observer?.disconnect(); previous.popup.remove(); controls.delete(id);
    }
    const wrapper = document.createElement('div');
    wrapper.className = `custom-select custom-select-${id}`;
    wrapper.dataset.select = id;
    select.before(wrapper); wrapper.append(select);
    select.classList.add('native-select');
    select.tabIndex = -1;
    select.setAttribute('aria-hidden', 'true');
    const trigger = document.createElement('button');
    trigger.type = 'button';
    trigger.id = `${id}-trigger`;
    trigger.className = 'custom-select-trigger';
    trigger.setAttribute('role', 'combobox');
    trigger.setAttribute('aria-haspopup', 'listbox');
    trigger.setAttribute('aria-expanded', 'false');
    trigger.setAttribute('aria-controls', `${id}-listbox`);
    const value = document.createElement('span');
    value.className = 'custom-select-value';
    const chevron = document.createElement('span');
    chevron.className = 'select-chevron'; chevron.innerHTML = svg('chevron');
    trigger.append(value, chevron);
    wrapper.append(trigger);
    const popup = document.createElement('div');
    popup.className = `select-popover select-popover-${id}`;
    popup.hidden = true;
    let search = null;
    if (['active-model', 'installed-model'].includes(id)) {
      const field = document.createElement('div'); field.className = 'select-search';
      field.innerHTML = svg('search');
      search = document.createElement('input'); search.type = 'search';
      search.setAttribute('aria-controls', `${id}-listbox`);
      search.setAttribute('autocomplete', 'off'); search.setAttribute('spellcheck', 'false');
      field.append(search); popup.append(field);
    }
    const list = document.createElement('div');
    list.id = `${id}-listbox`; list.className = 'custom-select-options';
    list.setAttribute('role', 'listbox');
    const empty = document.createElement('div'); empty.className = 'select-no-results'; empty.hidden = true;
    popup.append(list, empty);
    (select.closest('dialog') || document.body).append(popup);
    const control = { select, trigger, value, popup, list, search, empty, items: [], active: null };
    controls.set(id, control);
    trigger.addEventListener('click', event => {
      event.preventDefault(); if (opened === control) close(); else open(control);
    });
    trigger.addEventListener('keydown', event => keyboard(control, event));
    popup.addEventListener('keydown', event => keyboard(control, event));
    search?.addEventListener('input', () => filter(control));
    select.addEventListener('change', () => sync(id));
    document.querySelector(`label[for="${id}"]`)?.addEventListener('click', event => {
      if (event.target.closest('button, input')) return;
      event.preventDefault(); trigger.focus();
    });
    control.observer = new MutationObserver(() => sync(id));
    control.observer.observe(select, { childList: true, characterData: true,
      subtree: true, attributes: true, attributeFilter: ['disabled', 'label', 'selected'] });
    sync(id);
  }
  document.addEventListener('pointerdown', event => {
    if (opened && !opened.popup.contains(event.target) && !opened.trigger.contains(event.target)) close();
  });
  document.addEventListener('focusin', event => {
    if (opened && !opened.popup.contains(event.target) && !opened.trigger.contains(event.target)) close();
  });
  document.addEventListener('close', event => {
    if (opened && event.target.contains(opened.select)) close();
  }, true);
  addEventListener('resize', () => opened && place(opened));
  addEventListener('scroll', event => {
    if (opened && !opened.popup.contains(event.target)) place(opened);
  }, true);
  window.StudioControls = Object.freeze({
    init() {
      for (const [id, control] of controls) if (!control.select.isConnected) {
        if (opened === control) close();
        control.observer?.disconnect(); control.popup.remove(); controls.delete(id);
      }
      document.querySelectorAll('select[id]').forEach(select => enhance(select.id));
    },
    sync(id) { enhance(id); sync(id); },
    syncAll() { for (const id of controls.keys()) { enhance(id); sync(id); } },
    close,
  });
})();
