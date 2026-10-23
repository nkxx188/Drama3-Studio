/* One startup transaction: mount, hydrate, measure, then expose the workspace. */
(() => {
  'use strict';
  const root = document.documentElement, scrollKey = 'drama3.ui.scroll';
  let issue = '', scroll;
  try {
    history.scrollRestoration = 'manual';
    scroll = JSON.parse(sessionStorage.getItem(scrollKey) || 'null');
  } catch (_) { /* optional browser storage */ }
  const frame = () => new Promise(resolve => requestAnimationFrame(resolve));
  function labels() {
    if (!window.I18n) return;
    I18n.apply(); root.dataset.localized = 'true';
    const error = document.getElementById('startup-error');
    if (error && issue) error.textContent = I18n.message(issue);
  }
  function fail(error) {
    if (root.dataset.startup === 'ready') return;
    issue = error?.message || String(error || '');
    root.dataset.startup = 'error';
    const target = document.getElementById('startup-error');
    if (target) target.textContent = window.I18n?.message(issue) || issue;
    labels();
  }
  addEventListener('error', event => {
    if (event.target instanceof HTMLScriptElement) {
      fail(`Failed to load ${new URL(event.target.src, location.href).pathname.split('/').at(-1)}`);
    } else if (event.error) fail(event.error);
  }, true);
  addEventListener('unhandledrejection', event => fail(event.reason));
  addEventListener('pagehide', () => {
    if (root.dataset.startup !== 'ready') return;
    try { sessionStorage.setItem(scrollKey, JSON.stringify({ view: root.dataset.view, x: scrollX, y: scrollY })); }
    catch (_) { /* optional browser storage */ }
  });
  document.addEventListener('drama:languagechange', labels);
  document.addEventListener('DOMContentLoaded', labels, { once: true });
  window.StudioStartup = Object.freeze({
    fail,
    async commit() {
      if (root.dataset.startup === 'error') return;
      root.dataset.startup = 'measuring';
      labels();
      await document.fonts.ready;
      window.dispatchEvent(new Event('drama:studio-ready'));
      await frame(); await frame();
      if (root.dataset.startup === 'error') return;
      const saved = scroll?.view === root.dataset.view ? scroll : null;
      window.scrollTo({ left: Math.max(0, Number(saved?.x) || 0),
        top: Math.max(0, Number(saved?.y) || 0), behavior: 'instant' });
      await frame();
      if (root.dataset.startup === 'error') return;
      const content = document.getElementById('workspace-content');
      content.inert = false; content.setAttribute('aria-busy', 'false');
      root.dataset.startup = 'ready'; root.dataset.studioReady = 'true';
      window.dispatchEvent(new Event('drama:startup-ready'));
    },
  });
})();
