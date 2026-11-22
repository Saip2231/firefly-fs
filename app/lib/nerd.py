"""Nerd mode: a top-bar toggle beside the theme pill.

Off (the default) shows the crisp dashboard: charts, numbers, verdicts, the
one-sentence takeaways and every label the rules require (exploratory,
parallel load, PCA is a projection, the one-image anecdote line).  On shows
the extra explanation: method notes, captions, PREREG excerpts, tables.

Pages mark extra text with app.components.bento.nerd() (a keyed container
`st-key-ff-nerd-*`) or the `ff-nerd` class in raw HTML; text that only the
crisp view needs uses crisp() / `ff-crisp`.  The switch is pure CSS on
<html data-ff-nerd="on|off">, so it is instant, needs no rerun, and the
choice persists in localStorage for every page.  Styled with the secondary
colour (navy #2B4C7E in light, #5BA8AB in dark).
"""

from __future__ import annotations

STORAGE_KEY = "ff-nerd-mode"

NERD = '[class*="st-key-ff-nerd-"]'
CRISP = '[class*="st-key-ff-crisp-"]'
WRAP = '[data-testid="stLayoutWrapper"]'


def css(font: str) -> str:
    off = ':root:not([data-ff-nerd="on"])'
    on = ':root[data-ff-nerd="on"]'
    return f"""
/* Top bar: nerd-mode switch, then the theme pill. */
.ff-topbar {{
  position: fixed; top: 0.7rem; right: 3.6rem; z-index: 999991;
  display: flex; align-items: center; gap: 10px;
}}
.ff-nerd-pill {{
  display: inline-flex; align-items: center; gap: 0.45rem; cursor: pointer;
  padding: 0.22rem 0.75rem 0.22rem 0.5rem; border-radius: 999px;
  font: 500 13px/1.4 {font}; color: var(--ff-accent-2); background: transparent;
  border: 1px solid color-mix(in srgb, var(--ff-accent-2) 55%, transparent);
  transition: background-color 300ms ease-in-out, border-color 200ms ease,
              color 300ms ease-in-out, transform 120ms ease;
}}
.ff-nerd-pill:hover {{ border-color: var(--ff-accent-2); }}
.ff-nerd-pill:active {{ transform: scale(0.98); }}
.ff-nerd-pill:focus-visible {{ outline: 2px solid var(--ff-accent-2); outline-offset: 2px; }}
.ff-nerd-pill .ff-switch {{
  position: relative; width: 26px; height: 14px; border-radius: 999px;
  background: color-mix(in srgb, var(--ff-accent-2) 22%, transparent);
}}
.ff-nerd-pill .ff-switch::after {{
  content: ""; position: absolute; top: 2px; left: 2px; width: 10px; height: 10px;
  border-radius: 50%; background: var(--ff-accent-2); transition: transform 200ms ease;
}}
{on} .ff-nerd-pill {{
  background: color-mix(in srgb, var(--ff-accent-2) 14%, transparent);
  border-color: var(--ff-accent-2);
}}
{on} .ff-nerd-pill .ff-switch::after {{ transform: translateX(12px); }}
.ff-nerd-pill .ff-nerd-ico {{ position: relative; width: 1.05rem; height: 1.05rem; }}
.ff-nerd-pill .ff-nerd-ico span {{
  position: absolute; inset: 0; display: grid; place-items: center;
  transition: transform 450ms cubic-bezier(.2,.8,.2,1), opacity 300ms ease;
}}
{off} .ff-nerd-pill .ff-geek, {on} .ff-nerd-pill .ff-cool {{
  opacity: 0; transform: rotate(-90deg) scale(0.5);
}}
@media (max-width: 640px) {{ .ff-nerd-pill .ff-nerd-word {{ display: none; }} }}

/* Crisp view hides the extra text; nerd view hides the crisp-only lines. */
{off} {NERD}, {off} {WRAP}:has(> {NERD}), {off} .ff-nerd,
{on} {CRISP}, {on} {WRAP}:has(> {CRISP}), {on} .ff-crisp {{
  display: none !important;
}}
"""


JS = """
(() => {
  const root = document.documentElement;
  const apply = (on) => {
    root.dataset.ffNerd = on ? 'on' : 'off';
    document.querySelectorAll('.ff-nerd-pill').forEach((p) => {
      p.setAttribute('aria-pressed', String(on));
      p.setAttribute('title', on ? 'Nerd mode on: every note and table is shown'
                                 : 'Nerd mode off: the crisp view');
    });
  };
  let stored = null;
  try { stored = localStorage.getItem('__KEY__'); } catch (err) { stored = null; }
  apply(root.dataset.ffNerd ? root.dataset.ffNerd === 'on' : stored === 'on');
  if (window.__ffNerd) return;
  window.__ffNerd = true;
  document.addEventListener('click', (e) => {
    if (!e.target.closest || !e.target.closest('.ff-nerd-pill')) return;
    const on = root.dataset.ffNerd !== 'on';
    try { localStorage.setItem('__KEY__', on ? 'on' : 'off'); } catch (err) { /* session only */ }
    apply(on);
  });
})();
""".replace("__KEY__", STORAGE_KEY)

# 😎 while off (the crisp view), morphing into 🤓 when nerd mode is on.
BUTTON = """
<button type="button" class="ff-nerd-pill" aria-pressed="false"
        aria-label="Nerd mode: show every note and table">
  <span class="ff-nerd-ico" aria-hidden="true"><span class="ff-cool">😎</span><span class="ff-geek">🤓</span></span>
  <span class="ff-nerd-word">Nerd mode</span>
  <span class="ff-switch" aria-hidden="true"></span>
</button>
"""
