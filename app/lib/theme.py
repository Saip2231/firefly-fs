"""Design tokens, page CSS and the shared Plotly template (DASHBOARD.md section 5).

Two themes, defined once here and mirrored in .streamlit/config.toml:
light = warm beige / deep navy / orange, dark = rich black / teal.  Streamlit
switches between them (system default, choice persisted in the browser);
`page_head()` keeps a `data-ff-theme` attribute on <html> in sync, so the
custom CSS below and the protocol visualizer switch at the same moment.

Method colours stay Okabe-Ito (colour-blind safe).  Colour is never the only
signal: labels always carry a symbol and a word, PCA is also dashed.
"""

from __future__ import annotations

from typing import Any

METHOD_COLOURS: dict[str, str] = {
    "firefly": "#E69F00",
    "firefly_seeded": "#F0E442",
    "pso": "#0072B2",
    "pso_seeded": "#56B4E9",
    "anova": "#009E73",
    "mi": "#CC79A7",
    "l1": "#D55E00",
    "rf": "#8C564B",
    "all": "#7F7F7F",
    "random": "#7F7F7F",
}
# PCA is "black dashed" in the spec; black is invisible on the dark surface,
# so it uses the theme's text colour (navy in light mode) plus the dash.
PCA_DASH = "dash"

# Label badges: the fill carries the colour, the text is always dark so the
# word itself meets 4.5:1 on every fill (white on #D55E00 would not).
LABEL_STYLE: dict[str, dict[str, str]] = {
    "WIN": {"fill": "#009E73", "symbol": "✓", "text": "#0E1117"},
    "LOSS": {"fill": "#D55E00", "symbol": "✗", "text": "#0E1117"},
    "TIE": {"fill": "#BDBDBD", "symbol": "=", "text": "#0E1117"},
}

# Semantic tokens per theme.  `accent` fills and strokes (non-text, needs 3:1),
# `accent_text` is the accent for words (needs 4.5:1), `on_accent` is the text
# colour on an accent fill.  White on #E85D2A (3.5:1) or on #05979C (3.6:1)
# fails WCAG AA for body text, hence navy / black on the accent.
TOKENS: dict[str, dict[str, str]] = {
    "light": {
        "background": "#F6F1E7", "card": "#FCF9F2", "border": "#E4DCCB",
        "text": "#132238", "muted": "#4A5B73",
        "accent": "#E85D2A", "accent_hover": "#D14B1A", "accent_text": "#B4441A",
        "accent_soft": "rgba(232, 93, 42, 0.10)", "on_accent": "#132238",
        "accent_2": "#2B4C7E", "glow": "rgba(232, 93, 42, 0.05)",
        "hover_border": "#E85D2A",
        "shadow": "0 1px 2px rgba(19, 34, 56, 0.05), 0 6px 20px rgba(19, 34, 56, 0.06)",
    },
    "dark": {
        "background": "#080A0C", "card": "#111518", "border": "rgba(91, 168, 171, 0.18)",
        "text": "#F1F5F6", "muted": "#9BB0B3",
        "accent": "#05979C", "accent_hover": "#5BA8AB", "accent_text": "#5BA8AB",
        "accent_soft": "rgba(5, 151, 156, 0.14)", "on_accent": "#080A0C",
        "accent_2": "#5BA8AB", "glow": "rgba(5, 151, 156, 0.06)",
        "hover_border": "#5BA8AB",
        "shadow": "none",
    },
}
MODES = ("light", "dark")
DEFAULT_MODE = "dark"

# Kept for callers that only need the five surface colours.
SURFACES: dict[str, dict[str, str]] = {
    m: {k: TOKENS[m][k] for k in ("background", "card", "border", "text", "muted")}
    for m in MODES
}

# Four self-hosted families (app/static/fonts/, declared in config.toml
# [[theme.fontFaces]]), each with a system fallback.
_SYSTEM = 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
FONTS: dict[str, str] = {
    "display": f'"Sora", {_SYSTEM}',
    "heading": f'"Sora", {_SYSTEM}',
    "body": f'"Plus Jakarta Sans", {_SYSTEM}',
    "mono": ('"JetBrains Mono", ui-monospace, "SFMono-Regular", "Cascadia Mono", Menlo, '
             'Consolas, monospace'),
}
FONT_STACK = FONTS["body"]
MONO_STACK = FONTS["mono"]
# Body stays 16 px (DASHBOARD.md rule 9: body text at least 16 px).
BODY_PX = 16
TRANSITION_MS = 300


def current_mode() -> str:
    """'light' or 'dark' as Streamlit reports it for this session (falls back
    to DEFAULT_MODE outside a browser session, e.g. in AppTest)."""
    try:
        import streamlit as st  # noqa: PLC0415
        mode = st.context.theme.type
    except Exception:  # noqa: BLE001  (no runtime context)
        mode = None
    return mode if mode in MODES else DEFAULT_MODE


def _mode(mode: str | None) -> str:
    return mode if mode in MODES else current_mode()


def token(name: str, mode: str | None = None) -> str:
    return TOKENS[_mode(mode)][name]


def method_colour(method: str, mode: str | None = None) -> str:
    """Line/marker colour for a method; PCA follows the text colour."""
    m = _mode(mode)
    if method == "pca":
        return SURFACES[m]["text"]
    return METHOD_COLOURS.get(method, SURFACES[m]["muted"])


def _luminance(hex_colour: str) -> float:
    rgb = [int(hex_colour.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(fg: str, bg: str) -> float:
    """WCAG 2 contrast ratio between two hex colours."""
    hi, lo = sorted((_luminance(fg), _luminance(bg)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _vars(mode: str) -> str:
    return " ".join(f"--ff-{k.replace('_', '-')}: {v};" for k, v in TOKENS[mode].items())


def css(mode: str | None = None) -> str:
    """Global stylesheet: theme variables, type scale, cards, motion."""
    from app.lib.layout_css import layout_css  # noqa: PLC0415

    m = _mode(mode)
    t = f"{TRANSITION_MS}ms ease-in-out"
    block = '[data-testid="stMainBlockContainer"] > div > [data-testid="stVerticalBlock"]'
    stagger = "\n".join(f"{block} > :nth-child({i}) {{ animation-delay: {(i - 1) * 45}ms; }}"
                        for i in range(2, 9))
    return f"""
:root {{ {_vars(m)} }}
:root[data-ff-theme="light"] {{ {_vars("light")} }}
:root[data-ff-theme="dark"] {{ {_vars("dark")} }}
{layout_css(BODY_PX, FONTS)}
[data-testid="stAppViewContainer"] {{
  background: radial-gradient(1100px 620px at 12% -8%, var(--ff-glow), transparent 65%),
              var(--ff-background);
  transition: background-color {t};
}}
[data-testid="stHeader"] {{
  background: color-mix(in srgb, var(--ff-background) 82%, transparent);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid var(--ff-border);
  transition: background-color {t}, border-color {t};
}}
p, span, label, li, h1, h2, h3, h4, a {{ transition: color {t}; }}
a {{ color: var(--ff-accent-text); }}

/* Cards: soft shadow in light mode, a 1px teal-tinted border in dark mode. */
[data-testid="stMetric"], [data-testid="stExpander"] details {{
  background: var(--ff-card);
  border: 1px solid var(--ff-border) !important;
  border-radius: 1rem;
  box-shadow: var(--ff-shadow);
  transition: transform 200ms ease, border-color 200ms ease, box-shadow 200ms ease,
              background-color {t};
}}
[data-testid="stMetric"]:hover, [data-testid="stExpander"] details:hover {{
  transform: translateY(-2px);
  border-color: var(--ff-hover-border) !important;
}}
[data-testid="stMetric"] {{ padding: 0.9rem 1rem; }}

/* Controls: tactile press; readable text on the accent fill (WCAG AA). */
button {{ transition: transform 120ms ease, background-color {t}, border-color {t},
                     color {t}; }}
button:active {{ transform: scale(0.98); }}
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primary"] * {{
  color: var(--ff-on-accent) !important;
}}
:focus-visible {{ outline: 2px solid var(--ff-accent); outline-offset: 2px; }}

/* Entrance: staggered fade + 8px rise for the page's top-level blocks. */
@keyframes ff-rise {{ from {{ opacity: 0; transform: translateY(8px); }}
                      to {{ opacity: 1; transform: none; }} }}
{block} > [data-testid="stElementContainer"]:not(:has(.ff-theme-pill)),
{block} > [data-testid="stLayoutWrapper"] {{
  animation: ff-rise 350ms ease-out both;
}}
{stagger}
/* Theme pill (top bar): shows the active theme, opens the app menu. */
[data-testid="stElementContainer"]:has(.ff-theme-pill) {{
  position: absolute; width: 0; height: 0; margin: 0; overflow: visible;
}}
.ff-theme-pill {{
  position: fixed; top: 0.7rem; right: 3.6rem; z-index: 999991;
  display: inline-flex; align-items: center; gap: 0.4rem;
  padding: 0.22rem 0.7rem 0.22rem 0.5rem; border-radius: 999px; cursor: pointer;
  font: 500 13px/1.4 {FONT_STACK}; color: var(--ff-text);
  background: var(--ff-accent-soft); border: 1px solid var(--ff-border);
  transition: background-color {t}, border-color 200ms ease, color {t}, transform 120ms ease;
}}
.ff-theme-pill:hover {{ border-color: var(--ff-hover-border); }}
.ff-theme-pill:active {{ transform: scale(0.98); }}
.ff-theme-pill .ff-ico {{ position: relative; width: 1.05rem; height: 1.05rem; }}
.ff-theme-pill .ff-ico span {{
  position: absolute; inset: 0; display: grid; place-items: center;
  transition: transform 450ms cubic-bezier(.2,.8,.2,1), opacity 300ms ease;
}}
.ff-theme-pill .ff-word-dark, .ff-theme-pill .ff-word-light {{ display: none; }}
:root[data-ff-theme="light"] .ff-word-light, :root[data-ff-theme="dark"] .ff-word-dark {{
  display: inline;
}}
:root[data-ff-theme="light"] .ff-moon, :root[data-ff-theme="dark"] .ff-sun {{
  opacity: 0; transform: rotate(-90deg) scale(0.5);
}}
@media (max-width: 640px) {{ .ff-theme-pill .ff-label {{ display: none; }} }}
@media (prefers-reduced-motion: reduce) {{
  * {{ transition: none !important; animation: none !important; }}
}}
"""


# Keeps <html data-ff-theme> equal to the theme Streamlit is showing, read from
# the luminance of the .stApp background (Streamlit sets it; our CSS does not).
# Installed once per browser tab.
#
# The pill is a real toggle.  Streamlit has no API to change the theme, but
# its menu stores the choice in localStorage under
# `stActiveTheme-<path>-v<THEME_CACHE_VERSION>` as "Light" / "Dark" / "System"
# (streamlit 1.66 frontend, utils.*.js).  The toggle writes that value for the
# current path and every stored path, fades to the target background, then
# reloads so Streamlit starts in the chosen theme (no flash of the old one).
# Re-check the key format when upgrading Streamlit; test_dashboard_b0 pins it.
THEME_STORAGE_PREFIX = "stActiveTheme-"
THEME_CACHE_VERSION = 2
_TOGGLE_JS = """
  const toggle = () => {
    const next = root.dataset.ffTheme === 'dark' ? 'light' : 'dark';
    const value = JSON.stringify(next === 'light' ? 'Light' : 'Dark');
    const suffix = '-v__VER__';
    try {
      const keys = new Set([`__PREFIX__${window.location.pathname}${suffix}`]);
      for (let i = 0; i < localStorage.length; i++) {
        const k = localStorage.key(i);
        if (k && k.startsWith('__PREFIX__') && k.endsWith(suffix)) keys.add(k);
      }
      keys.forEach((k) => localStorage.setItem(k, value));
    } catch (err) {
      return;                          // storage blocked: nothing we can switch
    }
    const veil = document.createElement('div');
    veil.style.cssText = 'position:fixed;inset:0;z-index:2147483647;opacity:0;' +
      'pointer-events:none;transition:opacity 220ms ease-in-out;background:' +
      (next === 'light' ? '__LIGHT_BG__' : '__DARK_BG__');
    document.body.appendChild(veil);
    requestAnimationFrame(() => { veil.style.opacity = '1'; });
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    setTimeout(() => window.location.reload(), reduce ? 0 : 240);
  };
"""
_SYNC_JS = """
(() => {
  if (window.__ffThemeSync) { window.__ffThemeSync(); return; }
  const root = document.documentElement;
  const sync = () => {
    const app = document.querySelector('.stApp') || document.body;
    const m = getComputedStyle(app).backgroundColor.match(/[\\d.]+/g);
    if (!m || m.length < 3) return;
    const lum = (0.2126 * m[0] + 0.7152 * m[1] + 0.0722 * m[2]) / 255;
    const mode = lum < 0.5 ? 'dark' : 'light';
    if (root.dataset.ffTheme !== mode) root.dataset.ffTheme = mode;
    const label = mode === 'dark' ? 'Switch to light theme' : 'Switch to dark theme';
    document.querySelectorAll('.ff-theme-pill').forEach((p) => {
      p.setAttribute('aria-label', label);
      p.setAttribute('title', label);
    });
  };
__TOGGLE__
  window.__ffThemeSync = sync;
  sync();
  new MutationObserver(sync).observe(document.head,
    { childList: true, subtree: true, characterData: true });
  setInterval(sync, 1000);
  document.addEventListener('click', (e) => {
    if (e.target.closest && e.target.closest('.ff-theme-pill')) toggle();
  });
})();
"""

_PILL = """
<button type="button" class="ff-theme-pill" title="Switch theme" aria-label="Switch theme">
  <span class="ff-ico" aria-hidden="true"><span class="ff-sun">☀️</span><span class="ff-moon">🌙</span></span>
  <span class="ff-label"><span class="ff-word-light">Light</span><span class="ff-word-dark">Dark</span></span>
</button>
"""


def _sync_js() -> str:
    toggle = (_TOGGLE_JS.replace("__PREFIX__", THEME_STORAGE_PREFIX)
              .replace("__VER__", str(THEME_CACHE_VERSION))
              .replace("__LIGHT_BG__", TOKENS["light"]["background"])
              .replace("__DARK_BG__", TOKENS["dark"]["background"]))
    return _SYNC_JS.replace("__TOGGLE__", toggle)


def page_head() -> str:
    """Stylesheet, theme pill and theme-sync script for st.html in the entry app."""
    mode = current_mode()
    return (f'<style>{css(mode)}</style>{_PILL}'
            f'<script>document.documentElement.dataset.ffTheme ||= "{mode}";{_sync_js()}</script>')


def plotly_template(mode: str | None = None) -> Any:
    """Shared Plotly template: transparent paper, method colours, system font.

    Text, grid and hover colours are left to st.plotly_chart's Streamlit
    theme, which follows a light/dark switch live; setting them here would
    freeze them to the theme of the last rerun.
    """
    import plotly.graph_objects as go  # noqa: PLC0415  (only pages with charts need it)

    del mode  # colours that depend on the theme come from Streamlit's chart theme
    return go.layout.Template(layout=dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_STACK, size=BODY_PX - 2),
        colorway=list(dict.fromkeys(METHOD_COLOURS.values())),
        hoverlabel=dict(font=dict(size=BODY_PX - 2)),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    ))


REFERENCE_LINE = "rgba(127, 138, 153, 0.85)"   # reads on beige and on black
GRID_LINE = {"light": "rgba(19, 34, 56, 0.12)", "dark": "rgba(241, 245, 246, 0.12)"}


def chart_text(mode: str | None = None) -> dict[str, str]:
    """Explicit chart text colours (WCAG AA on the card surface) for the
    current theme; the theme toggle reloads the page, so these stay in step."""
    m = _mode(mode)
    t = TOKENS[m]
    return {"text": t["text"], "muted": t["muted"], "grid": GRID_LINE[m], "card": t["card"],
            "border": t["border"]}


def chart_accent(mode: str | None = None) -> str:
    """Highlight colour for charts (selected features, current frame)."""
    return token("accent", mode)


def heat_scale(mode: str | None = None, gap: bool = False) -> list[list]:
    """Heatmap colourscale from a neutral empty cell to the theme accent; with
    `gap`, zero stays neutral and any positive value starts at a soft tint."""
    empty = "rgba(127, 138, 153, 0.10)"
    if gap:
        return [[0, empty], [0.01, token("accent_soft", mode)], [1, chart_accent(mode)]]
    return [[0, empty], [1, chart_accent(mode)]]
