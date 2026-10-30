"""Design tokens, page CSS and the shared Plotly template (DASHBOARD.md section 5).

Colours are Okabe-Ito (colour-blind safe).  Colour is never the only signal:
labels always carry a symbol and a word, PCA is also dashed.
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
# so it uses the theme's text colour (black in light mode) plus the dash.
PCA_DASH = "dash"

# Label badges: the fill carries the colour, the text is always dark so the
# word itself meets 4.5:1 on every fill (white on #D55E00 would not).
LABEL_STYLE: dict[str, dict[str, str]] = {
    "WIN": {"fill": "#009E73", "symbol": "✓", "text": "#0E1117"},
    "LOSS": {"fill": "#D55E00", "symbol": "✗", "text": "#0E1117"},
    "TIE": {"fill": "#BDBDBD", "symbol": "=", "text": "#0E1117"},
}

SURFACES: dict[str, dict[str, str]] = {
    "dark": {"background": "#0E1117", "card": "#161B22", "border": "#30363D",
             "text": "#E6EDF3", "muted": "#9DA7B3"},
    "light": {"background": "#FFFFFF", "card": "#F6F8FA", "border": "#D0D7DE",
              "text": "#1F2328", "muted": "#57606A"},
}
ACCENT = "#E69F00"

FONT_STACK = ('system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", '
              'Arial, sans-serif')
BODY_PX = 16
HEADING_SCALE = 1.25
TRANSITION_MS = 600


def method_colour(method: str, mode: str = "dark") -> str:
    """Line/marker colour for a method; PCA follows the text colour."""
    if method == "pca":
        return SURFACES[mode]["text"]
    return METHOD_COLOURS.get(method, SURFACES[mode]["muted"])


def _luminance(hex_colour: str) -> float:
    rgb = [int(hex_colour.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(fg: str, bg: str) -> float:
    """WCAG 2 contrast ratio between two hex colours."""
    hi, lo = sorted((_luminance(fg), _luminance(bg)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def css() -> str:
    """Small global stylesheet: type scale, tabular numerals, focus, motion."""
    h = {n: BODY_PX * HEADING_SCALE ** p for n, p in (("h1", 3), ("h2", 2), ("h3", 1))}
    return f"""
html, body, [data-testid="stAppViewContainer"] {{
  font-family: {FONT_STACK};
  font-size: {BODY_PX}px;
}}
h1 {{ font-size: {h['h1']:.1f}px; }}
h2 {{ font-size: {h['h2']:.1f}px; }}
h3 {{ font-size: {h['h3']:.1f}px; }}
table, [data-testid="stDataFrame"], [data-testid="stMetricValue"] {{
  font-variant-numeric: tabular-nums;
}}
:focus-visible {{ outline: 2px solid {ACCENT}; outline-offset: 2px; }}
@media (prefers-reduced-motion: reduce) {{
  * {{ transition: none !important; animation: none !important; }}
}}
"""


def plotly_template(mode: str = "dark") -> Any:
    """Shared Plotly template: transparent paper, subtle grid, method colours."""
    import plotly.graph_objects as go  # noqa: PLC0415  (only pages with charts need it)

    s = SURFACES[mode]
    axis = dict(gridcolor=s["border"], zerolinecolor=s["border"], linecolor=s["border"],
                tickfont=dict(color=s["text"]), title=dict(font=dict(color=s["text"])))
    return go.layout.Template(layout=dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_STACK, size=BODY_PX - 2, color=s["text"]),
        colorway=list(dict.fromkeys(METHOD_COLOURS.values())),
        xaxis=axis, yaxis=axis,
        hoverlabel=dict(bgcolor=s["card"], bordercolor=s["border"],
                        font=dict(color=s["text"], size=BODY_PX - 2)),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    ))
