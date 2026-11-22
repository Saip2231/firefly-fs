"""Typography and bento-grid CSS (golden-ratio layout), used by theme.css().

Type: Sora for the page title (h1, 800) and for section and card titles
(600-700), Plus Jakarta Sans for body copy, JetBrains Mono for numbers,
badges and eyebrows (always tabular numerals).  Sizes step by sqrt(phi).

Layout: a centred 1320 px container; every block lives in a bento card
(app/components/bento.py) with 26 px padding; two-column sections split
61.8 % / 38.2 % (1.618fr : 1fr) and stack below 1024 px.  Spacing follows a
phi rhythm: 10 / 16 / 26 / 42 / 68 px.

Selectors rely on Streamlit's data-testid attributes and on the
`st-key-<key>` class that keyed containers receive (streamlit >= 1.37).
"""

from __future__ import annotations

PHI = (1 + 5 ** 0.5) / 2
SPACE = {"xs": 10, "sm": 16, "md": 26, "lg": 42, "xl": 68}   # px, phi rhythm

CARD = '[class*="st-key-ff-card-"]'
SPLIT = '[class*="st-key-ff-split-"]'
HERO = '[class*="st-key-ff-hero"]'
MAIN = '[data-testid="stMainBlockContainer"]'


def layout_css(body_px: int, fonts: dict[str, str]) -> str:
    s = SPACE
    return f"""
:root {{
  --font-display: {fonts["display"]};
  --font-heading: {fonts["heading"]};
  --font-body: {fonts["body"]};
  --font-mono: {fonts["mono"]};
}}
html, body, [data-testid="stAppViewContainer"] {{
  font-family: var(--font-body);
  font-size: {body_px}px;
}}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{
  line-height: 1.65;
}}

/* Type scale (sqrt(phi) steps). */
h1 {{
  font-family: var(--font-display) !important; font-weight: 800 !important;
  font-size: clamp(2.25rem, 4vw, 3.5rem) !important; line-height: 1.08 !important;
  letter-spacing: -0.03em !important;
}}
h2, h3, h4 {{
  font-family: var(--font-heading) !important; letter-spacing: -0.02em !important;
  line-height: 1.25 !important;
}}
h2 {{ font-size: clamp(1.5rem, 2.2vw, 2.05rem) !important; font-weight: 700 !important; }}
h3 {{ font-size: 1.25rem !important; font-weight: 600 !important; }}
h4 {{ font-size: 1.05rem !important; font-weight: 600 !important; }}
[data-testid="stHeader"] a, [data-testid="stHeader"] button,
[data-testid="stHeader"] [role="tab"] {{ font-family: var(--font-heading); }}
table, code, [data-testid="stDataFrame"], [data-testid="stMetricValue"],
[data-testid="stMetricDelta"], .ff-num {{
  font-family: var(--font-mono); font-variant-numeric: tabular-nums;
}}
[data-testid="stMetricValue"] {{ letter-spacing: -0.01em; }}
[data-testid="stMetricLabel"] p {{
  font-family: var(--font-mono); font-size: 0.75rem; font-weight: 500;
  letter-spacing: 0.08em; text-transform: uppercase;
}}

/* Container: centred 1320 px, 24 / 40 px gutters, phi section rhythm. */
{MAIN} {{
  max-width: 1320px; margin: 0 auto;
  padding: {s["lg"]}px 24px {s["xl"]}px !important;
}}
@media (min-width: 768px) {{
  {MAIN} {{ padding-left: 40px !important; padding-right: 40px !important; }}
}}
{MAIN} > div > [data-testid="stVerticalBlock"] {{ gap: {s["md"]}px; }}

/* Hero: the only text allowed straight on the page background. */
{HERO} {{ gap: {s["sm"]}px; padding-bottom: {s["sm"]}px; }}
{HERO} h1 {{ padding: 0 !important; margin: 0 0 6px !important; }}
.ff-lede {{
  max-width: 65ch; margin: 0; color: var(--ff-muted);
  font-size: 1.075rem; line-height: 1.65;
}}
.ff-lede-more {{ margin-top: {s["xs"]}px; font-size: 1rem; }}

/* Headings inside markdown (PREREG excerpts, notes): room before the first line. */
[data-testid="stMarkdownContainer"] :is(h2, h3, h4):not(.ff-card-title) {{
  padding-bottom: 0 !important; margin-bottom: 0.55em !important;
}}
[data-testid="stMarkdownContainer"] :is(h2, h3, h4):not(.ff-card-title):not(:first-child) {{
  margin-top: 1.1em !important;
}}
{CARD} [data-testid="stMarkdownContainer"] > :is(h2, h3, h4):first-child {{
  padding-top: 0 !important;
}}

/* Bento card: surface, 1px border, 26 px padding, 16 px inner rhythm. */
{CARD} {{
  background: var(--ff-card) !important;
  border: 1px solid var(--ff-border) !important;
  border-radius: 1rem !important;
  padding: {s["md"]}px !important;
  box-shadow: var(--ff-shadow);
  gap: {s["sm"]}px;
}}
{CARD} [data-testid="stMarkdownContainer"] p,
{CARD} [data-testid="stMarkdownContainer"] li,
{CARD} [data-testid="stCaptionContainer"] p {{ max-width: 65ch; }}
{CARD} [data-testid="stMetric"], {CARD} [data-testid="stExpander"] details {{
  background: var(--ff-background); box-shadow: none;
}}
/* Card head: eyebrow, title, then 26 px (16 gap + 10 margin) to the first line. */
.ff-card-head {{
  display: flex; flex-direction: column; gap: 8px; margin-bottom: {s["xs"]}px;
}}
.ff-card-head h3.ff-card-title {{ padding: 0 !important; margin: 0 !important; }}
.st-key-ff-nerd-prereg {{ gap: {s["md"]}px; }}
.ff-eyebrow {{
  display: inline-flex; align-items: center; gap: 8px; margin: 0;
  font: 500 0.75rem/1.2 var(--font-mono); letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--ff-muted);
}}
.ff-eyebrow.ff-dot::before {{
  content: ""; width: 6px; height: 6px; border-radius: 50%; background: var(--ff-accent);
}}
.ff-card-title {{
  margin: 0; font: 600 1.25rem/1.25 var(--font-heading); letter-spacing: -0.02em;
  color: var(--ff-text);
}}

/* Key-value rows: the label column never collapses (min 9rem, 38.2 %), long
   values wrap instead of squeezing the label. */
.ff-kv {{
  display: grid; grid-template-columns: minmax(9rem, 38.2%) minmax(0, 1fr);
  gap: {s["xs"]}px {s["sm"]}px; margin: 0;
}}
.ff-kv dt {{ color: var(--ff-muted); margin: 0; overflow-wrap: normal; word-break: normal; }}
.ff-kv dd {{
  margin: 0; text-align: right; color: var(--ff-text); overflow-wrap: anywhere;
  font-family: var(--font-mono); font-variant-numeric: tabular-nums; font-size: 0.9rem;
}}

/* Foldable card (Protocol page): the expander is the card, its chevron the toggle. */
[class*="st-key-ff-fold-"] [data-testid="stExpander"] details {{
  background: var(--ff-card) !important; border: 1px solid var(--ff-border) !important;
  border-radius: 1rem !important; box-shadow: var(--ff-shadow);
}}
[class*="st-key-ff-fold-"] [data-testid="stExpander"] details:hover {{ transform: none; }}
[class*="st-key-ff-fold-"] [data-testid="stExpander"] summary {{ padding: 18px {s["md"]}px; }}
[class*="st-key-ff-fold-"] [data-testid="stExpander"] summary p {{
  font: 600 1.25rem/1.25 var(--font-heading) !important; letter-spacing: -0.02em;
  color: var(--ff-text);
}}
[class*="st-key-ff-fold-"] [data-testid="stExpanderDetails"] {{
  padding: 0 {s["md"]}px {s["md"]}px !important;
}}
[class*="st-key-ff-fold-"] .ff-fold-meta {{ margin: 0 0 8px 4px; }}
[class*="st-key-ff-fold-"] [data-testid="stExpanderDetails"] > div {{ padding-top: 6px; }}

/* Charts: full card width; controls sit in one horizontal row above them. */
{CARD} [data-testid="stPlotlyChart"] {{ width: 100%; }}
.ff-kv dt, .ff-kv dd {{
  padding-bottom: {s["xs"]}px; border-bottom: 1px solid var(--ff-border);
}}
.ff-pill {{
  display: inline-flex; align-items: center; gap: 6px; padding: 2px 10px;
  border-radius: 999px; border: 1px solid var(--ff-border); background: var(--ff-accent-soft);
  font: 500 0.75rem/1.6 var(--font-mono); color: var(--ff-text);
}}

/* Golden split: 1.618fr : 1fr, 26 px gutter, equal-height cards, stacks < 1024 px. */
{SPLIT} [data-testid="stHorizontalBlock"] {{ gap: {s["md"]}px !important; align-items: stretch; }}
{SPLIT} [data-testid="stColumn"] > [data-testid="stVerticalBlock"] {{
  height: 100%; gap: {s["md"]}px;
}}
{SPLIT} [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > :last-child {{
  flex: 1 1 auto; display: flex; flex-direction: column;
}}
{SPLIT} [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > :last-child > {CARD},
{SPLIT} [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > {CARD}:last-child {{
  flex: 1 1 auto;
}}
@media (max-width: 1023px) {{
  {SPLIT} [data-testid="stHorizontalBlock"] {{ flex-direction: column; flex-wrap: wrap; }}
  {SPLIT} [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
    width: 100% !important; min-width: 100% !important; flex: 1 1 100% !important;
  }}
}}
{MAIN} img, {MAIN} iframe {{ max-width: 100%; }}
"""
