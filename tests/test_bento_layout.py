"""Bento layout and typography (golden-ratio grid, self-hosted fonts):
every page opens through the shared hero, cards and splits follow the phi
proportions, and fonts are served locally (DASHBOARD.md rule 8)."""

from __future__ import annotations

import math
import re
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("streamlit")
from app.components import bento  # noqa: E402
from app.lib import layout_css, theme  # noqa: E402

PAGES = sorted((ROOT / "app" / "pages").glob("*.py"))
FONT_FILES = ("sora", "plus-jakarta-sans", "jetbrains-mono")
FAMILIES = {"Sora", "Plus Jakarta Sans", "JetBrains Mono"}


def test_golden_ratio_split():
    assert math.isclose(layout_css.PHI, (1 + math.sqrt(5)) / 2)
    wide, narrow = bento.GOLDEN
    assert math.isclose(wide / (wide + narrow), 0.618, abs_tol=1e-3)
    s = layout_css.SPACE
    assert [s["xs"], s["sm"], s["md"], s["lg"], s["xl"]] == [10, 16, 26, 42, 68]
    for a, b in ((s["sm"], s["md"]), (s["md"], s["lg"]), (s["lg"], s["xl"])):
        assert abs(b / a - layout_css.PHI) < 0.03           # phi rhythm


def test_css_has_container_type_scale_and_breakpoint():
    css = theme.css("light")
    for needle in ("max-width: 1320px", "max-width: 65ch", "clamp(2.25rem, 4vw, 3.5rem)",
                   "clamp(1.5rem, 2.2vw, 2.05rem)", "max-width: 1023px", "padding: 26px",
                   "font-size: 16px", "tabular-nums"):
        assert needle in css, needle
    for family in FAMILIES:
        assert family in css, family
    # Key-value rows keep a minimum label column (the collapsed "F a s t e s t" bug).
    assert "minmax(9rem, 38.2%)" in css


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_every_page_uses_the_shared_hero_and_cards(page):
    text = page.read_text(encoding="utf-8")
    assert "hero(" in text and "st.title(" not in text
    assert "card(" in text
    assert "st.subheader(" not in text       # section titles live inside cards


CHART_PAGES = ("03_results.py", "04_cost.py", "05_stability.py", "06_transfer_ablations.py",
               "07_watch_search.py", "09_digits_live.py")


@pytest.mark.parametrize("name", CHART_PAGES)
def test_charts_are_not_squeezed_into_side_columns(name):
    """Charts sit at full card width with their controls in a row above;
    the golden split is not used on chart pages for chart + controls."""
    text = (ROOT / "app" / "pages" / name).read_text(encoding="utf-8")
    assert "golden(\"acc\")" not in text and "golden(\"scatter\")" not in text
    assert "row(" in text


def test_charts_set_explicit_text_colours():
    pytest.importorskip("plotly")
    from app.lib import charts  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415
    fig = charts.cost_scatter(pd.DataFrame(dict(
        method=["anova", "firefly"], budget=[float("nan"), 800.0], search_s_mean=[0.1, 30.0],
        acc_mean=[0.6, 0.6])), "t")
    for mode in theme.MODES:
        text = theme.chart_text(mode)["text"]
        assert theme.contrast(text, theme.TOKENS[mode]["card"]) >= 4.5
    assert fig.layout.font.color == theme.chart_text()["text"]
    assert fig.layout.xaxis.tickfont.color == theme.chart_text()["text"]
    assert all(t.mode == "markers" for t in fig.data)        # no overlapping text labels


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_every_page_has_a_crisp_lede_and_nerd_detail(page):
    """Nerd mode: each hero has a short lede plus `more=` detail for nerd mode."""
    text = page.read_text(encoding="utf-8")
    assert "more=" in text, "hero has no nerd-mode detail"


def test_nerd_mode_toggle_in_top_bar():
    from app.lib import nerd  # noqa: PLC0415
    head = theme.page_head()
    assert "ff-topbar" in head and "ff-nerd-pill" in head and "ff-theme-pill" in head
    assert nerd.STORAGE_KEY in head and "aria-pressed" in head
    assert "😎" in head and "🤓" in head                     # cool (off) -> nerd (on)
    css = nerd.css(theme.FONT_STACK)
    # crisp by default: nerd content hidden unless <html data-ff-nerd="on">
    assert ':root:not([data-ff-nerd="on"]) [class*="st-key-ff-nerd-"]' in css
    assert ':root[data-ff-nerd="on"] [class*="st-key-ff-crisp-"]' in css
    assert "var(--ff-accent-2)" in css                       # secondary colour
    for mode in theme.MODES:
        t = theme.TOKENS[mode]
        assert theme.contrast(t["accent_2"], t["background"]) >= 4.5, mode


def test_rule_required_text_is_never_nerd_only():
    """DASHBOARD.md rule 5: the one-image anecdote line stays in the crisp view."""
    text = (ROOT / "app" / "pages" / "08_try_image.py").read_text(encoding="utf-8")
    anecdote = text.index("One image is an anecdote")
    before = text[:anecdote]
    assert before.rfind('nerd("') < before.rfind('card("predictions"')


def test_limits_has_no_stretched_split():
    text = (ROOT / "app" / "pages" / "10_limits.py").read_text(encoding="utf-8")
    assert "golden(" not in text


def test_fonts_are_self_hosted():
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert cfg["server"]["enableStaticServing"] is True
    faces = cfg["theme"]["fontFaces"]
    assert {f["family"] for f in faces} == FAMILIES
    for f in faces:
        assert not re.match(r"https?://", f["url"]), f["url"]
        assert f["url"].startswith("app/static/fonts/")
    css = (ROOT / "app" / "frontend" / "src" / "index.css").read_text(encoding="utf-8")
    assert "fonts.googleapis" not in css and "fonts.gstatic" not in css


def test_font_files_present_after_build():
    folder = ROOT / "app" / "static" / "fonts"
    if not folder.exists():
        pytest.skip("fonts not copied yet (npm run build in app/frontend)")
    for name in FONT_FILES:
        path = folder / f"{name}.woff2"
        assert path.exists(), path
        assert path.read_bytes()[:4] == b"wOF2", path
