"""Bento layout primitives (styles in app/lib/layout_css.py).

    hero(...)                      page title band (the only text on the page background)
    with card("key", "Title", eyebrow="📊 Results"): ...   one bento box
    left, right = golden("key")    61.8 % / 38.2 % split; stacks below 1024 px
    a, b, c = grid("key", 3)       equal columns for supporting cards
    kv([("Label", "value"), ...])  key-value rows with mono, tabular values

Keys only need to be unique within a page; they become `st-key-ff-*`
classes that the CSS targets.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from html import escape

import streamlit as st

from app.lib.layout_css import PHI

GOLDEN = (PHI, 1.0)


def _eyebrow(text: str) -> str:
    """Eyebrow tag; a coloured dot stands in when the text has no emoji."""
    has_symbol = bool(text) and not text[0].isascii()
    cls = "ff-eyebrow" if has_symbol else "ff-eyebrow ff-dot"
    return f'<p class="{cls}">{escape(text)}</p>'


def hero(title: str, lede: str, eyebrow: str) -> None:
    """Eyebrow, page title (st.title, so it stays the page's h1) and a lede of
    two or three lines at most."""
    with st.container(key="ff-hero"):
        st.markdown(_eyebrow(eyebrow), unsafe_allow_html=True)
        st.title(title, anchor=False)
        st.markdown(f'<p class="ff-lede">{escape(lede)}</p>', unsafe_allow_html=True)


@contextmanager
def card(key: str, title: str | None = None, eyebrow: str | None = None) -> Iterator[None]:
    """One bento box: optional eyebrow and title, then whatever the block adds."""
    with st.container(border=True, key=f"ff-card-{key}"):
        if title or eyebrow:
            head = (_eyebrow(eyebrow) if eyebrow else "") + (
                f'<h3 class="ff-card-title">{escape(title)}</h3>' if title else "")
            st.markdown(f'<div class="ff-card-head">{head}</div>', unsafe_allow_html=True)
        yield


def golden(key: str, reverse: bool = False) -> list:
    """Two columns in the golden ratio (wide first unless `reverse`)."""
    with st.container(key=f"ff-split-{key}"):
        return st.columns(GOLDEN[::-1] if reverse else GOLDEN)


def grid(key: str, n: int) -> list:
    """`n` equal columns with the bento gutter."""
    with st.container(key=f"ff-split-{key}"):
        return st.columns(n)


def row(n: int | Sequence[float]) -> list:
    """Controls laid out horizontally (one row above a full-width chart)."""
    return st.columns(n, gap="medium", vertical_alignment="bottom")


@contextmanager
def fold(key: str, title: str, eyebrow: str | None = None,
         expanded: bool = False) -> Iterator[None]:
    """A card that collapses: the chevron (⌄) in its header shows or hides
    the body.  Used where a page would otherwise be a wall of text."""
    with st.container(key=f"ff-fold-{key}"):
        if eyebrow:
            st.markdown(_eyebrow(eyebrow).replace('class="', 'class="ff-fold-meta ', 1),
                        unsafe_allow_html=True)
        with st.expander(title, expanded=expanded):
            yield


def kv(rows: Sequence[tuple[str, object]]) -> None:
    """Structured key-value rows (labels left, mono values right)."""
    body = "".join(f"<dt>{escape(str(k))}</dt><dd>{escape(str(v))}</dd>" for k, v in rows)
    st.markdown(f'<dl class="ff-kv">{body}</dl>', unsafe_allow_html=True)


def pills(items: Sequence[str]) -> None:
    """A row of small pill badges (status, tags)."""
    body = " ".join(f'<span class="ff-pill">{escape(i)}</span>' for i in items)
    st.markdown(f'<p style="display:flex;flex-wrap:wrap;gap:10px;margin:0">{body}</p>',
                unsafe_allow_html=True)
