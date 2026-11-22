"""Stat tiles; values are computed by app.lib.facts, never typed."""

from __future__ import annotations

import streamlit as st


def kpi_strip(items: list[tuple[str, object, str]], per_row: int | None = None) -> None:
    """items: (label, value, help text); wraps onto rows of `per_row` tiles."""
    per_row = per_row or len(items)
    for start in range(0, len(items), per_row):
        row = items[start:start + per_row]
        for col, (label, value, help_text) in zip(st.columns(per_row), row):
            col.metric(label, value, help=help_text, border=True)
