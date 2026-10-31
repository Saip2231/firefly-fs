"""Row of KPI tiles; values are computed by app.lib.facts, never typed."""

from __future__ import annotations

import streamlit as st


def kpi_strip(items: list[tuple[str, object, str]]) -> None:
    """items: (label, value, help text)."""
    for col, (label, value, help_text) in zip(st.columns(len(items)), items):
        col.metric(label, value, help=help_text, border=True)
