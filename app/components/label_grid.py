"""Clickable WIN/LOSS/TIE grid: a chart click selects the family panel;
the two select boxes below do the same for keyboard users."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.lib import charts
from app.lib.methods import display


def label_grid(grid: pd.DataFrame, key: str = "rq1") -> pd.Series:
    """Render the grid and return the selected rq1 row."""
    event = st.plotly_chart(charts.label_grid(grid), on_select="rerun",
                            selection_mode="points", key=f"{key}_grid")
    points = event.get("selection", {}).get("points", []) if event else []
    clicked = tuple(points[0]["customdata"][:2]) if points and points[0].get("customdata") else None
    # The chart keeps its selection across reruns: apply a click only once, so
    # the select boxes below stay usable afterwards.
    if clicked and clicked != st.session_state.get(f"{key}_last_click"):
        st.session_state[f"{key}_last_click"] = clicked
        st.session_state[f"{key}_family"], st.session_state[f"{key}_comparator"] = clicked
    families = list(dict.fromkeys(grid["family"]))
    comparators = list(dict.fromkeys(grid["comparator"]))
    c1, c2 = st.columns(2)
    family = c1.selectbox("Family (dataset, k, budget)", families, key=f"{key}_family")
    comparator = c2.selectbox("Comparator", comparators, format_func=display,
                              key=f"{key}_comparator")
    return grid[(grid["family"] == family) & (grid["comparator"] == comparator)].iloc[0]
