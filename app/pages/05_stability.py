"""Stability (RQ3): overlap of selected subsets across seeds."""

import streamlit as st

from app.components.callout import placeholder

st.title("Stability (RQ3)")
st.markdown(
    "How similar the selected feature subsets are across seeds, with random selection "
    "as the reference (exploratory).")
placeholder("B4", "stability bars, feature frequency, and the Digits pixel heatmap")
