"""Cost (RQ2): search time, inference time, break-even."""

import streamlit as st

from app.components.callout import placeholder

st.title("Cost (RQ2)")
st.markdown(
    "Search time, inference time and break-even (exploratory). Search times were "
    "measured under 15-way parallel load.")
placeholder("B4", "cost vs accuracy scatter and the break-even calculator")
