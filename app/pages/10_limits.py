"""Limits: what the benchmark does not show."""

import streamlit as st

from app.components.callout import placeholder

st.title("Limits")
st.markdown(
    "What this benchmark does not show: statistical power is limited by the number of "
    "seeds, there is a single backbone and CIFAR-10 subset, and search times were "
    "measured under load.")
placeholder("B2", "limits and deviations, rendered from the project's markdown files")
