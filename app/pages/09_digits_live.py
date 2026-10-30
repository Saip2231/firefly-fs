"""Playground: capped live search on the Digits training split."""

import streamlit as st

from app.components.callout import placeholder

st.title("Digits live and build your own")
st.markdown(
    "Run a small, capped search live on the Digits training split and score it on a "
    "holdout carved from that training split.")
placeholder("B8", "live digit search, build-your-own sliders and the overfitting meter")
