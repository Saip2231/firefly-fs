"""Watch a real search: replay of logged traces plus an illustrative toy."""

import streamlit as st

from app.components.callout import placeholder

st.title("Watch a real search")
st.markdown(
    "Frame-by-frame replays of logged Firefly and PSO searches. Playback starts only "
    "when you press Play.")
placeholder("B6", "population grid replay, best-fitness curve and the match badge")
