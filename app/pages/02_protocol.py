"""Protocol: splits, selection on training data only, single test use."""

import streamlit as st

from app.components.callout import placeholder

st.title("Protocol")
st.markdown(
    "How the benchmark was run: fixed splits per seed, feature selection on training "
    "data only with an inner validation split, and the test split used once at the end.")
placeholder("B2", "protocol diagram and the deviations register")
