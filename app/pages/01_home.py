"""Home: what the dashboard is, and the integrity banner (KPIs and verdict in B2)."""

import streamlit as st

from app.components.callout import integrity_banner, placeholder
from app.lib.facts import integrity_status

st.title("Home")
st.markdown(
    "A leakage-safe benchmark of binary Firefly feature selection against standard "
    "baselines on Digits and a CIFAR-10 subset with ResNet-18 embeddings. Every "
    "number on these pages is read from the result files; nothing is recomputed.")
integrity_banner(integrity_status())
placeholder("B2", "headline KPIs and the verdict card, computed from the result tables")
