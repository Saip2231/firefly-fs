"""Results explorer (RQ1): accuracy vs k and the WIN/LOSS/TIE grid."""

import streamlit as st

from app.components.callout import placeholder

st.title("Results explorer (RQ1)")
st.markdown(
    "The pre-registered comparisons of plain Firefly against each comparator, for "
    "each dataset, k and search budget (paired Wilcoxon, Holm within each family).")
placeholder("B3", "accuracy vs k with filters and the clickable WIN/LOSS/TIE grid")
