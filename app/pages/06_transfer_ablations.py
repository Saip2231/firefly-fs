"""Transfer (RQ4) and ablations: all exploratory."""

import streamlit as st

from app.components.callout import placeholder

st.title("Transfer and ablations")
st.markdown(
    "How subsets chosen under logistic regression transfer to KNN and linear SVM, and "
    "the effect of seeding, re-ranking and search budget (all exploratory).")
placeholder("B5", "transfer panels, budget curve, seeding and re-rank effects")
