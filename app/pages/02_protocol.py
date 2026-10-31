"""Protocol: split, selection on training data only, inner validation, single
test use; the pre-registered design and the deviations register."""

import streamlit as st

from app.lib import data

DOT = """
digraph protocol {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#161B22", color="#30363D",
        fontcolor="#E6EDF3", fontname="Helvetica"];
  edge [color="#9DA7B3", fontcolor="#9DA7B3", fontname="Helvetica", fontsize=10];
  data [label="Dataset\\n(Digits or CIFAR-10 subset,\\nResNet-18 embeddings)"];
  split [label="Stratified train / test split\\n(one per seed, same for every method)"];
  train [label="Training split\\n(scaler fitted here only)"];
  test [label="Test split\\n(locked)", color="#D55E00"];
  base [label="Baseline selectors\\nANOVA, MI, L1, RF; PCA projection\\n(fit on training rows)"];
  inner [label="Inner split of training rows\\ninner-train / inner-validation"];
  search [label="Firefly / PSO search\\nfitness = inner-validation error\\n(budget of new fits)", color="#E69F00"];
  mask [label="Feature subset (exactly k)"];
  final [label="Final logistic regression\\non the full training split"];
  score [label="Scored ONCE on the test split", color="#D55E00"];
  data -> split; split -> train; split -> test [style=dashed, label="not used\\nuntil the end"];
  train -> base -> mask; train -> inner -> search -> mask;
  mask -> final -> score; test -> score [style=dashed];
}
"""

st.title("Protocol")
st.markdown(
    "How the benchmark was run: fixed splits per seed, feature selection on training "
    "data only with an inner validation split, and the test split used once at the end.")
st.graphviz_chart(DOT)
st.caption("Selection never sees the test split. The search scores candidate subsets on "
           "an inner validation split carved from training rows; the test split is "
           "touched once, by the final classifier.")

prereg = data.read_text(data.ROOT / "PREREG.md")
design = data.md_section(prereg, "Design fixed before data")
primary = data.md_section(prereg, "Primary comparison (RQ1)")
if design and primary:
    with st.expander("Pre-registered design (from PREREG.md)", expanded=True):
        st.markdown(design)
    with st.expander("Primary comparison (from PREREG.md)"):
        st.markdown(primary)
else:
    st.info("PREREG.md not found; the pre-registered design cannot be shown.")

st.subheader("Deviations register")
register = data.read_text(data.ROOT / "app" / "content" / "deviations.md")
if register:
    st.markdown(register)
else:
    st.info("app/content/deviations.md not found.")
