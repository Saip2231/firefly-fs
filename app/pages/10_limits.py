"""Limits: what the benchmark does not show, from PREREG.md plus computed notes."""

import streamlit as st

from app.lib import data, facts

st.title("Limits")
st.markdown(
    "What this benchmark does not show: statistical power is limited by the number of "
    "seeds, there is a single backbone and CIFAR-10 subset, and search times were "
    "measured under load.")

notes = []
try:
    rq1 = data.load_analysis("rq1_tests")
    n = int(rq1["n"].max())
    notes.append(f"With n={n} seeds the smallest two-sided Wilcoxon p is "
                 f"{facts.min_two_sided_p(n):.4f}; after Holm correction a WIN or LOSS "
                 "needs nearly every seed to agree. A TIE is not evidence of equivalence.")
except data.DataError:
    pass
factor = facts.contention_factor()
if factor:
    notes.append(f"Search times in the result files were measured under 15-way parallel "
                 f"load; on an idle machine the same searches ran about {factor:.2f}x faster "
                 "(A4 contention check).")
notes.append("Firefly does not run on a new image: the subset is chosen once on training "
             "data, then a fixed classifier uses those features.")
notes.append("PCA is a projection, not a selector: it needs the full embedding and a "
             "projection matrix at inference.")
with st.container(border=True):
    st.subheader("Computed notes")
    for note in notes:
        st.markdown(f"- {note}")

prereg = data.read_text(data.ROOT / "PREREG.md")
for heading in ("Limitations accepted in advance", "Disclosed asymmetries",
                "What we will do if Firefly does not win"):
    body = data.md_section(prereg, heading)
    if body:
        st.subheader(heading)
        st.markdown(body)
st.caption("Sections above are rendered from PREREG.md, written before the main runs.")
