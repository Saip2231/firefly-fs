"""Limits: what the benchmark does not show, from PREREG.md plus computed notes."""

import streamlit as st

from app.components.bento import card, golden, hero
from app.lib import data, facts

hero("Limits",
     "What this benchmark does not show: power is limited by the number of seeds, there is "
     "a single backbone and CIFAR-10 subset, and search times were measured under load.",
     eyebrow="📎 About")

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

prereg = data.read_text(data.ROOT / "PREREG.md")
sections = {h: data.md_section(prereg, h) for h in (
    "Limitations accepted in advance", "Disclosed asymmetries",
    "What we will do if Firefly does not win")}

left, right = golden("limits")
with left, card("accepted", "Limitations accepted in advance", eyebrow="📌 From PREREG.md"):
    body = sections["Limitations accepted in advance"]
    st.markdown(body if body else "Not found in PREREG.md.")
with right:
    with card("computed", "Computed notes", eyebrow="Computed from the results"):
        for note in notes:
            st.markdown(f"- {note}")
    with card("asymmetries", "Disclosed asymmetries", eyebrow="📌 From PREREG.md"):
        body = sections["Disclosed asymmetries"]
        st.markdown(body if body else "Not found in PREREG.md.")

with card("plan", "What we will do if Firefly does not win", eyebrow="📌 From PREREG.md"):
    body = sections["What we will do if Firefly does not win"]
    st.markdown(body if body else "Not found in PREREG.md.")
    st.caption("PREREG.md sections were written before the main runs.")
