"""Limits: what the benchmark does not show, from PREREG.md plus computed notes."""

import streamlit as st

from app.components.bento import card, fold, hero, nerd
from app.lib import data, facts

hero("Limits",
     "What this benchmark cannot tell you.",
     eyebrow="📎 About",
     more="What this benchmark does not show: power is limited by the number of seeds, there "
          "is a single backbone and CIFAR-10 subset, and search times were measured under "
          "load.")

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

# Full-width cards sized to their content: no stretched, half-empty columns.
with card("computed", "Computed notes", eyebrow="Computed from the results"):
    st.markdown("\n".join(f"- {note}" for note in notes))

prereg = data.read_text(data.ROOT / "PREREG.md")
with nerd("prereg"):
    for key, heading in (("accepted", "Limitations accepted in advance"),
                         ("asymmetries", "Disclosed asymmetries"),
                         ("plan", "What we will do if Firefly does not win")):
        with fold(key, heading, eyebrow="📌 From PREREG.md (written before the main runs)"):
            st.markdown(data.md_section(prereg, heading) or "Not found in PREREG.md.")
