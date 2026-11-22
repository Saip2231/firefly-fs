"""Protocol: split, selection on training data only, inner validation, single
test use; the pre-registered design and the deviations register."""

from string import Template

import streamlit as st

from app.components.bento import card, fold, hero, kv
from app.lib import data, pipeline_view, theme

# Static fallback when the animated bundle is not built; colours from the
# current theme's tokens.
DOT = Template("""
digraph protocol {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="$card", color="$line",
        fontcolor="$text", fontname="Helvetica"];
  edge [color="$muted", fontcolor="$muted", fontname="Helvetica", fontsize=10];
  data [label="Dataset\\n(Digits or CIFAR-10 subset,\\nResNet-18 embeddings)"];
  split [label="Stratified train / test split\\n(one per seed, same for every method)"];
  train [label="Training split\\n(scaler fitted here only)"];
  test [label="Test split\\n(locked)", color="$lock"];
  base [label="Baseline selectors\\nANOVA, MI, L1, RF; PCA projection\\n(fit on training rows)"];
  inner [label="Inner split of training rows\\ninner-train / inner-validation"];
  search [label="Firefly / PSO search\\nfitness = inner-validation error\\n(budget of new fits)", color="$accent"];
  mask [label="Feature subset (exactly k)"];
  final [label="Final logistic regression\\non the full training split"];
  score [label="Scored ONCE on the test split", color="$lock"];
  data -> split; split -> train; split -> test [style=dashed, label="not used\\nuntil the end"];
  train -> base -> mask; train -> inner -> search -> mask;
  mask -> final -> score; test -> score [style=dashed];
}
""")

hero("Protocol",
     "How the benchmark was run: fixed splits per seed, feature selection on training data "
     "only with an inner validation split, and the test split used once at the end.",
     eyebrow="⚡ Overview")

# The animated visualizer is its own bento card (border, surface, padding).
try:
    drawn = pipeline_view.render()
except data.DataError as exc:
    with card("pipeline-error", "Animated diagram unavailable"):
        st.warning(str(exc))
    drawn = False
if not drawn:      # bundle not built (npm run build in app/frontend) or data missing
    with card("pipeline", "Protocol diagram", eyebrow="⚡ Pipeline"):
        tok = theme.TOKENS[theme.current_mode()]
        st.graphviz_chart(DOT.substitute(
            card=tok["card"], line=tok["muted"], text=tok["text"], muted=tok["muted"],
            lock=tok["accent_text"], accent=tok["accent"]))

prereg = data.read_text(data.ROOT / "PREREG.md")
design = data.md_section(prereg, "Design fixed before data")
primary = data.md_section(prereg, "Primary comparison (RQ1)")

register = data.read_text(data.ROOT / "app" / "content" / "deviations.md")

# The short summary stays open; the long documents fold away behind a chevron.
with card("discipline", "The test split is used once", eyebrow="🔒 Locked test split"):
    st.markdown("Selection never sees the test split. The search scores candidate subsets "
                "on an inner validation split carved from training rows.")
    kv([("Scaler fitted on", "training rows"), ("Search fitness", "inner validation"),
        ("Test split touched", "once, final model")])

with fold("design", "Pre-registered design", eyebrow="📌 From PREREG.md"):
    st.markdown(design or "PREREG.md not found; the pre-registered design cannot be shown.")
with fold("primary", "Primary comparison (RQ1)", eyebrow="✓ Confirmatory"):
    st.markdown(primary or "PREREG.md not found.")
with fold("deviations", "Deviations register", eyebrow="📎 Disclosed changes"):
    st.markdown(register or "app/content/deviations.md not found.")
