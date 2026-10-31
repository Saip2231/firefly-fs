"""Home: what the benchmark is, KPIs, the computed verdict, integrity banner."""

import streamlit as st

from app.components.callout import data_error, integrity_banner, takeaway
from app.components.kpi_strip import kpi_strip
from app.lib import data, facts
from app.lib.methods import DATASET_DISPLAY, display

st.title("Home")
st.markdown(
    "A leakage-safe benchmark of binary Firefly feature selection against standard "
    "baselines on Digits and a CIFAR-10 subset with ResNet-18 embeddings. Every "
    "number on these pages is read from the result files; nothing is recomputed. "
    "Feature selection happens once, on training data: Firefly picks a subset of "
    "features, it does not run on a new image.")
integrity_banner(facts.integrity_status())

try:
    rq1 = data.load_analysis("rq1_tests")
    desc = data.load_analysis("descriptive")
    rows = facts.raw_row_counts()
except data.DataError as err:
    data_error(err)
    st.stop()

counts = facts.rq1_label_counts(rq1)
shape = facts.rq1_shape(rq1)
n_seeds = int(rq1["n"].max())
kpi_strip([
    ("Result rows", f"{rows['meta']:,}", "Metaheuristic runs in results/meta_*.csv "
     "(raw and re-ranked masks)"),
    ("Baseline rows", f"{rows['baselines']:,}", "Rows in results/baselines_*.csv"),
    ("Pre-registered tests", shape["tests"], "Families x comparators in rq1_tests.csv"),
    ("✓ WIN", counts["WIN"], "Holm-adjusted p < 0.05, Firefly higher"),
    ("✗ LOSS", counts["LOSS"], "Holm-adjusted p < 0.05, Firefly lower"),
    ("= TIE", counts["TIE"], "No detectable difference at n seeds"),
    ("Seeds per test", n_seeds, "Paired by seed; limited power"),
])

with st.container(border=True):
    st.subheader("Verdict (RQ1, confirmatory)")
    for line in facts.rq1_verdict(rq1):
        st.markdown(line)

st.subheader("Highest mean accuracy at each k")
budget = int(desc["budget"].max())
cols = st.columns(len(data.DATASETS))
for col, dataset in zip(cols, data.DATASETS):
    top = facts.top_method_per_k(desc, dataset, budget)
    with col:
        st.markdown(f"**{DATASET_DISPLAY[dataset]}** (metaheuristics at budget {budget})")
        st.dataframe(
            top.assign(method=top["method"].map(display)).rename(
                columns={"mean": "mean accuracy", "std": "std over seeds"}),
            hide_index=True, column_config={
                "mean accuracy": st.column_config.NumberColumn(format="%.3f"),
                "std over seeds": st.column_config.NumberColumn(format="%.3f")})
        leader = top["method"].value_counts()
        takeaway(f"{display(leader.index[0])} has the highest mean at {int(leader.iloc[0])} "
                 f"of {len(top)} k values; means only, see Results explorer for the tests.")
