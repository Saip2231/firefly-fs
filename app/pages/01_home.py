"""Home: what the benchmark is, KPIs, the computed verdict, integrity banner."""

import streamlit as st

from app.components.bento import card, golden, grid, hero, kv
from app.components.callout import data_error, integrity_banner, takeaway
from app.components.kpi_strip import kpi_strip
from app.lib import data, facts
from app.lib.methods import DATASET_DISPLAY, display

hero("Home",
     "A leakage-safe benchmark of binary Firefly feature selection against standard "
     "baselines on Digits and a CIFAR-10 subset with ResNet-18 embeddings. Every number "
     "here is read from the result files; nothing is recomputed.",
     eyebrow="⚡ Overview")

try:
    rq1 = data.load_analysis("rq1_tests")
    desc = data.load_analysis("descriptive")
    rows = facts.raw_row_counts()
except data.DataError as err:
    with card("error", "Data problem"):
        data_error(err)
    st.stop()

counts = facts.rq1_label_counts(rq1)
shape = facts.rq1_shape(rq1)
n_seeds = int(rq1["n"].max())

with card("kpis", "The benchmark at a glance", eyebrow="📊 Benchmark"):
    kpi_strip([
        ("Result rows", f"{rows['meta']:,}", "Metaheuristic runs in results/meta_*.csv "
         "(raw and re-ranked masks)"),
        ("Baseline rows", f"{rows['baselines']:,}", "Rows in results/baselines_*.csv"),
        ("Pre-registered tests", shape["tests"], "Families x comparators in rq1_tests.csv"),
        ("Seeds per test", n_seeds, "Paired by seed; limited power"),
        ("✓ WIN", counts["WIN"], "Holm-adjusted p < 0.05, Firefly higher"),
        ("✗ LOSS", counts["LOSS"], "Holm-adjusted p < 0.05, Firefly lower"),
        ("= TIE", counts["TIE"], "No detectable difference at n seeds"),
    ], per_row=4)

left, right = golden("verdict")
with left, card("verdict", "Verdict (RQ1, confirmatory)", eyebrow="✓ Confirmatory"):
    for line in facts.rq1_verdict(rq1):
        st.markdown(line)
with right, card("integrity", "Integrity and scope", eyebrow="🔒 Leakage-safe"):
    integrity_banner(facts.integrity_status())
    kv([("Datasets", len(data.DATASETS)), ("Seeds", n_seeds),
        ("Tests (families x comparators)", f"{shape['families']} x {shape['comparators']}")])
    st.caption("Feature selection happens once, on training data: Firefly picks a subset "
               "of features, it does not run on a new image.")

budget = int(desc["budget"].max())
for col, dataset in zip(grid("top", len(data.DATASETS)), data.DATASETS):
    top = facts.top_method_per_k(desc, dataset, budget)
    with col, card(f"top-{dataset}", DATASET_DISPLAY[dataset],
                   eyebrow=f"Highest mean accuracy at each k · budget {budget}"):
        st.dataframe(
            top.assign(method=top["method"].map(display)).rename(
                columns={"mean": "mean accuracy", "std": "std over seeds"}),
            hide_index=True, column_config={
                "mean accuracy": st.column_config.NumberColumn(format="%.3f"),
                "std over seeds": st.column_config.NumberColumn(format="%.3f")})
        leader = top["method"].value_counts()
        takeaway(f"{display(leader.index[0])} has the highest mean at {int(leader.iloc[0])} "
                 f"of {len(top)} k values; means only, see Results explorer for the tests.")
