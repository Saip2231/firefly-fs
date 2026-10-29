"""Firefly feature-selection benchmark dashboard (local only; not deployed).

Run from firefly-fs/ with the venv active:
    streamlit run app/streamlit_app.py

Tabs
  Overview            what exists in results/, baseline accuracy vs k
  Benchmark explorer  results/baselines_<ds>.csv + results/meta_<ds>.csv
                      (the second file may not exist yet)
  Live playground     one Firefly run on Digits or a planted toy, hard cap
                      of 300 evaluations, convergence from binary_firefly's
                      history

Reads only results/*.csv; all computation goes through src/ (via
app/dashboard_data.py).  The test split is touched only by
dashboard_data._score_on_test, through src.evaluate.evaluate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dashboard_data as dd  # noqa: E402

st.set_page_config(page_title="Firefly benchmark", layout="wide")


@st.cache_data
def load_csv(name: str) -> pd.DataFrame | None:
    return dd.read_results_csv(name)


@st.cache_data(show_spinner="Running Firefly (capped at 300 evaluations)...")
def cached_run(source: str, k: int, budget: int, n_fireflies: int, seed: int) -> dict:
    return dd.run_playground(source, k, budget, n_fireflies, seed)


def takeaway(text: str) -> None:
    st.caption(f"**Takeaway:** {text}")


def acc_vs_k_figure(summary: pd.DataFrame, title: str) -> go.Figure:
    sel = summary[~summary.method.isin(dd.REFERENCE_METHODS)].sort_values("k")
    fig = px.line(sel, x="k", y="mean", color="method", error_y="std", markers=True,
                  title=title, labels={"mean": "test accuracy (mean ± std over seeds)"})
    ref = summary[summary.method.isin(dd.REFERENCE_METHODS)]
    for _, r in ref.iterrows():
        fig.add_hline(y=r["mean"], line_dash="dash", line_color="grey",
                      annotation_text=f"All features (k={int(r.k)})")
    fig.update_xaxes(type="category")
    return fig


st.title("Binary Firefly feature selection: benchmark")
tab_over, tab_explore, tab_play = st.tabs(["Overview", "Benchmark explorer", "Live playground"])

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
with tab_over:
    st.markdown(
        "An honest benchmark of binary Firefly feature selection against standard "
        "baselines on Digits (64-d) and a 10,000-image CIFAR-10 subset embedded with "
        "pretrained ResNet-18 (512-d). Seeds 0-9, matched search budgets, leakage-safe "
        "splits. The primary analysis is pre-registered in `PREREG.md` (raw Firefly mask "
        "vs ANOVA, MI, L1, RF, PCA and PSO; paired Wilcoxon, Holm within each family). "
        "PCA is a projection, not a selection: it needs the full embedding plus a "
        "projection matrix at inference.")
    st.subheader("Result files")
    st.dataframe(dd.file_status(), hide_index=True)

    ds = st.selectbox("Dataset", dd.DATASETS, key="ov_ds")
    base = load_csv(f"baselines_{ds}.csv")
    if base is None:
        st.info(f"results/baselines_{ds}.csv not found.")
    else:
        summ = dd.summarise(dd.accuracy_table(base, None))
        st.plotly_chart(acc_vs_k_figure(summ, f"Baselines on {ds}: accuracy vs k"))
        takeaway(dd.takeaway_acc_vs_k(summ))

# ---------------------------------------------------------------------------
# Benchmark explorer
# ---------------------------------------------------------------------------
with tab_explore:
    c1, c2, c3, c4 = st.columns(4)
    ds = c1.selectbox("Dataset", dd.DATASETS, key="ex_ds")
    base = load_csv(f"baselines_{ds}.csv")
    meta = load_csv(f"meta_{ds}.csv")
    if meta is None:
        st.info(f"results/meta_{ds}.csv does not exist yet (Day-5 runs). "
                "Showing baselines only.")
    budgets = sorted(meta.budget.unique()) if meta is not None and len(meta) else []
    budget = c2.selectbox("Search budget", budgets, disabled=not budgets) if budgets else None
    mask_type = c3.selectbox("Mask", ["raw", "rerank"], disabled=meta is None,
                             help="raw = primary (matched budget); rerank costs 50 extra fits")
    metric = c4.selectbox("Classifier", ["acc_lr", "acc_knn", "acc_svm"],
                          format_func={"acc_lr": "Logistic regression",
                                       "acc_knn": "KNN (k=5)",
                                       "acc_svm": "Linear SVM"}.get,
                          help="Baselines exist for logistic regression only")
    if metric != "acc_lr":
        st.caption("Baseline CSVs hold logistic-regression accuracy only, so only "
                   "metaheuristics are shown for this classifier.")

    table = dd.accuracy_table(base, meta, budget=budget, mask_type=mask_type, metric=metric)
    if table.empty:
        st.warning("Nothing to show for this selection.")
    else:
        methods = sorted(table.method.unique())
        chosen = st.multiselect("Methods", methods, default=methods)
        table = table[table.method.isin(chosen)]
        summ = dd.summarise(table)
        title = f"{ds}: accuracy vs k" + (f" (budget {budget}, {mask_type} mask)" if budget else "")
        st.plotly_chart(acc_vs_k_figure(summ, title))
        takeaway(dd.takeaway_acc_vs_k(summ))

        ks = sorted(k for k in table.k.unique()
                    if k in set(table[~table.method.isin(dd.REFERENCE_METHODS)].k))
        if ks:
            k_sel = st.select_slider("k for the per-seed view", options=ks, value=ks[0])
            per_seed = table[(table.k == k_sel) & ~table.method.isin(dd.REFERENCE_METHODS)]
            fig = px.strip(per_seed, x="method", y="acc", color="method",
                           hover_data=["seed"], title=f"Per-seed accuracy at k={k_sel}")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)
            takeaway(dd.takeaway_spread(table, k_sel))

        with st.expander("Summary table"):
            st.dataframe(summ.round(4), hide_index=True)

# ---------------------------------------------------------------------------
# Live playground
# ---------------------------------------------------------------------------
with tab_play:
    st.markdown(
        f"One Firefly run with a hard cap of **{dd.BUDGET_CAP} evaluations**. "
        "Fitness = validation error of a logistic regression on an inner split of the "
        "training data. The held-out accuracy below is one run on one seed: an "
        "illustration, not a benchmark result.")
    c1, c2, c3, c4, c5 = st.columns(5)
    source = c1.radio("Data", ["digits", "toy"],
                      format_func={"digits": "Digits (64 features)",
                                   "toy": "Planted toy (50 features, 5 informative)"}.get)
    d_max = 63 if source == "digits" else 49
    k = c2.slider("k", 1, d_max, 16 if source == "digits" else 5)
    budget = c3.slider("Budget (evaluations)", 20, dd.BUDGET_CAP, 150, step=10)
    n_ff = c4.slider("Fireflies", 4, 30, 10)
    seed = c5.number_input("Seed", 0, 9, 0)

    if st.button("Run Firefly", type="primary"):
        st.session_state["play"] = cached_run(source, int(k), int(budget), int(n_ff), int(seed))

    res = st.session_state.get("play")
    if res:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Evaluations used", f"{res['evals_used']} / {res['budget']}")
        m2.metric("Best validation fitness", f"{res['best_fit']:.4f}",
                  delta=f"{res['best_fit'] - res['anova_fit']:+.4f} vs ANOVA top-k",
                  delta_color="inverse")
        m3.metric("Held-out accuracy (Firefly)", f"{res['test_acc']:.3f}")
        m4.metric("Held-out accuracy (ANOVA top-k)", f"{res['anova_test_acc']:.3f}")

        x = [res["init_evals"]] + res["evals_at"]
        y = [res["init_best"]] + res["history"]
        fig = go.Figure(go.Scatter(x=x, y=y, mode="lines+markers", name="Firefly best",
                                   line_shape="hv"))
        fig.add_hline(y=res["anova_fit"], line_dash="dash", line_color="grey",
                      annotation_text="ANOVA top-k (same inner split)")
        fig.update_layout(title="Convergence: best validation fitness vs evaluations",
                          xaxis_title="evaluations used", yaxis_title="validation fitness (lower is better)")
        st.plotly_chart(fig)
        gain = res["init_best"] - res["best_fit"]
        takeaway(
            f"The swarm improved its best validation fitness by {gain:.4f} over "
            f"{len(res['history'])} sweep(s) and {res['evals_used']} evaluations, ending "
            f"{'below' if res['best_fit'] < res['anova_fit'] else 'at or above'} the ANOVA "
            f"top-k mask ({res['anova_fit']:.4f}); validation fitness is in-sample and noisy "
            f"(about 2-3 points).")

        if res["source"] == "digits":
            img = res["mask"].reshape(8, 8).astype(int)
            fig = px.imshow(img, color_continuous_scale=["#f0f0f0", "#d95f02"],
                            title="Selected pixels (8 x 8 digit image)")
            fig.update_layout(coloraxis_showscale=False, width=420, height=420)
            st.plotly_chart(fig)
            takeaway(f"{int(img.sum())} of 64 pixels are selected; per row, top to bottom: "
                     f"{img.sum(axis=1).tolist()}.")
        else:
            hits = int(res["mask"][:5].sum())
            st.write(f"Selected columns: {np.flatnonzero(res['mask']).tolist()}")
            takeaway(f"{hits} of the 5 informative columns (0-4) were selected; not every "
                     f"informative column separates the classes (see src/synthetic.py).")
        if res["warnings"]:
            st.caption("Warnings during the run: " + ", ".join(res["warnings"]))
