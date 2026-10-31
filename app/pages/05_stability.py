"""Stability (RQ3, exploratory): Jaccard overlap across seeds, feature frequency,
and on Digits an 8x8 map of how often each pixel was chosen."""

import numpy as np
import pandas as pd
import streamlit as st

from app.components.callout import data_error, exploratory, takeaway
from app.lib import charts, data, results_view
from app.lib.methods import DATASET_DISPLAY, display, sort_key

st.title("Stability (RQ3)")
st.markdown(
    "How similar the selected feature subsets are across seeds, with random selection "
    "as the reference (exploratory).")

try:
    stab = data.load_analysis("rq3_stability")
except data.DataError as err:
    data_error(err)
    st.stop()

datasets = list(data.DATASETS)
c1, c2, c3 = st.columns(3)
dataset = c1.segmented_control("Dataset", datasets, format_func=DATASET_DISPLAY.get,
                               default=datasets[-1], key="stab_ds") or datasets[-1]
s = stab[(stab["dataset"] == dataset) & stab["mask_type"].isin(["raw", "-"])]
ks = sorted(int(k) for k in s["k"].unique())
k = c2.segmented_control("k", ks, default=ks[0], key=f"stab_k_{dataset}") or ks[0]
budgets = sorted(int(b) for b in s["budget"].dropna().unique())
budget = c3.segmented_control("Budget", budgets, default=budgets[0], key="stab_b") or budgets[0]
view = s[(s["k"] == k) & (s["budget"].isna() | (s["budget"] == budget))
         & (s["method"] != "random")].copy()
view = view.sort_values("method", key=lambda col: col.map(sort_key))
view["name"] = view["method"].map(display)
refs = {"random (empirical)": float(s[s["k"] == k]["random_empirical"].iloc[0]),
        "random (expected)": float(s[s["k"] == k]["random_analytic"].iloc[0])}
st.plotly_chart(charts.bars_with_reference(
    view, "jaccard_mean", "jaccard_std", f"Mean pairwise Jaccard across seeds, k={k}",
    "Jaccard overlap (1 = identical subsets)", refs), key="stab_bars")
exploratory(str(view["caveat"].iloc[0]) if "caveat" in view and len(view) else "stability")
top = view.loc[view["jaccard_mean"].idxmax()]
ff = view[view["method"] == "firefly"]
takeaway(f"{display(top['method'])} is the most stable at k={k} (Jaccard "
         f"{top['jaccard_mean']:.2f}); "
         + (f"plain Firefly reaches {ff['jaccard_mean'].iloc[0]:.2f}, "
            f"{ff['jaccard_mean'].iloc[0] / refs['random (expected)']:.1f}x the random "
            "expectation." if len(ff) else "plain Firefly is not in this view."))

st.subheader("How often each feature was chosen")
seeds = data.per_seed(dataset)
sel = seeds[(seeds["k"] == k) & seeds["method"].isin(view["method"])]
methods = sorted(sel["method"].unique(), key=sort_key)
method = st.selectbox("Method", methods, format_func=display, key=f"stab_m_{dataset}")
rows = sel[(sel["method"] == method)
           & (sel["budget"].isna() | (sel["budget"] == budget))]
d = int(seeds[seeds["method"] == "all"]["k"].iloc[0])
counts = results_view.selection_counts(rows["idx"], d)
n_seeds = len(rows)
order = np.argsort(-counts, kind="stable")
order = order[counts[order] > 0]
freq_df = pd.DataFrame(dict(name=[f"f{i}" for i in order], method=method,
                            count=counts[order]))
st.plotly_chart(charts.bars_with_reference(
    freq_df, "count", None, f"{display(method)}: seeds (of {n_seeds}) choosing each feature",
    "seeds choosing the feature"), key="stab_freq")
half = int(np.ceil(n_seeds / 2))
takeaway(f"{int((counts >= half).sum())} features were chosen in at least {half} of "
         f"{n_seeds} seeds and {int((counts > 0).sum())} features at least once, for "
         f"k={k} of d={d}.")

if dataset == "digits":
    st.subheader("Digits: selection frequency per pixel")
    st.plotly_chart(charts.pixel_grid(counts, f"{display(method)}: seeds choosing each "
                                      f"pixel (k={k})", zmax=n_seeds), key="stab_pixels")
    st.caption("Valid only for Digits, where features are pixels of the 8x8 image.")
else:
    st.caption("No pixel map for CIFAR: its features are embedding dimensions, not pixels.")
