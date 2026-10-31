"""Transfer (RQ4) and ablations: all exploratory, numbers from the analysis tables."""

import streamlit as st

from app.components.callout import data_error, exploratory, takeaway
from app.lib import charts, data
from app.lib.methods import DATASET_DISPLAY, display
from app.lib.stats_view import format_p

st.title("Transfer and ablations")
st.markdown(
    "How subsets chosen under logistic regression transfer to KNN and linear SVM, and "
    "the effect of seeding, re-ranking and search budget (all exploratory).")

try:
    transfer = data.load_analysis("rq4_transfer")
    abl = data.load_analysis("ablations")
except data.DataError as err:
    data_error(err)
    st.stop()

datasets = list(data.DATASETS)
VIEWS = {"transfer": "Transfer (RQ4)", "seeded_vs_plain": "Filter seeding",
         "rerank_vs_raw": "Re-ranking", "budget_3000_vs_800": "Budget 3000 vs 800"}
c1, c2 = st.columns(2)
view = c1.segmented_control("Panel", list(VIEWS), format_func=VIEWS.get,
                            default="transfer", key="ta_view") or "transfer"
dataset = c2.segmented_control("Dataset", datasets, format_func=DATASET_DISPLAY.get,
                               default=datasets[-1], key="ta_ds") or datasets[-1]

if view == "transfer":
    t = transfer[transfer["dataset"] == dataset]
    c1, c2, c3 = st.columns(3)
    ks = sorted(int(k) for k in t["k"].unique())
    k = c1.segmented_control("k", ks, default=ks[0], key=f"ta_k_{dataset}") or ks[0]
    budgets = sorted(int(b) for b in t["budget"].dropna().unique())
    budget = c2.segmented_control("Budget", budgets, default=budgets[0],
                                  key="ta_b") or budgets[0]
    clf = c3.segmented_control("Classifier", ["knn", "svm"], default="knn",
                               format_func={"knn": "KNN (k=5)", "svm": "Linear SVM"}.get,
                               key="ta_clf") or "knn"
    v = t[(t["k"] == k) & (t["budget"] == budget) & (t["classifier"] == clf)].copy()
    v["name"] = [f"{display(m)} − {display(c)}" for m, c in zip(v["method"], v["comparator"])]
    st.plotly_chart(charts.diff_bars(v, f"{'KNN' if clf == 'knn' else 'Linear SVM'} accuracy: "
                                        f"search subset minus filter subset, k={k}"),
                    key="ta_transfer")
    exploratory(str(v["note"].iloc[0]) if len(v) else "transfer")
    if len(v):
        lo = v.loc[v["mean_diff_pts"].idxmin()]
        hi = v.loc[v["mean_diff_pts"].idxmax()]
        takeaway(f"Differences range from {lo['mean_diff_pts']:+.2f} pts ({lo['name']}) to "
                 f"{hi['mean_diff_pts']:+.2f} pts ({hi['name']}); subsets were chosen for "
                 "logistic regression, not for this classifier.")
    st.dataframe(v[["name", "mean_method", "mean_comparator", "mean_diff_pts", "std_diff_pts",
                    "seeds_higher", "seeds_lower"]], hide_index=True)
else:
    a = abl[(abl["ablation"] == view) & (abl["dataset"] == dataset)].copy()
    if a.empty:
        st.info("No rows for this panel.")
        st.stop()
    a["name"] = [f"{display(m)}, k={int(k)}" + (f", b={int(b)}" if b == b else "")
                 for m, k, b in zip(a["method"], a["k"], a["budget"])]
    pairs = sorted({f"{v} minus {r}" for v, r in zip(a["variant"], a["reference"])})
    st.plotly_chart(charts.diff_bars(a, f"{VIEWS[view]}: {'; '.join(pairs)}"),
                    key=f"ta_{view}")
    exploratory("paired by seed; Wilcoxon p-values are not corrected across families")
    cols = ["name", "mean_variant", "mean_reference", "mean_diff_pts", "std_diff_pts",
            "seeds_higher", "seeds_lower", "p_raw"]
    if view == "rerank_vs_raw":
        cols += ["total_fits_variant", "total_fits_reference", "rerank_s_mean", "masks_changed"]
    if view == "budget_3000_vs_800":
        cols += ["search_s_variant", "search_s_reference"]
        st.caption("search_s measured under 15-way parallel load.")
    st.dataframe(a[[c for c in cols if c in a.columns]], hide_index=True)
    pos = int((a["mean_diff_pts"] > 0).sum())
    small = a[a["p_raw"] < 0.05]
    takeaway(f"The variant has the higher mean in {pos} of {len(a)} rows; "
             f"{len(small)} rows have an uncorrected p < 0.05"
             + (f" (smallest p {format_p(small['p_raw'].min())})" if len(small) else "")
             + ". Exploratory: these p-values are not corrected.")
    if view == "rerank_vs_raw" and "rerank_s_mean" in a:
        takeaway(f"Re-ranking adds {int(a['total_fits_variant'].iloc[0] - a['total_fits_reference'].iloc[0])} "
                 f"classifier fits and {a['rerank_s_mean'].min():.1f}-"
                 f"{a['rerank_s_mean'].max():.1f} s per run on this dataset.")
