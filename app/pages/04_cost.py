"""Cost (RQ2, exploratory): accuracy vs search time and the break-even calculator."""

import streamlit as st

from app.components.callout import data_error, exploratory, takeaway
from app.lib import charts, cost_view, data, facts
from app.lib.methods import DATASET_DISPLAY, display, sort_key
from app.lib.stats_view import format_break_even

st.title("Cost (RQ2)")
st.markdown(
    "Search time, inference time and break-even (exploratory). Search times were "
    "measured under 15-way parallel load.")
exploratory("cost and break-even are descriptive; no significance tests")

try:
    cost = data.load_analysis("rq2_cost")
except data.DataError as err:
    data_error(err)
    st.stop()
timing = data.load_optional("timing_idle")
factor = facts.contention_factor()

datasets = list(data.DATASETS)
c1, c2 = st.columns(2)
dataset = c1.segmented_control("Dataset", datasets, format_func=DATASET_DISPLAY.get,
                               default=datasets[-1], key="cost_ds") or datasets[-1]
c = cost[(cost["dataset"] == dataset) & (cost["method"] != "all")
         & cost["mask_type"].isin(["raw", "-"])]
ks = sorted(int(k) for k in c["k"].unique())
k = c2.segmented_control("k", ks, default=ks[0], key=f"cost_k_{dataset}") or ks[0]
view = c[c["k"] == k]
st.plotly_chart(charts.cost_scatter(view, f"Accuracy vs search time, k={k}"),
                key="cost_scatter")
fast = view.loc[view["search_s_mean"].idxmin()]
best = view.loc[view["acc_mean"].idxmax()]
takeaway(f"At k={k}, {display(fast['method'])} is fastest ({fast['search_s_mean']:.2f} s) "
         f"and {display(best['method'])} has the highest mean accuracy "
         f"({best['acc_mean']:.3f}); metaheuristic searches take "
         f"{view[view['budget'].notna()]['search_s_mean'].min():.0f}-"
         f"{view[view['budget'].notna()]['search_s_mean'].max():.0f} s under load.")

st.subheader("Break-even calculator")
if timing is None or factor is None:
    st.info("Idle timings (results/analysis/timing_idle.csv) or the contention factor "
            "are missing, so the calculator is hidden.")
    st.stop()
st.markdown(
    "How many single predictions must a smaller subset serve before its search time is "
    f"repaid? Search time is the logged time divided by the A4 contention factor "
    f"({factor:.2f}), an idle-machine estimate; per-prediction times are A4's idle "
    "single-sample medians on training rows (seed 0).")
methods = sorted(view["method"].unique(), key=sort_key)
c1, c2, c3 = st.columns(3)
method = c1.selectbox("Method", methods, format_func=display, key=f"cost_m_{dataset}_{k}")
budgets = sorted(int(b) for b in view[view["method"] == method]["budget"].dropna().unique())
budget = (c2.segmented_control("Budget", budgets, default=budgets[0], key="cost_b")
          or budgets[0]) if budgets else None
classifier = c3.segmented_control("Classifier", ["lr", "knn"], default="knn",
                                  format_func={"lr": "Logistic regression",
                                               "knn": "KNN (k=5)"}.get,
                                  key="cost_clf") or "knn"
n_pred = st.select_slider("Number of predictions served",
                          options=[10 ** i for i in range(2, 9)], value=10 ** 5,
                          format_func=lambda v: f"{v:,}")
r = cost_view.calculator(cost, timing, factor, dataset, method, k, budget, classifier, n_pred)
if not r["available"]:
    st.info("No idle timing for this combination.")
else:
    m = st.columns(4)
    m[0].metric("Search, logged (under load)", f"{r['search_logged_s']:.2f} s", border=True)
    m[1].metric("Search, idle estimate", f"{r['search_idle_s']:.2f} s", border=True)
    m[2].metric("Saving per prediction", f"{r['saving_s'] * 1e6:.1f} µs",
                help=f"all features {r['t_all_s'] * 1e6:.1f} µs, subset "
                     f"{r['t_selected_s'] * 1e6:.1f} µs", border=True)
    m[3].metric("Break-even", format_break_even(r["break_even"]),
                help="predictions needed to repay the idle search time", border=True)
    net = r["net_s"]
    st.markdown(f"After {n_pred:,} predictions the subset has "
                f"{'saved' if net > 0 else 'cost'} {abs(net):.2f} s net of its search.")
    takeaway(cost_view.sentence(r, display(method),
                                "logistic regression" if classifier == "lr" else "KNN"))
