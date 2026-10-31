"""Results explorer (RQ1): accuracy vs k with filters, the clickable
WIN/LOSS/TIE grid, and a per-family panel with paired per-seed differences."""

import streamlit as st

from app.components.callout import data_error, takeaway
from app.components.label_grid import label_grid
from app.lib import charts, data, facts, results_view
from app.lib.methods import DATASET_DISPLAY, display, sort_key
from app.lib.stats_view import format_p, label_text

st.title("Results explorer (RQ1)")
st.markdown(
    "The pre-registered comparisons of plain Firefly against each comparator, for "
    "each dataset, k and search budget (paired Wilcoxon, Holm within each family).")

try:
    desc = data.load_analysis("descriptive")
    rq1 = data.load_analysis("rq1_tests")
except data.DataError as err:
    data_error(err)
    st.stop()

# -- accuracy vs k -----------------------------------------------------------
c1, c2 = st.columns(2)
datasets = list(data.DATASETS)
dataset = c1.segmented_control("Dataset", datasets, format_func=DATASET_DISPLAY.get,
                               default=datasets[-1], key="res_ds") or datasets[-1]
budgets = sorted(int(b) for b in desc["budget"].dropna().unique())
budget = c2.segmented_control("Search budget (metaheuristics)", budgets,
                              default=budgets[0], key="res_budget") or budgets[0]
view = desc[(desc["dataset"] == dataset)
            & (desc["budget"].isna() | (desc["budget"] == budget))]
methods = sorted(view["method"].unique(), key=sort_key)
chosen = st.pills("Methods", methods, selection_mode="multi", default=methods,
                  format_func=display, key=f"res_methods_{dataset}")
view = view[view["method"].isin(chosen or methods)]
st.plotly_chart(charts.acc_vs_k(view, f"Accuracy vs k: {DATASET_DISPLAY[dataset]}, "
                                      f"budget {budget}"), key="res_acc_vs_k")
takeaway(results_view.explorer_takeaway(view))
st.caption("Baselines have no search budget; the same baseline rows appear at both "
           "budgets. PCA is a projection, not a selector (dashed).")

# -- grid ----------------------------------------------------------------------
st.subheader("Pre-registered tests: plain Firefly vs each comparator")
grid = results_view.grid_table(rq1)
counts = results_view.counts_by_label(grid)
n = int(rq1["n"].max())
st.markdown(
    f"{len(grid)} tests: {label_text('WIN')} {counts['WIN']}, {label_text('LOSS')} "
    f"{counts['LOSS']}, {label_text('TIE')} {counts['TIE']}. Each cell shows the label "
    f"symbol and the mean difference in accuracy points (Firefly minus comparator). "
    f"n={n} seeds per test; the smallest possible p is "
    f"{facts.min_two_sided_p(n):.4f}, so power is limited.")
row = label_grid(grid)

# -- family panel ----------------------------------------------------------------
with st.container(border=True):
    st.subheader(f"{row['family']}: Firefly vs {display(row['comparator'])}")
    seeds = data.per_seed(row["dataset"])
    try:
        diffs = results_view.paired_diffs(seeds, int(row["k"]), int(row["budget"]),
                                          row["comparator"])
    except data.DataError as err:
        data_error(err)
        st.stop()
    m = st.columns(4)
    m[0].metric("Label", label_text(row["label"]), border=True)
    m[1].metric("Mean difference", f"{row['mean_diff_pts']:+.2f} pts",
                help="Firefly minus comparator, accuracy points", border=True)
    m[2].metric("Holm-adjusted p", format_p(row["p_holm"]),
                help=f"raw p {format_p(row['p_raw'])}", border=True)
    m[3].metric("Non-zero pairs", f"{int(row['nonzero_pairs'])} of {int(row['n'])}",
                border=True)
    st.markdown(results_view.panel_sentence(row, diffs))
    st.plotly_chart(charts.seed_diffs(diffs, row["comparator"]), key="res_seed_diffs")
    table = diffs.rename(columns={"acc_lr_firefly": "firefly accuracy",
                                  "acc_lr_comparator": f"{row['comparator']} accuracy",
                                  "diff_pts": "difference (pts)"})
    st.download_button("Download per-seed differences (CSV)", table.to_csv(index=False),
                       file_name=f"rq1_{row['dataset']}_k{int(row['k'])}_b{int(row['budget'])}"
                                 f"_{row['comparator']}.csv",
                       mime="text/csv", icon=":material/download:")

with st.expander(f"All {len(grid)} tests as a table"):
    st.dataframe(grid.drop(columns=["family"]), hide_index=True)
