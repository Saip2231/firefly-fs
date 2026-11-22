"""Results explorer (RQ1): accuracy vs k with filters, the clickable
WIN/LOSS/TIE grid, and a per-family panel with paired per-seed differences."""

import streamlit as st

from app.components.bento import card, hero, row
from app.components.callout import data_error, takeaway
from app.components.label_grid import label_grid
from app.lib import charts, data, facts, results_view
from app.lib.methods import DATASET_DISPLAY, DATASET_SHORT, display, sort_key
from app.lib.stats_view import format_p, label_text

hero("Results explorer (RQ1)",
     "The pre-registered comparisons of plain Firefly against each comparator, for each "
     "dataset, k and search budget (paired Wilcoxon, Holm within each family).",
     eyebrow="📊 Results")

try:
    desc = data.load_analysis("descriptive")
    rq1 = data.load_analysis("rq1_tests")
except data.DataError as err:
    with card("error", "Data problem"):
        data_error(err)
    st.stop()

# -- accuracy vs k: controls in one row, chart at full card width -----------
datasets = list(data.DATASETS)
budgets = sorted(int(b) for b in desc["budget"].dropna().unique())
with card("acc", "Accuracy vs k", eyebrow="📊 Mean ± std over seeds"):
    c1, c2 = row([1, 1])
    dataset = c1.segmented_control("Dataset", datasets, format_func=DATASET_SHORT.get,
                                   default=datasets[-1], key="res_ds") or datasets[-1]
    budget = c2.segmented_control("Search budget (metaheuristics)", budgets,
                                  default=budgets[0], key="res_budget") or budgets[0]
    view = desc[(desc["dataset"] == dataset)
                & (desc["budget"].isna() | (desc["budget"] == budget))]
    methods = sorted(view["method"].unique(), key=sort_key)
    chosen = st.pills("Methods", methods, selection_mode="multi", default=methods,
                      format_func=display, key=f"res_methods_{dataset}")
    view = view[view["method"].isin(chosen or methods)]
    st.plotly_chart(charts.acc_vs_k(view, f"{DATASET_DISPLAY[dataset]}, budget {budget}"),
                    key="res_acc_vs_k")
    takeaway(results_view.explorer_takeaway(view))
    st.caption("Baselines have no search budget; the same baseline rows appear at both "
               "budgets. PCA is a projection, not a selector (dashed).")

# -- grid ----------------------------------------------------------------------
grid = results_view.grid_table(rq1)
counts = results_view.counts_by_label(grid)
n = int(rq1["n"].max())
with card("grid", "Pre-registered tests: plain Firefly vs each comparator",
          eyebrow="✓ Confirmatory"):
    m = row(4)
    m[0].metric("Tests", len(grid), border=True)
    m[1].metric(label_text("WIN"), counts["WIN"], border=True)
    m[2].metric(label_text("LOSS"), counts["LOSS"], border=True)
    m[3].metric(label_text("TIE"), counts["TIE"], border=True)
    st.caption(f"Each cell shows the label symbol and the mean difference in accuracy points "
               f"(Firefly minus comparator). n={n} seeds per test; the smallest possible p is "
               f"{facts.min_two_sided_p(n):.4f}, so power is limited.")
    sel = label_grid(grid)

# -- family panel: stat tiles in a row, chart at full width -------------------
seeds = data.per_seed(sel["dataset"])
try:
    diffs = results_view.paired_diffs(seeds, int(sel["k"]), int(sel["budget"]),
                                      sel["comparator"])
except data.DataError as err:
    with card("family-error", "Data problem"):
        data_error(err)
    st.stop()
with card("family", f"{sel['family']}: Firefly vs {display(sel['comparator'])}",
          eyebrow="Selected cell · per-seed differences"):
    m = row(4)
    m[0].metric("Label", label_text(sel["label"]), border=True)
    m[1].metric("Mean difference", f"{sel['mean_diff_pts']:+.2f} pts",
                help="Firefly minus comparator, accuracy points", border=True)
    m[2].metric("Holm-adjusted p", format_p(sel["p_holm"]),
                help=f"raw p {format_p(sel['p_raw'])}", border=True)
    m[3].metric("Non-zero pairs", f"{int(sel['nonzero_pairs'])} of {int(sel['n'])}",
                border=True)
    st.markdown(results_view.panel_sentence(sel, diffs))
    st.plotly_chart(charts.seed_diffs(diffs, sel["comparator"]), key="res_seed_diffs")
    table = diffs.rename(columns={"acc_lr_firefly": "firefly accuracy",
                                  "acc_lr_comparator": f"{sel['comparator']} accuracy",
                                  "diff_pts": "difference (pts)"})
    st.download_button("Download per-seed differences (CSV)", table.to_csv(index=False),
                       file_name=f"rq1_{sel['dataset']}_k{int(sel['k'])}_b{int(sel['budget'])}"
                                 f"_{sel['comparator']}.csv",
                       mime="text/csv", icon=":material/download:")

with card("table", f"All {len(grid)} tests", eyebrow="Table"):
    with st.expander("Show the full table"):
        st.dataframe(grid.drop(columns=["family"]), hide_index=True)
