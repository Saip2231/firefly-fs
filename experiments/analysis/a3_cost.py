"""Stage A3: RQ2 search cost and break-even (EXPLORATORY; ANALYSIS.md A3).

results/analysis/rq2_cost.csv, one row per (dataset, method, k, budget,
mask_type); baselines have no budget and mask_type "-":
  mean / std of search_s, rerank_s, infer_s, evals_used, rerank_fits,
  total_fits (= evals_used + rerank_fits) and total_s (= search_s +
  rerank_s), the mean accuracy (baselines `acc`, meta `acc_lr`), and
  break_even    N = total_s / (infer_all - infer_selected) test samples, where
                infer_all is the mean infer_s of the dataset's 'all' rows;
                "never" when the denominator is <= 0.  Indicative only (A4
                re-times inference on an idle machine).
  extra_s_per_pt_vs_anova
                (total_s_method - search_s_anova) / mean_diff_pts when the
                method's mean accuracy exceeds ANOVA's at the same k;
                otherwise "no accuracy gain".  mean_diff_vs_anova_pts is
                reported next to it.
Timing caveats written into the CSV and figure captions:
  * search_s was measured under parallel load (meta runs: 15 workers;
    baselines: joblib n_jobs=-1), so it is not idle-machine time.
  * infer_s = batch predict() time / number of test rows for the logistic
    regression; it excludes the column slice and, for PCA, the projection.
PCA is a projection, not a selector: its inference also needs the full
embedding and a projection matrix, which infer_s does not include.

Figures (exploratory):
  F4 figures/F4_acc_vs_search_<dataset>.{png,pdf}: mean accuracy vs mean
     search time (log x), one panel per k, all methods (raw masks).
  F5 figures/F5_break_even_<dataset>.{png,pdf}: break-even N (log y) per
     method and k; "never" drawn as a hatched bar.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from experiments.analysis import a0_integrity  # noqa: E402
from experiments.analysis import common as C  # noqa: E402
from experiments.analysis import style as S  # noqa: E402
from experiments.analysis import takeaway as T  # noqa: E402

OUT = C.ANALYSIS_DIR / "rq2_cost.csv"
EXPLORATORY = "exploratory"
LOAD_NOTE = ("search_s measured under parallel load (meta runs: 15 workers; baselines: "
             "joblib n_jobs=-1); not idle-machine time")
INFER_NOTE = ("infer_s = batch predict time / test rows, logistic regression, excluding the "
              "column slice and the PCA projection; indicative only, see A4")
STATS = ("search_s", "rerank_s", "infer_s", "evals_used", "rerank_fits", "total_fits", "total_s")
F5_METHODS = ("random", "anova", "mi", "l1", "rf", "pca", "firefly", "pso")


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def per_run(baselines: dict, meta: dict) -> pd.DataFrame:
    """One row per run with the cost columns filled for both sources."""
    parts = []
    for ds, b in baselines.items():
        x = b.assign(dataset=ds, budget=np.nan, mask_type="-", rerank_s=0.0,
                     evals_used=np.nan, rerank_fits=0)
        parts.append(x[["dataset", "method", "k", "budget", "mask_type", "seed", "acc",
                        "search_s", "rerank_s", "infer_s", "evals_used", "rerank_fits"]])
    for ds, m in meta.items():
        x = m.assign(dataset=ds).rename(columns={"acc_lr": "acc"})
        parts.append(x[["dataset", "method", "k", "budget", "mask_type", "seed", "acc",
                        "search_s", "rerank_s", "infer_s", "evals_used", "rerank_fits"]])
    df = pd.concat(parts, ignore_index=True)
    df["total_fits"] = df.evals_used + df.rerank_fits
    df["total_s"] = df.search_s + df.rerank_s
    return df


def cost_table(baselines: dict, meta: dict) -> pd.DataFrame:
    runs = per_run(baselines, meta)
    keys = ["dataset", "method", "k", "budget", "mask_type"]
    agg = {f"{c}_{s}": (c, s) for c in STATS for s in ("mean", "std")}
    t = runs.groupby(keys, dropna=False).agg(n=("seed", "count"), acc_mean=("acc", "mean"),
                                             acc_std=("acc", "std"), **agg).reset_index()
    infer_all = t[t.method == "all"].set_index("dataset")["infer_s_mean"]
    anova = t[t.method == "anova"].set_index(["dataset", "k"])
    t["infer_all_mean"] = t.dataset.map(infer_all)

    be_num, be_txt, diff, extra, extra_txt = [], [], [], [], []
    for r in t.itertuples():
        if r.method == "all":
            be_num.append(np.nan); be_txt.append("-")
            diff.append(np.nan); extra.append(np.nan); extra_txt.append("-")
            continue
        denom = r.infer_all_mean - r.infer_s_mean
        if denom > 0:
            n = r.total_s_mean / denom
            be_num.append(n); be_txt.append(f"{n:.3g}")
        else:
            be_num.append(np.nan); be_txt.append("never")
        a = anova.loc[(r.dataset, r.k)] if (r.dataset, r.k) in anova.index else None
        if a is None or r.method == "anova":
            diff.append(np.nan); extra.append(np.nan); extra_txt.append("-")
            continue
        d_pts = (r.acc_mean - a.acc_mean) * 100
        diff.append(d_pts)
        if d_pts > 0:
            e = (r.total_s_mean - a.search_s_mean) / d_pts
            extra.append(e); extra_txt.append(f"{e:.3g} s per pt")
        else:
            extra.append(np.nan); extra_txt.append("no accuracy gain")
    t["break_even_samples"] = be_num
    t["break_even"] = be_txt
    t["mean_diff_vs_anova_pts"] = diff
    t["extra_s_per_pt_vs_anova"] = extra
    t["extra_cost_vs_anova"] = extra_txt
    t["display"] = t.method.map(C.DISPLAY)
    t["analysis"] = EXPLORATORY
    t["search_s_note"] = LOAD_NOTE
    t["infer_s_note"] = INFER_NOTE
    return t.sort_values(["dataset", "k", "method", "budget", "mask_type"],
                         na_position="first").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _points(t: pd.DataFrame, ds: str, k: int) -> pd.DataFrame:
    """Methods shown in F4/F5 at one (dataset, k): baselines + raw meta rows."""
    sel = t[(t.dataset == ds) & (t.k == k) & (t.method != "all")
            & t.mask_type.isin(["-", "raw"])].copy()
    sel["series"] = np.where(sel.budget.isna(), sel.method,
                             sel.method + " b" + sel.budget.fillna(0).astype(int).astype(str))
    return sel


def _ks(t: pd.DataFrame, ds: str) -> list[int]:
    """k values with search runs (digits k = 64 = d has baselines only)."""
    return sorted(int(k) for k in t[(t.dataset == ds) & (t.mask_type == "raw")].k.unique())


def plot_f4(t: pd.DataFrame, ds: str):
    ks = _ks(t, ds)
    fig, axes = plt.subplots(1, len(ks), figsize=(3.4 * len(ks) + 1.8, 3.8), sharey=False,
                             squeeze=False)
    allrow = t[(t.dataset == ds) & (t.method == "all")].iloc[0]
    for ax, k in zip(axes[0], ks):
        p = _points(t, ds, k)
        for r in p.itertuples():
            st = S.style(r.method)
            hollow = r.budget == 800
            ax.errorbar(r.total_s_mean, r.acc_mean, xerr=r.total_s_std, yerr=r.acc_std,
                        marker=st["marker"], color=st["color"], linestyle="none",
                        markersize=7, capsize=2, markerfacecolor="white" if hollow else st["color"],
                        label=st["label"] + ("" if np.isnan(r.budget) else f", budget {int(r.budget)}"))
        ax.axhline(allrow.acc_mean, color="black", linestyle=":", linewidth=1)
        S.log_x(ax, "mean search time, s (log)")
        ax.set_title(f"k = {k}", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("test accuracy, logistic regression\n(mean ± std, 10 seeds)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="black", linestyle=":"))
    labels.append(f"All features (d={int(allrow.k)})")
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7,
               frameon=False)
    fig.suptitle(f"Accuracy vs search time ({EXPLORATORY}): {C.DATASET_DISPLAY[ds]}\n"
                 f"{LOAD_NOTE}; hollow = budget 800, filled = 3000; raw masks", fontsize=8.5)
    fig.tight_layout()
    return fig


def plot_f5(t: pd.DataFrame, ds: str):
    ks = _ks(t, ds)
    fig, axes = plt.subplots(1, len(ks), figsize=(3.4 * len(ks) + 1.5, 4.0), sharey=True,
                             squeeze=False)
    finite = t.loc[(t.dataset == ds) & t.break_even_samples.notna(), "break_even_samples"]
    top = float(finite.max()) * 3 if len(finite) else 1e6
    for ax, k in zip(axes[0], ks):
        p = _points(t, ds, k)
        p = p[p.method.isin(F5_METHODS)]
        for i, r in enumerate(p.itertuples()):
            st = S.style(r.method)
            never = r.break_even == "never"
            ax.bar(i, top if never else r.break_even_samples, color="white" if never else st["color"],
                   edgecolor=st["color"], hatch="///" if never else None, linewidth=1.2)
            if never:
                ax.text(i, top, "never", rotation=90, ha="center", va="top", fontsize=6.5)
        ax.set_xticks(range(len(p)), [s.replace("_", " ") for s in p.series], rotation=70,
                      fontsize=6.5)
        S.log_y(ax)
        ax.set_title(f"k = {k}", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].set_ylabel("break-even N, test samples (log)")
    fig.legend(handles=[Patch(facecolor="white", edgecolor="black", hatch="///",
                              label="never (selected inference not faster than all features)")],
               loc="lower center", bbox_to_anchor=(0.5, -0.04), fontsize=7, frameon=False)
    fig.suptitle(f"Break-even N = search time / (infer_all - infer_selected) ({EXPLORATORY}): "
                 f"{C.DATASET_DISPLAY[ds]}\nlogistic regression; {LOAD_NOTE}; inference times "
                 "indicative only (see A4)", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Takeaways
# ---------------------------------------------------------------------------

def f4_takeaway(t: pd.DataFrame, ds: str) -> str:
    raw = t[(t.dataset == ds) & (t.method.isin(["firefly", "pso"])) & (t.mask_type == "raw")]
    base = t[(t.dataset == ds) & t.method.isin(["anova", "mi", "l1", "rf"])]
    ff = raw[raw.method == "firefly"]
    return (f"{C.DATASET_DISPLAY[ds]} ({EXPLORATORY}): mean Firefly search time ranges from "
            f"{ff.total_s_mean.min():.1f} s to {ff.total_s_mean.max():.1f} s across k and budget, "
            f"against {base.total_s_mean.min():.3g} s to {base.total_s_mean.max():.3g} s for the "
            f"ANOVA, MI, L1 and RF baselines; {LOAD_NOTE}.")


def f5_takeaway(t: pd.DataFrame, ds: str) -> str:
    sel = t[(t.dataset == ds) & (t.method != "all") & t.mask_type.isin(["-", "raw"])
            & t.method.isin(F5_METHODS) & t.k.isin(_ks(t, ds))]
    n_never = int((sel.break_even == "never").sum())
    ff = sel[(sel.method == "firefly") & sel.break_even_samples.notna()]
    rng_txt = (f"Firefly's finite break-even N ranges from {ff.break_even_samples.min():.3g} to "
               f"{ff.break_even_samples.max():.3g} test samples" if len(ff)
               else "Firefly never breaks even")
    return (f"{C.DATASET_DISPLAY[ds]} ({EXPLORATORY}): {n_never} of {len(sel)} method x k x budget "
            f"bars are 'never' (selected-feature inference not faster than all features); "
            f"{rng_txt}. With logistic regression the per-sample saving is microseconds or less, "
            f"so these numbers are indicative only (see A4).")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def make_outputs(baselines: dict, meta: dict, out_csv=OUT, fig_dir=C.FIGURES,
                 takeaways=None) -> dict:
    t = cost_table(baselines, meta)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out_csv, index=False)
    takeaways = takeaways or fig_dir / "takeaways.csv"
    made = {}
    for ds in baselines:
        for name, plot, tk in ((f"F4_acc_vs_search_{ds}", plot_f4, f4_takeaway),
                               (f"F5_break_even_{ds}", plot_f5, f5_takeaway)):
            fig = plot(t, ds)
            paths = S.save_fig(fig, name, fig_dir)
            plt.close(fig)
            text = tk(t, ds)
            T.record(name, [p.name for p in paths], text, takeaways)
            made[name] = dict(files=paths, takeaway=text)
    return dict(table=t, figures=made)


def run() -> dict:
    a0_integrity.validate()
    baselines = {ds: C.load_baselines(ds) for ds in C.DATASETS}
    meta = {ds: C.load_meta(ds) for ds in C.DATASETS}
    out = make_outputs(baselines, meta)
    t = out["table"]
    show = t[t.mask_type.isin(["-", "raw"]) & (t.method != "all")]
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    print(f"RQ2 cost ({EXPLORATORY}); {LOAD_NOTE}.\n{INFER_NOTE}.\n")
    print(show[["dataset", "method", "k", "budget", "total_s_mean", "infer_s_mean",
                "break_even", "mean_diff_vs_anova_pts", "extra_cost_vs_anova"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nWrote {OUT} ({len(t)} rows)")
    for name, m in out["figures"].items():
        print(f"{name}: {', '.join(p.name for p in m['files'])}\n  Takeaway: {m['takeaway']}")
    return out
