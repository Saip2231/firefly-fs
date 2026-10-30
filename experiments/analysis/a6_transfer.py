"""Stage A6: RQ4 transfer of selected subsets to KNN and linear SVM
(EXPLORATORY; ANALYSIS.md A6).  Reads CSVs only.

Subsets were chosen under logistic regression (wrapper fitness for Firefly
and PSO; the filters/L1/RF are classifier-free or LR-based).  Here the same
subsets are scored with KNN (k=5) and a linear SVM:
  meta methods    acc_knn / acc_svm from results/meta_<ds>.csv, raw masks
  baselines       acc_knn / acc_svm from results/baselines_transfer_<ds>.csv
                  (Stage A6a, scored with the runner's evaluate_on_test)

Outputs (exploratory; NO p-values, by design)
  results/analysis/rq4_transfer.csv  for each family (dataset x k x budget):
      firefly and pso vs anova, mi, l1, rf, per classifier (knn, svm; lr
      repeated for reference -- the confirmatory LR comparison is A1):
      mean and std of each side, paired mean difference and std in accuracy
      points, number of seeds where the search method is higher / lower.
  results/analysis/rq4_gaps.csv      within-method gaps acc_lr - acc_knn and
      acc_lr - acc_svm (paired by seed, points), per method x k [x budget].
  figures/F8_transfer_<dataset>.{png,pdf}  KNN and SVM accuracy vs k.
Pairing: both sides must hold seeds 0-9 exactly once (else ValueError).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.analysis import a0_integrity  # noqa: E402
from experiments.analysis import common as C  # noqa: E402
from experiments.analysis import style as S  # noqa: E402
from experiments.analysis import takeaway as T  # noqa: E402

EXPLORATORY = "exploratory"
SEARCH_METHODS = ("firefly", "pso")
TRANSFER_COMPARATORS = ("anova", "mi", "l1", "rf")
CLASSIFIERS = {"knn": "KNN (k=5)", "svm": "linear SVM", "lr": "logistic regression"}
OUT_TRANSFER = C.ANALYSIS_DIR / "rq4_transfer.csv"
OUT_GAPS = C.ANALYSIS_DIR / "rq4_gaps.csv"
F8_METHODS = ("random", "anova", "mi", "l1", "rf", "firefly", "pso")
NOTE = ("subsets selected under logistic regression, scored with other classifiers; "
        "descriptive only, no significance tests")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def transfer_long(transfer: dict, meta: dict) -> pd.DataFrame:
    """dataset, method, k, budget (NaN for baselines), seed, acc_lr, acc_knn, acc_svm."""
    cols = ["dataset", "method", "k", "budget", "seed", "acc_lr", "acc_knn", "acc_svm"]
    parts = [t.assign(dataset=ds, budget=np.nan)[cols] for ds, t in transfer.items()]
    parts += [m[m.mask_type == "raw"].assign(dataset=ds)[cols] for ds, m in meta.items()]
    return pd.concat(parts, ignore_index=True)


def side(long: pd.DataFrame, ds: str, method: str, k: int, budget, clf: str) -> pd.Series:
    sel = (long.dataset == ds) & (long.method == method) & (long.k == k)
    if method in C.META_METHODS:
        sel &= long.budget == budget
    s = long.loc[sel].set_index("seed")[f"acc_{clf}"].sort_index()
    if list(s.index) != list(C.SEEDS):
        raise ValueError(f"{ds}/{method}/k={k}/b={budget}: seeds {list(s.index)}, "
                         f"expected {list(C.SEEDS)}")
    return s


def compare(a: pd.Series, b: pd.Series) -> dict:
    diff = a.values - b.values
    return dict(n=len(diff), mean_method=float(a.mean()), std_method=float(a.std()),
                mean_comparator=float(b.mean()), std_comparator=float(b.std()),
                mean_diff_pts=float(diff.mean() * 100),
                std_diff_pts=float(diff.std(ddof=1) * 100),
                seeds_higher=int((diff > 0).sum()), seeds_lower=int((diff < 0).sum()))


def transfer_table(transfer: dict, meta: dict) -> pd.DataFrame:
    long = transfer_long(transfer, meta)
    fams = sorted({(ds, int(k), int(b)) for ds, m in meta.items()
                   for k, b in m[["k", "budget"]].drop_duplicates().itertuples(index=False)})
    rows = []
    for ds, k, b in fams:
        for method in SEARCH_METHODS:
            for comp in TRANSFER_COMPARATORS:
                for clf in CLASSIFIERS:
                    r = compare(side(long, ds, method, k, b, clf),
                                side(long, ds, comp, k, b, clf))
                    rows.append(dict(dataset=ds, k=k, budget=b, method=method,
                                     comparator=comp, classifier=clf, **r,
                                     role="reference (confirmatory test in A1)" if clf == "lr"
                                     else EXPLORATORY))
    return pd.DataFrame(rows)


def gaps_table(transfer: dict, meta: dict) -> pd.DataFrame:
    long = transfer_long(transfer, meta)
    rows = []
    for (ds, method, k, b), g in long.groupby(["dataset", "method", "k", "budget"], dropna=False):
        g = g.sort_values("seed")
        knn = (g.acc_lr - g.acc_knn) * 100
        svm = (g.acc_lr - g.acc_svm) * 100
        rows.append(dict(dataset=ds, method=method, display=C.DISPLAY[method], k=int(k),
                         budget=b, n=len(g), acc_lr=g.acc_lr.mean(), acc_knn=g.acc_knn.mean(),
                         acc_svm=g.acc_svm.mean(), gap_lr_knn_pts=knn.mean(),
                         gap_lr_knn_std_pts=knn.std(), gap_lr_svm_pts=svm.mean(),
                         gap_lr_svm_std_pts=svm.std(), analysis=EXPLORATORY))
    return pd.DataFrame(rows).sort_values(["dataset", "k", "method", "budget"],
                                          na_position="first").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_f8(transfer: dict, meta: dict, ds: str):
    long = transfer_long({ds: transfer[ds]}, {ds: meta[ds]})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=False)
    for ax, clf in zip(axes, ("knn", "svm")):
        for method in F8_METHODS:
            st = S.style(method)
            g = long[long.method == method]
            budgets = [np.nan] if method not in C.META_METHODS else sorted(g.budget.unique())
            for b in budgets:
                gb = g if np.isnan(b) else g[g.budget == b]
                agg = gb.groupby("k")[f"acc_{clf}"].agg(["mean", "std"]).reset_index()
                label = st["label"] + ("" if np.isnan(b) else f", budget {int(b)}")
                ls = "--" if b == 800 else st["linestyle"]
                ax.plot(agg.k, agg["mean"], color=st["color"], marker=st["marker"],
                        linestyle=ls, label=label, linewidth=1.4)
                ax.fill_between(agg.k, agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                                color=st["color"], alpha=0.10, linewidth=0)
        ks = sorted(long.k.unique())
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks, [str(int(k)) for k in ks])
        ax.minorticks_off()
        ax.set_xlabel("k (selected features)")
        ax.set_ylabel(f"test accuracy, {CLASSIFIERS[clf]}\n(mean ± 1 std, 10 seeds)")
        ax.set_title(CLASSIFIERS[clf], fontsize=9)
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.suptitle(f"Transfer of subsets to other classifiers ({EXPLORATORY}): "
                 f"{C.DATASET_DISPLAY[ds]}\n{NOTE}; Firefly/PSO raw masks, dashed = budget 800; "
                 "PCA excluded (projection, not a selector)", fontsize=8.5)
    fig.tight_layout()
    return fig


def f8_takeaway(tt: pd.DataFrame, gaps: pd.DataFrame, ds: str) -> str:
    t = tt[(tt.dataset == ds) & (tt.method == "firefly") & tt.classifier.isin(["knn", "svm"])]
    parts = []
    for clf in ("knn", "svm"):
        c = t[t.classifier == clf]
        parts.append(f"with {CLASSIFIERS[clf]} Firefly's mean is above the comparator's in "
                     f"{int((c.mean_diff_pts > 0).sum())} of {len(c)} family x comparator cells "
                     f"(differences {T.pts(c.mean_diff_pts.min() / 100)} to "
                     f"{T.pts(c.mean_diff_pts.max() / 100)})")
    g = gaps[(gaps.dataset == ds) & (gaps.method == "firefly")]
    return (f"{C.DATASET_DISPLAY[ds]} ({EXPLORATORY}, descriptive): " + "; ".join(parts)
            + f"; Firefly's own LR-minus-KNN gap is {g.gap_lr_knn_pts.min():+.1f} to "
              f"{g.gap_lr_knn_pts.max():+.1f} pts and LR-minus-SVM {g.gap_lr_svm_pts.min():+.1f} "
              f"to {g.gap_lr_svm_pts.max():+.1f} pts. No significance tests.")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def make_outputs(transfer: dict, meta: dict, out_dir=C.ANALYSIS_DIR, fig_dir=C.FIGURES,
                 takeaways=None) -> dict:
    tt = transfer_table(transfer, meta)
    gaps = gaps_table(transfer, meta)
    out_dir.mkdir(parents=True, exist_ok=True)
    tt.assign(note=NOTE).to_csv(out_dir / OUT_TRANSFER.name, index=False)
    gaps.assign(note=NOTE).to_csv(out_dir / OUT_GAPS.name, index=False)
    takeaways = takeaways or fig_dir / "takeaways.csv"
    made = {}
    for ds in meta:
        name = f"F8_transfer_{ds}"
        fig = plot_f8(transfer, meta, ds)
        paths = S.save_fig(fig, name, fig_dir)
        plt.close(fig)
        text = f8_takeaway(tt, gaps, ds)
        T.record(name, [p.name for p in paths], text, takeaways)
        made[name] = dict(files=paths, takeaway=text)
    return dict(transfer=tt, gaps=gaps, figures=made)


def run() -> dict:
    a0_integrity.validate()
    transfer = {ds: C.load_baseline_transfer(ds) for ds in C.DATASETS}
    meta = {ds: C.load_meta(ds) for ds in C.DATASETS}
    out = make_outputs(transfer, meta)
    tt = out["transfer"]
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    print(f"RQ4 transfer ({EXPLORATORY}): {NOTE}.\n")
    for clf in ("knn", "svm"):
        c = tt[(tt.classifier == clf)]
        piv = c.pivot_table(index=["dataset", "k", "budget", "method"], columns="comparator",
                            values="mean_diff_pts").round(1)[list(TRANSFER_COMPARATORS)]
        print(f"Mean difference, search method minus comparator, {CLASSIFIERS[clf]} "
              f"(accuracy points; raw masks):\n{piv.to_string()}\n")
    g = out["gaps"]
    print("Within-method gaps (points, mean over seeds, and over both budgets for search "
          "methods; positive = LR higher; per-budget values in rq4_gaps.csv):")
    print(g.pivot_table(index=["dataset", "k"], columns="method",
                        values=["gap_lr_knn_pts", "gap_lr_svm_pts"]).round(1).to_string())
    print(f"\nWrote {OUT_TRANSFER} ({len(tt)} rows), {OUT_GAPS} ({len(g)} rows)")
    for name, m in out["figures"].items():
        print(f"{name}: {', '.join(p.name for p in m['files'])}\n  Takeaway: {m['takeaway']}")
    return out
