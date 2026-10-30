"""Stage A2: figures F1 and F2 (ANALYSIS.md section 4, A2).

F1  figures/F1_acc_vs_k_<dataset>_b<budget>.{png,pdf}
    Test accuracy (logistic regression) vs k, mean with a +/- 1 std band over
    seeds 0-9, for random, anova, mi, l1, rf, pca (dashed, "projection, not a
    selector"), firefly and pso (raw masks at that budget), with 'all'
    features as a horizontal line and band.  Seeded variants are not shown.
    Baselines have no budget: the same baseline curves appear in both budget
    panels of a dataset.
F2  figures/F2_rq1_grid.{png,pdf}
    The 12 RQ1 families (rows) x 6 comparators (columns), coloured by the
    pre-registered label (WIN / LOSS / TIE), annotated with the mean
    difference in points (Firefly minus comparator) and the Holm-adjusted p.
    The tests are recomputed by a1_rq1.run_rq1 (deterministic; identical to
    results/analysis/rq1_tests.csv).

Each figure's takeaway is computed from the same dataframe that is plotted
and stored in figures/takeaways.csv.  The plotted frames are returned by
f1_frame() / f2_frame() so A9 can compare figures by their data.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")                      # files only, no window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from experiments.analysis import a0_integrity, a1_rq1  # noqa: E402
from experiments.analysis import common as C  # noqa: E402
from experiments.analysis import style as S  # noqa: E402
from experiments.analysis import takeaway as T  # noqa: E402

F1_METHODS = ("random", "anova", "mi", "l1", "rf", "pca", "firefly", "pso")
SHORT = {"anova": "ANOVA F", "mi": "MI", "l1": "L1", "rf": "RF",
         "pca": "PCA\n(projection)", "pso": "PSO"}
F2_LEGEND = "paired Wilcoxon, Holm within family, n=10"


# ---------------------------------------------------------------------------
# F1
# ---------------------------------------------------------------------------

def f1_frame(baselines: dict, meta: dict, dataset: str, budget: int) -> pd.DataFrame:
    """mean / std / n of the primary accuracy per (method, k) for one panel."""
    long = a1_rq1.primary_long({dataset: baselines[dataset]}, {dataset: meta[dataset]})
    ks = C.selection_ks(baselines[dataset])
    d = C.dimension(baselines[dataset])
    base = long[long.budget.isna() & long.method.isin(F1_METHODS) & long.k.isin(ks)]
    srch = long[(long.budget == budget) & long.method.isin(F1_METHODS) & long.k.isin(ks)]
    allf = long[long.budget.isna() & (long.method == "all") & (long.k == d)]
    df = pd.concat([base, srch, allf])
    out = (df.groupby(["method", "k"])["acc"].agg(mean="mean", std="std", n="count")
             .reset_index())
    out.insert(0, "budget", budget)
    out.insert(0, "dataset", dataset)
    return out


def f1_takeaway(frame: pd.DataFrame) -> str:
    sel = frame[frame.method != "all"]
    allf = frame[frame.method == "all"].iloc[0]
    parts = []
    for k in (sel.k.min(), sel.k.max()):
        at = sel[sel.k == k].sort_values("mean", ascending=False).reset_index(drop=True)
        rank = int(at.index[at.method == "firefly"][0]) + 1
        ff = float(at.loc[at.method == "firefly", "mean"].iloc[0])
        top, top_v = T.best_mean(at)
        parts.append(f"at k={int(k)} Firefly averages {T.acc(ff)} (rank {rank} of {len(at)} "
                     f"methods shown) and the highest mean is {top} at {T.acc(top_v)}")
    return (f"{C.DATASET_DISPLAY[frame.dataset.iloc[0]]}, budget {int(frame.budget.iloc[0])}: "
            + "; ".join(parts) + f"; all {int(allf.k)} features give {T.acc(allf['mean'])}. "
            "Means over 10 seeds; see F2 for the paired tests.")


def plot_f1(frame: pd.DataFrame):
    ds, b = frame.dataset.iloc[0], int(frame.budget.iloc[0])
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    allf = frame[frame.method == "all"].iloc[0]
    st = S.style("all")
    ax.axhline(allf["mean"], color=st["color"], linestyle=st["linestyle"],
               label=f"{st['label']} (d={int(allf.k)})")
    ax.axhspan(allf["mean"] - allf["std"], allf["mean"] + allf["std"],
               color=st["color"], alpha=0.06)
    for method in F1_METHODS:
        g = frame[frame.method == method].sort_values("k")
        if g.empty:
            continue
        st = S.style(method)
        ax.plot(g.k, g["mean"], color=st["color"], marker=st["marker"],
                linestyle=st["linestyle"], label=st["label"], linewidth=1.6)
        ax.fill_between(g.k, g["mean"] - g["std"], g["mean"] + g["std"],
                        color=st["color"], alpha=0.12, linewidth=0)
    ks = sorted(frame.loc[frame.method != "all", "k"].unique())
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks, [str(int(k)) for k in ks])
    ax.minorticks_off()
    ax.set_xlabel("k (selected features; PCA: components)")
    ax.set_ylabel("test accuracy, logistic regression\n(mean ± 1 std over 10 seeds)")
    ax.set_title(f"Accuracy vs k: {C.DATASET_DISPLAY[ds]}\n"
                 f"search budget {b} (Firefly, PSO raw masks); baselines have no budget",
                 fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# F2
# ---------------------------------------------------------------------------

def f2_frame(baselines: dict, meta: dict) -> pd.DataFrame:
    return a1_rq1.run_rq1(baselines, meta, C.load_split_hashes())


def plot_f2(tests: pd.DataFrame):
    fams = tests[["dataset", "k", "budget"]].drop_duplicates().sort_values(
        ["dataset", "k", "budget"]).itertuples(index=False)
    fams = list(fams)
    comps = list(C.COMPARATORS)
    fig, ax = plt.subplots(figsize=(8.6, 0.48 * len(fams) + 1.9))
    for i, (ds, k, b) in enumerate(fams):
        for j, comp in enumerate(comps):
            r = tests[(tests.dataset == ds) & (tests.k == k) & (tests.budget == b)
                      & (tests.comparator == comp)].iloc[0]
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=S.LABEL_COLOURS[r.label],
                                       edgecolor="white", linewidth=1.5))
            colour = "white" if r.label != "TIE" else "black"
            ax.text(j + 0.5, i + 0.5, f"{r.mean_diff_pts:+.1f}\n{T.pval(r.p_holm)}",
                    ha="center", va="center", fontsize=6.8, color=colour)
    ax.set_xlim(0, len(comps))
    ax.set_ylim(len(fams), 0)
    ax.set_xticks(np.arange(len(comps)) + 0.5, [SHORT[c] for c in comps], fontsize=8)
    ax.set_yticks(np.arange(len(fams)) + 0.5,
                  [f"{'CIFAR-10/R18' if ds == 'cifar_r18' else 'Digits'}  k={k}  b={b}"
                   for ds, k, b in fams], fontsize=7.5)
    ax.xaxis.tick_top()
    for i in range(1, len(fams)):
        if fams[i][0] != fams[i - 1][0]:
            ax.axhline(i, color="black", linewidth=1.2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("RQ1: plain Firefly vs comparators, test accuracy (logistic regression)\n"
                 "cell = Firefly minus comparator (accuracy points), Holm-adjusted p",
                 fontsize=9.5, pad=28)
    handles = [Patch(facecolor=S.LABEL_COLOURS[x], label=x) for x in ("WIN", "LOSS", "TIE")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3,
              fontsize=8, frameon=False, title=F2_LEGEND, title_fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def make_figures(baselines: dict, meta: dict, out_dir=C.FIGURES, takeaways=None) -> list[dict]:
    """Draw and save F1 (per dataset x budget) and F2; record takeaways."""
    takeaways = takeaways or out_dir / "takeaways.csv"
    made, f1_frames = [], []
    for ds in C.DATASETS:
        for b in sorted(int(x) for x in meta[ds].budget.unique()):
            frame = f1_frame(baselines, meta, ds, b)
            fig = plot_f1(frame)
            name = f"F1_acc_vs_k_{ds}_b{b}"
            paths = S.save_fig(fig, name, out_dir)
            plt.close(fig)
            text = f1_takeaway(frame)
            T.record(name, [p.name for p in paths], text, takeaways)
            made.append(dict(figure=name, files=paths, takeaway=text, frame=frame))
            f1_frames.append(frame)
    # F1's plotted numbers, so A9 can compare the figures by their data
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(f1_frames, ignore_index=True).to_csv(data_dir / "F1_frames.csv", index=False)
    tests = f2_frame(baselines, meta)
    fig = plot_f2(tests)
    paths = S.save_fig(fig, "F2_rq1_grid", out_dir)
    plt.close(fig)
    text = a1_rq1.rq1_takeaway(tests) + f" ({F2_LEGEND}; {a1_rq1.POWER_NOTE})"
    T.record("F2_rq1_grid", [p.name for p in paths], text, takeaways)
    made.append(dict(figure="F2_rq1_grid", files=paths, takeaway=text, frame=tests))
    return made


def run() -> list[dict]:
    a0_integrity.validate()
    baselines = {ds: C.load_baselines(ds) for ds in C.DATASETS}
    meta = {ds: C.load_meta(ds) for ds in C.DATASETS}
    made = make_figures(baselines, meta)
    for m in made:
        print(f"{m['figure']}: {', '.join(str(p) for p in m['files'])}")
        print(f"  Takeaway: {m['takeaway']}\n")
    print(f"Wrote {C.FIGURES / 'takeaways.csv'}")
    return made
