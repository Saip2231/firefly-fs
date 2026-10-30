"""Stage A5: RQ3 subset stability across seeds (EXPLORATORY; ANALYSIS.md A5).

Stability of a method at (dataset, k[, budget]) = mean pairwise Jaccard
similarity |A & B| / |A | B| of its selected sets over all 45 pairs of
seeds 0-9 (std, min and max of the 45 values reported too).  Meta methods:
per budget and mask_type (raw is the primary mask; rerank listed for
completeness).  Baselines: per k.  PCA (projection, not a selector) and
'all' have no subset and are excluded.

Reference values
  * the empirical Jaccard of the `random` baseline at the same k;
  * the analytic value for two independent uniform k-subsets of d features,
    (k^2/d) / (2k - k^2/d), i.e. E|A & B| / E|A | B| (a ratio of
    expectations, an approximation of the expected Jaccard).
Caveat (written into every output): each seed uses a different
train/test split, so baseline stability includes sensitivity to resampling
the training data, not search noise alone.  The same holds for the meta
methods, whose search also depends on the seed.

Outputs (all labelled exploratory)
  results/analysis/rq3_stability.csv
  results/analysis/rq3_feature_frequency.csv  per (dataset, method, k,
      budget): features chosen in >= 5 of 10 seeds, and in all 10
  figures/F6_stability_<dataset>.{png,pdf}    bars per k with the random
      reference (empirical bar + analytic line)
  figures/F7_feature_frequency_<dataset>_k<k>.{png,pdf}  how often each
      feature was chosen across seeds, sorted, per method
"""

from __future__ import annotations

from itertools import combinations

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
CAVEAT = ("each seed uses a different train/test split, so stability includes sensitivity to "
          "resampling the training data, not search noise alone")
OUT_STAB = C.ANALYSIS_DIR / "rq3_stability.csv"
OUT_FREQ = C.ANALYSIS_DIR / "rq3_feature_frequency.csv"
F7_METHODS = ("random", "anova", "mi", "l1", "rf", "firefly", "pso")
MIN_SEEDS = 5


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def jaccard(a: frozenset, b: frozenset) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 1.0


def pairwise(sets: list[frozenset]) -> np.ndarray:
    return np.array([jaccard(a, b) for a, b in combinations(sets, 2)])


def analytic_random(k: int, d: int) -> float:
    """(k^2/d) / (2k - k^2/d): E|A&B| / E|A|B| for independent uniform k-subsets."""
    inter = k * k / d
    return inter / (2 * k - inter)


def frequency(sets: list[frozenset], d: int) -> np.ndarray:
    """Times each feature 0..d-1 was selected across the sets."""
    f = np.zeros(d, dtype=int)
    for s in sets:
        f[list(s)] += 1
    return f


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _groups(baselines: dict, meta: dict):
    """Yield (dataset, method, k, budget, mask_type, d, sets ordered by seed)."""
    for ds, b in baselines.items():
        d = C.dimension(b)
        ks = C.selection_ks(b)
        sel = b[~b.method.isin(C.NO_IDX_METHODS) & b.k.isin(ks)]
        for (method, k), g in sel.groupby(["method", "k"]):
            yield ds, method, int(k), np.nan, "-", d, list(g.sort_values("seed").idx)
    for ds, m in meta.items():
        d = C.dimension(baselines[ds])
        for (method, k, budget, mt), g in m.groupby(["method", "k", "budget", "mask_type"]):
            yield ds, method, int(k), int(budget), mt, d, list(g.sort_values("seed").idx)


def stability_table(baselines: dict, meta: dict) -> pd.DataFrame:
    rows = []
    for ds, method, k, budget, mt, d, sets in _groups(baselines, meta):
        j = pairwise(sets)
        rows.append(dict(dataset=ds, method=method, display=C.DISPLAY[method], k=k,
                         budget=budget, mask_type=mt, d=d, n_seeds=len(sets), n_pairs=len(j),
                         jaccard_mean=float(j.mean()), jaccard_std=float(j.std(ddof=1)),
                         jaccard_min=float(j.min()), jaccard_max=float(j.max())))
    t = pd.DataFrame(rows)
    rnd = t[t.method == "random"].set_index(["dataset", "k"])["jaccard_mean"]
    t["random_empirical"] = [rnd.get((r.dataset, r.k), np.nan) for r in t.itertuples()]
    t["random_analytic"] = [analytic_random(r.k, r.d) for r in t.itertuples()]
    t["ratio_to_random_analytic"] = t.jaccard_mean / t.random_analytic
    t["analysis"] = EXPLORATORY
    t["caveat"] = CAVEAT
    return t.sort_values(["dataset", "k", "method", "budget", "mask_type"],
                         na_position="first").reset_index(drop=True)


def frequency_table(baselines: dict, meta: dict) -> pd.DataFrame:
    rows = []
    for ds, method, k, budget, mt, d, sets in _groups(baselines, meta):
        if mt == "rerank":
            continue
        f = frequency(sets, d)
        rows.append(dict(dataset=ds, method=method, k=k, budget=budget, d=d,
                         n_features_ge5=int((f >= MIN_SEEDS).sum()),
                         n_features_all10=int((f == len(sets)).sum()),
                         n_features_ever=int((f > 0).sum()), analysis=EXPLORATORY))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _series(t: pd.DataFrame, ds: str, k: int) -> pd.DataFrame:
    s = t[(t.dataset == ds) & (t.k == k) & t.mask_type.isin(["-", "raw"])].copy()
    s["series"] = np.where(s.budget.isna(), s.method,
                           s.method + " b" + s.budget.fillna(0).astype(int).astype(str))
    order = {m: i for i, m in enumerate(["random", "anova", "mi", "l1", "rf", "firefly",
                                         "pso", "firefly_seeded", "pso_seeded"])}
    return s.sort_values(["method", "budget"], key=lambda c: c.map(order) if c.name == "method"
                         else c, na_position="first")


def plot_f6(t: pd.DataFrame, ds: str):
    ks = sorted(t[(t.dataset == ds) & (t.mask_type == "raw")].k.unique())
    fig, axes = plt.subplots(1, len(ks), figsize=(3.3 * len(ks) + 0.8, 4.0), sharey=True,
                             squeeze=False)
    for ax, k in zip(axes[0], ks):
        s = _series(t, ds, k)
        for i, r in enumerate(s.itertuples()):
            st = S.style(r.method)
            hollow = r.budget == 800
            ax.bar(i, r.jaccard_mean, yerr=r.jaccard_std, capsize=2,
                   color="white" if hollow else st["color"], edgecolor=st["color"],
                   linewidth=1.2, hatch="//" if r.method.endswith("_seeded") else None)
        ax.axhline(s.random_analytic.iloc[0], color="black", linestyle=":", linewidth=1)
        ax.set_xticks(range(len(s)), [x.replace("_", " ") for x in s.series], rotation=75,
                      fontsize=6.5)
        ax.set_title(f"k = {int(k)}", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].set_ylabel("mean pairwise Jaccard over 45 seed pairs\n(± std)")
    fig.suptitle(f"Subset stability across seeds ({EXPLORATORY}): {C.DATASET_DISPLAY[ds]}\n"
                 "dotted = analytic random reference; hollow = budget 800, filled = 3000, "
                 "hatched = filter-seeded; raw masks; PCA and all excluded\n"
                 f"note: {CAVEAT}", fontsize=7.5)
    fig.tight_layout()
    return fig


def plot_f7(baselines: dict, meta: dict, ds: str, k: int):
    d = C.dimension(baselines[ds])
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    curves = {}
    for g in _groups({ds: baselines[ds]}, {ds: meta[ds]}):
        _, method, kk, budget, mt, _, sets = g
        if kk != k or mt == "rerank" or method not in F7_METHODS:
            continue
        f = np.sort(frequency(sets, d))[::-1]
        label = C.DISPLAY[method] + ("" if np.isnan(budget) else f", budget {budget}")
        st = S.style(method)
        ax.step(np.arange(1, d + 1), f, where="post", color=st["color"], label=label,
                linestyle="--" if budget == 800 else st["linestyle"], linewidth=1.3)
        curves[label] = f
    ax.axhline(MIN_SEEDS, color="grey", linewidth=0.8, linestyle=":")
    ax.set_xscale("log")
    ax.set_xlabel("feature rank by selection frequency (log)")
    ax.set_ylabel("seeds (of 10) that selected the feature")
    ax.set_ylim(0, 10.5)
    ax.set_title(f"Feature selection frequency across seeds ({EXPLORATORY}): "
                 f"{C.DATASET_DISPLAY[ds]}, k = {k}\nraw masks; Firefly/PSO dashed = budget 800, "
                 f"solid = 3000; dotted line = {MIN_SEEDS} seeds", fontsize=8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    return fig, curves


# ---------------------------------------------------------------------------
# Takeaways
# ---------------------------------------------------------------------------

def f6_takeaway(t: pd.DataFrame, ds: str) -> str:
    s = t[(t.dataset == ds) & t.mask_type.isin(["-", "raw"])]
    ff = s[s.method == "firefly"]
    top = s.sort_values("jaccard_mean", ascending=False).iloc[0]
    rnd = s[s.method == "random"]
    return (f"{C.DATASET_DISPLAY[ds]} ({EXPLORATORY}): Firefly's mean pairwise Jaccard ranges "
            f"from {ff.jaccard_mean.min():.2f} to {ff.jaccard_mean.max():.2f} across k and budget, "
            f"against {rnd.jaccard_mean.min():.2f} to {rnd.jaccard_mean.max():.2f} for the random "
            f"baseline; the highest value is {top.display} at k={int(top.k)} "
            f"({top.jaccard_mean:.2f}). Note: {CAVEAT}.")


def f7_takeaway(freq: pd.DataFrame, ds: str, k: int) -> str:
    f = freq[(freq.dataset == ds) & (freq.k == k) & freq.method.isin(F7_METHODS)]
    parts = [f"{C.DISPLAY[r.method]}{'' if np.isnan(r.budget) else f' b{int(r.budget)}'} "
             f"{r.n_features_ge5}" for r in f.itertuples()]
    return (f"{C.DATASET_DISPLAY[ds]}, k={k} ({EXPLORATORY}): features chosen in at least "
            f"{MIN_SEEDS} of 10 seeds: " + ", ".join(parts) + ".")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def make_outputs(baselines: dict, meta: dict, out_dir=C.ANALYSIS_DIR, fig_dir=C.FIGURES,
                 takeaways=None) -> dict:
    t = stability_table(baselines, meta)
    freq = frequency_table(baselines, meta)
    out_dir.mkdir(parents=True, exist_ok=True)
    t.to_csv(out_dir / OUT_STAB.name, index=False)
    freq.to_csv(out_dir / OUT_FREQ.name, index=False)
    takeaways = takeaways or fig_dir / "takeaways.csv"
    made = {}
    for ds in baselines:
        name = f"F6_stability_{ds}"
        fig = plot_f6(t, ds)
        paths = S.save_fig(fig, name, fig_dir)
        plt.close(fig)
        text = f6_takeaway(t, ds)
        T.record(name, [p.name for p in paths], text, takeaways)
        made[name] = dict(files=paths, takeaway=text)
        for k in C.selection_ks(baselines[ds]):
            name = f"F7_feature_frequency_{ds}_k{k}"
            fig, _ = plot_f7(baselines, meta, ds, k)
            paths = S.save_fig(fig, name, fig_dir)
            plt.close(fig)
            text = f7_takeaway(freq, ds, k)
            T.record(name, [p.name for p in paths], text, takeaways)
            made[name] = dict(files=paths, takeaway=text)
    return dict(stability=t, frequency=freq, figures=made)


def run() -> dict:
    a0_integrity.validate()
    baselines = {ds: C.load_baselines(ds) for ds in C.DATASETS}
    meta = {ds: C.load_meta(ds) for ds in C.DATASETS}
    out = make_outputs(baselines, meta)
    t = out["stability"]
    show = t[t.mask_type.isin(["-", "raw"])]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    print(f"RQ3 stability ({EXPLORATORY}): mean pairwise Jaccard over 45 seed pairs; "
          f"raw masks.\nNote: {CAVEAT}.\n")
    piv = show.assign(col=np.where(show.budget.isna(), show.method,
                                   show.method + "_b" + show.budget.fillna(0).astype(int).astype(str)))
    print(piv.pivot_table(index=["dataset", "k"], columns="col", values="jaccard_mean")
             .round(3).to_string())
    print("\nRandom reference (empirical / analytic):")
    ref = show[show.method == "random"][["dataset", "k", "jaccard_mean", "random_analytic"]]
    print(ref.round(3).to_string(index=False))
    print(f"\nFeatures chosen in >= {MIN_SEEDS} of 10 seeds (raw masks):")
    fq = out["frequency"].assign(col=lambda f: np.where(
        f.budget.isna(), f.method, f.method + "_b" + f.budget.fillna(0).astype(int).astype(str)))
    print(fq.pivot_table(index=["dataset", "k"], columns="col", values="n_features_ge5")
            .to_string())
    print(f"\nWrote {OUT_STAB}, {OUT_FREQ}")
    for name, m in out["figures"].items():
        print(f"{name}: {', '.join(p.name for p in m['files'])}\n  Takeaway: {m['takeaway']}")
    return out
