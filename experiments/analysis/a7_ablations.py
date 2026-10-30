"""Stage A7: ablations (EXPLORATORY; ANALYSIS.md A7).  Reads CSVs only.

All comparisons are paired by seed (0-9), on test accuracy of the final
logistic regression (acc_lr), with a two-sided Wilcoxon p-value labelled
"exploratory, not corrected across families" (same routine as A1:
experiments._common.paired_wilcoxon; all-zero differences give p = n/a).

  seeded_vs_plain   firefly_seeded vs firefly and pso_seeded vs pso, raw
                    masks, per dataset x k x budget
  rerank_vs_raw     rerank vs raw mask of the same run, per method x dataset
                    x k x budget; each row reports total_fits (raw = search
                    evals; rerank = search evals + rerank fits), mean
                    rerank_s, and in how many seeds the rerank changed the mask
  budget_3000_vs_800  each method's raw mask at 3000 vs 800, per dataset x k,
                    with the mean search_s at each budget (measured under
                    15-way parallel load, see A4)
  levy_vs_plain     only if firefly_levy rows exist (they do not in the Day-5
                    grid; reported as "not run")

Outputs
  results/analysis/ablations.csv
  figures/F3_budget_curve_<dataset>.{png,pdf}  acc_lr vs budget per k
  figures/F9_ablations_<dataset>.{png,pdf}     paired mean differences
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments._common import paired_wilcoxon  # noqa: E402
from experiments.analysis import a0_integrity  # noqa: E402
from experiments.analysis import common as C  # noqa: E402
from experiments.analysis import style as S  # noqa: E402
from experiments.analysis import takeaway as T  # noqa: E402

LABEL = "exploratory, not corrected across families"
OUT = C.ANALYSIS_DIR / "ablations.csv"
PAIRS_SEEDED = (("firefly_seeded", "firefly"), ("pso_seeded", "pso"))
SEARCH = ("firefly", "pso", "firefly_seeded", "pso_seeded")
LOAD_NOTE = "search_s measured under 15-way parallel load (contention factor in A4)"


def _seeds(df: pd.DataFrame, what: str) -> pd.DataFrame:
    df = df.sort_values("seed")
    if list(df.seed) != list(C.SEEDS):
        raise ValueError(f"{what}: seeds {list(df.seed)}, expected {list(C.SEEDS)}")
    return df


def _pair(v_df: pd.DataFrame, r_df: pd.DataFrame, **info) -> dict:
    """Paired comparison of two seed-aligned frames; info holds the labels
    (incl. the 'variant' / 'reference' names, hence the parameter names)."""
    a, b = v_df.acc_lr.values, r_df.acc_lr.values
    res = paired_wilcoxon(a, b)
    diff = a - b
    return dict(**info, n=len(diff), mean_variant=float(a.mean()),
                mean_reference=float(b.mean()), mean_diff_pts=float(diff.mean() * 100),
                std_diff_pts=float(diff.std(ddof=1) * 100),
                seeds_higher=int((diff > 0).sum()), seeds_lower=int((diff < 0).sum()),
                nonzero_pairs=res["n_nonzero"], p_raw=res["p"], label=LABEL)


def ablation_table(meta: dict) -> pd.DataFrame:
    rows = []
    for ds, m in meta.items():
        raw = m[m.mask_type == "raw"]
        for (k, b), g in raw.groupby(["k", "budget"]):
            for seeded, plain in PAIRS_SEEDED:
                v = _seeds(g[g.method == seeded], f"{ds}/{seeded}/k={k}/b={b}")
                r = _seeds(g[g.method == plain], f"{ds}/{plain}/k={k}/b={b}")
                rows.append(_pair(v, r, ablation="seeded_vs_plain", dataset=ds, method=plain,
                                  k=int(k), budget=int(b), variant=seeded, reference=plain,
                                  total_fits_variant=v.evals_used.mean(),
                                  total_fits_reference=r.evals_used.mean()))
        for (method, k, b), g in m.groupby(["method", "k", "budget"]):
            v = _seeds(g[g.mask_type == "rerank"], f"{ds}/{method}/k={k}/b={b} rerank")
            r = _seeds(g[g.mask_type == "raw"], f"{ds}/{method}/k={k}/b={b} raw")
            changed = int(sum(a != c for a, c in zip(v.idx, r.idx)))
            rows.append(_pair(v, r, ablation="rerank_vs_raw", dataset=ds, method=method,
                              k=int(k), budget=int(b), variant="rerank", reference="raw",
                              total_fits_variant=(v.evals_used + v.rerank_fits).mean(),
                              total_fits_reference=r.evals_used.mean(),
                              rerank_s_mean=v.rerank_s.mean(), masks_changed=changed))
        budgets = sorted(int(x) for x in raw.budget.unique())
        if budgets[:2] == [800, 3000]:
            for (method, k), g in raw.groupby(["method", "k"]):
                v = _seeds(g[g.budget == 3000], f"{ds}/{method}/k={k}/b=3000")
                r = _seeds(g[g.budget == 800], f"{ds}/{method}/k={k}/b=800")
                rows.append(_pair(v, r, ablation="budget_3000_vs_800", dataset=ds,
                                  method=method, k=int(k), budget=np.nan, variant="budget 3000",
                                  reference="budget 800", total_fits_variant=v.evals_used.mean(),
                                  total_fits_reference=r.evals_used.mean(),
                                  search_s_variant=v.search_s.mean(),
                                  search_s_reference=r.search_s.mean()))
        if "firefly_levy" in set(m.method):
            for (k, b), g in raw.groupby(["k", "budget"]):
                lv = g[g.method == "firefly_levy"]
                if len(lv):
                    rows.append(_pair(_seeds(lv, "levy"), _seeds(g[g.method == "firefly"], "ff"),
                                      ablation="levy_vs_plain", dataset=ds, method="firefly",
                                      k=int(k), budget=int(b), variant="firefly_levy",
                                      reference="firefly"))
    t = pd.DataFrame(rows)
    t["search_s_note"] = np.where(t.ablation == "budget_3000_vs_800", LOAD_NOTE, "")
    return t


def levy_status(meta: dict) -> str:
    return ("firefly_levy present" if any("firefly_levy" in set(m.method) for m in meta.values())
            else "firefly_levy not run in the Day-5 grid: long-jump mutation ablation not available")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_f3(meta: dict, baselines: dict, ds: str):
    raw = meta[ds][meta[ds].mask_type == "raw"]
    ks = sorted(int(k) for k in raw.k.unique())
    fig, axes = plt.subplots(1, len(ks), figsize=(3.2 * len(ks) + 1.6, 3.8), squeeze=False)
    b = baselines[ds]
    for ax, k in zip(axes[0], ks):
        for method in SEARCH:
            st = S.style(method)
            agg = raw[(raw.method == method) & (raw.k == k)].groupby("budget").acc_lr.agg(
                ["mean", "std"]).reset_index()
            ax.errorbar(agg.budget, agg["mean"], yerr=agg["std"], color=st["color"],
                        marker=st["marker"], linestyle=st["linestyle"], capsize=3,
                        label=st["label"])
        for ref in ("anova", "rf"):
            st = S.style(ref)
            ax.axhline(b[(b.method == ref) & (b.k == k)].acc.mean(), color=st["color"],
                       linestyle=":", linewidth=1, label=f"{st['label']} (no budget)")
        ax.set_xscale("log")
        ax.set_xticks([800, 3000], ["800", "3000"])
        ax.minorticks_off()
        ax.set_xlabel("search budget (evaluations)")
        ax.set_title(f"k = {k}", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("test accuracy, logistic regression\n(mean ± std, 10 seeds)")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7, frameon=False)
    fig.suptitle(f"Accuracy vs search budget ({LABEL}): {C.DATASET_DISPLAY[ds]}; raw masks",
                 fontsize=8.5)
    fig.tight_layout()
    return fig


def plot_f9(t: pd.DataFrame, ds: str):
    abls = [a for a in ("seeded_vs_plain", "rerank_vs_raw", "budget_3000_vs_800")
            if a in set(t.ablation)]
    fig, axes = plt.subplots(1, len(abls), figsize=(4.6 * len(abls) + 1.5, 4.0), squeeze=False)
    titles = {"seeded_vs_plain": "filter-seeded minus plain",
              "rerank_vs_raw": "re-ranked minus raw mask\n(+50 fits per run)",
              "budget_3000_vs_800": "budget 3000 minus 800"}
    for ax, abl in zip(axes[0], abls):
        s = t[(t.dataset == ds) & (t.ablation == abl)].copy()
        s["series"] = s.method + np.where(s.budget.isna(), "",
                                          " b" + s.budget.fillna(0).astype(int).astype(str))
        series = list(dict.fromkeys(s.sort_values(["method", "budget"]).series))
        ks = sorted(s.k.unique())
        width = 0.8 / len(series)
        for i, name in enumerate(series):
            g = s[s.series == name].set_index("k").reindex(ks)
            method = g.method.dropna().iloc[0]
            st = S.style(method)
            hollow = name.endswith("b800")
            ax.bar(np.arange(len(ks)) + i * width, g.mean_diff_pts, width, yerr=g.std_diff_pts,
                   capsize=2, color="white" if hollow else st["color"], edgecolor=st["color"],
                   label=name.replace("_", " "))
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(np.arange(len(ks)) + 0.4 - width / 2, [f"k={int(k)}" for k in ks])
        ax.set_title(titles[abl], fontsize=9)
        ax.set_ylabel("paired mean difference, accuracy points (± std)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=6.5, frameon=False)
    fig.suptitle(f"Ablations ({LABEL}): {C.DATASET_DISPLAY[ds]}; logistic regression, "
                 "paired by seed, n=10", fontsize=8.5)
    fig.tight_layout()
    return fig


def f3_takeaway(t: pd.DataFrame, ds: str) -> str:
    b = t[(t.dataset == ds) & (t.ablation == "budget_3000_vs_800")]
    ff = b[b.method == "firefly"]
    ps = b[b.method == "pso"]
    return (f"{C.DATASET_DISPLAY[ds]} ({LABEL}): going from budget 800 to 3000 changes Firefly's "
            f"mean accuracy by {T.pts(ff.mean_diff_pts.min() / 100)} to "
            f"{T.pts(ff.mean_diff_pts.max() / 100)} across k and PSO's by "
            f"{T.pts(ps.mean_diff_pts.min() / 100)} to {T.pts(ps.mean_diff_pts.max() / 100)}, "
            f"for about {b.total_fits_variant.mean() / b.total_fits_reference.mean():.2f}x the "
            f"evaluations.")


def f9_takeaway(t: pd.DataFrame, ds: str) -> str:
    s = t[t.dataset == ds]
    sp = s[s.ablation == "seeded_vs_plain"]
    rr = s[s.ablation == "rerank_vs_raw"]
    n_cells = len(rr)
    n_changed = int(rr.masks_changed.sum())
    return (f"{C.DATASET_DISPLAY[ds]} ({LABEL}): filter seeding changes mean accuracy by "
            f"{T.pts(sp.mean_diff_pts.min() / 100)} to {T.pts(sp.mean_diff_pts.max() / 100)}; "
            f"re-ranking changes it by {T.pts(rr.mean_diff_pts.min() / 100)} to "
            f"{T.pts(rr.mean_diff_pts.max() / 100)} at +50 fits and {rr.rerank_s_mean.mean():.1f} s "
            f"per run on average, and picked a different mask than the raw search in {n_changed} "
            f"of {n_cells * 10} runs; {sum(s.p_raw < 0.05)} of {len(s)} comparisons have raw "
            f"p < 0.05 (uncorrected).")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def make_outputs(meta: dict, baselines: dict, out_csv=OUT, fig_dir=C.FIGURES,
                 takeaways=None) -> dict:
    t = ablation_table(meta)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out_csv, index=False)
    takeaways = takeaways or fig_dir / "takeaways.csv"
    made = {}
    for ds in meta:
        for name, fig, text in ((f"F3_budget_curve_{ds}", plot_f3(meta, baselines, ds),
                                 f3_takeaway(t, ds)),
                                (f"F9_ablations_{ds}", plot_f9(t, ds), f9_takeaway(t, ds))):
            paths = S.save_fig(fig, name, fig_dir)
            plt.close(fig)
            T.record(name, [p.name for p in paths], text, takeaways)
            made[name] = dict(files=paths, takeaway=text)
    return dict(table=t, figures=made)


def run() -> dict:
    a0_integrity.validate()
    meta = {ds: C.load_meta(ds) for ds in C.DATASETS}
    baselines = {ds: C.load_baselines(ds) for ds in C.DATASETS}
    out = make_outputs(meta, baselines)
    t = out["table"]
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 300)
    print(f"Ablations ({LABEL}); acc_lr, paired by seed, n=10.\n{levy_status(meta)}.\n")
    cols = ["dataset", "method", "k", "budget", "mean_diff_pts", "std_diff_pts",
            "seeds_higher", "seeds_lower", "p_raw"]
    for abl, extra in (("seeded_vs_plain", []),
                       ("rerank_vs_raw", ["masks_changed", "rerank_s_mean", "total_fits_variant"]),
                       ("budget_3000_vs_800", ["search_s_reference", "search_s_variant"])):
        print(f"--- {abl} ---")
        print(t[t.ablation == abl][cols + extra].to_string(
            index=False, float_format=lambda v: f"{v:.4g}"))
        print()
    print(f"Wrote {OUT} ({len(t)} rows)")
    for name, m in out["figures"].items():
        print(f"{name}: {', '.join(p.name for p in m['files'])}\n  Takeaway: {m['takeaway']}")
    return out
