"""Post-hoc re-scoring of Stage 2 (validate_clean_signal.csv) - no searches re-run.

For each (d, seed) the Stage 2 data are regenerated with identical settings
and the separating informative columns are found with
src.synthetic.separating_columns on the 3000 NON-holdout rows only.
Every saved mask (column "idx") is then scored on:
  sep_hits   = |selected  ∩ separating|
  n_sep      = number of separating columns for that seed
  sep_recall = sep_hits / n_sep
  dead_hits  = hits - sep_hits   (informative but non-separating picks)
Original "hits" is kept unchanged.  No pass/fail rule on recall is applied.

Outputs (new files; the Stage 2 CSV is not modified):
  results/separating_columns_clean_signal.csv
  results/validate_clean_signal_rescored.csv
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from _common import RESULTS_DIR, paired_wilcoxon, warn_once
from validate_clean_signal import N_HOLDOUT, N_INF, N_TOTAL

from src.results_store import log_experiment
from src.synthetic import separating_columns

CLASS_SEP, N_CLUSTERS = 2.0, 1     # must match validate_clean_signal.py
COMPARE = ("firefly", "firefly_levy")


def non_holdout_rows(d: int, seed: int):
    """Regenerate Stage 2 data and return only the rows outside the holdout."""
    X, y = make_classification(
        n_samples=N_TOTAL, n_features=d, n_informative=N_INF, n_redundant=0,
        n_repeated=0, n_clusters_per_class=N_CLUSTERS, class_sep=CLASS_SEP,
        shuffle=False, random_state=seed)
    X_rest, _, y_rest, _ = train_test_split(
        X, y, test_size=N_HOLDOUT, stratify=y, random_state=seed)
    return X_rest, y_rest


def main():
    warn_once()
    t0 = time.perf_counter()
    df = pd.read_csv(RESULTS_DIR / "validate_clean_signal.csv")

    # -- separating columns per (d, seed) ------------------------------------
    sep_rows, sep_sets = [], {}
    for (d, seed), _ in df.groupby(["d", "seed"]):
        Xr, yr = non_holdout_rows(int(d), int(seed))
        s = separating_columns(Xr, yr, N_INF, CLASS_SEP, N_CLUSTERS)
        sep_sets[(d, seed)] = set(s["sep_idx"].tolist())
        sep_rows.append(dict(d=d, seed=seed, n_rows=len(yr), threshold=s["threshold"],
                             sep_idx=",".join(map(str, s["sep_idx"])),
                             n_sep=len(s["sep_idx"]),
                             ambiguous=",".join(map(str, s["ambiguous"])),
                             **{f"gap_{j}": g for j, g in enumerate(s["gaps"])}))
    sep_df = pd.DataFrame(sep_rows)
    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print(f"SEPARATING COLUMNS (threshold = class_sep/n_clusters_per_class = "
          f"{CLASS_SEP / N_CLUSTERS:.2f}; gaps measured on non-holdout rows only)")
    print(sep_df.to_string(index=False, float_format=fmt))
    n_amb = int((sep_df.ambiguous != "").sum())
    print(f"\nAmbiguous labels: {n_amb} (expected 0)")
    same = all(sep_sets[(50, s)] == sep_sets[(500, s)] for s in sorted(df.seed.unique()))
    print(f"Separating set identical for d=50 and d=500 at every seed: {same} "
          f"(centroids are drawn before the noise columns, so this should be True)")
    print(f"Mean separating columns per seed: {sep_df[sep_df.d == 50].n_sep.mean():.2f} "
          f"(theory for 2 distinct random vertices of a 5-cube: {5 * 16 / 31:.2f})")

    # -- re-score every saved mask --------------------------------------------
    def score(r):
        chosen = set(map(int, str(r.idx).split(",")))
        sep = sep_sets[(r.d, r.seed)]
        sh = len(chosen & sep)
        return pd.Series(dict(n_sep=len(sep), sep_hits=sh,
                              sep_recall=sh / len(sep) if sep else np.nan,
                              dead_hits=int(r.hits) - sh))
    scored = df.apply(score, axis=1)
    # Post-F5 validate_clean_signal.csv already has sep_hits / n_sep computed
    # the same way; they must agree, then the rescored values replace them.
    for c in ("sep_hits", "n_sep"):
        if c in df:
            assert (df[c].astype(int).values == scored[c].astype(int).values).all(), \
                f"{c} in validate_clean_signal.csv disagrees with the rescore"
    out = pd.concat([df.drop(columns=[c for c in scored.columns if c in df]), scored], axis=1)

    print("\nMEAN +/- STD per (d, n_search, method)   [n=10 seeds]")
    order = {m: i for i, m in enumerate(["firefly", "firefly_levy", "random_search", "true_mask"])}
    summ = (out.groupby(["d", "n_search", "method"])[["hits", "sep_hits", "n_sep", "sep_recall", "dead_hits"]]
               .agg(["mean", "std"]).reset_index())
    summ.columns = ["_".join(c).strip("_") for c in summ.columns]
    summ = summ.sort_values(["d", "n_search", "method"],
                            key=lambda s: s.map(order) if s.name == "method" else s)
    print(summ.to_string(index=False, float_format=fmt))

    print("\nPAIRED WILCOXON vs random_search (two-sided, paired by seed, n=10; "
          "mean_diff = method - random; zero differences dropped)")
    tests = []
    for (d, ns), g in out.groupby(["d", "n_search"]):
        rs = g[g.method == "random_search"].sort_values("seed")
        for m in COMPARE:
            fm = g[g.method == m].sort_values("seed")
            assert list(fm.seed) == list(rs.seed)
            for metric in ("sep_hits", "sep_recall"):
                tests.append(dict(d=d, n_search=ns, method=m, metric=metric,
                                  **paired_wilcoxon(fm[metric].values, rs[metric].values)))
    print(pd.DataFrame(tests).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    csv_sep = RESULTS_DIR / "separating_columns_clean_signal.csv"
    csv_out = RESULTS_DIR / "validate_clean_signal_rescored.csv"
    sep_df.to_csv(csv_sep, index=False)
    out.to_csv(csv_out, index=False)
    total = time.perf_counter() - t0
    print(f"\nSaved {csv_sep}\nSaved {csv_out}\nTotal runtime: {total:.1f} s")
    log_experiment(
        name="rescore_clean_signal",
        params=dict(source="results/validate_clean_signal.csv", class_sep=CLASS_SEP,
                    n_clusters_per_class=N_CLUSTERS, n_informative=N_INF,
                    threshold=CLASS_SEP / N_CLUSTERS, gap_rows="non-holdout (3000)",
                    ks=[5]),
        seeds=sorted(int(s) for s in df.seed.unique()),
        csv_paths=[str(p.relative_to(RESULTS_DIR.parent)) for p in (csv_sep, csv_out)],
        extra=dict(runtime_s=round(total, 1)),
    )


if __name__ == "__main__":
    main()
