"""Stage 2 - Validate binary Firefly on clean, strong-signal data.

Data (per seed, random_state=seed):
  make_classification(n_samples=8000, n_features=d, n_informative=5,
                      n_redundant=0, n_repeated=0, n_clusters_per_class=1,
                      class_sep=2.0, shuffle=False)   -> informative cols 0..4
Splits (stratified, nested):
  holdout  5000 rows  - never seen by search; used once per final mask
  search   2000 rows  - from the remaining 3000
  search   1000 rows  - stratified subset of the 2000
Fitness: make_fitness(max_rows=n_search) -> inner 75/25 split gives
  250 (n_search=1000) or 500 (n_search=2000) validation rows.
Methods: firefly, firefly_levy, random_search; k=5, budget=600, d in {50, 500}.
Reference: true_mask (cols 0..4), excluded from all statistics.

Metrics (Day-3 review, F5)
  PRIMARY   holdout_acc  - final_clf() accuracy on the 5000-row holdout
            sep_hits     - selected columns among the SEPARATING informative
                           columns (src.synthetic.separating_columns on the
                           3000 non-holdout rows, threshold class_sep/1 = 2.0;
                           same rows and rule as rescore_clean_signal.py)
  SECONDARY hits (any informative column), val_fitness.  val_fitness is the
            objective the search minimised on the same validation rows that
            chose the mask, so it is in-sample and never a basis for a claim.

Single primary test (named before the F5 rerun):
  holdout_acc, firefly vs random_search, d=500, n_search=2000; paired
  two-sided Wilcoxon over seeds 0-9.
All 32 tests in this script (2 d x 2 n_search x 2 methods x 4 metrics) are
Holm-corrected as one family; results are reported as "higher / lower /
equal, mean_diff, p, Holm-adjusted p", never as "beats".  With n=10 the
smallest two-sided p is about 0.002, so power is limited.

(Superseded Day-3 rules, kept for the record: "reaches >= 4/5" on mean hits
at d=50 and "beats random" on val fitness at d=500.  They were judged on
in-sample val fitness and uncorrected p-values.)

Comparison rule (Day-3 review, F2): all comparisons use RAW masks (equal
search budget); no method is reranked here.  Rows report evals_used,
rerank_fits (always 0) and total_fits = evals_used + rerank_fits.

Output: results/validate_clean_signal.csv, entry in results/runs.json.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from _common import (RESULTS_DIR, direction, holm, paired_wilcoxon,
                     random_search, warn_once)

from src.classifiers import clf_configs, final_clf
from src.firefly import binary_firefly, make_fitness
from src.results_store import log_experiment
from src.synthetic import separating_columns

DS = [50, 500]
N_SEARCH = [1000, 2000]
SEEDS = list(range(10))
N_INF, K, BUDGET, N_FIREFLIES = 5, 5, 600, 20
N_TOTAL, N_HOLDOUT = 8000, 5000
CLASS_SEP, N_CLUSTERS = 2.0, 1
METHODS = ["firefly", "firefly_levy", "random_search"]
TEST_METRICS = ["holdout_acc", "sep_hits", "hits", "val_fitness"]
PRIMARY = dict(d=500, n_search=2000, method="firefly", metric="holdout_acc")


def _rest_and_holdout(d: int, seed: int):
    X, y = make_classification(
        n_samples=N_TOTAL, n_features=d, n_informative=N_INF,
        n_redundant=0, n_repeated=0, n_clusters_per_class=N_CLUSTERS,
        class_sep=CLASS_SEP, shuffle=False, random_state=seed,
    )
    return train_test_split(X, y, test_size=N_HOLDOUT, stratify=y, random_state=seed)


def separating_set(d: int, seed: int) -> set[int]:
    """Separating informative columns, from the 3000 non-holdout rows only."""
    X_rest, _, y_rest, _ = _rest_and_holdout(d, seed)
    s = separating_columns(X_rest, y_rest, N_INF, CLASS_SEP, N_CLUSTERS)
    return set(s["sep_idx"].tolist())


def make_splits(d: int, seed: int):
    X_rest, X_hold, y_rest, y_hold = _rest_and_holdout(d, seed)
    X2k, _, y2k, _ = train_test_split(
        X_rest, y_rest, train_size=2000, stratify=y_rest, random_state=seed)
    X1k, _, y1k, _ = train_test_split(
        X2k, y2k, train_size=1000, stratify=y2k, random_state=seed)
    return {1000: (X1k, y1k), 2000: (X2k, y2k)}, (X_hold, y_hold)


def holdout_acc(mask, Xs, ys, Xh, yh, seed):
    # Shared final classifier (src.classifiers); seed kept for callers.
    clf = final_clf()
    clf.fit(Xs[:, mask], ys)
    return float(clf.score(Xh[:, mask], yh))


def run_one(method, Xs, ys, d, seed):
    fitness, state = make_fitness(Xs, ys, k=K, seed=seed, max_rows=len(ys))
    t0 = time.perf_counter()
    if method == "random_search":
        out = random_search(fitness, state, d=d, k=K, budget=BUDGET, seed=seed)
        mask, val_fit, sweeps = out["best_mask"], out["best_fit"], 0
    else:
        mask, val_fit, hist = binary_firefly(
            fitness, state, d=d, k=K, budget=BUDGET, n_fireflies=N_FIREFLIES,
            levy=(method == "firefly_levy"), seed=seed)
        sweeps = len(hist)
    return mask, float(val_fit), state["evals"], sweeps, time.perf_counter() - t0


def main():
    warn_once()
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []
    t_all = time.perf_counter()

    for d in DS:
        for seed in SEEDS:
            searches, (Xh, yh) = make_splits(d, seed)
            sep = separating_set(d, seed)
            true_mask = np.zeros(d, dtype=bool)
            true_mask[:N_INF] = True
            for n_search, (Xs, ys) in searches.items():
                base = dict(d=d, n_search=n_search, n_val=len(ys) // 4,
                            seed=seed, k=K, budget=BUDGET, n_sep=len(sep))
                # Reference only: fresh fitness, same inner split as the methods
                ref_fit, _ = make_fitness(Xs, ys, k=K, seed=seed, max_rows=len(ys))
                rows.append({**base, "method": "true_mask", "hits": N_INF,
                             "sep_hits": len(sep),
                             "val_fitness": float(ref_fit(true_mask)),
                             "holdout_acc": holdout_acc(true_mask, Xs, ys, Xh, yh, seed),
                             "evals_used": 1, "rerank_fits": 0, "total_fits": 1,
                             "sweeps": 0, "seconds": 0.0, "idx": "0,1,2,3,4"})
                for method in METHODS:
                    mask, vf, ev, sw, secs = run_one(method, Xs, ys, d, seed)
                    rows.append({**base, "method": method,
                                 "hits": int(mask[:N_INF].sum()),
                                 "sep_hits": len(set(np.flatnonzero(mask).tolist()) & sep),
                                 "val_fitness": vf,
                                 "holdout_acc": holdout_acc(mask, Xs, ys, Xh, yh, seed),
                                 "evals_used": ev, "rerank_fits": 0, "total_fits": ev,
                                 "sweeps": sw, "seconds": secs,
                                 "idx": ",".join(map(str, np.flatnonzero(mask)))})
            done = DS.index(d) * len(SEEDS) + SEEDS.index(seed) + 1
            total_units = len(DS) * len(SEEDS)
            el = time.perf_counter() - t_all
            eta = el / done * (total_units - done)
            print(f"[{done:>2}/{total_units}] d={d} seed {seed} done | "
                  f"{6 * done}/{6 * total_units} runs | elapsed {el / 60:.1f} min | "
                  f"ETA ~{eta / 60:.1f} min", flush=True)

    df = pd.DataFrame(rows)
    csv = RESULTS_DIR / "validate_clean_signal.csv"
    df.to_csv(csv, index=False)
    fmt = lambda v: f"{v:.4f}"  # noqa: E731
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)

    print("\nPER-RUN ROWS")
    print(df[["d", "n_search", "seed", "method", "holdout_acc", "sep_hits", "n_sep", "hits",
              "val_fitness", "evals_used", "sweeps"]].to_string(index=False, float_format=fmt))

    print("\nMEAN +/- STD per (d, n_search, method)   [n=10 seeds]")
    order = {m: i for i, m in enumerate(METHODS + ["true_mask"])}
    summ = (df.groupby(["d", "n_search", "method"])[["holdout_acc", "sep_hits", "hits",
                                                     "val_fitness", "evals_used"]]
              .agg(["mean", "std"]).reset_index())
    summ.columns = ["_".join(c).strip("_") for c in summ.columns]
    summ = summ.sort_values(["d", "n_search", "method"],
                            key=lambda s: s.map(order) if s.name == "method" else s)
    print(summ.to_string(index=False, float_format=fmt))

    tests = []
    for (d, ns), g in df.groupby(["d", "n_search"]):
        rs = g[g.method == "random_search"].sort_values("seed")
        for m in ("firefly", "firefly_levy"):
            fm = g[g.method == m].sort_values("seed")
            assert list(fm.seed) == list(rs.seed)
            for metric in TEST_METRICS:
                t = paired_wilcoxon(fm[metric].values, rs[metric].values)
                tests.append(dict(d=d, n_search=ns, method=m, metric=metric,
                                  role="secondary" if metric in ("hits", "val_fitness")
                                  else "primary-metric", **t))
    tdf = pd.DataFrame(tests)
    tdf["p_holm"] = holm(tdf.p.values)
    tdf["direction"] = tdf.mean_diff.map(direction)
    is_primary = np.logical_and.reduce([tdf[c] == v for c, v in PRIMARY.items()])
    tdf.loc[is_primary, "role"] = "PRIMARY TEST"
    assert int(is_primary.sum()) == 1

    print(f"\nPAIRED WILCOXON vs random_search (two-sided, paired by seed, n=10; "
          f"mean_diff = method - random; zero differences dropped; "
          f"Holm across all {len(tdf)} tests)")
    print(tdf[["d", "n_search", "method", "metric", "role", "direction", "mean_diff",
               "n_nonzero", "p", "p_holm"]].to_string(index=False, float_format=fmt))

    t = tdf[is_primary].iloc[0]
    print("\nPRIMARY TEST (named before running): holdout_acc, firefly vs random_search, "
          "d=500, n_search=2000")
    print(f"  firefly holdout accuracy is {t.direction} than random search: mean_diff "
          f"{t.mean_diff:+.4f}, p={t.p:.4f}, Holm-adjusted p={t.p_holm:.4f} "
          f"(n={t.n}, nonzero={t.n_nonzero}; with n=10 power is limited)")

    csv_tests = RESULTS_DIR / "validate_clean_signal_tests.csv"
    tdf.to_csv(csv_tests, index=False)
    total = time.perf_counter() - t_all
    print(f"\nSaved {csv}\nSaved {csv_tests}\nTotal runtime: {total:.1f} s")
    log_experiment(
        name="validate_clean_signal",
        params=dict(ds=DS, n_search=N_SEARCH, n_informative=N_INF, ks=[K],
                    budget=BUDGET, n_fireflies=N_FIREFLIES, n_total=N_TOTAL,
                    n_holdout=N_HOLDOUT, class_sep=2.0, n_clusters_per_class=1,
                    methods=METHODS, data_random_state="seed",
                    holdout_clf=clf_configs()["final_clf"],
                    primary_test=PRIMARY, multiple_testing="holm, all tests in script"),
        seeds=SEEDS,
        csv_paths=[str(p.relative_to(RESULTS_DIR.parent)) for p in (csv, csv_tests)],
        extra=dict(runtime_s=round(total, 1)),
    )


if __name__ == "__main__":
    main()
