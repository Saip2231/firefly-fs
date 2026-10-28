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

Decision rules (fixed before running):
  d=50  : "reaches >= 4/5"   <=> firefly mean hits >= 4.0
  d=500 : "beats random"     <=> lower mean val fitness AND Wilcoxon p < 0.05

Output: results/validate_clean_signal.csv, entry in results/runs.json.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from _common import RESULTS_DIR, paired_wilcoxon, random_search, warn_once

from src.firefly import binary_firefly, make_fitness
from src.results_store import log_experiment

DS = [50, 500]
N_SEARCH = [1000, 2000]
SEEDS = list(range(10))
N_INF, K, BUDGET, N_FIREFLIES = 5, 5, 600, 20
N_TOTAL, N_HOLDOUT = 8000, 5000
METHODS = ["firefly", "firefly_levy", "random_search"]


def make_splits(d: int, seed: int):
    X, y = make_classification(
        n_samples=N_TOTAL, n_features=d, n_informative=N_INF,
        n_redundant=0, n_repeated=0, n_clusters_per_class=1,
        class_sep=2.0, shuffle=False, random_state=seed,
    )
    X_rest, X_hold, y_rest, y_hold = train_test_split(
        X, y, test_size=N_HOLDOUT, stratify=y, random_state=seed)
    X2k, _, y2k, _ = train_test_split(
        X_rest, y_rest, train_size=2000, stratify=y_rest, random_state=seed)
    X1k, _, y1k, _ = train_test_split(
        X2k, y2k, train_size=1000, stratify=y2k, random_state=seed)
    return {1000: (X1k, y1k), 2000: (X2k, y2k)}, (X_hold, y_hold)


def holdout_acc(mask, Xs, ys, Xh, yh, seed):
    # Same LogisticRegression settings as make_fitness.
    clf = LogisticRegression(max_iter=100, random_state=seed)
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
            true_mask = np.zeros(d, dtype=bool)
            true_mask[:N_INF] = True
            for n_search, (Xs, ys) in searches.items():
                base = dict(d=d, n_search=n_search, n_val=len(ys) // 4,
                            seed=seed, k=K, budget=BUDGET)
                # Reference only: fresh fitness, same inner split as the methods
                ref_fit, _ = make_fitness(Xs, ys, k=K, seed=seed, max_rows=len(ys))
                rows.append({**base, "method": "true_mask", "hits": N_INF,
                             "val_fitness": float(ref_fit(true_mask)),
                             "holdout_acc": holdout_acc(true_mask, Xs, ys, Xh, yh, seed),
                             "evals_used": 1, "sweeps": 0, "seconds": 0.0, "idx": "0,1,2,3,4"})
                for method in METHODS:
                    mask, vf, ev, sw, secs = run_one(method, Xs, ys, d, seed)
                    rows.append({**base, "method": method,
                                 "hits": int(mask[:N_INF].sum()),
                                 "val_fitness": vf,
                                 "holdout_acc": holdout_acc(mask, Xs, ys, Xh, yh, seed),
                                 "evals_used": ev, "sweeps": sw, "seconds": secs,
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
    print(df[["d", "n_search", "seed", "method", "hits", "val_fitness",
              "holdout_acc", "evals_used", "sweeps"]].to_string(index=False, float_format=fmt))

    print("\nMEAN +/- STD per (d, n_search, method)   [n=10 seeds]")
    order = {m: i for i, m in enumerate(METHODS + ["true_mask"])}
    summ = (df.groupby(["d", "n_search", "method"])[["hits", "val_fitness", "holdout_acc", "evals_used"]]
              .agg(["mean", "std"]).reset_index())
    summ.columns = ["_".join(c).strip("_") for c in summ.columns]
    summ = summ.sort_values(["d", "n_search", "method"],
                            key=lambda s: s.map(order) if s.name == "method" else s)
    print(summ.to_string(index=False, float_format=fmt))

    print("\nPAIRED WILCOXON vs random_search (two-sided, paired by seed, n=10; "
          "mean_diff = method - random; zero differences dropped)")
    tests = []
    for (d, ns), g in df.groupby(["d", "n_search"]):
        rs = g[g.method == "random_search"].sort_values("seed")
        for m in ("firefly", "firefly_levy"):
            fm = g[g.method == m].sort_values("seed")
            assert list(fm.seed) == list(rs.seed)
            for metric in ("hits", "val_fitness"):
                t = paired_wilcoxon(fm[metric].values, rs[metric].values)
                tests.append(dict(d=d, n_search=ns, method=m, metric=metric, **t))
    tdf = pd.DataFrame(tests)
    print(tdf.to_string(index=False, float_format=fmt))

    print("\nPLAIN ANSWERS (rules fixed before running)")
    for ns in N_SEARCH:
        for m in ("firefly", "firefly_levy"):
            h = df[(df.d == 50) & (df.n_search == ns) & (df.method == m)].hits.mean()
            print(f"  d=50,  n_search={ns}, {m:<12}: mean hits {h:.2f}/5 -> "
                  f"{'REACHES' if h >= 4.0 else 'DOES NOT reach'} >= 4/5")
        for m in ("firefly", "firefly_levy"):
            t = tdf[(tdf.d == 500) & (tdf.n_search == ns) & (tdf.method == m)
                    & (tdf.metric == "val_fitness")].iloc[0]
            beats = t.mean_diff < 0 and not np.isnan(t.p) and t.p < 0.05
            print(f"  d=500, n_search={ns}, {m:<12}: val-fitness diff {t.mean_diff:+.4f}, "
                  f"p={t.p:.4f} (n=10, nonzero={t.n_nonzero}) -> "
                  f"{'BEATS' if beats else 'does NOT significantly beat'} random search")

    total = time.perf_counter() - t_all
    print(f"\nSaved {csv}\nTotal runtime: {total:.1f} s")
    log_experiment(
        name="validate_clean_signal",
        params=dict(ds=DS, n_search=N_SEARCH, n_informative=N_INF, ks=[K],
                    budget=BUDGET, n_fireflies=N_FIREFLIES, n_total=N_TOTAL,
                    n_holdout=N_HOLDOUT, class_sep=2.0, n_clusters_per_class=1,
                    methods=METHODS, data_random_state="seed",
                    holdout_clf="LogisticRegression(max_iter=100, random_state=seed)"),
        seeds=SEEDS,
        csv_paths=[str(csv.relative_to(RESULTS_DIR.parent))],
        extra=dict(runtime_s=round(total, 1)),
    )


if __name__ == "__main__":
    main()
