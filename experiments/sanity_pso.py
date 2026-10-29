"""Day 4 Stage 2 sanity check: PSO with random-key (top-k) decoding on the
clean-signal toy, against random search and Firefly.

Data and splits: exactly those of experiments/validate_clean_signal.py
(make_splits / separating_set imported from it): make_classification
(8000 rows, 5 informative cols 0..4, class_sep=2, 1 cluster/class,
shuffle=False, random_state=seed), d in {50, 500}, seeds 0-9, search sets of
1000 and 2000 rows, 5000-row holdout.
Search: k=5, budget=600, 20 fireflies / particles, default hyperparameters,
fresh make_fitness per run (same inner split for all methods of a seed).
For each seed, Firefly and PSO start from the same initial population.

Metrics
  val_fitness  objective on the inner-validation rows that chose the mask
               (in-sample; a sanity signal, not a basis for any claim)
  sep_hits     selected SEPARATING informative columns
               (src.synthetic.separating_columns, as validate_clean_signal)
  hits         selected informative columns (any of 0..4)
  holdout_acc  final_clf() accuracy on the 5000-row holdout (reported only)

Tests: paired two-sided Wilcoxon, PSO vs random search, on val_fitness and
sep_hits for each (d, n_search) -> 8 tests, Holm-corrected as one family.
Exploratory sanity check; results are "higher / lower / equal", never
"beats".  With n=10 the smallest two-sided p is about 0.002.
Expectation from CLAUDE.md Stage 2: PSO val_fitness at or below random search.

History
  v0 (plain PSO, results/sanity_pso_v0_noreset.csv): swarm collapsed, budget
     30-80% unspent, val_fitness above random search in 4/4 settings.
  v1 (velocity redraw on stagnation, run 20261029_133804_sanity_pso): did
     not fix it, 2/40 runs spent the budget
     (results/sanity_pso_v1_velreset.csv).
  v2 (now): Firefly-style stall swap (src/pso.py, stall_swap=True).
Acceptance for the fix, fixed BEFORE this run and judged on budget use only
(not on fitness): at least 38/40 PSO runs spend the full budget.

Output: results/sanity_pso.csv (per-run rows), results/sanity_pso_tests.csv,
entry in results/runs.json.

Usage (from firefly-fs/, venv active):
    python experiments/sanity_pso.py
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _common import (RESULTS_DIR, direction, holm, paired_wilcoxon,  # noqa: E402
                     random_search, warn_once)
from validate_clean_signal import (CLASS_SEP, N_CLUSTERS, N_INF,  # noqa: E402
                                   holdout_acc, make_splits, separating_set)

from src.classifiers import clf_configs  # noqa: E402
from src.firefly import binary_firefly, make_fitness  # noqa: E402
from src.pso import binary_pso  # noqa: E402
from src.results_store import log_experiment  # noqa: E402

DS = [50, 500]
SEEDS = list(range(10))
K, BUDGET, N_POP = 5, 600, 20
METHODS = ["random_search", "firefly", "pso"]
TEST_METRICS = ["val_fitness", "sep_hits"]


def run_one(method, Xs, ys, d, seed):
    fitness, state = make_fitness(Xs, ys, k=K, seed=seed, max_rows=len(ys))
    t0 = time.perf_counter()
    if method == "random_search":
        out = random_search(fitness, state, d=d, k=K, budget=BUDGET, seed=seed)
        mask, val_fit, iters = out["best_mask"], out["best_fit"], 0
    elif method == "firefly":
        mask, val_fit, hist = binary_firefly(
            fitness, state, d=d, k=K, budget=BUDGET, n_fireflies=N_POP, seed=seed)
        iters = len(hist)
    else:
        mask, val_fit, hist = binary_pso(
            fitness, state, d=d, k=K, budget=BUDGET, n_particles=N_POP, seed=seed)
        iters = len(hist)
    div = state.get("diversity", [])
    return dict(mask=mask, val_fitness=float(val_fit), evals_used=state["evals"],
                iters=iters, stall_iters=int(state.get("pso_stall_iters", 0)),
                diversity_last=float(div[-1]) if div else np.nan,
                seconds=time.perf_counter() - t0)


def main():
    warn_once()
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []
    t_all = time.perf_counter()

    for d in DS:
        for seed in SEEDS:
            searches, (Xh, yh) = make_splits(d, seed)
            sep = separating_set(d, seed)
            for n_search, (Xs, ys) in searches.items():
                for method in METHODS:
                    r = run_one(method, Xs, ys, d, seed)
                    mask = r.pop("mask")
                    assert int(mask.sum()) == K and r["evals_used"] <= BUDGET
                    rows.append(dict(
                        d=d, n_search=n_search, n_val=len(ys) // 4, seed=seed, k=K,
                        budget=BUDGET, method=method, n_sep=len(sep),
                        sep_hits=len(set(np.flatnonzero(mask).tolist()) & sep),
                        hits=int(mask[:N_INF].sum()),
                        holdout_acc=holdout_acc(mask, Xs, ys, Xh, yh, seed),
                        idx=",".join(map(str, np.flatnonzero(mask))), **r))
            print(f"d={d} seed {seed} done | elapsed "
                  f"{(time.perf_counter() - t_all) / 60:.1f} min", flush=True)

    df = pd.DataFrame(rows)
    csv = RESULTS_DIR / "sanity_pso.csv"
    df.to_csv(csv, index=False)
    fmt = lambda v: f"{v:.4f}"  # noqa: E731
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)

    print("\nMEAN +/- STD per (d, n_search, method)   [n=10 seeds]")
    order = {m: i for i, m in enumerate(METHODS)}
    summ = (df.groupby(["d", "n_search", "method"])
              [["val_fitness", "sep_hits", "hits", "holdout_acc", "evals_used", "iters",
                "stall_iters"]]
              .agg(["mean", "std"]).reset_index())
    summ.columns = ["_".join(c).strip("_") for c in summ.columns]
    summ = summ.sort_values(["d", "n_search", "method"],
                            key=lambda s: s.map(order) if s.name == "method" else s)
    print(summ.to_string(index=False, float_format=fmt))

    tests = []
    for (d, ns), g in df.groupby(["d", "n_search"]):
        rs = g[g.method == "random_search"].sort_values("seed")
        ps = g[g.method == "pso"].sort_values("seed")
        assert list(ps.seed) == list(rs.seed)
        for metric in TEST_METRICS:
            t = paired_wilcoxon(ps[metric].values, rs[metric].values)
            tests.append(dict(d=d, n_search=ns, comparison="pso - random_search",
                              metric=metric, **t))
    tdf = pd.DataFrame(tests)
    tdf["p_holm"] = holm(tdf.p.values)
    tdf["direction"] = tdf.mean_diff.map(direction)

    print(f"\nPAIRED WILCOXON, PSO vs random search (two-sided, paired by seed, n=10; "
          f"mean_diff = pso - random; zero differences dropped; Holm across "
          f"{len(tdf)} tests; exploratory)")
    print("  val_fitness: lower is better (in-sample).  sep_hits: higher is better.")
    print(tdf[["d", "n_search", "metric", "direction", "mean_diff", "n_nonzero",
               "p", "p_holm"]].to_string(index=False, float_format=fmt))

    vf = tdf[tdf.metric == "val_fitness"]
    n_ok = int((vf.mean_diff <= 0).sum())
    print(f"\nExpectation check (CLAUDE.md Stage 2): PSO mean val_fitness <= random "
          f"search in {n_ok}/{len(vf)} (d, n_search) settings.")
    pso = df[df.method == "pso"]
    n_full = int((pso.evals_used == BUDGET).sum())
    print(f"PSO budget use: {n_full}/{len(pso)} runs spent the full budget of {BUDGET}; "
          f"min evals_used {int(pso.evals_used.min())}.")
    print(f"Acceptance (fixed before the run, >= 38/{len(pso)} full-budget runs): "
          f"{'PASS' if n_full >= 38 else 'FAIL'}")

    csv_tests = RESULTS_DIR / "sanity_pso_tests.csv"
    tdf.to_csv(csv_tests, index=False)
    total = time.perf_counter() - t_all
    print(f"\nSaved {csv}\nSaved {csv_tests}\nTotal runtime: {total:.1f} s")
    log_experiment(
        name="sanity_pso",
        params=dict(ds=DS, n_search=[1000, 2000], n_informative=N_INF, ks=[K],
                    budget=BUDGET, n_population=N_POP, class_sep=CLASS_SEP,
                    n_clusters_per_class=N_CLUSTERS, methods=METHODS,
                    data="validate_clean_signal.make_splits",
                    pso="random-key (top-k) decoding, defaults, stall_swap=True",
                    acceptance=">= 38/40 PSO runs spend the full budget",
                    holdout_clf=clf_configs()["final_clf"],
                    multiple_testing="holm, 8 tests (pso vs random_search)"),
        seeds=SEEDS,
        csv_paths=[str(p.relative_to(RESULTS_DIR.parent)) for p in (csv, csv_tests)],
        extra=dict(runtime_s=round(total, 1)),
    )


if __name__ == "__main__":
    main()
