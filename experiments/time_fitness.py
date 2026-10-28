"""Stage 4 - Time the fitness function on the real data (Day 5 compute plan).

Single-thread timing: BLAS/OpenMP thread env vars are set to 1 BEFORE numpy
is imported, and all timing runs inside threadpool_limits(1).
Training split only (load_dataset(name, seed=0)); the test split is unused.

Per dataset and k:
  * make_fitness(max_rows=2000) setup time (paid once per run)
  * 1 warm-up eval (excluded), then 30 distinct random k-masks, one NEW
    evaluation each -> mean / median / p95 seconds, share hitting max_iter
  * one rerank(top_m=10, n_splits=5) on the FULL training set
Estimate: seconds/run = 800*mean_eval + setup + rerank;
  total = sum_k 10 seeds * 3 metaheuristics * seconds/run, for 1 worker and
  for (cpu_count-1) workers (ideal linear scaling; a lower bound).
Outputs: results/time_fitness.csv (raw per-mask), results/time_fitness.md,
  entry in results/runs.json.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from _common import RESULTS_DIR, warn_once  # noqa: E402

from src.data import load_dataset  # noqa: E402
from src.firefly import make_fitness, rerank  # noqa: E402
from src.results_store import log_experiment  # noqa: E402

DATASETS = {"cifar_r18": [16, 32, 64, 128], "digits": [8, 16, 32]}
N_MASKS, MAX_ROWS, SEED = 30, 2000, 0
BUDGET, N_SEEDS, N_METHODS = 800, 10, 3          # firefly, firefly_levy, pso
TOP_M, N_SPLITS = 10, 5
LIMIT_H = 6.0


def random_masks(d: int, k: int, n: int, rng) -> list[np.ndarray]:
    """n+1 distinct random k-masks (first one is the warm-up)."""
    seen, out = set(), []
    while len(out) < n + 1:
        m = np.zeros(d, dtype=bool)
        m[rng.choice(d, size=k, replace=False)] = True
        if m.tobytes() not in seen:
            seen.add(m.tobytes())
            out.append(m)
    return out


def time_one_k(Xtr, ytr, k, rng):
    t0 = time.perf_counter()
    fitness, state = make_fitness(Xtr, ytr, k=k, seed=SEED, max_rows=MAX_ROWS)
    setup = time.perf_counter() - t0

    masks = random_masks(Xtr.shape[1], k, N_MASKS, rng)
    fitness(masks[0])                                   # warm-up, excluded
    rows = []
    for i, m in enumerate(masks[1:]):
        before = state["evals"]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t = time.perf_counter()
            fitness(m)
            secs = time.perf_counter() - t
        assert state["evals"] == before + 1, "expected a NEW (uncached) evaluation"
        rows.append(dict(mask_i=i, seconds=secs,
                         hit_max_iter=any(issubclass(w.category, ConvergenceWarning) for w in caught)))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t = time.perf_counter()
        _, _, rr_fits = rerank(state, Xtr, ytr, k, top_m=TOP_M, n_splits=N_SPLITS, seed=SEED)
        rr_secs = time.perf_counter() - t
    rr_conv = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    return setup, rows, rr_secs, rr_fits, rr_conv


def hms(s: float) -> str:
    return f"{s / 3600:.2f} h" if s >= 3600 else f"{s / 60:.1f} min"


def main():
    warn_once()
    RESULTS_DIR.mkdir(exist_ok=True)
    cpu = os.cpu_count() or 1
    workers = max(1, cpu - 1)
    print(f"os.cpu_count() = {cpu}  -> parallel estimate uses {workers} workers")
    print(f"numpy {np.__version__}, scikit-learn {sklearn.__version__}, single-thread BLAS")

    raw, summary = [], []
    t_all = time.perf_counter()
    with threadpool_limits(limits=1):
        for name, ks in DATASETS.items():
            Xtr, _, ytr, _ = load_dataset(name, seed=SEED)       # test split unused
            print(f"\n{name}: X_train {Xtr.shape}, classes {len(np.unique(ytr))}, "
                  f"fitness rows {min(MAX_ROWS, len(ytr))}", flush=True)
            rng = np.random.default_rng(SEED)
            for k in ks:
                setup, rows, rr_secs, rr_fits, rr_conv = time_one_k(Xtr, ytr, k, rng)
                s = np.array([r["seconds"] for r in rows])
                hit = np.mean([r["hit_max_iter"] for r in rows])
                raw += [dict(dataset=name, k=k, **r) for r in rows]
                per_run = BUDGET * s.mean() + setup + rr_secs
                summary.append(dict(
                    dataset=name, k=k, n_train=len(ytr), mean_s=s.mean(),
                    median_s=float(np.median(s)), p95_s=float(np.percentile(s, 95)),
                    hit_max_iter=hit, setup_s=setup, rerank_s=rr_secs,
                    rerank_fits=rr_fits, rerank_hit_max_iter=rr_conv / rr_fits,
                    per_run_s=per_run, runs=N_SEEDS * N_METHODS,
                    total_1w_s=N_SEEDS * N_METHODS * per_run))
                print(f"  k={k:<4} mean {s.mean():.4f}s  median {np.median(s):.4f}s  "
                      f"p95 {np.percentile(s, 95):.4f}s  hit max_iter {hit:.0%}  "
                      f"rerank {rr_secs:.1f}s  | elapsed {(time.perf_counter() - t_all) / 60:.1f} min",
                      flush=True)

    sm = pd.DataFrame(summary)
    lines = [
        "# Fitness timing (Stage 4)", "",
        f"- os.cpu_count() = {cpu}; parallel column uses {workers} workers (ideal scaling, lower bound)",
        f"- numpy {np.__version__}, scikit-learn {sklearn.__version__}; single-thread BLAS",
        f"- make_fitness(max_rows={MAX_ROWS}), LogisticRegression(max_iter=100); "
        f"{N_MASKS} distinct random masks per k, 1 warm-up excluded",
        f"- rerank: top_m={TOP_M}, {N_SPLITS}-fold CV on the full training set",
        f"- per run = {BUDGET} x mean eval + setup + rerank; "
        f"{N_SEEDS} seeds x {N_METHODS} metaheuristics per k (PSO assumed same per-eval cost)",
        "",
        "## Per k", "",
        "| dataset | k | mean s/eval | median | p95 | evals hitting max_iter | setup s | rerank s | s/run | 30 runs, 1 worker |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sm.itertuples():
        lines.append(f"| {r.dataset} | {r.k} | {r.mean_s:.4f} | {r.median_s:.4f} | {r.p95_s:.4f} | "
                     f"{r.hit_max_iter:.0%} | {r.setup_s:.2f} | {r.rerank_s:.1f} | "
                     f"{r.per_run_s:.0f} | {hms(r.total_1w_s)} |")
    lines += ["", "## Total wall time per dataset", "",
              f"| dataset | runs | 1 worker | {workers} workers |", "|---|---|---|---|"]
    over = []
    for name, g in sm.groupby("dataset", sort=False):
        t1 = g.total_1w_s.sum()
        tw = t1 / workers
        lines.append(f"| {name} | {int(g.runs.sum())} | {hms(t1)} | {hms(tw)} |")
        for label, t in (("1 worker", t1), (f"{workers} workers", tw)):
            if t > LIMIT_H * 3600:
                over.append((name, label, t, g))

    lines += ["", f"## Reductions (suggested only, not applied; limit {LIMIT_H:.0f} h)", ""]
    if not over:
        lines.append(f"No estimate exceeds {LIMIT_H:.0f} h; no reduction needed.")
    for name, label, t, g in over:
        t1 = g.total_1w_s.sum()
        div = t1 / t
        biggest = g.loc[g.total_1w_s.idxmax()]
        lines += [
            f"**{name}, {label}: {hms(t)} > {LIMIT_H:.0f} h.** In order:",
            f"1. max_rows: eval time grows with rows; {MAX_ROWS} -> 1000 should cut most of the "
            f"{BUDGET} x mean-eval term (not measured here; re-time before relying on it). "
            f"Also widens validation noise (250 vs 500 val rows).",
            f"2. LogisticRegression max_iter / tol: {g.hit_max_iter.mean():.0%} of timed evals hit "
            f"max_iter=100, so lowering max_iter caps those fits; raising tol helps fits that converge "
            f"early. Saving not measured.",
            f"3. seeds {N_SEEDS} -> 5: exactly halves -> {hms(t / 2)}.",
            f"4. drop k={int(biggest.k)} (largest share): -> {hms((t1 - biggest.total_1w_s) / div)}.",
            "",
        ]
    md = "\n".join(lines)
    print("\n" + md)

    csv = RESULTS_DIR / "time_fitness.csv"
    md_path = RESULTS_DIR / "time_fitness.md"
    pd.DataFrame(raw).to_csv(csv, index=False)
    md_path.write_text(md + "\n", encoding="utf-8")
    total = time.perf_counter() - t_all
    print(f"\nSaved {csv}\nSaved {md_path}\nTotal runtime: {total:.1f} s")
    log_experiment(
        name="time_fitness",
        params=dict(datasets=DATASETS, n_masks=N_MASKS, max_rows=MAX_ROWS, budget=BUDGET,
                    n_seeds_planned=N_SEEDS, n_methods=N_METHODS, rerank_top_m=TOP_M,
                    n_splits=N_SPLITS, threads=1, cpu_count=cpu, workers=workers,
                    ks=sorted({k for ks in DATASETS.values() for k in ks})),
        seeds=[SEED],
        csv_paths=[str(csv.relative_to(RESULTS_DIR.parent)),
                   str(md_path.relative_to(RESULTS_DIR.parent))],
        extra=dict(runtime_s=round(total, 1),
                   per_k=sm[["dataset", "k", "mean_s", "median_s", "p95_s", "rerank_s",
                             "per_run_s"]].to_dict(orient="records")),
    )


if __name__ == "__main__":
    main()
