"""Day 4 Stage 6: compute plan for the Day-5 metaheuristic grid.

Inputs
  results/time_fitness.md   per (dataset, k): mean s/eval, setup s, rerank s
                            (single process, single-thread BLAS)
  results/tuning_runs.csv   observed search_s of the budget-800 tuning runs,
                            15 workers in parallel -> contention factor
  results/baselines_<ds>.csv  k lists (k >= d skipped, as in the runner)

Estimate per run = (budget * s_eval + setup + rerank) * contention * 1.2
  contention  = mean(observed tuning search_s / (800 * s_eval + setup)) per
                dataset, floored at 1.0 (parallel runs are not faster than a
                lone process).  Digits tuning searched 75 % of Xtr, which can
                only make this factor optimistic for digits.
  1.2         = the 20 % safety margin.
  Not measured: the final test-set evaluation (3 classifiers x 2 masks on the
  full training split); assumed small next to the search and covered by the
  margin.  Check it against the first Day-5 rows.
Grid: datasets {digits, cifar_r18} x budgets {800, 3000} x 4 methods (firefly,
pso, firefly_seeded, pso_seeded) x seeds 0-9 x k.  Hours with W workers =
single-worker hours / W, W = cpu_count - 1 (ideal packing; a lower bound).
Optional firefly_levy (budget 800 only) is listed separately.
If a total with W workers exceeds 8 h, reductions are listed in this order:
fewer budgets, drop firefly_levy, fewer seeds, fewer k values.

Output: results/compute_plan.md (+ runs.json entry).

Usage (from firefly-fs/, venv active):
    python experiments/plan_compute.py
"""

from __future__ import annotations

import os
import re

import pandas as pd

from _common import RESULTS_DIR, git_hash

from src.results_store import log_experiment

DATASETS = ["digits", "cifar_r18"]
BUDGETS = [800, 3000]
METHODS = ["firefly", "pso", "firefly_seeded", "pso_seeded"]
N_SEEDS = 10
MARGIN = 1.2
LIMIT_H = 8.0
TUNING_BUDGET = 800
OUT = RESULTS_DIR / "compute_plan.md"


def read_timing(path=RESULTS_DIR / "time_fitness.md") -> pd.DataFrame:
    """Parse the 'Per k' table: dataset, k, s_eval, setup_s, rerank_s."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 8 and re.fullmatch(r"\d+", cells[1] or ""):
            rows.append(dict(dataset=cells[0], k=int(cells[1]), s_eval=float(cells[2]),
                             setup_s=float(cells[6]), rerank_s=float(cells[7])))
    if not rows:
        raise SystemExit(f"no per-k timing rows found in {path}")
    return pd.DataFrame(rows)


def grid_ks(dataset: str) -> list[int]:
    df = pd.read_csv(RESULTS_DIR / f"baselines_{dataset}.csv")
    d = int(df.loc[df.method == "All Features", "k"].iloc[0])
    return sorted(int(k) for k in df.loc[df.method != "All Features", "k"].unique() if k < d)


def contention(timing: pd.DataFrame) -> dict[str, float]:
    """Observed / predicted search time of the tuning runs, per dataset (>= 1)."""
    tr = pd.read_csv(RESULTS_DIR / "tuning_runs.csv")
    m = tr.merge(timing, on=["dataset", "k"], how="inner")
    m["ratio"] = m.search_s / (TUNING_BUDGET * m.s_eval + m.setup_s)
    return {ds: max(1.0, float(g.ratio.mean())) for ds, g in m.groupby("dataset")}


def per_run_seconds(t: pd.Series, budget: int, factor: float) -> float:
    return (budget * t.s_eval + t.setup_s + t.rerank_s) * factor * MARGIN


def main():
    cpu = os.cpu_count() or 2
    workers = max(1, cpu - 1)
    timing = read_timing()
    factors = contention(timing)

    rows, missing = [], []
    for ds in DATASETS:
        ks = grid_ks(ds)
        tds = timing[timing.dataset == ds].set_index("k")
        absent = [k for k in ks if k not in tds.index]
        if absent:
            missing.append(f"{ds} k={absent}")
            continue
        f = factors.get(ds, 1.0)
        for b in BUDGETS:
            secs = [per_run_seconds(tds.loc[k], b, f) for k in ks]
            n_runs = len(METHODS) * N_SEEDS * len(ks)
            single_h = sum(secs) * len(METHODS) * N_SEEDS / 3600
            rows.append(dict(dataset=ds, budget=b, ks=ks, runs=n_runs, factor=f,
                             s_min=min(secs), s_mean=sum(secs) / len(secs), s_max=max(secs),
                             single_h=single_h, workers_h=single_h / workers))
    if missing:
        raise SystemExit(f"time_fitness.md has no timing for {missing}; rerun time_fitness.py")

    # optional firefly_levy, budget 800 only
    levy_single = sum(r["single_h"] for r in rows if r["budget"] == 800) / len(METHODS)

    tot_single = sum(r["single_h"] for r in rows)
    tot_workers = tot_single / workers
    lines = ["# Compute plan (Day 4 Stage 6)", "",
             f"- os.cpu_count() = {cpu}; workers = cpu_count - 1 = {workers}; "
             f"single-thread BLAS per worker.",
             f"- Per run = (budget x s/eval + setup + rerank) x contention x {MARGIN} "
             f"(20 % margin); timing from results/time_fitness.md.",
             "- Contention factor (observed / predicted search time of the budget-800 tuning "
             "runs with 15 workers, floored at 1.0): "
             + ", ".join(f"{ds} {f:.2f}" for ds, f in factors.items()) + ".",
             "- Not measured: final test evaluation (3 classifiers x 2 masks); assumed covered "
             "by the margin. Hours with workers assume ideal packing (lower bound).",
             f"- Grid per dataset x budget: {len(METHODS)} methods ({', '.join(METHODS)}) x "
             f"{N_SEEDS} seeds x k list.", "",
             f"| dataset | budget | k values | runs | s/run min | s/run mean | s/run max | "
             f"1-worker h | {workers}-worker h |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['budget']} | {', '.join(map(str, r['ks']))} | "
                     f"{r['runs']} | {r['s_min']:.0f} | {r['s_mean']:.0f} | {r['s_max']:.0f} | "
                     f"{r['single_h']:.2f} | {r['workers_h']:.2f} |")
    lines += [f"| **total** | | | {sum(r['runs'] for r in rows)} | | | | "
              f"**{tot_single:.2f}** | **{tot_workers:.2f}** |", "",
              f"Optional firefly_levy (budget 800 only): +{levy_single:.2f} h on 1 worker, "
              f"+{levy_single / workers:.2f} h on {workers} workers.", ""]

    over = [r for r in rows if r["workers_h"] > LIMIT_H]
    if over or tot_workers > LIMIT_H:
        where = [f"{r['dataset']}/{r['budget']}" for r in over]
        if tot_workers > LIMIT_H:
            where.append("grand total")
        lines += [f"**Over the {LIMIT_H:.0f} h limit** ({', '.join(where)}). "
                  "Reductions to consider, in this order:",
                  "1. fewer budgets (drop 3000 where it is over)",
                  "2. drop firefly_levy",
                  "3. fewer seeds",
                  "4. fewer k values", ""]
    else:
        lines += [f"No total exceeds {LIMIT_H:.0f} h with {workers} workers; "
                  "no reduction needed.", ""]

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Saved {OUT}")
    log_experiment(
        name="plan_compute",
        params=dict(datasets=DATASETS, budgets=BUDGETS, methods=METHODS, n_seeds=N_SEEDS,
                    ks=sorted({k for r in rows for k in r["ks"]}), margin=MARGIN,
                    limit_h=LIMIT_H, contention=factors, workers=workers),
        seeds=list(range(N_SEEDS)),
        csv_paths=[str(OUT.relative_to(RESULTS_DIR.parent))],
        extra=dict(total_single_h=round(tot_single, 2),
                   total_workers_h=round(tot_workers, 2), git_hash=git_hash()),
    )


if __name__ == "__main__":
    main()
