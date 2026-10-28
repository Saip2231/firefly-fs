"""Stage 1 - Diagnose the d=500 failure (diagnosis only, no algorithm changes).

Configs (make_classification, n_redundant=0, n_repeated=0, shuffle=False, so
informative columns are 0..n_informative-1; data random_state = seed):
  A: d=500, n_informative=5,  k=5,  budget=600, n=1000
  B: d=500, n_informative=10, k=10, budget=800, n=2000
Methods: firefly, firefly_levy, random_search (same budget, same fitness).
Seeds: 0-4.  Only training rows are used (no test / holdout in this stage).

Outputs
  results/diagnostics_d500.csv         one row per (config, seed, method)
  results/diagnostics_d500_sweeps.csv  per-sweep trace for the firefly runs
  results/runs.json                    entry via src.results_store.log_experiment

Extra diagnostic columns (beyond the requested ones):
  inf_cols_tried    distinct informative columns that appeared in at least
                    one NEW evaluation (did the search ever try them?)
  mean_inf_per_eval mean number of informative columns per evaluated mask
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from _common import RESULTS_DIR, EvalRecorder, random_search, warn_once

from src.firefly import binary_firefly, make_fitness
from src.results_store import log_experiment

CONFIGS = [
    dict(d=500, n_informative=5,  k=5,  budget=600, n=1000),
    dict(d=500, n_informative=10, k=10, budget=800, n=2000),
]
SEEDS = list(range(5))
N_FIREFLIES = 20


def verdict(row: dict) -> str:
    """Rules applied in the order given in the Stage 1 spec.
    Rules whose inputs are undefined for a method (NaN) are skipped."""
    if row["evals_used"] < 0.9 * row["budget"]:
        return "BUDGET UNDERUSED"
    if row["best_end"] - row["best_init"] > -0.01:
        return "NO IMPROVEMENT"
    df_, dl_ = row["diversity_first"], row["diversity_last"]
    if not (np.isnan(df_) or np.isnan(dl_)) and dl_ < 0.05 * df_:
        return "COLLAPSE"
    acc = row["accepted_moves"]
    if not np.isnan(acc) and row["proposals"] > 0 and acc < 0.02 * row["proposals"]:
        return "LOW ACCEPTANCE"
    return "OK"


def run_firefly(X, y, cfg, seed, levy, name):
    fitness, state = make_fitness(X, y, k=cfg["k"], seed=seed)
    rec = EvalRecorder(fitness, state)
    trace: dict = {}
    t0 = time.perf_counter()
    mask, best_fit, history = binary_firefly(
        rec, state, d=cfg["d"], k=cfg["k"], budget=cfg["budget"],
        n_fireflies=N_FIREFLIES, levy=levy, seed=seed, trace=trace,
    )
    secs = time.perf_counter() - t0
    sw = trace["sweeps"]
    proposals = sum(s["proposals"] for s in sw)
    tried, mean_inf = rec.informative_coverage(cfg["n_informative"])
    row = dict(
        method=name,
        evals_used=state["evals"],
        sweeps=len(sw),
        best_init=trace["init_best"],
        best_end=best_fit,
        accepted_moves=sum(s["accepted_moves"] for s in sw),
        proposals=proposals,
        # cache-hit rate over all fitness calls (init + sweeps)
        cache_hit_rate=rec.cache_hits / rec.calls if rec.calls else 0.0,
        diversity_first=sw[0]["diversity"] if sw else np.nan,
        diversity_last=sw[-1]["diversity"] if sw else np.nan,
        brightest_changes=sum(s["brightest_changed"] for s in sw),
        hits_informative=int(mask[: cfg["n_informative"]].sum()),
        inf_cols_tried=tried,
        mean_inf_per_eval=mean_inf,
        seconds=secs,
    )
    sweep_rows = [dict(method=name, sweep=i, **s) for i, s in enumerate(sw)]
    return row, sweep_rows


def run_random(X, y, cfg, seed):
    fitness, state = make_fitness(X, y, k=cfg["k"], seed=seed)
    rec = EvalRecorder(fitness, state)
    t0 = time.perf_counter()
    out = random_search(rec, state, d=cfg["d"], k=cfg["k"],
                        budget=cfg["budget"], seed=seed, init_n=N_FIREFLIES)
    secs = time.perf_counter() - t0
    tried, mean_inf = rec.informative_coverage(cfg["n_informative"])
    return dict(
        method="random_search",
        evals_used=state["evals"],
        sweeps=0,
        best_init=out["best_init"],
        best_end=out["best_fit"],
        accepted_moves=np.nan,          # no acceptance step in random search
        proposals=out["proposals"],
        cache_hit_rate=rec.cache_hits / rec.calls if rec.calls else 0.0,
        diversity_first=np.nan,
        diversity_last=np.nan,
        brightest_changes=out["improvements"],
        hits_informative=int(out["best_mask"][: cfg["n_informative"]].sum()),
        inf_cols_tried=tried,
        mean_inf_per_eval=mean_inf,
        seconds=secs,
    )


def main():
    warn_once()
    RESULTS_DIR.mkdir(exist_ok=True)
    rows, sweep_rows = [], []
    t_all = time.perf_counter()

    for cfg in CONFIGS:
        tag = f"d{cfg['d']}_ni{cfg['n_informative']}_k{cfg['k']}_n{cfg['n']}_b{cfg['budget']}"
        print(f"\n=== {tag}  seeds={SEEDS} ===", flush=True)
        for seed in SEEDS:
            X, y = make_classification(
                n_samples=cfg["n"], n_features=cfg["d"],
                n_informative=cfg["n_informative"], n_redundant=0,
                n_repeated=0, shuffle=False, random_state=seed,
            )
            base = dict(config=tag, seed=seed, **cfg)
            for levy, name in ((False, "firefly"), (True, "firefly_levy")):
                r, sw = run_firefly(X, y, cfg, seed, levy, name)
                rows.append({**base, **r})
                sweep_rows += [{**base, **s} for s in sw]
            rows.append({**base, **run_random(X, y, cfg, seed)})
            print(f"  seed {seed} done", flush=True)

    df = pd.DataFrame(rows)
    df["verdict"] = [verdict(r) for r in df.to_dict(orient="records")]

    cols = ["config", "seed", "method", "evals_used", "sweeps", "best_init",
            "best_end", "accepted_moves", "proposals", "cache_hit_rate",
            "diversity_first", "diversity_last", "hits_informative",
            "inf_cols_tried", "mean_inf_per_eval", "verdict"]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.max_rows", 200)
    print("\nDIAGNOSTICS TABLE")
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\nMEANS per (config, method)")
    agg_cols = ["evals_used", "sweeps", "best_init", "best_end", "proposals",
                "accepted_moves", "cache_hit_rate", "diversity_last",
                "hits_informative", "inf_cols_tried", "mean_inf_per_eval"]
    print(df.groupby(["config", "method"], sort=False)[agg_cols].mean()
            .to_string(float_format=lambda v: f"{v:.4f}"))

    print("\nVerdict counts")
    print(df.groupby(["config", "method", "verdict"], sort=False).size().to_string())

    csv = RESULTS_DIR / "diagnostics_d500.csv"
    csv_sw = RESULTS_DIR / "diagnostics_d500_sweeps.csv"
    df.to_csv(csv, index=False)
    pd.DataFrame(sweep_rows).to_csv(csv_sw, index=False)
    total = time.perf_counter() - t_all
    print(f"\nSaved {csv}\nSaved {csv_sw}\nTotal runtime: {total:.1f} s")

    log_experiment(
        name="diagnose_d500",
        params=dict(configs=CONFIGS, n_fireflies=N_FIREFLIES,
                    methods=["firefly", "firefly_levy", "random_search"],
                    data_random_state="seed", fitness_seed="seed",
                    ks=[c["k"] for c in CONFIGS]),
        seeds=SEEDS,
        csv_paths=[str(csv.relative_to(RESULTS_DIR.parent)),
                   str(csv_sw.relative_to(RESULTS_DIR.parent))],
        extra=dict(runtime_s=round(total, 1)),
    )


if __name__ == "__main__":
    main()
