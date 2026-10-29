"""Day-3 golden outputs: regression fixture recorded BEFORE the day3-fixes.

Runs (fresh make_fitness per run, same seed for fitness and search):
  toy_d50   make_classification(n_samples=1000, n_features=50,  n_informative=5,
            n_redundant=0, n_repeated=0, shuffle=False, random_state=seed),
            all 1000 rows are search rows (max_rows=1000 -> 250 validation rows)
            k=5, budget=600, seeds 0-4
  toy_d500  same with n_features=500
  digits    Xtr, ytr from load_dataset("digits", seed=0) (Xte discarded),
            k=16, budget=300, seed 0, default max_rows
Methods: firefly (levy=False), firefly_levy (levy=True), random_search
(experiments/_common.py); n_fireflies=20, all other hyperparameters default.

Saved per run: idx (selected columns), best_fit, history, evals_used, n_sweeps.
evals_used < budget marks a run whose search stopped early (tie-stall break).

Usage
-----
    python experiments/golden_day3.py            # write results/golden_day3.json (refuses to overwrite)
    python experiments/golden_day3.py --compare  # rerun, write results/golden_day3_compare.json, list diffs
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.datasets import make_classification  # noqa: E402

from _common import RESULTS_DIR, random_search, warn_once  # noqa: E402

from src.data import load_dataset  # noqa: E402
from src.firefly import binary_firefly, make_fitness  # noqa: E402

GOLDEN = RESULTS_DIR / "golden_day3.json"
COMPARE = RESULTS_DIR / "golden_day3_compare.json"
METHODS = ("firefly", "firefly_levy", "random_search")
N_FIREFLIES = 20

# (case, d or None, k, budget, seeds)
CASES = [
    ("toy_d50", 50, 5, 600, [0, 1, 2, 3, 4]),
    ("toy_d500", 500, 5, 600, [0, 1, 2, 3, 4]),
    ("digits", None, 16, 300, [0]),
]


def get_data(case, d, seed):
    if case == "digits":
        Xtr, _Xte, ytr, _yte = load_dataset("digits", seed=seed)
        return Xtr, ytr, {}
    X, y = make_classification(
        n_samples=1000, n_features=d, n_informative=5, n_redundant=0,
        n_repeated=0, shuffle=False, random_state=seed)
    return X, y, {"max_rows": 1000}


def run_one(method, X, y, k, budget, seed, fit_kw):
    d = X.shape[1]
    fitness, state = make_fitness(X, y, k=k, seed=seed, **fit_kw)
    if method == "random_search":
        out = random_search(fitness, state, d=d, k=k, budget=budget, seed=seed)
        mask, best_fit, history = out["best_mask"], out["best_fit"], []
    else:
        mask, best_fit, history = binary_firefly(
            fitness, state, d=d, k=k, budget=budget, n_fireflies=N_FIREFLIES,
            levy=(method == "firefly_levy"), seed=seed)
    return {
        "idx": [int(i) for i in np.flatnonzero(mask)],
        "best_fit": float(best_fit),
        "history": [float(h) for h in history],
        "evals_used": int(state["evals"]),
        "n_sweeps": len(history),
    }


def run_all():
    runs = {}
    for case, d, k, budget, seeds in CASES:
        for seed in seeds:
            X, y, fit_kw = get_data(case, d, seed)
            for method in METHODS:
                key = f"{case}|{method}|seed{seed}"
                r = run_one(method, X, y, k, budget, seed, fit_kw)
                runs[key] = dict(case=case, method=method, seed=seed, d=X.shape[1],
                                 k=k, budget=budget, **r)
                flag = "  EARLY STOP" if r["evals_used"] < budget else ""
                print(f"{key:<36} best_fit={r['best_fit']:.6f} evals={r['evals_used']:>4} "
                      f"sweeps={r['n_sweeps']:>3}{flag}", flush=True)
    return runs


def diff(golden, new):
    fields = ("idx", "best_fit", "history", "evals_used", "n_sweeps")
    changed = []
    for key in sorted(set(golden) | set(new)):
        g, n = golden.get(key), new.get(key)
        if g is None or n is None:
            changed.append((key, "missing", g is None))
            continue
        bad = [f for f in fields if g[f] != n[f]]
        if bad:
            changed.append((key, ",".join(bad), g["evals_used"] < g["budget"]))
    return changed


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--compare", action="store_true",
                    help="rerun and compare against results/golden_day3.json")
    args = ap.parse_args()
    warn_once()
    RESULTS_DIR.mkdir(exist_ok=True)

    if not args.compare and GOLDEN.exists():
        sys.exit(f"{GOLDEN} exists; refusing to overwrite. Use --compare.")

    runs = run_all()
    out = COMPARE if args.compare else GOLDEN
    out.write_text(json.dumps(runs, indent=1))
    n_early = sum(r["evals_used"] < r["budget"] for r in runs.values())
    print(f"\nSaved {out} ({len(runs)} runs, {n_early} stopped before budget)")

    if args.compare:
        golden = json.loads(GOLDEN.read_text())
        changed = diff(golden, runs)
        print(f"\nCOMPARE vs {GOLDEN.name}: {len(runs) - len(changed)} identical, "
              f"{len(changed)} changed")
        for key, fields, early in changed:
            print(f"  {key:<36} changed: {fields:<40} golden stopped early: {early}")


if __name__ == "__main__":
    main()
