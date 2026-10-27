"""run_and_store.py — Run baselines + pytest for one or both datasets
and persist everything (rows, metrics, timing, test results) to disk.

Usage
-----
    # Run both datasets (all k, 1 seed each) + pytest, store everything
    python experiments/run_and_store.py

    # Only digits, 5 seeds, k in {16,32,64}, skip tests
    python experiments/run_and_store.py --datasets digits --seeds 5 --ks 16 32 64 --no-tests

    # Only CIFAR, 3 seeds, run tests, use 4 parallel workers
    python experiments/run_and_store.py --datasets cifar_r18 --seeds 3 --jobs 4

    # Query past runs
    python experiments/run_and_store.py --history
    python experiments/run_and_store.py --best digits
"""

# Force UTF-8 on Windows consoles
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import sys
import os
import argparse
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import load_dataset                # noqa: E402
from src.baselines import (                      # noqa: E402
    select_all,
    select_random,
    select_kbest_anova,
    select_kbest_mi,
    select_model_lasso,
    pca_transform,
)
from src.evaluate import evaluate                # noqa: E402
from src.results_store import (                  # noqa: E402
    save_run,
    capture_pytest_results,
    print_run_history,
    best_per_method,
)


# ---------------------------------------------------------------------------
# RF importance (local, mirrors run_baselines.py)
# ---------------------------------------------------------------------------

def select_rf_importance(X, y, k, seed=42, n_estimators=200):
    """Select top-k features by Random Forest feature importance."""
    rf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=seed, n_jobs=1,
    )
    rf.fit(X, y)
    importance = rf.feature_importances_
    top_k      = np.argsort(importance)[-k:]
    mask       = np.zeros(X.shape[1], dtype=bool)
    mask[top_k] = True
    return mask


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------
METHODS = [
    ("Random",        select_random,        True,  True),
    ("ANOVA F",       select_kbest_anova,   False, True),
    ("Mutual Info",   select_kbest_mi,      True,  True),
    ("L1 Logistic",   select_model_lasso,   True,  True),
    ("RF Importance", select_rf_importance,  True,  True),
    ("PCA",           pca_transform,         True,  False),
]


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------

def _run_one(dataset_name, seed, k, method_name, method_fn, needs_seed,
             returns_mask, X_train, X_test, y_train, y_test):
    """Execute one (seed, k, method) job. Returns (row_dict, warnings_list)."""
    caught = []
    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")

        clf = LogisticRegression(max_iter=500)

        t0 = time.perf_counter()
        if method_name == "All Features":
            result_obj = select_all(X_train, y_train)
        elif needs_seed:
            result_obj = method_fn(X_train, y_train, k, seed=seed)
        else:
            result_obj = method_fn(X_train, y_train, k)
        search_s = time.perf_counter() - t0

        if returns_mask and method_name != "All Features":
            idx_str = ",".join(map(str, sorted(np.where(result_obj)[0])))
        else:
            idx_str = ""

        res = evaluate(result_obj, X_train, y_train, X_test, y_test, clf)

    for wi in w_list:
        caught.append((wi.category.__name__, str(wi.message)))

    row = {
        "dataset":  dataset_name,
        "method":   method_name,
        "seed":     seed,
        "k":        res["n_features"] if method_name == "All Features" else k,
        "acc":      res["accuracy"],
        "n_feat":   res["n_features"],
        "fit_s":    res["fit_time"],
        "infer_s":  res["inference_time_per_sample"],
        "search_s": search_s,
        "idx":      idx_str,
    }
    return row, caught


# ---------------------------------------------------------------------------
# Per-dataset runner
# ---------------------------------------------------------------------------

def run_dataset(dataset_name: str, ks: list[int], n_seeds: int,
                n_jobs: int = -1) -> list[dict]:
    """Run all seeds × k × methods for one dataset. Returns list of row dicts."""
    print(f"\n{'='*60}")
    print(f"DATASET: {dataset_name}  |  seeds: {n_seeds}  |  k: {ks}")
    print(f"{'='*60}")

    # Pre-load data for every seed
    print(f"Loading '{dataset_name}' for {n_seeds} seed(s)...")
    data_by_seed = {}
    for seed in range(n_seeds):
        data_by_seed[seed] = load_dataset(dataset_name, seed=seed)

    n_features = data_by_seed[0][0].shape[1]
    valid_ks   = [k for k in ks if k <= n_features]
    skipped_ks = [k for k in ks if k > n_features]
    if skipped_ks:
        print(f"  Skipping k={skipped_ks} (dataset has only {n_features} features)")

    # Build job list
    jobs = []
    for seed in range(n_seeds):
        Xtr, Xte, ytr, yte = data_by_seed[seed]
        jobs.append((
            dataset_name, seed, n_features,
            "All Features", select_all, False, True,
            Xtr, Xte, ytr, yte,
        ))
        for k in valid_ks:
            for method_name, method_fn, needs_seed, returns_mask in METHODS:
                jobs.append((
                    dataset_name, seed, k,
                    method_name, method_fn, needs_seed, returns_mask,
                    Xtr, Xte, ytr, yte,
                ))

    print(f"Running {len(jobs)} jobs (n_jobs={n_jobs})...\n")

    rows        = []
    all_warnings: set[tuple] = set()

    for row, caught in tqdm(
        Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(_run_one)(*job) for job in jobs
        ),
        total=len(jobs),
        desc=dataset_name,
        unit="job",
    ):
        rows.append(row)
        all_warnings.update(caught)

    if all_warnings:
        print(f"\n  {len(all_warnings)} distinct warning(s):")
        for cat, msg in sorted(all_warnings):
            display = msg if len(msg) <= 100 else msg[:97] + "..."
            print(f"    [{cat}] {display}")

    return rows


# ---------------------------------------------------------------------------
# Summary printer (reused from run_baselines.py)
# ---------------------------------------------------------------------------

def _print_summary(df: pd.DataFrame, n_seeds: int) -> None:
    print("\n" + "=" * 60)
    print("SUMMARY — mean (± std) accuracy per method × k")
    print("=" * 60)

    summary = (
        df.groupby(["method", "k"], sort=False)["acc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["_is_all"] = summary["method"] == "All Features"
    summary = summary.sort_values(
        ["_is_all", "k", "method"], ascending=[False, True, True]
    )

    if n_seeds == 1:
        header = f"{'Method':<20} {'k':>5} {'Accuracy':>10}"
        print(header)
        print("-" * len(header))
        for _, r in summary.iterrows():
            print(f"{r['method']:<20} {int(r['k']):>5} {r['mean']:>10.4f}")
    else:
        header = f"{'Method':<20} {'k':>5} {'Mean Acc':>10} {'± Std':>8}"
        print(header)
        print("-" * len(header))
        for _, r in summary.iterrows():
            std_str = f"{r['std']:.4f}" if not np.isnan(r["std"]) else "n/a"
            print(
                f"{r['method']:<20} {int(r['k']):>5} "
                f"{r['mean']:>10.4f} {std_str:>8}"
            )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run baselines + tests for cifar_r18 and/or digits; "
                    "store all results (metrics, timing, test outcomes) to disk.",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["cifar_r18", "digits"],
        metavar="DS",
        help="Datasets to run (default: cifar_r18 digits)",
    )
    parser.add_argument(
        "--ks", nargs="+", type=int, default=[16, 32, 64, 128],
        help="k values to evaluate (default: 16 32 64 128)",
    )
    parser.add_argument(
        "--seeds", type=int, default=1,
        help="Number of seeds 0..seeds-1 (default: 1)",
    )
    parser.add_argument(
        "--jobs", type=int, default=-1,
        help="Parallel workers for joblib (default: -1 = all CPUs)",
    )
    parser.add_argument(
        "--no-tests", action="store_true",
        help="Skip running the pytest test suite",
    )
    parser.add_argument(
        "--tests-only", action="store_true",
        help="Only run pytest; skip baseline experiments",
    )
    # Query commands
    parser.add_argument(
        "--history", action="store_true",
        help="Print a table of all past runs and exit",
    )
    parser.add_argument(
        "--best", metavar="DATASET",
        help="Print best-per-method accuracy for a past dataset run and exit",
    )
    args = parser.parse_args()

    # ---- Query modes -------------------------------------------------------
    if args.history:
        print_run_history()
        return

    if args.best:
        try:
            df = best_per_method(args.best)
            print(f"\nBest k per method for '{args.best}' (most recent run):\n")
            print(df.to_string(index=False))
        except ValueError as exc:
            print(f"Error: {exc}")
        return

    # ---- Optionally run pytest first (shared across datasets) --------------
    test_run = None
    if not args.no_tests:
        print("\n" + "=" * 60)
        print("RUNNING PYTEST TEST SUITE")
        print("=" * 60)
        test_run = capture_pytest_results(["tests/", "-v", "--tb=short"])
        status = "PASSED" if test_run["exit_code"] == 0 else "FAILED"
        print(
            f"\nTests: {test_run['passed']} passed, "
            f"{test_run['failed']} failed, "
            f"{test_run['errors']} errors  [{status}]  "
            f"({test_run['duration_s']:.1f}s)"
        )

    if args.tests_only:
        return

    # ---- Run each dataset --------------------------------------------------
    ks = sorted(args.ks)
    for dataset_name in args.datasets:
        rows = run_dataset(
            dataset_name=dataset_name,
            ks=ks,
            n_seeds=args.seeds,
            n_jobs=args.jobs,
        )

        # Persist: baselines rows + test results (shared test_run for all datasets)
        run_id = save_run(
            dataset=dataset_name,
            n_seeds=args.seeds,
            ks=ks,
            rows=rows,
            test_run=test_run,
        )

        # Print summary
        df = pd.DataFrame(rows)
        _print_summary(df, args.seeds)

    print("All done. Use --history to view all past runs.\n")


if __name__ == "__main__":
    main()
