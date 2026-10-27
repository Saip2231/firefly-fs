"""Run all baseline feature selectors on a given dataset.

Prints a summary table and writes detailed per-row results to CSV.

Usage
-----
    python experiments/run_baselines.py --dataset digits --ks 16 32 --seeds 1
    python experiments/run_baselines.py --dataset cifar_r18 --seeds 10
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

from src.data import load_dataset  # noqa: E402
from src.baselines import (  # noqa: E402
    select_all,
    select_random,
    select_kbest_anova,
    select_kbest_mi,
    select_model_lasso,
    pca_transform,
)
from src.evaluate import evaluate  # noqa: E402
from src.results_store import save_run  # noqa: E402


# ---------------------------------------------------------------------------
# Additional selector: Random Forest importance (not yet in baselines.py)
# ---------------------------------------------------------------------------

def select_rf_importance(X, y, k, seed=42, n_estimators=200):
    """Select top-k features by Random Forest feature importance.

    Parameters
    ----------
    X : np.ndarray   – training features only.
    y : np.ndarray   – training labels.
    k : int          – number of features to select.
    seed : int       – random state for reproducibility.
    n_estimators : int – number of trees (default 200).

    Returns
    -------
    mask : np.ndarray of bool, shape (n_features,)
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=seed, n_jobs=1,
    )
    rf.fit(X, y)
    importance = rf.feature_importances_
    top_k = np.argsort(importance)[-k:]
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[top_k] = True
    return mask


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------
# (display_name, callable, needs_seed, returns_mask)
METHODS = [
    ("Random",        select_random,        True,  True),
    ("ANOVA F",       select_kbest_anova,   False, True),
    ("Mutual Info",   select_kbest_mi,      True,  True),
    ("L1 Logistic",   select_model_lasso,   True,  True),
    ("RF Importance", select_rf_importance,  True,  True),
    ("PCA",           pca_transform,         True,  False),
]


# ---------------------------------------------------------------------------
# Worker function (called by joblib in each process)
# ---------------------------------------------------------------------------

def _run_one(dataset_name, seed, k, method_name, method_fn, needs_seed,
             returns_mask, X_train, X_test, y_train, y_test):
    """Execute one (seed, k, method) job.

    Returns
    -------
    row : dict   – one result row for the DataFrame.
    caught_warnings : list[tuple]  – (category_name, message) pairs.
    """
    caught = []
    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")

        clf = LogisticRegression(max_iter=500)

        # ---- Feature selection / transform (timed) ----
        t0 = time.perf_counter()
        if method_name == "All Features":
            result_obj = select_all(X_train, y_train)
        elif needs_seed:
            result_obj = method_fn(X_train, y_train, k, seed=seed)
        else:
            result_obj = method_fn(X_train, y_train, k)
        search_s = time.perf_counter() - t0

        # ---- Build comma-separated index string ----
        if returns_mask and method_name != "All Features":
            idx_str = ",".join(map(str, sorted(np.where(result_obj)[0])))
        else:
            idx_str = ""

        # ---- Evaluate (classifier fit + predict on test set) ----
        res = evaluate(result_obj, X_train, y_train, X_test, y_test, clf)

    # Collect unique (category, message) pairs from this worker call
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run baseline feature selectors and report accuracy.",
    )
    parser.add_argument(
        "--dataset", default="cifar_r18",
        help="Dataset name passed to load_dataset (default: cifar_r18)",
    )
    parser.add_argument(
        "--ks", nargs="+", type=int, default=[16, 32, 64, 128],
        help="k values to evaluate (default: 16 32 64 128)",
    )
    parser.add_argument(
        "--seeds", type=int, default=1,
        help="Number of seeds 0..seeds-1 (default: 1)",
    )
    args = parser.parse_args()

    dataset_name = args.dataset
    ks = sorted(args.ks)
    n_seeds = args.seeds

    # ---- Preload data for every seed ----
    print(f"Loading '{dataset_name}' for {n_seeds} seed(s)…")
    data_by_seed = {}
    for seed in range(n_seeds):
        data_by_seed[seed] = load_dataset(dataset_name, seed=seed)

    n_features = data_by_seed[0][0].shape[1]

    # ---- Validate k values against number of features ----
    valid_ks = [k for k in ks if k <= n_features]
    skipped_ks = [k for k in ks if k > n_features]
    if skipped_ks:
        print(f"⚠  Skipping k={skipped_ks} (exceeds {n_features} features)")
    ks = valid_ks

    # ---- Build job list ----
    jobs = []
    for seed in range(n_seeds):
        Xtr, Xte, ytr, yte = data_by_seed[seed]
        # "All Features" — once per seed (k is irrelevant)
        jobs.append((
            dataset_name, seed, n_features,
            "All Features", select_all, False, True,
            Xtr, Xte, ytr, yte,
        ))
        for k in ks:
            for method_name, method_fn, needs_seed, returns_mask in METHODS:
                jobs.append((
                    dataset_name, seed, k,
                    method_name, method_fn, needs_seed, returns_mask,
                    Xtr, Xte, ytr, yte,
                ))

    print(f"Running {len(jobs)} jobs (n_jobs=-1)…\n")

    # ---- Run in parallel with tqdm progress ----
    rows = []
    all_warnings = set()

    for row, caught in tqdm(
        Parallel(n_jobs=-1, return_as="generator")(
            delayed(_run_one)(*job) for job in jobs
        ),
        total=len(jobs),
        desc="Baselines",
        unit="job",
    ):
        rows.append(row)
        all_warnings.update(caught)

    # ---- Print unique warnings ----
    if all_warnings:
        print(f"\n⚠  {len(all_warnings)} distinct warning(s) encountered:")
        for cat, msg in sorted(all_warnings):
            # Truncate very long messages
            display = msg if len(msg) <= 120 else msg[:117] + "…"
            print(f"   [{cat}] {display}")

    # ---- Persist to structured store (runs.json + CSV + summary CSV) ----
    run_id = save_run(
        dataset=dataset_name,
        n_seeds=n_seeds,
        ks=ks,
        rows=rows,
    )

    df = pd.DataFrame(rows)
    csv_path = os.path.join("results", f"baselines_{dataset_name}.csv")
    print(f"\nResults saved → {csv_path}  ({len(df)} rows)")
    print(f"Run ID        → {run_id}  (see results/runs.json)")

    # ---- Print summary table ----
    print("\n" + "=" * 60)
    print("SUMMARY — mean (± std) accuracy per method × k")
    print("=" * 60)

    summary = (
        df.groupby(["method", "k"], sort=False)["acc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    # Sort: "All Features" first, then by k ascending, method alphabetical
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


if __name__ == "__main__":
    main()
