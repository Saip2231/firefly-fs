"""Results storage for the feature-selection benchmark.

Writes experiment runs (baselines + test suite) to a structured JSON store
so that every run can be reproduced, compared, and queried later.

Layout inside ``results/``
--------------------------
results/
  runs.json              <- append-only log of every run (human-readable)
  baselines_<ds>.csv     <- flat per-row results (existing format, unchanged)
  summary_<ds>.csv       <- mean+-std summary table per method x k
  test_results_<ds>.json <- per-dataset pytest outcome log

JSON record format (one entry in ``runs.json``)
-----------------------------------------------
{
  "run_id":    "20241027_103045_cifar_r18",
  "timestamp": "2024-10-27T10:30:45+05:30",
  "dataset":   "cifar_r18",
  "n_seeds":   3,
  "ks":        [16, 32, 64, 128],
  "python":    "3.11.4",
  "platform":  "Windows-11",
  "rows": [
    { "method": "ANOVA F", "seed": 0, "k": 16,
      "acc": 0.58, "n_feat": 16,
      "fit_s": 0.18, "infer_s": 7.2e-07, "search_s": 0.09,
      "idx": "17,32,88,..." },
    ...
  ],
  "summary": [
    { "method": "ANOVA F", "k": 16, "mean_acc": 0.58, "std_acc": 0.01,
      "mean_fit_s": 0.18, "mean_search_s": 0.09 },
    ...
  ],
  "test_run": {                   <- present only if pytest was executed
    "exit_code": 0,
    "passed": 6,
    "failed": 0,
    "errors": 0,
    "duration_s": 12.4,
    "tests": [
      { "nodeid": "tests/test_leakage.py::...", "outcome": "passed",
        "duration_s": 1.2 }
    ]
  }
}
"""

from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
_RUNS_FILE   = os.path.join(_RESULTS_DIR, "runs.json")


def _ensure_results_dir() -> None:
    os.makedirs(_RESULTS_DIR, exist_ok=True)


def _load_runs() -> list[dict]:
    """Load existing runs from ``runs.json`` (returns [] if not present)."""
    if not os.path.exists(_RUNS_FILE):
        return []
    with open(_RUNS_FILE, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, list) else [data]


def _save_runs(runs: list[dict]) -> None:
    with open(_RUNS_FILE, "w", encoding="utf-8") as fh:
        json.dump(runs, fh, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    """JSON serialisation fallback for numpy scalar types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


def _make_run_id(dataset: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{dataset}"


def _local_iso() -> str:
    return datetime.now().astimezone().isoformat()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_run(
    *,
    dataset: str,
    n_seeds: int,
    ks: list[int],
    rows: list[dict],
    test_run: dict | None = None,
) -> str:
    """Persist one complete experiment run to disk.

    Parameters
    ----------
    dataset   : Dataset name, e.g. ``"cifar_r18"`` or ``"digits"``.
    n_seeds   : Number of seeds used.
    ks        : List of k values evaluated.
    rows      : Per-job result dicts (same schema as the CSV rows).
    test_run  : Optional dict produced by :func:`capture_pytest_results`.

    Returns
    -------
    run_id : str
        The unique identifier assigned to this run.
    """
    _ensure_results_dir()

    run_id = _make_run_id(dataset)
    df     = pd.DataFrame(rows)

    # -- Build summary (mean +- std per method x k) --------------------------
    metric_cols = [c for c in ["acc", "fit_s", "infer_s", "search_s"] if c in df.columns]
    agg = (
        df.groupby(["method", "k"], sort=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = ["_".join(c).strip("_") for c in agg.columns]
    summary = agg.to_dict(orient="records")

    # -- Assemble run record -------------------------------------------------
    record: dict[str, Any] = {
        "run_id":    run_id,
        "timestamp": _local_iso(),
        "dataset":   dataset,
        "n_seeds":   n_seeds,
        "ks":        sorted(ks),
        "python":    sys.version.split()[0],
        "platform":  platform.platform(terse=True),
        "rows":      rows,
        "summary":   summary,
    }
    if test_run is not None:
        record["test_run"] = test_run

    # -- Append to runs.json -------------------------------------------------
    runs = _load_runs()
    runs.append(record)
    _save_runs(runs)

    # -- Write flat CSV (all rows) -------------------------------------------
    col_order = ["dataset", "method", "seed", "k",
                 "acc", "n_feat", "fit_s", "infer_s", "search_s", "idx"]
    csv_cols  = [c for c in col_order if c in df.columns]
    csv_path  = os.path.join(_RESULTS_DIR, f"baselines_{dataset}.csv")
    df[csv_cols].to_csv(csv_path, index=False)

    # -- Write summary CSV ---------------------------------------------------
    summary_path = os.path.join(_RESULTS_DIR, f"summary_{dataset}.csv")
    pd.DataFrame(summary).to_csv(summary_path, index=False)

    # -- Write test JSON (if present) ----------------------------------------
    if test_run is not None:
        test_path = os.path.join(_RESULTS_DIR, f"test_results_{dataset}.json")
        with open(test_path, "w", encoding="utf-8") as fh:
            json.dump(
                {"run_id": run_id, "dataset": dataset, **test_run},
                fh, indent=2, default=_json_default,
            )

    print(f"\n[results_store] Run '{run_id}' saved to results/")
    print(f"  -> runs.json         (append-only log)")
    print(f"  -> baselines_{dataset}.csv")
    print(f"  -> summary_{dataset}.csv")
    if test_run is not None:
        print(f"  -> test_results_{dataset}.json")

    return run_id


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def load_all_runs() -> list[dict]:
    """Return every recorded run from ``runs.json``."""
    return _load_runs()


def runs_as_dataframe() -> pd.DataFrame:
    """Return a DataFrame of run *metadata* (no per-row detail).

    Each DataFrame row = one experiment run.
    """
    runs     = _load_runs()
    meta_keys = ["run_id", "timestamp", "dataset", "n_seeds", "ks",
                 "python", "platform"]
    records = []
    for r in runs:
        rec = {k: r.get(k) for k in meta_keys}
        tr  = r.get("test_run") or {}
        rec["tests_passed"]  = tr.get("passed",     None)
        rec["tests_failed"]  = tr.get("failed",     None)
        rec["tests_errors"]  = tr.get("errors",     None)
        rec["test_duration"] = tr.get("duration_s", None)
        records.append(rec)
    return pd.DataFrame(records)


def best_per_method(dataset: str) -> pd.DataFrame:
    """Return the best (highest mean accuracy) k per method for *dataset*.

    Uses the most recent run that matches the given dataset.
    """
    runs     = _load_runs()
    matching = [r for r in runs if r["dataset"] == dataset]
    if not matching:
        raise ValueError(f"No runs found for dataset {dataset!r}")
    latest   = matching[-1]
    df       = pd.DataFrame(latest["summary"])
    if "acc_mean" in df.columns:
        df = df.rename(columns={"acc_mean": "mean_acc", "acc_std": "std_acc"})
    idx      = df.groupby("method")["mean_acc"].idxmax()
    return (
        df.loc[idx]
        .sort_values("mean_acc", ascending=False)
        .reset_index(drop=True)
    )


def print_run_history() -> None:
    """Print a short human-readable table of all past runs."""
    df = runs_as_dataframe()
    if df.empty:
        print("No runs recorded yet.")
        return
    print("\n" + "=" * 72)
    print("RUN HISTORY")
    print("=" * 72)
    fmt = f"{'run_id':<30} {'dataset':<12} {'seeds':<7} {'pass':<6} {'fail':<6}"
    print(fmt)
    print("-" * 72)
    for _, r in df.iterrows():
        p = r["tests_passed"] if r["tests_passed"] is not None else "-"
        f = r["tests_failed"] if r["tests_failed"] is not None else "-"
        print(
            f"{str(r['run_id']):<30} {str(r['dataset']):<12} "
            f"{str(r['n_seeds']):<7} {str(p):<6} {str(f):<6}"
        )
    print()


# ---------------------------------------------------------------------------
# pytest integration
# ---------------------------------------------------------------------------

def capture_pytest_results(pytest_args: list[str] | None = None) -> dict:
    """Run pytest programmatically and return a structured result dict.

    Parameters
    ----------
    pytest_args : list of str, optional
        Arguments forwarded to pytest (e.g. ``["tests/", "-v"]``).
        Defaults to ``["tests/", "-v", "--tb=short"]``.

    Returns
    -------
    dict
        Keys: ``exit_code``, ``passed``, ``failed``, ``errors``,
        ``duration_s``, ``tests`` (list of per-test dicts).
    """
    import time
    import pytest as _pytest  # noqa: PLC0415

    class _Collector:
        """Minimal pytest plugin that collects per-test outcomes."""

        def __init__(self):
            self.tests: list[dict] = []

        def pytest_runtest_logreport(self, report):
            # Capture: call phase for pass/fail; setup phase for early errors
            if report.when == "call" or (
                report.when == "setup" and report.failed
            ):
                self.tests.append({
                    "nodeid":     report.nodeid,
                    "outcome":    report.outcome,   # "passed"/"failed"/"error"
                    "duration_s": round(report.duration, 6),
                    "longrepr":   (
                        str(report.longrepr)
                        if report.failed or report.outcome == "error"
                        else None
                    ),
                })

    collector = _Collector()
    args      = list(pytest_args) if pytest_args else ["tests/", "-v", "--tb=short"]

    t0        = time.perf_counter()
    exit_code = _pytest.main(args, plugins=[collector])
    duration  = time.perf_counter() - t0

    outcomes = [t["outcome"] for t in collector.tests]
    return {
        "exit_code":  int(exit_code),
        "passed":     outcomes.count("passed"),
        "failed":     outcomes.count("failed"),
        "errors":     outcomes.count("error"),
        "duration_s": round(duration, 3),
        "tests":      collector.tests,
    }
