"""Tests for experiments/tune.py and experiments/tune_summary.py.

Protocol checks (no search is run here; the dry run exercises run_job):
(a) 9 configs per method, default first, each variant changes exactly its
    named factor, defaults equal the function defaults
(b) 162 jobs on tuning seeds 100-102 only; dry run = 6 jobs
(c) resume: jobs already in the CSV are skipped
(d) tuning split is stratified 75/25, disjoint and deterministic
(e) tune.py never names a test-split variable (static AST check)
(f) decision rule: default kept below the 0.003 margin, winner at/above it,
    ties go to the default; incomplete grids are refused
"""

from __future__ import annotations

import ast
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
import tune  # noqa: E402
import tune_summary  # noqa: E402

from src.firefly import binary_firefly  # noqa: E402
from src.pso import binary_pso  # noqa: E402


# (a) -----------------------------------------------------------------------

def test_configs_one_factor_at_a_time():
    funcs = {"firefly": binary_firefly, "pso": binary_pso}
    for method, configs in tune.CONFIGS.items():
        assert len(configs) == 9
        ids = [cid for cid, _ in configs]
        assert ids[0] == "default" and len(set(ids)) == 9
        default = configs[0][1]
        sig = inspect.signature(funcs[method]).parameters
        for key, val in default.items():
            assert sig[key].default == val, f"{method}.{key}"
        for cid, hp in configs[1:]:
            changed = {k for k in hp if hp[k] != default[k]}
            assert set(hp) == set(default)
            named = set(cid.split("=")[:-1])            # "c1=c2=1.0" -> {c1, c2}
            assert changed == named, cid


# (b) -----------------------------------------------------------------------

def test_jobs_use_tuning_seeds_only():
    jobs = tune.build_jobs()
    assert len(jobs) == 2 * 9 * 3 * 3
    assert {j["seed"] for j in jobs} == {100, 101, 102}
    assert len({tuple(j[c] for c in tune.KEY) for j in jobs}) == len(jobs)
    dry = tune.build_jobs(dry_run=True)
    assert len(dry) == 6
    assert {j["seed"] for j in dry} == {100} and {j["config_id"] for j in dry} == {"default"}
    assert all(j["seed"] not in range(10) for j in jobs)


# (c) -----------------------------------------------------------------------

def test_pending_jobs_resume(tmp_path):
    jobs = tune.build_jobs()
    path = tmp_path / "runs.csv"
    assert tune.pending_jobs(jobs, path) == jobs              # no file yet
    done = [dict(j, tuning_acc=0.5, val_fitness=0.1, evals_used=800,
                 search_s=1.0, git_hash="x") for j in jobs[:7]]
    for row in done:
        tune._append(row, path)
    left = tune.pending_jobs(jobs, path)
    assert left == jobs[7:]
    assert list(pd.read_csv(path).columns) == tune.COLUMNS


# (d) -----------------------------------------------------------------------

def test_tuning_split_stratified_and_deterministic():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 6))
    y = np.repeat(np.arange(4), 100)
    a = tune.tuning_split(X, y, seed=100)
    b = tune.tuning_split(X, y, seed=100)
    for u, v in zip(a, b):
        np.testing.assert_array_equal(u, v)
    Xs, Xh, ys, yh = a
    assert len(Xs) == 300 and len(Xh) == 100
    assert np.all(np.bincount(yh) == 25)
    rows_s = {r.tobytes() for r in Xs}
    assert not any(r.tobytes() in rows_s for r in Xh)


# (e) -----------------------------------------------------------------------

def test_tune_never_names_test_split():
    banned = {"Xte", "yte", "X_test", "y_test", "X_te", "y_te"}
    for name in ("tune.py", "tune_summary.py"):
        tree = ast.parse((ROOT / "experiments" / name).read_text(encoding="utf-8"))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        names |= {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}
        assert not names & banned, f"{name}: {names & banned}"


def test_worker_pickles_by_reference():
    """Regression (dry-run crash): the worker must be importable as tune.run_job,
    so loky pickles it by name, not by value from __main__."""
    import pickle

    assert tune.run_job.__module__ == "tune"
    assert pickle.loads(pickle.dumps(tune.run_job)) is tune.run_job
    assert pickle.loads(pickle.dumps(tune._tuning_data)) is tune._tuning_data
    src = (ROOT / "experiments" / "tune.py").read_text(encoding="utf-8")
    assert "import tune as _tune" in src and "delayed(worker)" in src


def test_run_job_returns_warnings(monkeypatch):
    """run_job records warnings and hands them back (parent prints each
    category once) instead of letting every worker print them."""
    import warnings

    def fake(job, git):
        warnings.warn("constant columns", UserWarning)
        warnings.warn("constant columns", UserWarning)
        return {"x": 1}

    monkeypatch.setattr(tune, "_run_job", fake)
    row = tune.run_job({}, "g")
    assert row["x"] == 1
    assert row["warnings"] == [("UserWarning", "constant columns")] * 2


# (f) -----------------------------------------------------------------------

ORDER =["default", "a", "b"]


def test_decide_keeps_default_below_margin():
    d = tune_summary.decide({"default": 0.800, "a": 0.802, "b": 0.790}, ORDER)
    assert d["best"] == "a" and d["chosen"] == "default"


def test_decide_picks_winner_at_or_above_margin():
    d = tune_summary.decide({"default": 0.800, "a": 0.790, "b": 0.8040}, ORDER)
    assert d["best"] == "b" and d["chosen"] == "b"


def test_decide_default_best_and_ties():
    assert tune_summary.decide({"default": 0.81, "a": 0.80, "b": 0.79}, ORDER)["chosen"] == "default"
    d = tune_summary.decide({"default": 0.80, "a": 0.80, "b": 0.80}, ORDER)
    assert d["best"] == "default" and d["chosen"] == "default"
    d = tune_summary.decide({"default": 0.70, "a": 0.80, "b": 0.80}, ORDER)
    assert d["best"] == "a" and d["chosen"] == "a"            # tie -> earlier config
    with pytest.raises(ValueError):
        tune_summary.decide({"default": 0.8, "a": 0.9}, ["a", "default"])


def _full_grid_df():
    rows = []
    for method, configs in tune.CONFIGS.items():
        for cid, hp in configs:
            for ds, k in tune.CONDITIONS:
                for s in tune.TUNING_SEEDS:
                    rows.append(dict(method=method, config_id=cid,
                                     config_json=json.dumps(hp), dataset=ds, k=k,
                                     seed=s, tuning_acc=0.5, val_fitness=0.1,
                                     evals_used=800, search_s=1.0, git_hash="x"))
    return pd.DataFrame(rows)


def test_check_complete():
    df = _full_grid_df()
    tune_summary.check_complete(df)                          # complete: no error
    with pytest.raises(SystemExit):
        tune_summary.check_complete(df.iloc[1:])             # one run missing
    bad = df.copy()
    bad.loc[0, "seed"] = 0                                   # evaluation seed
    with pytest.raises(SystemExit):
        tune_summary.check_complete(bad)
