"""Tests for experiments/eval_baseline_transfer.py (Stage A6a, touches the
test split only through run_metaheuristics.evaluate_on_test)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
import eval_baseline_transfer as EBT  # noqa: E402

from experiments.analysis import common as C  # noqa: E402

SRC = (ROOT / "experiments" / "eval_baseline_transfer.py").read_text(encoding="utf-8")


def test_no_direct_test_split_reference():
    """Xte / yte never appear as names; the load_dataset tuple goes straight
    into the runner's evaluate_on_test, nothing else opens it."""
    tree = ast.parse(SRC)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}
    assert not names & {"Xte", "yte", "X_test", "y_test", "X_te", "y_te"}
    for bad in ("[1]", "[3]", "from src.evaluate", ".predict(", ".fit("):
        assert bad not in SRC, bad
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", getattr(n.func, "attr", None)) == "load_dataset"]
    assert len(calls) == 1                                     # one load ...
    assert "rm.evaluate_on_test(mask, load_dataset(" in SRC    # ... passed straight in


def test_jobs_cover_selectors_only_and_k_below_d():
    jobs = EBT.jobs_for("digits")
    assert {j["method"] for j in jobs} == {"random", "anova", "mi", "l1", "rf"}
    assert {j["k"] for j in jobs} == {16, 32}                  # k = 64 = d excluded
    assert len(jobs) == 5 * 2 * 10
    assert all(len(j["idx"]) == j["k"] for j in jobs)
    cifar = EBT.jobs_for("cifar_r18", seeds=[0])
    assert len(cifar) == 5 * 4


def test_one_real_job_reproduces_baseline_lr_accuracy():
    """The runner's evaluation on the stored ANOVA subset gives exactly the
    baseline CSV's logistic-regression accuracy (same classifier and split)."""
    job = next(j for j in EBT.jobs_for("digits", seeds=[0])
               if j["method"] == "anova" and j["k"] == 16)
    row = EBT.score_job(job)
    assert row["acc_lr"] == job["acc_baseline"] and row["acc_lr_matches_baseline"]
    assert 0 < row["acc_knn"] <= 1 and 0 < row["acc_svm"] <= 1
    assert set(EBT.COLUMNS) <= set(row)
