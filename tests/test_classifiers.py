"""Tests for the shared classifier configs (src/classifiers.py) and the
convergence-warning counter in make_fitness.  Exact checks only."""

from __future__ import annotations

import ast
import warnings
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

import src.firefly as ff
from src.classifiers import (FINAL_CLF_PARAMS, SEARCH_CLF_PARAMS, clf_configs,
                             final_clf, search_clf)
from src.firefly import make_fitness

ROOT = Path(__file__).resolve().parent.parent


def _lr_calls(source: str) -> list[int]:
    """Line numbers of real LogisticRegression(...) calls (strings/comments ignored)."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
            if name == "LogisticRegression":
                lines.append(node.lineno)
    return lines


def test_no_experiment_builds_its_own_logistic_regression():
    """Final/holdout accuracy must come from src.classifiers.final_clf()."""
    offenders = []
    for path in sorted((ROOT / "experiments").glob("*.py")):
        offenders += [f"{path.name}:{no}" for no in _lr_calls(path.read_text(encoding="utf-8"))]
    assert not offenders, "use src.classifiers.final_clf():\n" + "\n".join(offenders)


def test_lr_call_detector():
    assert _lr_calls("from x import LogisticRegression\nc = LogisticRegression(max_iter=5)\n") == [2]
    assert _lr_calls("import sklearn.linear_model as lm\nc = lm.LogisticRegression()\n") == [2]
    assert _lr_calls('s = "LogisticRegression(max_iter=100)"  # LogisticRegression()\n') == []


def test_classifier_settings():
    f, s = final_clf(), search_clf(7)
    assert f.max_iter == 500 and f.random_state == 0
    assert s.max_iter == 100 and s.random_state == 7
    assert f is not final_clf()                      # fresh object each call
    assert FINAL_CLF_PARAMS["max_iter"] == 500 and SEARCH_CLF_PARAMS == {"max_iter": 100}
    cfg = clf_configs()
    assert "max_iter=500" in cfg["final_clf"] and "max_iter=100" in cfg["search_clf"]


def _data(seed=0, d=20):
    X, y = make_classification(n_samples=300, n_features=d, n_informative=4,
                               n_redundant=0, n_repeated=0, shuffle=False,
                               random_state=seed)
    return X, y


def _masks(d, k, n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        m = np.zeros(d, dtype=bool)
        m[rng.choice(d, size=k, replace=False)] = True
        out.append(m)
    return out


def test_conv_warn_counted_and_fitness_unchanged(monkeypatch):
    """Every new fit that raises ConvergenceWarning is counted once; cache
    hits are not; fitness values equal those of an unpatched run."""
    X, y = _data()
    masks = _masks(20, 4, 6, seed=1)

    fit_ref, st_ref = make_fitness(X, y, k=4, seed=0)
    ref = [fit_ref(m) for m in masks]

    class WarningLR(LogisticRegression):
        def fit(self, *a, **kw):
            warnings.warn("forced", ConvergenceWarning)
            return super().fit(*a, **kw)

    monkeypatch.setattr(ff, "LogisticRegression", WarningLR)
    fit, st = make_fitness(X, y, k=4, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = [fit(m) for m in masks] + [fit(masks[0])]   # last is a cache hit
    assert got[:-1] == ref and got[-1] == ref[0]
    assert st["evals"] == st_ref["evals"] == 6
    assert st["n_conv_warn"] == 6


def test_conv_warn_zero_when_no_warning():
    X, y = _data()
    fit, st = make_fitness(X, y, k=4, seed=0)
    assert st["n_conv_warn"] == 0
    fit(np.zeros(20, dtype=bool))             # infeasible: no fit, no count
    assert st["n_conv_warn"] == 0 and st["evals"] == 0
