"""Tests for the re-ranking step (rerank / run_firefly) and the exposed cache.

All checks are exact properties (membership, equality, counts); none uses a
tuned numeric threshold.  Synthetic data only; no filesystem access.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

import src.firefly as ff
from src.firefly import binary_firefly, make_fitness, rerank, run_firefly

K, D, BUDGET, SEED = 5, 30, 150, 3


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(n_samples=600, n_features=D, n_informative=5,
                               n_redundant=0, n_repeated=0, shuffle=False,
                               random_state=SEED)
    return X.astype(np.float64), y


@pytest.fixture(scope="module")
def searched(data):
    """One finished search: (state, raw_mask, raw_fit, history)."""
    X, y = data
    fitness, state = make_fitness(X, y, k=K, seed=SEED)
    mask, fit, hist = binary_firefly(fitness, state, d=D, k=K, budget=BUDGET,
                                     n_fireflies=10, seed=SEED)
    return state, mask, fit, hist


def test_cache_exposed_and_consistent(data, searched):
    """state['cache'] holds exactly one entry per NEW eval, with the same
    fitness the closure returns."""
    X, y = data
    state, mask, fit, _ = searched
    assert len(state["cache"]) == state["evals"]
    assert state["cache"][mask.tobytes()] == fit


def test_reranked_mask_is_cached_and_has_k(data, searched):
    X, y = data
    state, *_ = searched
    best, scores, _ = rerank(state, X, y, K, top_m=10, n_splits=5, seed=SEED)
    assert best.dtype == bool and int(best.sum()) == K
    assert best.tobytes() in state["cache"]
    # candidates are the lowest cached fitness values, in order
    lowest = sorted(state["cache"].values())[:10]
    assert [s["cached_fitness"] for s in scores] == lowest


def test_rerank_deterministic(data, searched):
    X, y = data
    state, *_ = searched
    a = rerank(state, X, y, K, top_m=10, n_splits=5, seed=SEED)
    b = rerank(state, X, y, K, top_m=10, n_splits=5, seed=SEED)
    np.testing.assert_array_equal(a[0], b[0])
    assert [s["cv_acc"] for s in a[1]] == [s["cv_acc"] for s in b[1]]

    r1 = run_firefly(X, y, K, budget=BUDGET, seed=SEED, n_fireflies=10)
    r2 = run_firefly(X, y, K, budget=BUDGET, seed=SEED, n_fireflies=10)
    np.testing.assert_array_equal(r1["reranked_mask"], r2["reranked_mask"])
    np.testing.assert_array_equal(r1["raw_mask"], r2["raw_mask"])


def test_rerank_cost_accounting(data, searched, monkeypatch):
    """rerank_fits equals the number of LogisticRegression.fit calls actually
    made, equals top_m*n_splits, and the search budget is untouched."""
    X, y = data
    state, *_ = searched
    evals_before = state["evals"]
    calls = {"n": 0}

    class CountingLR(LogisticRegression):
        def fit(self, *a, **kw):
            calls["n"] += 1
            return super().fit(*a, **kw)

    monkeypatch.setattr(ff, "LogisticRegression", CountingLR)
    top_m, n_splits = 7, 4
    assert len(state["cache"]) >= top_m
    _, scores, fits = rerank(state, X, y, K, top_m=top_m, n_splits=n_splits, seed=SEED)
    assert fits == calls["n"] == top_m * n_splits
    assert len(scores) == top_m
    assert state["evals"] == evals_before


def test_run_firefly_separates_costs(data):
    X, y = data
    r = run_firefly(X, y, K, budget=BUDGET, seed=SEED, n_fireflies=10,
                    rerank_top_m=10, n_splits=5)
    assert r["search_evals"] <= BUDGET
    assert r["rerank_fits"] == 10 * 5
    assert int(r["reranked_mask"].sum()) == K


def test_run_firefly_total_fits(data):
    """evals_used / rerank_fits / total_fits are reported and add up (F2)."""
    X, y = data
    for top_m, n_splits in ((10, 5), (3, 2), (0, 5)):
        r = run_firefly(X, y, K, budget=BUDGET, seed=SEED, n_fireflies=10,
                        rerank_top_m=top_m, n_splits=n_splits)
        assert r["evals_used"] == r["search_evals"] <= BUDGET
        assert r["rerank_fits"] == top_m * n_splits
        assert r["total_fits"] == r["evals_used"] + r["rerank_fits"]
        assert r["search_s"] > 0 and r["rerank_s"] >= 0


def test_top_m_zero_returns_raw_unchanged(data):
    """rerank_top_m=0 -> re-ranked mask is the raw best, no extra fits, and
    the raw search is identical to calling binary_firefly directly."""
    X, y = data
    r = run_firefly(X, y, K, budget=BUDGET, seed=SEED, n_fireflies=10,
                    rerank_top_m=0)
    np.testing.assert_array_equal(r["reranked_mask"], r["raw_mask"])
    assert r["rerank_fits"] == 0 and r["rerank_scores"] == []

    fitness, state = make_fitness(X, y, k=K, seed=SEED)
    mask, fit, hist = binary_firefly(fitness, state, d=D, k=K, budget=BUDGET,
                                     n_fireflies=10, seed=SEED)
    np.testing.assert_array_equal(r["raw_mask"], mask)
    assert r["raw_fit"] == fit
    assert r["history"] == hist
    assert r["search_evals"] == state["evals"]
