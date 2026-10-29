"""Tests for src/init_pop.py and the Stage-1 hooks in src/firefly.py.

(a) shapes and exactly k bits for both kinds (incl. edge sizes); seeded
    masks distinct where possible; score bias points the right way
(b) seeded population contains the exact top-k mask (member 0)
(c) same rng seed -> same population
(d) state["filter_scores"] == f_classif on the inner-train rows, differs from
    full-train scores, and ignores inner-validation rows entirely
(e) init_masks=None reproduces the recorded pre-Stage-1 Firefly output
    (results/golden_day3.json) exactly
(f) seeded Firefly's best fitness before / after sweep 0 is <= the fitness of
    the (inner-train) ANOVA top-k mask
(g) shared init: binary_firefly(init_masks=None) starts from
    init_population("random", ..., default_rng(seed)); init_masks is used
    as given and never mutated; bad init_masks raise ValueError

Exact checks only (equalities, counts); no tuned thresholds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedShuffleSplit

from src.firefly import binary_firefly, make_fitness
from src.init_pop import check_init_masks, init_population, top_k_mask

GOLDEN = Path(__file__).resolve().parent.parent / "results" / "golden_day3.json"


def _data(n=600, d=30, seed=0):
    X, y = make_classification(n_samples=n, n_features=d, n_informative=5,
                               n_redundant=0, n_repeated=0, shuffle=False,
                               random_state=seed)
    return X.astype(np.float64), y


def _scores(d, seed=0):
    """Distinct, finite scores (no ties)."""
    return np.random.default_rng(seed).permutation(d).astype(float) + 0.5


class _Recorder:
    """Records every mask passed to fitness (cache hits included)."""

    def __init__(self, fitness):
        self.fitness = fitness
        self.masks: list[np.ndarray] = []

    def __call__(self, mask):
        self.masks.append(mask.copy())
        return self.fitness(mask)


# ---------------------------------------------------------------------------
# (a) shapes, exactly k, distinctness, bias direction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["random", "seeded"])
@pytest.mark.parametrize("n,d,k,frac", [
    (20, 30, 5, 0.5), (1, 30, 5, 0.5), (7, 10, 9, 0.5),
    (10, 64, 16, 0.0), (10, 64, 16, 1.0), (10, 4, 2, 0.5),
])
def test_shape_and_exactly_k(kind, n, d, k, frac):
    pop = init_population(kind, n, d, k, np.random.default_rng(0),
                          filter_scores=_scores(d), frac_seeded=frac)
    assert pop.shape == (n, d) and pop.dtype == bool
    assert np.all(pop.sum(axis=1) == k)


def test_seeded_distinct_where_possible():
    pop = init_population("seeded", 20, 30, 5, np.random.default_rng(1),
                          filter_scores=_scores(30))
    assert len({row.tobytes() for row in pop}) == 20
    # d=4, k=2 has only C(4,2)=6 masks: all 6 appear, the rest are duplicates
    small = init_population("seeded", 10, 4, 2, np.random.default_rng(1),
                            filter_scores=_scores(4))
    assert len({row.tobytes() for row in small}) == 6


def test_seeded_bias_favours_high_scores():
    """frac_seeded=1: top-10 columns are picked more often than bottom-10
    (expected probability ratio ~ (1)^3 / (0.2)^3 = 125)."""
    d, k = 50, 5
    s = _scores(d, seed=3)
    pop = init_population("seeded", 200, d, k, np.random.default_rng(2),
                          filter_scores=s, frac_seeded=1.0)
    order = np.argsort(-s, kind="stable")
    counts = pop[1:].sum(axis=0)
    assert counts[order[:10]].sum() > counts[order[-10:]].sum()


def test_bad_arguments_raise():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        init_population("nope", 5, 10, 3, rng)
    with pytest.raises(ValueError):
        init_population("seeded", 5, 10, 3, rng)                    # no scores
    with pytest.raises(ValueError):
        init_population("seeded", 5, 10, 3, rng, filter_scores=np.ones(9))
    with pytest.raises(ValueError):
        init_population("seeded", 5, 10, 3, rng,
                        filter_scores=np.full(10, np.nan))
    with pytest.raises(ValueError):
        init_population("random", 5, 10, 10, rng)                   # k >= d


# ---------------------------------------------------------------------------
# (b) exact top-k mask is member 0
# ---------------------------------------------------------------------------

def test_seeded_contains_exact_top_k():
    d, k = 40, 7
    s = _scores(d, seed=5)
    expected = np.zeros(d, dtype=bool)
    expected[np.argsort(s)[-k:]] = True          # no ties, so unambiguous
    np.testing.assert_array_equal(top_k_mask(s, k), expected)
    for frac in (0.0, 0.5, 1.0):
        pop = init_population("seeded", 12, d, k, np.random.default_rng(9),
                              filter_scores=s, frac_seeded=frac)
        np.testing.assert_array_equal(pop[0], expected)


# ---------------------------------------------------------------------------
# (c) determinism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["random", "seeded"])
def test_same_seed_same_population(kind):
    s = _scores(50)
    a = init_population(kind, 20, 50, 8, np.random.default_rng(42), filter_scores=s)
    b = init_population(kind, 20, 50, 8, np.random.default_rng(42), filter_scores=s)
    c = init_population(kind, 20, 50, 8, np.random.default_rng(43), filter_scores=s)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a[1:], c[1:])      # RNG path is live


# ---------------------------------------------------------------------------
# (d) filter scores: inner-train rows only
# ---------------------------------------------------------------------------

def _inner_rows(X, y, seed, max_rows):
    """Row indices (into X) of the inner-train and inner-val rows, rebuilt
    with the same splitters and seed as make_fitness."""
    if len(X) > max_rows:
        sub, _ = next(StratifiedShuffleSplit(
            n_splits=1, train_size=max_rows, random_state=seed).split(X, y))
    else:
        sub = np.arange(len(X))
    tr, va = next(StratifiedShuffleSplit(
        n_splits=1, test_size=0.25, random_state=seed).split(X[sub], y[sub]))
    return sub[tr], sub[va]


@pytest.mark.parametrize("n,max_rows", [(600, 2000), (1200, 800)])
def test_filter_scores_inner_train_only(n, max_rows):
    seed, k = 4, 5
    X, y = _data(n=n, d=30, seed=seed)
    _, state = make_fitness(X, y, k=k, seed=seed, max_rows=max_rows)
    tr, va = _inner_rows(X, y, seed, max_rows)

    expected = np.nan_to_num(f_classif(X[tr], y[tr])[0])
    np.testing.assert_array_equal(state["filter_scores"], expected)
    full = np.nan_to_num(f_classif(X, y)[0])
    assert not np.allclose(state["filter_scores"], full)
    assert state["evals"] == 0                   # scoring is not a classifier fit

    # Overwrite every row except inner-train (inner-val and unused rows):
    # the scores must not move.  The split depends on y only, so it is fixed.
    keep = np.zeros(n, dtype=bool)
    keep[tr] = True
    X2 = X.copy()
    X2[~keep] = np.random.default_rng(0).normal(size=((~keep).sum(), X.shape[1]))
    _, state2 = make_fitness(X2, y, k=k, seed=seed, max_rows=max_rows)
    np.testing.assert_array_equal(state2["filter_scores"], state["filter_scores"])

    # ...while touching inner-train rows does move them (the check is live).
    X3 = X.copy()
    X3[tr[:50]] += 3.0
    _, state3 = make_fitness(X3, y, k=k, seed=seed, max_rows=max_rows)
    assert not np.array_equal(state3["filter_scores"], state["filter_scores"])


# ---------------------------------------------------------------------------
# (e) init_masks=None reproduces the recorded old output
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["firefly", "firefly_levy"])
def test_init_masks_none_matches_golden(method):
    """Same settings as experiments/golden_day3.py, case toy_d50, seed 0
    (recorded before any Day-4 change)."""
    golden = json.loads(GOLDEN.read_text())[f"toy_d50|{method}|seed0"]
    X, y = make_classification(n_samples=1000, n_features=50, n_informative=5,
                               n_redundant=0, n_repeated=0, shuffle=False,
                               random_state=0)
    for kw in ({}, {"init_masks": None}):
        fitness, state = make_fitness(X, y, k=5, seed=0, max_rows=1000)
        mask, fit, hist = binary_firefly(
            fitness, state, d=50, k=5, budget=600, n_fireflies=20,
            levy=(method == "firefly_levy"), seed=0, **kw)
        assert [int(i) for i in np.flatnonzero(mask)] == golden["idx"]
        assert float(fit) == golden["best_fit"]
        assert [float(h) for h in hist] == golden["history"]
        assert state["evals"] == golden["evals_used"]


# ---------------------------------------------------------------------------
# (f) seeded Firefly starts at or below the ANOVA top-k mask
# ---------------------------------------------------------------------------

def test_seeded_firefly_starts_at_or_below_anova():
    seed, d, k, n_ff = 6, 40, 5, 10
    X, y = _data(n=600, d=d, seed=seed)
    fitness, state = make_fitness(X, y, k=k, seed=seed)
    init = init_population("seeded", n_ff, d, k, np.random.default_rng(seed),
                           filter_scores=state["filter_scores"])
    trace: dict = {}
    _, _, hist = binary_firefly(fitness, state, d=d, k=k, budget=150,
                                n_fireflies=n_ff, seed=seed, trace=trace,
                                init_masks=init)
    anova = top_k_mask(state["filter_scores"], k)
    evals = state["evals"]
    f_anova = fitness(anova)
    assert state["evals"] == evals               # cached: member 0 was scored
    assert trace["init_best"] <= f_anova
    assert hist[0] <= f_anova


# ---------------------------------------------------------------------------
# (g) shared init and init_masks handling
# ---------------------------------------------------------------------------

def test_default_init_equals_init_population_random():
    seed, d, k, n_ff = 8, 30, 5, 10
    X, y = _data(n=500, d=d, seed=seed)
    fitness, state = make_fitness(X, y, k=k, seed=seed)
    rec = _Recorder(fitness)
    binary_firefly(rec, state, d=d, k=k, budget=60, n_fireflies=n_ff, seed=seed)
    expected = init_population("random", n_ff, d, k, np.random.default_rng(seed))
    np.testing.assert_array_equal(np.array(rec.masks[:n_ff]), expected)


def test_init_masks_used_and_not_mutated():
    seed, d, k, n_ff = 2, 30, 5, 8
    X, y = _data(n=500, d=d, seed=seed)
    init = init_population("seeded", n_ff, d, k, np.random.default_rng(123),
                           filter_scores=_scores(d))
    before = init.copy()
    fitness, state = make_fitness(X, y, k=k, seed=seed)
    rec = _Recorder(fitness)
    mask, _, _ = binary_firefly(rec, state, d=d, k=k, budget=80,
                                n_fireflies=n_ff, seed=seed, init_masks=init)
    np.testing.assert_array_equal(np.array(rec.masks[:n_ff]), before)
    np.testing.assert_array_equal(init, before)
    assert int(mask.sum()) == k and state["evals"] <= 80


def test_bad_init_masks_raise():
    d, k, n_ff = 20, 4, 5
    good = init_population("random", n_ff, d, k, np.random.default_rng(0))
    wrong_k = good.copy()
    wrong_k[2, np.flatnonzero(~wrong_k[2])[0]] = True
    with pytest.raises(ValueError):
        check_init_masks(wrong_k, n_ff, d, k)
    with pytest.raises(ValueError):
        check_init_masks(good[:-1], n_ff, d, k)
    fitness = lambda m: 0.5                      # never reached
    with pytest.raises(ValueError):
        binary_firefly(fitness, {"evals": 0}, d=d, k=k, budget=10,
                       n_fireflies=n_ff, seed=0, init_masks=wrong_k)
