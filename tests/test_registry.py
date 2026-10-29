"""Tests for src/search_registry.py (run_search dispatcher).

(a) all five methods: exactly k bits, evals <= budget, history/diversity
    lengths agree, init_best >= best_fit
(b) firefly, firefly_levy and pso evaluate identical initial populations
(c) firefly_seeded and pso_seeded evaluate identical initial populations,
    member 0 is the inner-train ANOVA top-k mask, and they start at or below
    its fitness
(d) registry firefly is byte-identical to calling binary_firefly directly
(e) unknown method, unknown hyperparameter, reused state -> ValueError
Exact checks only; synthetic data; no filesystem access.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.firefly import binary_firefly, make_fitness
from src.init_pop import init_population, top_k_mask
from src.search_registry import METHODS, run_search

D, K, BUDGET, SEED, N = 30, 5, 120, 3, 10
HP = {
    "firefly":        {"n_fireflies": N},
    "firefly_levy":   {"n_fireflies": N},
    "pso":            {"n_particles": N},
    "firefly_seeded": {"n_fireflies": N},
    "pso_seeded":     {"n_particles": N},
}


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(n_samples=500, n_features=D, n_informative=5,
                               n_redundant=0, n_repeated=0, shuffle=False,
                               random_state=SEED)
    return X.astype(np.float64), y


class _Recorder:
    """Records every mask passed to fitness (cache hits included)."""

    def __init__(self, fitness):
        self.fitness = fitness
        self.masks: list[np.ndarray] = []

    def __call__(self, mask):
        self.masks.append(mask.copy())
        return self.fitness(mask)


def _run(method, data, hp=None):
    X, y = data
    fitness, state = make_fitness(X, y, k=K, seed=SEED)
    rec = _Recorder(fitness)
    out = run_search(method, rec, state, D, K, BUDGET, SEED,
                     HP[method] if hp is None else hp)
    return out, rec, state


# (a) -----------------------------------------------------------------------

@pytest.mark.parametrize("method", METHODS)
def test_all_methods_exactly_k_and_budget(method, data):
    out, rec, state = _run(method, data)
    assert out["method"] == method
    assert out["best_mask"].dtype == bool and int(out["best_mask"].sum()) == K
    assert all(int(m.sum()) == K for m in rec.masks)
    assert out["evals_used"] == state["evals"] <= BUDGET
    assert len(out["history"]) == len(out["diversity"]) > 0
    assert out["init_best"] >= out["best_fit"] == out["history"][-1]


# (b) -----------------------------------------------------------------------

def test_plain_methods_share_initial_population(data):
    first = {m: np.array(_run(m, data)[1].masks[:N])
             for m in ("firefly", "firefly_levy", "pso")}
    expected = init_population("random", N, D, K, np.random.default_rng(SEED))
    for m, masks in first.items():
        np.testing.assert_array_equal(masks, expected, err_msg=m)


# (c) -----------------------------------------------------------------------

def test_seeded_methods_share_initial_population(data):
    runs = {m: _run(m, data) for m in ("firefly_seeded", "pso_seeded")}
    fa, ps = (np.array(runs[m][1].masks[:N]) for m in ("firefly_seeded", "pso_seeded"))
    np.testing.assert_array_equal(fa, ps)

    state = runs["firefly_seeded"][2]
    expected = init_population("seeded", N, D, K, np.random.default_rng(SEED),
                               filter_scores=state["filter_scores"])
    np.testing.assert_array_equal(fa, expected)
    anova = top_k_mask(state["filter_scores"], K)
    np.testing.assert_array_equal(fa[0], anova)

    X, y = data
    ref_fit, _ = make_fitness(X, y, k=K, seed=SEED)      # same inner split
    f_anova = ref_fit(anova)
    for m, (out, _, _) in runs.items():
        assert out["init_best"] <= f_anova, m


def test_frac_seeded_passed_through(data):
    out_a, rec_a, st = _run("pso_seeded", data, {"n_particles": N, "frac_seeded": 1.0})
    expected = init_population("seeded", N, D, K, np.random.default_rng(SEED),
                               filter_scores=st["filter_scores"], frac_seeded=1.0)
    np.testing.assert_array_equal(np.array(rec_a.masks[:N]), expected)


# (d) -----------------------------------------------------------------------

def test_registry_firefly_matches_direct_call(data):
    out, _, state = _run("firefly", data)
    X, y = data
    fitness, st2 = make_fitness(X, y, k=K, seed=SEED)
    mask, fit, hist = binary_firefly(fitness, st2, d=D, k=K, budget=BUDGET,
                                     n_fireflies=N, seed=SEED)
    np.testing.assert_array_equal(out["best_mask"], mask)
    assert out["best_fit"] == fit and out["history"] == hist
    assert out["evals_used"] == st2["evals"]
    assert out["diversity"] == st2["diversity"]


# (e) -----------------------------------------------------------------------

def test_unknown_method_raises(data):
    X, y = data
    fitness, state = make_fitness(X, y, k=K, seed=SEED)
    with pytest.raises(ValueError):
        run_search("ga", fitness, state, D, K, BUDGET, SEED, {})


def test_unknown_hyperparameter_raises(data):
    X, y = data
    for method, hp in (("firefly", {"n_particles": 10}),
                       ("pso", {"beta0": 0.4}),
                       ("firefly", {"frac_seeded": 0.5})):
        fitness, state = make_fitness(X, y, k=K, seed=SEED)
        with pytest.raises(ValueError):
            run_search(method, fitness, state, D, K, BUDGET, SEED, hp)


def test_reused_state_raises(data):
    X, y = data
    fitness, state = make_fitness(X, y, k=K, seed=SEED)
    run_search("firefly", fitness, state, D, K, BUDGET, SEED, HP["firefly"])
    with pytest.raises(ValueError):
        run_search("pso", fitness, state, D, K, BUDGET, SEED, HP["pso"])
