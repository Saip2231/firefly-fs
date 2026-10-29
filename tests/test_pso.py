"""Tests for src/pso.py (PSO with random-key (top-k) decoding).

(a) every evaluated mask and the returned mask have exactly k bits
(b) evals never exceed the budget (incl. budget < n_particles)
(c) same seed -> identical mask, fitness and history
(d) history non-increasing; one diversity entry per iteration, in [0, 1]
(e) PSO and Firefly evaluate the identical initial population for the same
    seed (= init_population("random", ..., default_rng(seed)))
(f) init_masks is respected: the first n evaluated masks are init_masks,
    trace["init_best"] equals their best fitness, the array is not mutated
(g) all-cached loop stops at the 10*T iteration cap
(h) stall swap: identical to stall_swap=False up to the first stalled
    iteration; exact bookkeeping; each proposal is a one-swap move; the
    toy search spends its budget
Exact checks only; synthetic data; no filesystem access.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.firefly import binary_firefly, make_fitness
from src.init_pop import init_population
from src.pso import _decode, binary_pso


def _data(n=500, d=30, seed=0):
    X, y = make_classification(n_samples=n, n_features=d, n_informative=5,
                               n_redundant=0, n_repeated=0, shuffle=False,
                               random_state=seed)
    return X.astype(np.float64), y


class _Recorder:
    """Records every mask passed to fitness (cache hits included)."""

    def __init__(self, fitness):
        self.fitness = fitness
        self.masks: list[np.ndarray] = []

    def __call__(self, mask):
        self.masks.append(mask.copy())
        return self.fitness(mask)


def _const_fitness(k: int, value: float = 0.5):
    """Constant fitness with make_fitness's cache / evals semantics."""
    cache: dict[bytes, float] = {}
    state: dict = {"evals": 0}

    def fitness(mask):
        assert int(mask.sum()) == k
        key = mask.tobytes()
        if key not in cache:
            cache[key] = value
            state["evals"] += 1
        return value

    return fitness, state, cache


# (a) + (b) -----------------------------------------------------------------

@pytest.mark.parametrize("budget,n_p", [(150, 10), (5, 20), (61, 20)])
def test_exactly_k_and_budget(budget, n_p):
    seed, d, k = 1, 30, 6
    X, y = _data(d=d, seed=seed)
    fitness, state = make_fitness(X, y, k=k, seed=seed)
    rec = _Recorder(fitness)
    mask, fit, hist = binary_pso(rec, state, d=d, k=k, budget=budget,
                                 n_particles=n_p, seed=seed)
    assert all(int(m.sum()) == k for m in rec.masks)
    assert mask.dtype == bool and int(mask.sum()) == k
    assert state["evals"] <= budget
    assert fit == state["cache"][mask.tobytes()]


def test_decode_exactly_k():
    rng = np.random.default_rng(0)
    for k in (1, 5, 29):
        for _ in range(20):
            assert int(_decode(rng.normal(size=30), k).sum()) == k


# (c) -----------------------------------------------------------------------

def test_same_seed_identical():
    d, k = 30, 5
    X, y = _data(d=d, seed=7)
    out = []
    for _ in range(2):
        fitness, state = make_fitness(X, y, k=k, seed=0)
        mask, fit, hist = binary_pso(fitness, state, d=d, k=k, budget=200,
                                     n_particles=10, seed=42)
        out.append((mask.tobytes(), fit, hist, state["evals"], state["diversity"]))
    assert out[0] == out[1]


# (d) -----------------------------------------------------------------------

def test_history_non_increasing_and_diversity():
    d, k = 40, 5
    X, y = _data(n=600, d=d, seed=3)
    fitness, state = make_fitness(X, y, k=k, seed=3)
    _, fit, hist = binary_pso(fitness, state, d=d, k=k, budget=300,
                              n_particles=12, seed=3)
    assert len(hist) > 0 and hist[-1] == fit
    assert all(hist[t] <= hist[t - 1] for t in range(1, len(hist)))
    div = state["diversity"]
    assert len(div) == len(hist) == state["pso_iters"]
    assert all(0.0 <= v <= 1.0 for v in div)


# (e) -----------------------------------------------------------------------

def test_pso_and_firefly_share_initial_population():
    seed, d, k, n = 5, 30, 5, 10
    X, y = _data(d=d, seed=seed)
    fa_fit, fa_state = make_fitness(X, y, k=k, seed=seed)
    fa_rec = _Recorder(fa_fit)
    binary_firefly(fa_rec, fa_state, d=d, k=k, budget=60, n_fireflies=n, seed=seed)
    ps_fit, ps_state = make_fitness(X, y, k=k, seed=seed)
    ps_rec = _Recorder(ps_fit)
    binary_pso(ps_rec, ps_state, d=d, k=k, budget=60, n_particles=n, seed=seed)

    expected = init_population("random", n, d, k, np.random.default_rng(seed))
    np.testing.assert_array_equal(np.array(fa_rec.masks[:n]), expected)
    np.testing.assert_array_equal(np.array(ps_rec.masks[:n]), expected)


# (f) -----------------------------------------------------------------------

def test_init_masks_respected():
    seed, d, k, n = 2, 30, 5, 8
    X, y = _data(d=d, seed=seed)
    fitness, state = make_fitness(X, y, k=k, seed=seed)
    init = init_population("seeded", n, d, k, np.random.default_rng(11),
                           filter_scores=state["filter_scores"])
    before = init.copy()
    rec = _Recorder(fitness)
    trace: dict = {}
    _, fit, hist = binary_pso(rec, state, d=d, k=k, budget=100, n_particles=n,
                              seed=seed, init_masks=init, trace=trace)
    np.testing.assert_array_equal(np.array(rec.masks[:n]), before)
    np.testing.assert_array_equal(init, before)

    ref_fit, _ = make_fitness(X, y, k=k, seed=seed)       # fresh, same split
    assert trace["init_best"] == min(ref_fit(m) for m in before)
    assert trace["init_evals"] == len({m.tobytes() for m in before})
    assert hist[0] <= trace["init_best"] and fit <= trace["init_best"]


def test_bad_init_masks_raise():
    d, k, n = 20, 4, 5
    bad = init_population("random", n, d, k, np.random.default_rng(0))
    bad[0, np.flatnonzero(~bad[0])[0]] = True
    with pytest.raises(ValueError):
        binary_pso(lambda m: 0.5, {"evals": 0}, d=d, k=k, budget=10,
                   n_particles=n, init_masks=bad)


# (g) -----------------------------------------------------------------------

def test_all_cached_stops_at_iteration_cap():
    """d=4, k=2: only C(4,2)=6 masks exist, so a budget of 50 can never be
    spent; the loop must stop at 10*T iterations, T = 50 // 5 = 10."""
    d, k, budget, n = 4, 2, 50, 5
    fitness, state, cache = _const_fitness(k)
    mask, fit, hist = binary_pso(fitness, state, d=d, k=k, budget=budget,
                                 n_particles=n, seed=1)
    assert state["evals"] == len(cache) <= 6
    assert len(hist) == 10 * (budget // n)
    assert int(mask.sum()) == k and fit == 0.5


# (h) stall swap ------------------------------------------------------------

def _run_pair(seed=4, d=30, k=5, n=10, budget=300):
    """Same search with stall_swap False / True; returns {flag: (..)}."""
    X, y = _data(d=d, seed=seed)
    runs = {}
    for flag in (False, True):
        fitness, state = make_fitness(X, y, k=k, seed=seed)
        rec = _Recorder(fitness)
        trace: dict = {}
        mask, fit, hist = binary_pso(rec, state, d=d, k=k, budget=budget,
                                     n_particles=n, seed=seed, trace=trace,
                                     stall_swap=flag)
        runs[flag] = dict(mask=mask, fit=fit, hist=hist, trace=trace,
                          state=state, rec=rec)
    return runs


def test_stall_swap_identical_until_first_stall():
    """stall_swap draws nothing before the first iteration whose moves gave 0
    new evals: both settings agree on every iteration before it, and the
    fallback fires in that iteration."""
    budget, n = 300, 10
    runs = _run_pair(budget=budget, n=n)
    off, on = runs[False]["trace"], runs[True]["trace"]
    assert runs[False]["state"]["pso_stall_iters"] == 0
    assert all(r["stall_swaps"] == 0 for r in off["iters"])

    used, s = off["init_evals"], None
    for t, r in enumerate(off["iters"]):
        used += r["move_evals"]
        if r["move_evals"] == 0 and used < budget:
            s = t
            break
    if s is None:                                  # never stalled: identical
        assert runs[False]["hist"] == runs[True]["hist"]
        np.testing.assert_array_equal(runs[False]["mask"], runs[True]["mask"])
    else:
        assert off["iters"][:s] == on["iters"][:s]
        assert on["iters"][s]["move_evals"] == 0
        assert on["iters"][s]["stall_swaps"] == n


def test_stall_swap_bookkeeping():
    """The fallback fires exactly when the moves give 0 new evals (budget left),
    makes one swap per particle, and the counter matches the trace; cap and
    exactly-k still hold."""
    d, k, budget, n = 4, 2, 50, 5
    fitness, state, cache = _const_fitness(k)       # asserts exactly k per call
    trace: dict = {}
    mask, _, hist = binary_pso(fitness, state, d=d, k=k, budget=budget,
                               n_particles=n, seed=1, trace=trace)
    its = trace["iters"]                            # budget can never be spent
    assert all((r["stall_swaps"] > 0) == (r["move_evals"] == 0) for r in its)
    assert all(r["stall_swaps"] in (0, n) for r in its)
    assert state["pso_stall_iters"] == sum(r["stall_swaps"] > 0 for r in its) > 0
    assert len(hist) == 10 * (budget // n) and state["evals"] == len(cache) <= 6
    assert int(mask.sum()) == k


def test_stall_swap_proposals_are_one_swap_moves():
    """Each stall-swap proposal differs from the particle's previous evaluated
    mask by exactly one swap (Hamming distance 2) and has exactly k bits."""
    budget, n = 300, 10
    run = _run_pair(budget=budget, n=n)[True]
    its, masks = run["trace"]["iters"], run["rec"].masks
    # Call order: n initial masks, then per iteration n move calls (fewer only
    # in the last iteration, when the budget runs out) and, if the fallback
    # fired, stall_swaps swap calls.  A stalled iteration always completes
    # its n moves (budget was left), so the offsets below are exact for it.
    pos = n
    checked = 0
    for r in its:
        moves = masks[pos:pos + n]
        pos += n
        if r["stall_swaps"]:
            swaps = masks[pos:pos + r["stall_swaps"]]
            pos += r["stall_swaps"]
            for before, after in zip(moves, swaps):
                assert int(after.sum()) == int(before.sum())
                assert int(np.sum(before != after)) == 2
                checked += 1
    assert 0 <= pos - len(masks) < n                # only the last iteration short
    assert checked > 0


def test_stall_swap_spends_budget():
    """With the fallback the toy search uses its whole budget (the original
    PSO stopped at the iteration cap with budget left on such runs)."""
    run = _run_pair(budget=300, n=10)[True]
    assert run["state"]["evals"] == 300


def test_bad_arguments_raise():
    f, st, _ = _const_fitness(3)
    for kw in (dict(budget=0), dict(k=10), dict(n_particles=0)):
        args = dict(d=10, k=3, budget=10, n_particles=4) | kw
        with pytest.raises(ValueError):
            binary_pso(f, st, **args)
