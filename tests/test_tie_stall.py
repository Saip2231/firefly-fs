"""Tests for the F3 tie-stall fallback in binary_firefly.

A stub fitness that returns the same value for every valid mask makes every
sweep a tie-stall (no firefly strictly brighter than another).  Before F3 the
search stopped after the initial swarm; now it must keep spending budget,
or stop at the hard sweep cap when every proposal is a cache hit.
Exact checks only (counts and equalities).
"""

from __future__ import annotations

import numpy as np

from src.firefly import binary_firefly


def _const_fitness(k: int, value: float = 0.5):
    """Constant fitness with the same cache / evals semantics as make_fitness."""
    cache: dict[bytes, float] = {}
    state: dict = {"evals": 0}

    def fitness(mask: np.ndarray) -> float:
        assert int(mask.sum()) == k, "candidate without exactly k bits"
        key = mask.tobytes()
        if key not in cache:
            cache[key] = value
            state["evals"] += 1
        return value

    return fitness, state, cache


def test_constant_fitness_uses_whole_budget():
    d, k, budget, n = 30, 5, 200, 10
    fitness, state, _ = _const_fitness(k)
    trace: dict = {}
    mask, fit, hist = binary_firefly(fitness, state, d=d, k=k, budget=budget,
                                     n_fireflies=n, seed=0, trace=trace)
    assert state["evals"] == budget                    # whole budget, never more
    assert state["stall_sweeps"] == len(trace["sweeps"]) > 0
    assert int(mask.sum()) == k and fit == 0.5
    assert len(hist) <= budget                         # within the sweep cap
    sw = trace["sweeps"]
    assert all(s["proposals"] == s["stall_proposals"] for s in sw)
    assert trace["init_evals"] + sum(s["new_evals"] for s in sw) == state["evals"]
    for s in sw:
        assert 0 <= s["cache_hits"] <= s["proposals"]
        assert s["new_evals"] <= s["proposals"]


def test_constant_fitness_stops_at_sweep_cap_when_all_cached():
    """d=4, k=2 has only C(4,2)=6 masks: the budget can never be spent, so
    the loop must end at the hard cap (max_sweeps = budget) with evals = 6."""
    d, k, budget, n = 4, 2, 50, 5
    fitness, state, cache = _const_fitness(k)
    mask, fit, hist = binary_firefly(fitness, state, d=d, k=k, budget=budget,
                                     n_fireflies=n, seed=1)
    assert state["evals"] == len(cache) <= 6 <= budget
    assert len(hist) == budget                         # hard cap reached
    assert int(mask.sum()) == k


def test_constant_fitness_deterministic():
    d, k, budget = 30, 5, 120
    out = []
    for _ in range(2):
        fitness, state, cache = _const_fitness(k)
        mask, _, hist = binary_firefly(fitness, state, d=d, k=k, budget=budget,
                                       n_fireflies=8, seed=4)
        out.append((mask.tobytes(), hist, sorted(cache)))
    assert out[0] == out[1]
