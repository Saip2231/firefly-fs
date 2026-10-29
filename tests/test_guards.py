"""F4 tests: loop guards in random_search, budget / k validation, and the
2*min(k, d-k) distance normalisation.  Exact checks only."""

from __future__ import annotations

import sys
from math import comb
from pathlib import Path

import numpy as np
import pytest

from src.firefly import binary_firefly

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from _common import random_search  # noqa: E402


def _const_fitness(k: int, value: float = 0.5):
    """Constant fitness with make_fitness cache / evals semantics."""
    cache: dict[bytes, float] = {}
    state: dict = {"evals": 0}

    def fitness(mask: np.ndarray) -> float:
        assert int(mask.sum()) == k
        key = mask.tobytes()
        if key not in cache:
            cache[key] = value
            state["evals"] += 1
        return value

    return fitness, state


# -- random_search loop guards ------------------------------------------------

def test_random_search_terminates_when_budget_exceeds_mask_count():
    """d=6, k=3 has C(6,3)=20 masks < budget=50: without a guard this loops
    forever.  It must stop on the consecutive-cache-hit guard."""
    fitness, state = _const_fitness(3)
    out = random_search(fitness, state, d=6, k=3, budget=50, seed=0)
    assert out["evals_used"] == state["evals"] <= comb(6, 3) < 50
    assert out["stopped"] == "cache_hits"
    assert out["proposals"] <= 100 * 50
    assert int(out["best_mask"].sum()) == 3


def test_random_search_normal_run_spends_exact_budget():
    fitness, state = _const_fitness(5)
    out = random_search(fitness, state, d=200, k=5, budget=60, seed=1)
    assert out["evals_used"] == 60 and out["stopped"] == "budget"


# -- budget and k validation ---------------------------------------------------

def test_budget_must_be_positive():
    fitness, state = _const_fitness(3)
    with pytest.raises(ValueError):
        random_search(fitness, state, d=20, k=3, budget=0, seed=0)
    with pytest.raises(ValueError):
        binary_firefly(fitness, state, d=20, k=3, budget=0, seed=0)
    assert state["evals"] == 0


@pytest.mark.parametrize("k,d", [(10, 10), (12, 10), (0, 10)])
def test_firefly_rejects_k_out_of_range(k, d):
    fitness, state = _const_fitness(k)
    with pytest.raises(ValueError):
        binary_firefly(fitness, state, d=d, k=k, budget=10, seed=0)
    assert state["evals"] == 0


# -- 2*min(k, d-k) normalisation ----------------------------------------------

def test_diversity_in_unit_interval_when_k_above_half_d():
    """For k > d/2 the max Hamming distance is 2*(d-k), not 2k.  With the
    correct normaliser diversity stays in [0, 1] and can exceed the old
    2k-normalised ceiling of (d-k)/k."""
    d, k = 10, 8
    fitness, state = _const_fitness(k)
    binary_firefly(fitness, state, d=d, k=k, budget=30, n_fireflies=6, seed=2)
    div = state["diversity"]
    assert div and all(0.0 <= v <= 1.0 for v in div)
    assert max(div) > (d - k) / k       # unreachable under the old 2k divisor
