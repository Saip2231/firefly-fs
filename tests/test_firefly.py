"""Tests for src/firefly.py (v2 -- three targeted fixes).

Test inventory
--------------
(a) planted_features
    Uses make_classification with 5 informative features placed at the
    first 5 columns (shuffle=False). Firefly with budget=600 and k=5
    averaged across 5 seeds must recover at least 3.0 of them on average
    and stay within budget.

(b) beats_random
    Averaged over 5 seeds, Firefly should achieve lower mean fitness
    than pure random search given the same evaluation budget.
    Random search baseline: sample budget random valid masks, keep best.

(c) reproducibility
    Same seed => identical masks; different seeds are allowed to differ
    (not required to, but this verifies the RNG path is seed-controlled).

(d) invariants
    history is non-increasing; every mask returned by binary_firefly has
    exactly k True values.

(e) diversity_logged
    state["diversity"] is populated after each sweep, values in [0, 1],
    and diversity does not collapse to zero within the first few sweeps
    (confirms FIX 3: lower beta0 preserves population variety).

No test ever touches real datasets or the filesystem.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.firefly import binary_firefly, make_fitness, repair


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _synthetic(n=1000, d=50, n_informative=5, seed=0):
    """Return (Xtr, ytr) from a controlled synthetic dataset."""
    X, y = make_classification(
        n_samples=n,
        n_features=d,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        shuffle=False,        # informative features stay at columns 0..n_inf-1
        random_state=seed,
    )
    return X.astype(np.float64), y


# ---------------------------------------------------------------------------
# (a) Planted features
# ---------------------------------------------------------------------------

def test_planted_features():
    """Firefly recovers planted informative columns averaged over 5 seeds (mean >= 3.0 / 5)."""
    k = 5
    budget = 600
    informative_cols = np.arange(5)
    hits = []

    for s in range(5):
        Xtr, ytr = _synthetic(n=1000, d=50, n_informative=5, seed=s)
        fitness, state = make_fitness(Xtr, ytr, k=k, seed=s)
        best_mask, best_fit, history = binary_firefly(
            fitness, state, d=50, k=k,
            budget=budget, n_fireflies=20,
            seed=s,
        )

        overlap = int(best_mask[informative_cols].sum())
        hits.append(overlap)

        # Budget respected on every seed
        assert state["evals"] <= budget, (
            f"Seed {s}: Used {state['evals']} evals, budget was {budget}"
        )

    mean_hits = float(np.mean(hits))
    assert mean_hits >= 3.0, (
        f"Only {mean_hits:.2f}/5 planted features recovered on average across 5 seeds: {hits}. "
        f"Expected >= 3.0 (random search averages ~0.5)."
    )


# ---------------------------------------------------------------------------
# (b) Firefly beats pure random search
# ---------------------------------------------------------------------------

def _random_search_fitness(fitness, state, d, k, budget, seed):
    """Baseline: evaluate random valid masks until budget exhausted."""
    rng      = np.random.default_rng(seed)
    best_fit = np.inf
    while state["evals"] < budget:
        mask = np.zeros(d, dtype=bool)
        chosen = rng.choice(d, size=k, replace=False)
        mask[chosen] = True
        f = fitness(mask)
        if f < best_fit:
            best_fit = f
    return best_fit


def test_firefly_beats_random():
    """Firefly mean fitness < random-search mean fitness (averaged over 5 seeds)."""
    d      = 50
    k      = 5
    budget = 400    # shared budget between FA and random search

    fa_fits  = []
    rnd_fits = []

    for seed in range(5):
        Xtr, ytr = _synthetic(n=800, d=d, n_informative=5, seed=seed)

        # --- Firefly ---
        fitness_fa, state_fa = make_fitness(Xtr, ytr, k=k, seed=seed)
        _, fa_best, _ = binary_firefly(
            fitness_fa, state_fa, d=d, k=k,
            budget=budget, n_fireflies=15,
            seed=seed,
        )
        fa_fits.append(fa_best)

        # --- Random search (fresh fitness / state, same budget) ---
        fitness_rnd, state_rnd = make_fitness(Xtr, ytr, k=k, seed=seed)
        rnd_best = _random_search_fitness(
            fitness_rnd, state_rnd, d=d, k=k, budget=budget, seed=seed
        )
        rnd_fits.append(rnd_best)

    mean_fa  = float(np.mean(fa_fits))
    mean_rnd = float(np.mean(rnd_fits))

    assert mean_fa < mean_rnd, (
        f"Firefly ({mean_fa:.4f}) did not beat random search ({mean_rnd:.4f})"
    )


# ---------------------------------------------------------------------------
# (c) Reproducibility
# ---------------------------------------------------------------------------

def test_same_seed_identical():
    """Same seed => identical best masks."""
    Xtr, ytr = _synthetic(n=500, d=30, seed=7)
    k        = 5

    def _run(seed_fa):
        fitness, state = make_fitness(Xtr, ytr, k=k, seed=0)
        mask, _, _ = binary_firefly(
            fitness, state, d=30, k=k,
            budget=200, n_fireflies=10,
            seed=seed_fa,
        )
        return mask

    mask1 = _run(seed_fa=42)
    mask2 = _run(seed_fa=42)
    np.testing.assert_array_equal(mask1, mask2, err_msg="Same seed gave different masks")


def test_different_seeds_can_differ():
    """Different seeds are allowed to produce different masks.

    We run with two very different seeds and check that at least one pair
    of runs differs (over 3 attempts), confirming the RNG path is live.
    """
    Xtr, ytr = _synthetic(n=500, d=30, seed=7)
    k        = 5

    any_diff = False
    for _ in range(3):
        fitness_a, state_a = make_fitness(Xtr, ytr, k=k, seed=0)
        mask_a, _, _ = binary_firefly(
            fitness_a, state_a, d=30, k=k,
            budget=150, n_fireflies=10, seed=1,
        )
        fitness_b, state_b = make_fitness(Xtr, ytr, k=k, seed=0)
        mask_b, _, _ = binary_firefly(
            fitness_b, state_b, d=30, k=k,
            budget=150, n_fireflies=10, seed=99,
        )
        if not np.array_equal(mask_a, mask_b):
            any_diff = True
            break

    assert any_diff, "Different seeds always gave identical masks — RNG may be broken"


# ---------------------------------------------------------------------------
# (d) Invariants: history non-increasing, masks have exactly k features
# ---------------------------------------------------------------------------

def test_history_non_increasing():
    """History values must be non-increasing (greedy acceptance guarantee)."""
    Xtr, ytr = _synthetic(n=600, d=40, seed=3)
    k        = 5

    fitness, state = make_fitness(Xtr, ytr, k=k, seed=3)
    _, _, history = binary_firefly(
        fitness, state, d=40, k=k,
        budget=300, n_fireflies=12,
        seed=3,
    )

    assert len(history) > 0, "History is empty"

    for t in range(1, len(history)):
        assert history[t] <= history[t - 1] + 1e-12, (
            f"History increased at step {t}: "
            f"{history[t-1]:.6f} -> {history[t]:.6f}"
        )


def test_returned_mask_has_exactly_k():
    """best_mask returned by binary_firefly must have exactly k True values."""
    Xtr, ytr = _synthetic(n=600, d=40, seed=5)
    k        = 8

    fitness, state = make_fitness(Xtr, ytr, k=k, seed=5)
    best_mask, _, _ = binary_firefly(
        fitness, state, d=40, k=k,
        budget=250, n_fireflies=10,
        seed=5,
    )

    n_selected = int(best_mask.sum())
    assert n_selected == k, (
        f"best_mask has {n_selected} True values, expected {k}"
    )


# ---------------------------------------------------------------------------
# (e) Diversity logging
# ---------------------------------------------------------------------------

def test_diversity_logged():
    """state['diversity'] is populated, values in [0,1], swarm stays diverse.

    FIX 3 (beta0=0.4) should prevent immediate collapse.  We check that the
    mean diversity over all recorded sweeps is above a small threshold,
    confirming the population has not been cloned into a single point.
    """
    Xtr, ytr = _synthetic(n=600, d=40, seed=11)
    k        = 6

    fitness, state = make_fitness(Xtr, ytr, k=k, seed=11)
    _, _, history = binary_firefly(
        fitness, state, d=40, k=k,
        budget=300, n_fireflies=12,
        beta0=0.4,   # new default; explicitly stated for clarity
        seed=11,
    )

    div = state.get("diversity", [])

    # diversity list must have one entry per recorded sweep
    assert len(div) == len(history), (
        f"len(diversity)={len(div)} != len(history)={len(history)}"
    )

    # All values must be in [0, 1]
    for t, v in enumerate(div):
        assert 0.0 <= v <= 1.0 + 1e-9, (
            f"diversity[{t}]={v:.4f} is outside [0, 1]"
        )

    # Mean diversity must be above a small threshold -- swarm not collapsed
    mean_div = float(np.mean(div)) if div else 0.0
    assert mean_div > 0.05, (
        f"Mean diversity={mean_div:.4f} is too low; swarm may have collapsed "
        f"(check beta0 and gamma settings)"
    )


# ---------------------------------------------------------------------------
# Bonus: repair unit tests (fast, no fitness calls)
# ---------------------------------------------------------------------------

def test_repair_adds_missing():
    """repair adds True values when mask has too few."""
    rng  = np.random.default_rng(0)
    mask = np.zeros(20, dtype=bool)
    mask[:2] = True           # only 2 True, target k=5
    fixed = repair(mask, k=5, rng=rng)
    assert fixed.sum() == 5
    assert not np.shares_memory(fixed, mask)


def test_repair_removes_extras():
    """repair removes True values when mask has too many."""
    rng  = np.random.default_rng(0)
    mask = np.ones(20, dtype=bool)   # all True, target k=5
    fixed = repair(mask, k=5, rng=rng)
    assert fixed.sum() == 5
    assert not np.shares_memory(fixed, mask)


def test_repair_exact_noop():
    """repair leaves mask unchanged (but still returns a copy) when sum==k."""
    rng  = np.random.default_rng(0)
    mask = np.zeros(20, dtype=bool)
    mask[:5] = True
    fixed = repair(mask, k=5, rng=rng)
    assert fixed.sum() == 5
    np.testing.assert_array_equal(fixed, mask)
    assert not np.shares_memory(fixed, mask)
