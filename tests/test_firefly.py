"""Tests for src/firefly.py (v2 -- three targeted fixes).

Test inventory
--------------
(a) planted_features  [noisy, clean]
    make_classification with 5 informative columns at 0..4 (shuffle=False),
    on a noisy (class_sep=1, 2 clusters/class) and a clean (class_sep=2,
    1 cluster/class) dataset.  Firefly (budget=600, k=5), 5-seed mean recall
    of the SEPARATING informative columns must be >= 0.5 (clean: capped at
    2 columns); budget respected.  Threshold reasons are documented beside
    PLANTED_CONFIGS.

(b) not_worse_than_random
    Averaged over 5 seeds, Firefly's mean validation fitness should be lower
    than pure random search's at the same evaluation budget.  A smoke check,
    not a statistical claim.
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

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.firefly import binary_firefly, make_fitness, repair

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from _common import random_search  # noqa: E402


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _synthetic(n=1000, d=50, n_informative=5, seed=0,
               class_sep=1.0, n_clusters_per_class=2):
    """Return (Xtr, ytr) from a controlled synthetic dataset.

    class_sep / n_clusters_per_class default to sklearn's defaults, so every
    existing call is unchanged.
    """
    X, y = make_classification(
        n_samples=n,
        n_features=d,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=n_clusters_per_class,
        class_sep=class_sep,
        shuffle=False,        # informative features stay at columns 0..n_inf-1
        random_state=seed,
    )
    return X.astype(np.float64), y


# ---------------------------------------------------------------------------
# (a) Planted features
# ---------------------------------------------------------------------------

# Ground truth: only informative columns whose class means differ carry
# linear signal (see src/synthetic.py), so recall is measured on the
# SEPARATING columns, not on all n_informative columns.
#
# THRESHOLD = 0.5 (both datasets).  Reason, fixed from chance level and not
# from any observed run: a random k-subset has expected recall k/d = 5/50 =
# 0.10, and a 5-seed mean of 0.5 under random choice is vanishingly unlikely,
# so 0.5 separates "found signal" from chance with a wide margin while staying
# well below the maximum the fitness can reward (1.0).
#
# CAP (clean only) = 2.  Reason: with class_sep=2 each separating column puts
# the class means 4 units apart, so two such columns already drive the Bayes
# error to ~0; on 250 validation rows (1 row = 0.4 pts) the fitness cannot
# reward a third, so recall beyond 2 columns would test the data, not the
# search.  Noisy data (class_sep=1) does not saturate, so it is uncapped.
PLANTED_CONFIGS = {
    "noisy": dict(class_sep=1.0, n_clusters_per_class=2, cap=None),
    "clean": dict(class_sep=2.0, n_clusters_per_class=1, cap=2),
}
PLANTED_MIN_RECALL = 0.5


@pytest.mark.parametrize("name", list(PLANTED_CONFIGS))
def test_planted_features(name):
    """Firefly recovers planted SEPARATING columns: 5-seed mean recall >= 0.5."""
    from src.synthetic import separating_columns

    cfg = PLANTED_CONFIGS[name]
    d, k, budget = 50, 5, 600
    recalls = []

    for s in range(5):
        Xtr, ytr = _synthetic(n=1000, d=d, n_informative=5, seed=s,
                              class_sep=cfg["class_sep"],
                              n_clusters_per_class=cfg["n_clusters_per_class"])
        sep = separating_columns(Xtr, ytr, 5, cfg["class_sep"],
                                 cfg["n_clusters_per_class"])
        # A doubtful ground-truth label must fail loudly, not miscount silently
        assert sep["ambiguous"].size == 0, (
            f"{name} seed {s}: ambiguous separating label(s) {sep['ambiguous']}, "
            f"gaps {np.round(sep['gaps'], 3)}"
        )
        n_sep = len(sep["sep_idx"])
        assert n_sep > 0, f"{name} seed {s}: no separating column"

        fitness, state = make_fitness(Xtr, ytr, k=k, seed=s)
        best_mask, _, _ = binary_firefly(
            fitness, state, d=d, k=k, budget=budget, n_fireflies=20, seed=s,
        )
        # Budget respected on every seed
        assert state["evals"] <= budget, (
            f"{name} seed {s}: used {state['evals']} evals, budget was {budget}"
        )

        sep_hits = int(best_mask[sep["sep_idx"]].sum())
        if cfg["cap"] is not None:
            recalls.append(min(sep_hits, cfg["cap"]) / min(n_sep, cfg["cap"]))
        else:
            recalls.append(sep_hits / n_sep)

    mean_recall = float(np.mean(recalls))
    assert mean_recall >= PLANTED_MIN_RECALL, (
        f"{name}: mean separating-column recall {mean_recall:.2f} over 5 seeds "
        f"{np.round(recalls, 2).tolist()} < {PLANTED_MIN_RECALL} (chance = {k / d:.2f})"
    )


# ---------------------------------------------------------------------------
# (b) Firefly not worse than pure random search (smoke check, no claim)
# ---------------------------------------------------------------------------

def _random_search_fitness(fitness, state, d, k, budget, seed):
    """Baseline: the guarded experiments/_common.random_search (same RNG draws
    and strict-improvement rule as the old local loop, plus loop guards)."""
    return random_search(fitness, state, d=d, k=k, budget=budget, seed=seed)["best_fit"]


def test_firefly_not_worse_than_random():
    """Firefly mean val fitness < random-search mean val fitness over 5 seeds.

    Smoke check only: it makes NO statistical claim (no test, n=5, in-sample
    validation fitness on one toy).  The threshold is unchanged from Day 3.
    """
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
        f"Firefly mean val fitness ({mean_fa:.4f}) not below random search ({mean_rnd:.4f})"
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


# ---------------------------------------------------------------------------
# (f) trace=None leaves algorithm byte-for-byte unchanged
# ---------------------------------------------------------------------------

def test_trace_none_unchanged():
    """binary_firefly with trace=None and trace={} must produce identical
    best_mask, best_fit, and history (same seed => same RNG path).

    This verifies that the tracing instrumentation has zero effect on
    algorithm behaviour: no extra RNG draws, no branching in the hot path.
    """
    Xtr, ytr = _synthetic(n=600, d=40, seed=17)
    k = 6

    # Run 1: no trace (baseline)
    fitness1, state1 = make_fitness(Xtr, ytr, k=k, seed=17)
    mask1, fit1, hist1 = binary_firefly(
        fitness1, state1, d=40, k=k,
        budget=300, n_fireflies=10,
        seed=17,
        trace=None,      # explicit None
    )

    # Run 2: with trace dict
    trace2: dict = {}
    fitness2, state2 = make_fitness(Xtr, ytr, k=k, seed=17)
    mask2, fit2, hist2 = binary_firefly(
        fitness2, state2, d=40, k=k,
        budget=300, n_fireflies=10,
        seed=17,
        trace=trace2,
    )

    np.testing.assert_array_equal(
        mask1, mask2,
        err_msg="trace=None vs trace={} produced different best_mask",
    )
    assert fit1 == fit2, (
        f"trace=None fit {fit1:.8f} != trace={{}} fit {fit2:.8f}"
    )
    assert hist1 == hist2, "trace=None vs trace={} produced different history"
    assert state1["evals"] == state2["evals"], (
        f"evals differ: {state1['evals']} vs {state2['evals']}"
    )

    # Trace must have been populated
    assert "sweeps" in trace2 and len(trace2["sweeps"]) > 0, (
        "trace dict was not populated with sweep records"
    )
    # Spot-check trace record structure
    for rec in trace2["sweeps"]:
        for key in ("best_fitness", "proposals", "cache_hits",
                    "accepted_moves", "new_evals", "diversity",
                    "brightest_changed"):
            assert key in rec, f"trace sweep record missing key {key!r}"


def test_trace_accounting_consistent():
    """Trace counters must agree with state/history (exact bookkeeping,
    so these are equalities, not tuned thresholds)."""
    Xtr, ytr = _synthetic(n=600, d=40, seed=19)
    k = 5
    trace: dict = {}
    fitness, state = make_fitness(Xtr, ytr, k=k, seed=19)
    _, best_fit, history = binary_firefly(
        fitness, state, d=40, k=k, budget=250, n_fireflies=10,
        seed=19, trace=trace,
    )
    sweeps = trace["sweeps"]

    # One record per sweep, best_fitness mirrors history exactly
    assert [s["best_fitness"] for s in sweeps] == history
    # Every new eval is attributed either to init or to exactly one sweep
    assert trace["init_evals"] + sum(s["new_evals"] for s in sweeps) == state["evals"]
    # Init best is before any move, so it can only be >= the final best
    assert trace["init_best"] >= best_fit
    # Per-sweep counters are internally consistent
    for s in sweeps:
        assert 0 <= s["cache_hits"] <= s["proposals"]
        assert 0 <= s["accepted_moves"] <= s["proposals"]
        assert s["new_evals"] <= s["proposals"]

