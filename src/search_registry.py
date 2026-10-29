"""Thin dispatcher over the metaheuristics (Day 4, Stage 3).

run_search(method, fitness, state, d, k, budget, seed, hp=None) -> dict

Methods
    firefly         binary_firefly, random initial swarm
    firefly_levy    binary_firefly(levy=True)  ("long-jump mutation", not a
                    Levy flight)
    pso             binary_pso (random-key top-k decoding), random init
    firefly_seeded  binary_firefly from init_population("seeded")   (ablation)
    pso_seeded      binary_pso from init_population("seeded")       (ablation)

Shared initial populations (fairness rule 5)
    Plain methods pass init_masks=None.  Both binary_firefly and binary_pso
    then build init_population("random", n, d, k, default_rng(seed)) with the
    same RNG calls (tests/test_init_pop.py, tests/test_pso.py), so firefly,
    firefly_levy and pso start from the SAME masks.  None is passed instead of
    the explicit masks because an explicit init_masks skips the init draws and
    would shift Firefly's move stream away from the recorded Day-3 behaviour.
    Seeded methods build init_population("seeded", n, d, k, default_rng(seed),
    state["filter_scores"], frac_seeded) here and pass it as init_masks, so
    firefly_seeded and pso_seeded also start from the same masks.
    If the population sizes differ (n_fireflies != n_particles), plain
    methods still share the smaller population (it equals the first rows of
    the larger one).  Seeded methods share fewer rows: the number of
    score-biased members scales with n, so the populations agree only up to
    the smaller seeded count (e.g. 5 rows for n = 10 vs 20 at frac 0.5).

Budget hygiene
    run_search refuses a state that has already been used (state["evals"] !=
    0 or a "diversity" log present): a reused make_fitness state would carry
    its cache and eval count into the next run and silently shrink the
    budget.  Build a fresh make_fitness per run.

hp
    Hyperparameters for the method; unknown keys raise ValueError.
    firefly*: n_fireflies, beta0, gamma, alpha;  pso*: n_particles, w_start,
    w_end, c1, c2, vmax;  seeded methods also: frac_seeded (default 0.5).

Returns dict: method, best_mask, best_fit, history, diversity (per sweep /
iteration), evals_used, init_best (best fitness of the initial population).
"""

from __future__ import annotations

import numpy as np

from src.firefly import binary_firefly
from src.init_pop import init_population
from src.pso import binary_pso

METHODS = ("firefly", "firefly_levy", "pso", "firefly_seeded", "pso_seeded")
FIREFLY_HP = {"n_fireflies", "beta0", "gamma", "alpha"}
PSO_HP = {"n_particles", "w_start", "w_end", "c1", "c2", "vmax"}
DEFAULT_N = 20          # binary_firefly / binary_pso default population size


def run_search(method, fitness, state, d, k, budget, seed, hp=None) -> dict:
    """Run one search method on a fresh (fitness, state) pair; see module doc."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; use one of {METHODS}")
    if state.get("evals", 0) != 0 or "diversity" in state:
        raise ValueError("state was already used by a search; "
                         "build a fresh make_fitness per run")

    hp = dict(hp or {})
    seeded = method.endswith("_seeded")
    is_pso = method.startswith("pso")
    frac_seeded = hp.pop("frac_seeded", 0.5) if seeded else None
    allowed = PSO_HP if is_pso else FIREFLY_HP
    unknown = set(hp) - allowed
    if unknown:
        raise ValueError(f"unknown hyperparameter(s) {sorted(unknown)} for {method}")

    init_masks = None
    if seeded:
        n = hp.get("n_particles" if is_pso else "n_fireflies", DEFAULT_N)
        init_masks = init_population("seeded", n, d, k, np.random.default_rng(seed),
                                     filter_scores=state["filter_scores"],
                                     frac_seeded=frac_seeded)

    trace: dict = {}
    if is_pso:
        mask, fit, hist = binary_pso(fitness, state, d=d, k=k, budget=budget,
                                     init_masks=init_masks, seed=seed,
                                     trace=trace, **hp)
    else:
        mask, fit, hist = binary_firefly(fitness, state, d=d, k=k, budget=budget,
                                         levy=(method == "firefly_levy"),
                                         seed=seed, trace=trace,
                                         init_masks=init_masks, **hp)

    return {
        "method":     method,
        "best_mask":  mask,
        "best_fit":   float(fit),
        "history":    list(hist),
        "diversity":  list(state.get("diversity", [])),
        "evals_used": int(state["evals"]),
        "init_best":  float(trace["init_best"]),
    }
