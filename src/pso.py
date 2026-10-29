"""PSO with random-key (top-k) decoding for feature-mask optimisation.

Not classic sigmoid binary PSO: each particle moves in R^d and its mask is
the top-k entries of its position, so every candidate has exactly k True
bits by construction (no repair needed).

binary_pso(fitness, state, d, k, ...) has the same fitness/state interface
and return value as src.firefly.binary_firefly.

Design notes
------------
Initial population
    init_masks=None -> init_population("random", n_particles, d, k, rng) on
    rng = default_rng(seed): the same RNG calls binary_firefly(seed=seed)
    makes, so both methods start from the SAME masks for the same seed.
    Otherwise a validated copy of init_masks is used.
Initial position and velocity
    x0 = mask + 0.1 * N(0, 1), with the noise clipped to [-0.45, 0.45] so that
    selected entries (>= 0.55) always rank above unselected ones (<= 0.45) and
    decoding x0 gives back exactly the initial mask (unclipped N(0, 1) could,
    with tiny probability, swap a pair).  The clip only touches draws beyond
    4.5 sigma.  v0 = 0.  The initial masks themselves are what is evaluated.
Update (asynchronous: gbest can change inside an iteration)
    v = w*v + c1*r1*(pbest_x - x) + c2*r2*(gbest_x - x),  r1, r2 ~ U(0,1)^d
    v clipped to [-vmax, vmax];  x += v;  mask = top-k(x) via argpartition.
    pbest and gbest are updated when the new fitness is <= the stored one.
Inertia
    w decays linearly from w_start (iteration 0) to w_end (iteration T-1),
    T = max(1, budget // n_particles), and stays at w_end afterwards.
Budget and termination
    Budget = NEW fitness evaluations (cache hits are free).  The guard is
    checked BEFORE every fitness call, so state["evals"] <= budget.  The loop
    is capped at 10*T iterations, so it ends even when every decoded mask is
    already cached (converged swarm).
Stall swap (stall_swap=True, approved Day 4 Stage 2 fix)
    Plain gbest PSO collapses onto the gbest mask within a few iterations;
    every decoded mask is then a cache hit and the budget goes unspent
    (first sanity run: 30-80% unspent, results/sanity_pso_v0_noreset.csv).
    A velocity redraw from U(-vmax, vmax) was tried first and did NOT fix it
    (2/40 runs spent the budget): kicks of <= w*vmax are pulled straight
    back and never cross the top-k threshold.
    Rule now, the same as Firefly's F3 tie-stall fallback: when the PSO moves
    of an iteration produce 0 new evaluations (budget not yet spent), each
    particle swaps the keys of one random selected and one random unselected
    entry, which changes its mask by exactly one swap; the new mask is
    evaluated at once (budget guard first) with the usual <= pbest/gbest
    update.  Unlike Firefly the position always moves (PSO semantics); the
    pbest keeps the better point.  Fixed rule, not tuned.  No RNG draw is
    spent before the first stalled iteration, so runs are identical to
    stall_swap=False up to that point.
history / diversity
    history[t] = best fitness after iteration t (non-increasing).
    state["diversity"][t] = mean pairwise Hamming distance of the decoded
    masks / (2*min(k, d-k)) after iteration t (same normaliser as Firefly;
    = 2k for k <= d/2).
"""

from __future__ import annotations

import numpy as np

from src.init_pop import check_init_masks, init_population

INIT_NOISE = 0.1
INIT_NOISE_CLIP = 0.45      # keeps decode(x0) == init mask exactly


def _decode(x: np.ndarray, k: int) -> np.ndarray:
    """Top-k entries of x as a bool mask (exactly k True)."""
    mask = np.zeros(x.shape[0], dtype=bool)
    mask[np.argpartition(-x, k - 1)[:k]] = True
    return mask


def _diversity(masks: np.ndarray, k: int, d: int) -> float:
    """Mean pairwise Hamming / (2*min(k, d-k)) over rows of exact-k masks."""
    n = masks.shape[0]
    if n < 2:
        return 0.0
    M = masks.astype(np.int64)
    overlap = M @ M.T                        # |A & B|
    iu = np.triu_indices(n, 1)
    hamming = 2 * k - 2 * overlap[iu]        # |A ^ B| for two k-subsets
    return float(hamming.mean()) / (2 * min(k, d - k))


def binary_pso(
    fitness,
    state: dict,
    d: int,
    k: int,
    budget: int       = 800,
    n_particles: int  = 20,
    w_start: float    = 0.9,
    w_end: float      = 0.4,
    c1: float         = 1.5,
    c2: float         = 1.5,
    vmax: float       = 0.5,
    init_masks: np.ndarray | None = None,
    seed: int         = 0,
    trace: dict | None = None,
    stall_swap: bool  = True,
) -> tuple[np.ndarray, float, list[float]]:
    """PSO with random-key (top-k) decoding (see module docstring).

    Returns
    -------
    best_mask : np.ndarray of bool, shape (d,), exactly k True
    best_fit  : float
    history   : list[float] -- best fitness after each iteration.

    trace : dict or None
        When a dict is passed, "init_best" (best fitness of the initial
        population), "init_evals" (new evals used to score it) and "iters"
        are added.  One record per iteration: best_fitness, new_evals (all
        new evals in the iteration), move_evals (new evals from PSO moves
        only), stall_swaps (swap proposals made by the stall fallback),
        cache_hits, w, diversity.  trace=None changes nothing.
    stall_swap : one random key swap per particle after an iteration whose
        moves gave 0 new evaluations (see module docstring).  False gives
        the original Stage-2 behaviour.

    Side-effects on *state*: state["evals"] (via fitness), state["diversity"]
    (one entry per iteration), state["pso_iters"] (iterations run),
    state["pso_stall_iters"] (iterations that used the stall swap).
    """
    if budget < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")
    if not 1 <= k < d:
        raise ValueError(f"need 1 <= k < d, got k={k}, d={d}")
    if n_particles < 1:
        raise ValueError(f"need n_particles >= 1, got {n_particles}")

    rng = np.random.default_rng(seed)

    # -- Initial population (shared with Firefly for the same seed) ----------
    if init_masks is None:
        masks = init_population("random", n_particles, d, k, rng)
    else:
        masks = check_init_masks(init_masks, n_particles, d, k)

    noise = np.clip(INIT_NOISE * rng.standard_normal((n_particles, d)),
                    -INIT_NOISE_CLIP, INIT_NOISE_CLIP)
    x = masks.astype(float) + noise
    v = np.zeros((n_particles, d))

    # Evaluate the initial masks (budget guard before every call)
    fits = np.full(n_particles, np.inf)
    for i in range(n_particles):
        if state["evals"] >= budget:
            break
        fits[i] = fitness(masks[i])

    pbest_x = x.copy()
    pbest_f = fits.copy()
    g = int(np.argmin(pbest_f))
    gbest_x    = pbest_x[g].copy()
    gbest_f    = float(pbest_f[g])
    gbest_mask = masks[g].copy()

    if trace is not None:
        trace.setdefault("iters", [])
        trace["init_best"]  = gbest_f
        trace["init_evals"] = state["evals"]

    T = max(1, budget // n_particles)
    max_iter = 10 * T
    history: list[float] = []
    state.setdefault("diversity", [])
    n_iter = 0
    n_stall_iters = 0

    for t in range(max_iter):
        if state["evals"] >= budget:
            break
        frac = min(t / (T - 1), 1.0) if T > 1 else 1.0
        w = w_start + (w_end - w_start) * frac
        evals_start = state["evals"]
        cache_hits = 0

        # -- PSO moves ---------------------------------------------------------
        for i in range(n_particles):
            r1 = rng.random(d)
            r2 = rng.random(d)
            v[i] = w * v[i] + c1 * r1 * (pbest_x[i] - x[i]) + c2 * r2 * (gbest_x - x[i])
            np.clip(v[i], -vmax, vmax, out=v[i])
            x[i] += v[i]
            masks[i] = _decode(x[i], k)

            # -- Budget guard BEFORE the fitness call ------------------------
            if state["evals"] >= budget:
                break
            before = state["evals"]
            f = fitness(masks[i])
            if state["evals"] == before:
                cache_hits += 1

            if f <= pbest_f[i]:
                pbest_f[i] = f
                pbest_x[i] = x[i].copy()
            if f <= gbest_f:
                gbest_f    = float(f)
                gbest_x    = x[i].copy()
                gbest_mask = masks[i].copy()

        move_evals = state["evals"] - evals_start

        # -- Stall swap: moves gave no new eval -> one key swap per particle -
        n_swaps = 0
        if stall_swap and move_evals == 0 and state["evals"] < budget:
            n_stall_iters += 1
            for i in range(n_particles):
                if state["evals"] >= budget:
                    break
                on_i  = np.flatnonzero(masks[i])
                off_i = np.flatnonzero(~masks[i])
                a = rng.choice(on_i)
                b = rng.choice(off_i)
                # Swapping the two keys keeps the multiset of key values, so
                # the top-k set loses exactly a and gains exactly b.
                x[i, a], x[i, b] = x[i, b], x[i, a]
                masks[i, a], masks[i, b] = False, True

                n_swaps += 1
                before = state["evals"]
                f = fitness(masks[i])
                if state["evals"] == before:
                    cache_hits += 1

                if f <= pbest_f[i]:
                    pbest_f[i] = f
                    pbest_x[i] = x[i].copy()
                if f <= gbest_f:
                    gbest_f    = float(f)
                    gbest_x    = x[i].copy()
                    gbest_mask = masks[i].copy()

        n_iter += 1
        history.append(gbest_f)
        div = _diversity(masks, k, d)
        state["diversity"].append(div)

        if trace is not None:
            trace["iters"].append({
                "best_fitness": gbest_f,
                "new_evals":    state["evals"] - evals_start,
                "move_evals":   move_evals,
                "stall_swaps":  n_swaps,
                "cache_hits":   cache_hits,
                "w":            w,
                "diversity":    div,
            })

    state["pso_iters"] = state.get("pso_iters", 0) + n_iter
    state["pso_stall_iters"] = state.get("pso_stall_iters", 0) + n_stall_iters
    return gbest_mask, gbest_f, history
