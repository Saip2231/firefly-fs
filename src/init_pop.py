"""Shared initial populations for the metaheuristics (Day 4, Stage 1).

init_population(kind, n, d, k, rng, filter_scores=None, frac_seeded=0.5)
    kind="random"
        n uniform random k-masks, drawn with exactly the same RNG calls as the
        original binary_firefly initialisation (one rng.choice(d, size=k,
        replace=False) per member).  So init_population("random", n, d, k,
        np.random.default_rng(seed)) is the swarm binary_firefly(seed=seed,
        init_masks=None) starts from.  Duplicates are NOT removed here, to
        keep that equivalence.
    kind="seeded"  (ablation, not the method under test)
        member 0 is the exact top-k mask by filter score.  Members
        1..n_seeded-1 are score-biased: k features sampled without
        replacement with p proportional to (1 - rank/d)**3 (rank 0 = highest
        score).  n_seeded = round_half_up(frac_seeded * n), at least 1 (member
        0).  The remaining members are uniform random k-masks.  Every member is
        redrawn up to MAX_TRIES times to be distinct from earlier members; if
        no new mask turns up (e.g. n > C(d, k)) a duplicate is kept.

filter_scores must come from training rows the fitness does NOT validate on
(make_fitness exposes inner-train ANOVA F scores as state["filter_scores"]).

top_k_mask(scores, k)
    The exact top-k mask; ties are broken towards the lower column index
    (stable sort), so it is deterministic.

check_init_masks(masks, n, d, k)
    Validate a caller-supplied population (shape, exactly k per row) and
    return a boolean copy, so the caller's array is never mutated.
"""

from __future__ import annotations

import numpy as np

MAX_TRIES = 100     # redraws per seeded member before a duplicate is accepted


def top_k_mask(scores: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask of the k highest scores (ties -> lower column index)."""
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="stable")
    mask = np.zeros(scores.shape[0], dtype=bool)
    mask[order[:k]] = True
    return mask


def check_init_masks(masks, n: int, d: int, k: int) -> np.ndarray:
    """Return a bool copy of *masks* after checking shape (n, d) and row sums == k."""
    m = np.array(masks, dtype=bool, copy=True)
    if m.shape != (n, d):
        raise ValueError(f"init_masks shape {m.shape} != ({n}, {d})")
    sums = m.sum(axis=1)
    if not np.all(sums == k):
        bad = np.flatnonzero(sums != k).tolist()
        raise ValueError(f"init_masks rows {bad} do not have exactly k={k} True bits")
    return m


def init_population(
    kind: str,
    n: int,
    d: int,
    k: int,
    rng: np.random.Generator,
    filter_scores: np.ndarray | None = None,
    frac_seeded: float = 0.5,
) -> np.ndarray:
    """Return an (n, d) bool array of k-masks (see module docstring)."""
    if n < 1:
        raise ValueError(f"need n >= 1, got {n}")
    if not 1 <= k < d:
        raise ValueError(f"need 1 <= k < d, got k={k}, d={d}")

    if kind == "random":
        # Same calls, same order as binary_firefly's original initialisation.
        positions = np.zeros((n, d), dtype=bool)
        for i in range(n):
            chosen = rng.choice(d, size=k, replace=False)
            positions[i, chosen] = True
        return positions

    if kind != "seeded":
        raise ValueError(f"unknown init kind {kind!r} (use 'random' or 'seeded')")

    if filter_scores is None:
        raise ValueError("kind='seeded' needs filter_scores")
    scores = np.asarray(filter_scores, dtype=float)
    if scores.shape != (d,) or not np.all(np.isfinite(scores)):
        raise ValueError(f"filter_scores must be finite with shape ({d},)")
    if not 0.0 <= frac_seeded <= 1.0:
        raise ValueError(f"frac_seeded must be in [0, 1], got {frac_seeded}")

    n_seeded = min(n, max(1, int(np.floor(frac_seeded * n + 0.5))))

    order = np.argsort(-scores, kind="stable")      # order[r] = column of rank r
    rank = np.empty(d, dtype=float)
    rank[order] = np.arange(d)
    p = (1.0 - rank / d) ** 3                       # > 0 for every rank <= d-1
    p /= p.sum()

    positions = np.zeros((n, d), dtype=bool)
    positions[0] = top_k_mask(scores, k)
    seen = {positions[0].tobytes()}

    for i in range(1, n):
        biased = i < n_seeded
        for _ in range(MAX_TRIES):
            m = np.zeros(d, dtype=bool)
            if biased:
                m[rng.choice(d, size=k, replace=False, p=p)] = True
            else:
                m[rng.choice(d, size=k, replace=False)] = True
            if m.tobytes() not in seen:
                break
        positions[i] = m
        seen.add(m.tobytes())

    return positions
