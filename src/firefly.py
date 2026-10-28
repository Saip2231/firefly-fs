"""Binary Firefly Algorithm for feature selection.

Design notes
------------
make_fitness
    Builds a closure over a stratified *inner* validation split carved
    entirely from the TRAINING data (outer test set is never touched).
    Subsampling to max_rows keeps each evaluation fast.  Results are cached
    by mask.tobytes() so repeated proposals cost zero extra classifier fits.
    Invalid masks (sum != k) are penalised without training and always lose
    to any valid mask (penalty > 1.0 >= worst valid fitness).

    The w term (sparsity weight) is retained in the fitness formula for
    completeness, but because repair forces every valid mask to have exactly
    k features the n_selected/d component is a constant across all valid
    candidates.  It cannot break ties during the search; it only shifts all
    valid fitness values by the same constant (w * k/d).  Do NOT claim it
    differentiates masks in any report.

    Validation noise: the inner validation set contains ~500 samples
    (25% of 2,000 subsampled rows).  Fitness differences smaller than
    ~1-2 percentage points are within noise.  Use the held-out outer test
    set as the sole source of final accuracy judgement.

repair
    Ensures exactly k True values by randomly dropping extras or adding
    from the False indices.  Operates on a copy to avoid aliasing bugs.

binary_firefly  (v2 — three targeted fixes)
    FIX 1 — Swap-based mutation scaled to k (not d).
        The old per-bit flip probability alpha=0.02 caused ~0.02*d expected
        flips.  On Digits (d=64) that is ~1 flip, fine.  On CIFAR (d=512)
        it is ~10 flips for k=16-128, effectively a random restart after
        repair.  Replacement: apply exactly n_swaps random swaps, where each
        swap removes one selected feature and adds one unselected one.
        n_swaps is drawn from Uniform{1, max_swaps} with max_swaps=max(1,
        round(alpha*k)).  This keeps mutation proportional to the subset
        size, not the ambient dimension.

        The optional long-jump mutation (levy=True) uses 3x the normal swap
        count on 10% of moves.  This is a heavy-tailed discrete step, NOT
        a true Levy flight (which requires a power-law step-size distribution).
        Call it "long-jump" or "heavy-tailed mutation" in any report.

    FIX 2 — Distance normalised by 2k (max possible Hamming between
        k-subsets).
        Two k-subsets can differ by at most 2k bits (all k bits of one are
        absent from the other).  Dividing raw Hamming by d makes r <= 2k/d,
        which is tiny on CIFAR (2*16/512 = 0.0625) and gamma has no effect.
        Dividing by 2k instead gives r in [0,1] for any k and d, and gamma
        becomes a scale-invariant absorption coefficient.

    FIX 3 — Lower default beta0=0.4.
        beta0=0.9 copies ~90% of differing bits, collapsing diversity in
        a few sweeps.  beta0=0.4 preserves more structural variety while
        still providing meaningful attraction.

    Budget enforcement: the budget guard is placed BEFORE each fitness call
    so that state["evals"] can never exceed the requested budget.

    Diversity logging: state["diversity"] is a list of mean pairwise
    Hamming distances (normalised by 2k) recorded after each sweep so that
    population collapse is visible.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


# ---------------------------------------------------------------------------
# Fitness factory
# ---------------------------------------------------------------------------

def make_fitness(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    k: int,
    seed: int,
    max_rows: int = 2000,
    w: float = 0.01,
):
    """Return a fitness callable and a shared state dict.

    Parameters
    ----------
    Xtr, ytr    : Full (outer) training features and labels.
    k           : Target number of features.
    seed        : Random seed for subsampling and the inner split.
    max_rows    : Maximum training rows used per fitness evaluation.
    w           : Weight on the sparsity term w*(n_selected/d).  Because
                  repair enforces exactly k features for all valid masks,
                  this term is a constant offset and does NOT differentiate
                  candidates during search.  It is retained for formula
                  completeness only.

    Returns
    -------
    fitness : callable
        fitness(mask) -> float for valid masks (validation error + constant),
        or > 1.0 for infeasible masks (sum != k).  All valid mask scores
        are below 1.0 because error < 1 and w*k/d < w < 1.
    state : dict
        {"evals": int}  -- count of NEW (uncached) classifier trainings.
        {"cache": dict} -- mask.tobytes() -> fitness for every valid mask
                           evaluated so far (len(cache) == evals).
    """
    n, d = Xtr.shape

    # -- Subsample to max_rows (stratified) ----------------------------------
    if n > max_rows:
        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=max_rows, random_state=seed
        )
        idx, _ = next(sss.split(Xtr, ytr))
        X_sub, y_sub = Xtr[idx], ytr[idx]
    else:
        X_sub, y_sub = Xtr, ytr

    # -- Inner stratified 75/25 split (from subsampled training data only) ---
    inner_split = StratifiedShuffleSplit(
        n_splits=1, test_size=0.25, random_state=seed
    )
    inner_tr_idx, inner_val_idx = next(inner_split.split(X_sub, y_sub))

    X_in_tr  = X_sub[inner_tr_idx]
    y_in_tr  = y_sub[inner_tr_idx]
    X_in_val = X_sub[inner_val_idx]
    y_in_val = y_sub[inner_val_idx]

    _cache: dict[bytes, float] = {}
    # state["cache"] is the SAME dict object as _cache (read-only use by
    # rerank); exposing it does not change any fitness value or count.
    state: dict = {"evals": 0, "cache": _cache}

    def fitness(mask: np.ndarray) -> float:
        """Evaluate a boolean feature mask.

        Parameters
        ----------
        mask : np.ndarray of bool, shape (d,)

        Returns
        -------
        float
            (1-w)*validation_error + w*(k/d)  for valid masks (w*k/d is a
            constant because repair enforces sum==k; it does not rank masks).
            1.0 + abs(n_selected - k) / d     for infeasible masks (sum!=k).
        """
        n_selected = int(mask.sum())

        # -- Infeasibility penalty (no training) -----------------------------
        if n_selected != k:
            return 1.0 + abs(n_selected - k) / d

        # -- Cache lookup -----------------------------------------------------
        key = mask.tobytes()
        if key in _cache:
            return _cache[key]

        # -- Train LogisticRegression on inner split --------------------------
        clf = LogisticRegression(max_iter=100, random_state=seed)
        clf.fit(X_in_tr[:, mask], y_in_tr)

        val_acc   = clf.score(X_in_val[:, mask], y_in_val)
        val_error = 1.0 - val_acc

        # w*(n_selected/d) is a constant (n_selected==k after repair) and
        # does not differentiate valid masks; included for formula consistency.
        fit_val         = (1.0 - w) * val_error + w * (n_selected / d)
        _cache[key]     = fit_val
        state["evals"] += 1
        return fit_val

    return fitness, state


# ---------------------------------------------------------------------------
# Repair operator
# ---------------------------------------------------------------------------

def repair(mask: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Return a copy of *mask* with exactly *k* True values.

    Parameters
    ----------
    mask : np.ndarray of bool, shape (d,)
    k    : Desired number of selected features.
    rng  : numpy Generator used for random choices.

    Returns
    -------
    np.ndarray of bool, shape (d,)  -- a new array, never the same object.
    """
    m = mask.copy()
    on_idx  = np.where(m)[0]
    off_idx = np.where(~m)[0]
    n_on    = len(on_idx)

    if n_on > k:
        # Remove random extras
        drop = rng.choice(on_idx, size=n_on - k, replace=False)
        m[drop] = False
    elif n_on < k:
        # Add from random non-selected indices
        add = rng.choice(off_idx, size=k - n_on, replace=False)
        m[add] = True

    return m


# ---------------------------------------------------------------------------
# Binary Firefly Algorithm
# ---------------------------------------------------------------------------

def binary_firefly(
    fitness,
    state: dict,
    d: int,
    k: int,
    budget: int      = 800,
    n_fireflies: int = 20,
    beta0: float     = 0.4,
    gamma: float     = 2.0,
    alpha: float     = 0.02,
    levy: bool       = False,
    seed: int        = 0,
    trace: dict | None = None,
) -> tuple[np.ndarray, float, list[float]]:
    """Binary Firefly Algorithm for feature-mask optimisation (v2).

    Parameters
    ----------
    fitness     : callable  mask -> float (lower is better).
    state       : dict with key "evals" (shared with fitness).
    d           : Total number of features.
    k           : Number of features to select.
    budget      : Maximum number of NEW fitness evaluations.
    n_fireflies : Swarm size.
    beta0       : Base attraction strength [0, 1].  Default 0.4 (was 0.9);
                  lower value preserves swarm diversity.
    gamma       : Light absorption coefficient.  Acts on r = Hamming/2k,
                  which is now in [0, 1] for any k and d.
    alpha       : Controls mutation magnitude.  Expected swaps per move =
                  max(1, round(alpha * k)).  Scale-invariant w.r.t. d.
    levy        : If True, 10% of moves use 3x the normal swap count.
    seed        : RNG seed for full reproducibility.

    Returns
    -------
    best_mask : np.ndarray of bool, shape (d,)
    best_fit  : float
    history   : list[float] -- best fitness after each sweep (non-increasing).

    trace : dict or None
        When a dict is passed (e.g. ``trace={}``), the keys ``"init_best"``
        (best fitness of the initial swarm), ``"init_evals"`` (new evals used
        to score it) and ``"sweeps"`` are added; a per-sweep record is
        appended to ``"sweeps"`` after each sweep.
        Each record contains:
          best_fitness    -- best fitness seen so far after the sweep
          proposals       -- number of (i, j) pairs where j brighter than i
          cache_hits      -- proposals whose fitness() was a cache hit
          accepted_moves  -- proposals whose candidate was accepted by firefly i
          new_evals       -- new (uncached) classifier fits in this sweep
          diversity       -- mean pairwise Hamming/2k after the sweep
          brightest_changed -- True if the global best mask changed this sweep
        When ``trace=None`` (the default) behaviour is byte-for-byte unchanged.

    Side-effects on *state*
    -----------------------
    state["evals"]     : int  -- cumulative new fitness evaluations.
    state["diversity"] : list[float] -- mean pairwise Hamming/2k per sweep.
    """
    rng = np.random.default_rng(seed)

    # -- Initialise swarm: n_fireflies random masks each with exactly k True -
    positions = np.zeros((n_fireflies, d), dtype=bool)
    for i in range(n_fireflies):
        chosen = rng.choice(d, size=k, replace=False)
        positions[i, chosen] = True

    # Evaluate initial positions
    fits = np.full(n_fireflies, np.inf)
    for i in range(n_fireflies):
        if state["evals"] >= budget:
            break
        fits[i] = fitness(positions[i])

    best_idx  = int(np.argmin(fits))
    best_mask = positions[best_idx].copy()
    best_fit  = float(fits[best_idx])

    history: list[float] = []
    state.setdefault("diversity", [])
    max_sweeps = budget          # hard cap: cannot spin more times than budget
    alpha_cur  = alpha
    max_hamming = 2 * k          # FIX 2: max possible Hamming between two k-subsets

    # Initialise trace container (no-op when trace is None).
    # init_best / init_evals describe the swarm BEFORE the first sweep
    # (history[0] is already after one sweep of moves).
    if trace is not None:
        trace.setdefault("sweeps", [])
        trace["init_best"]  = best_fit
        trace["init_evals"] = state["evals"]

    for _sweep in range(max_sweeps):
        if state["evals"] >= budget:
            break

        # Per-sweep accounting (computed regardless of trace to keep hot path
        # identical; overhead is a few integer increments per proposal).
        _sw_proposals = 0
        _sw_cache_hits = 0
        _sw_accepted = 0
        _sw_new_evals_start = state["evals"]
        _sw_brightest_changed = False

        moved_any = False

        for i in range(n_fireflies):
            for j in range(n_fireflies):
                if state["evals"] >= budget:
                    break
                if i == j:
                    continue
                # Firefly j must be brighter (lower fitness) than i
                if fits[j] >= fits[i]:
                    continue

                _sw_proposals += 1
                moved_any = True

                # -- Attraction move -----------------------------------------
                # FIX 2: normalise Hamming by 2k so r is in [0, 1] for any
                # k-subset pair, making gamma scale-invariant across datasets.
                raw_hamming = float(np.sum(positions[i] != positions[j]))
                r    = raw_hamming / max_hamming
                beta = beta0 * np.exp(-gamma * r * r)

                candidate = positions[i].copy()

                # For each bit where i and j differ, copy j's bit with
                # probability beta (vectorised over the differing positions)
                diff_idx = np.where(positions[i] != positions[j])[0]
                if len(diff_idx) > 0:
                    probs     = rng.random(len(diff_idx))
                    copy_bits = diff_idx[probs < beta]
                    candidate[copy_bits] = positions[j][copy_bits]

                # -- FIX 1: Swap-based mutation scaled to k ------------------
                # n_swaps ~ Uniform{1, max_swaps}, max_swaps proportional to k.
                # This keeps mutation magnitude constant relative to the
                # subset size, independent of ambient dimension d.
                max_swaps = max(1, round(alpha_cur * k))
                if levy and rng.random() < 0.1:
                    max_swaps = max(1, 3 * max_swaps)   # long-jump: 3x swaps
                n_swaps = int(rng.integers(1, max_swaps + 1))

                on_c   = np.where(candidate)[0]
                off_c  = np.where(~candidate)[0]
                # Clamp to available bits (edge case: very small d or k)
                n_swaps = min(n_swaps, len(on_c), len(off_c))
                if n_swaps > 0:
                    drop = rng.choice(on_c,  size=n_swaps, replace=False)
                    add  = rng.choice(off_c, size=n_swaps, replace=False)
                    candidate[drop] = False
                    candidate[add]  = True
                # After swaps the mask already has exactly k True values,
                # so repair is only needed to fix the attraction step.

                # -- Repair to exactly k (attraction may have broken count) --
                candidate = repair(candidate, k, rng)

                # -- Budget guard BEFORE the fitness call --------------------
                # Placed here so state["evals"] can never exceed budget.
                if state["evals"] >= budget:
                    break

                # -- Evaluate candidate (track cache hits via evals delta) --
                _evals_before = state["evals"]
                new_fit = fitness(candidate)
                if state["evals"] == _evals_before:
                    _sw_cache_hits += 1

                # -- Greedy acceptance ---------------------------------------
                if new_fit <= fits[i]:
                    positions[i] = candidate
                    fits[i]      = new_fit
                    _sw_accepted += 1

                    if new_fit < best_fit:
                        best_fit  = new_fit
                        best_mask = candidate.copy()
                        _sw_brightest_changed = True

            if state["evals"] >= budget:
                break

        # Record best fitness after this sweep.
        # best_fit is only ever updated downward, so history is non-increasing.
        history.append(best_fit)

        # -- Diversity: mean pairwise Hamming distance (normalised by 2k) ----
        if n_fireflies > 1:
            total_hamming = 0.0
            n_pairs = 0
            for pi in range(n_fireflies):
                for pj in range(pi + 1, n_fireflies):
                    total_hamming += float(np.sum(positions[pi] != positions[pj]))
                    n_pairs += 1
            mean_div = (total_hamming / n_pairs) / max_hamming
        else:
            mean_div = 0.0
        state["diversity"].append(mean_div)

        # -- Append trace record (no-op when trace is None) ------------------
        if trace is not None:
            trace["sweeps"].append({
                "best_fitness":      best_fit,
                "proposals":         _sw_proposals,
                "cache_hits":        _sw_cache_hits,
                "accepted_moves":    _sw_accepted,
                "new_evals":         state["evals"] - _sw_new_evals_start,
                "diversity":         mean_div,
                "brightest_changed": _sw_brightest_changed,
            })

        # Geometric alpha decay: reduce exploration over time
        alpha_cur *= 0.98

        # Early termination: if no firefly moved, the swarm has converged
        if not moved_any:
            break

    return best_mask, best_fit, history


# ---------------------------------------------------------------------------
# Re-ranking against validation overfitting (additive; search unchanged)
# ---------------------------------------------------------------------------

def rerank(
    state: dict,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    k: int,
    top_m: int = 10,
    n_splits: int = 5,
    seed: int = 0,
):
    """Re-score the top_m cached masks with stratified k-fold CV on Xtr.

    Candidates are the distinct cached masks with exactly k features and the
    lowest cached (single-split validation) fitness; ties are broken by the
    mask bytes so the order is deterministic.  Each candidate is scored by
    mean stratified n_splits-fold CV accuracy on ALL rows of (Xtr, ytr) with
    the same LogisticRegression settings as make_fitness, using the same
    folds for every candidate.  Highest CV accuracy wins; ties go to the
    candidate with the lower cached fitness.

    Only training data is passed in; the search fitness is never called, so
    state["evals"] (the search budget) is unchanged.

    Returns
    -------
    best_mask   : np.ndarray of bool
    scores      : list of dicts {mask, cached_fitness, cv_acc}, in candidate order
    rerank_fits : int -- classifier fits used = n_candidates * n_splits
                  (= top_m * n_splits when the cache holds >= top_m masks)
    """
    cands = sorted(
        (fit, key) for key, fit in state["cache"].items()
        if int(np.frombuffer(key, dtype=bool).sum()) == k
    )[:top_m]
    if not cands:
        raise ValueError("rerank: no cached mask with exactly k features")

    folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True,
                                 random_state=seed).split(Xtr, ytr))
    scores = []
    for fit, key in cands:
        mask = np.frombuffer(key, dtype=bool).copy()
        Xm = Xtr[:, mask]
        accs = []
        for tr, va in folds:
            clf = LogisticRegression(max_iter=100, random_state=seed)
            clf.fit(Xm[tr], ytr[tr])
            accs.append(clf.score(Xm[va], ytr[va]))
        scores.append({"mask": mask, "cached_fitness": float(fit),
                       "cv_acc": float(np.mean(accs))})

    # max() returns the first maximal element -> lower cached fitness on ties
    best = max(range(len(scores)), key=lambda i: scores[i]["cv_acc"])
    return scores[best]["mask"].copy(), scores, len(cands) * n_splits


def run_firefly(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    k: int,
    budget: int = 800,
    seed: int = 0,
    max_rows: int = 2000,
    rerank_top_m: int = 10,
    n_splits: int = 5,
    **ff_kwargs,
) -> dict:
    """Search with binary_firefly, then optionally re-rank (training data only).

    ff_kwargs are passed unchanged to binary_firefly (n_fireflies, beta0,
    gamma, alpha, levy, trace).  rerank_top_m=0 skips re-ranking and returns
    the raw best as the re-ranked mask (rerank_fits=0).

    Returns dict with raw_mask, raw_fit, history, reranked_mask,
    rerank_scores, search_evals, rerank_fits.
    """
    d = Xtr.shape[1]
    fitness, state = make_fitness(Xtr, ytr, k=k, seed=seed, max_rows=max_rows)
    raw_mask, raw_fit, history = binary_firefly(
        fitness, state, d=d, k=k, budget=budget, seed=seed, **ff_kwargs)
    search_evals = state["evals"]

    if rerank_top_m > 0:
        rr_mask, scores, rr_fits = rerank(state, Xtr, ytr, k, top_m=rerank_top_m,
                                          n_splits=n_splits, seed=seed)
    else:
        rr_mask, scores, rr_fits = raw_mask.copy(), [], 0

    return {
        "raw_mask":      raw_mask,
        "raw_fit":       raw_fit,
        "history":       history,
        "reranked_mask": rr_mask,
        "rerank_scores": scores,
        "search_evals":  search_evals,
        "rerank_fits":   rr_fits,
    }
