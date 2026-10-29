"""Shared helpers for experiment scripts (no algorithm logic lives here)."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def git_hash() -> str:
    """Short HEAD hash, with "+dirty" if tracked files have uncommitted
    changes; "unknown" when git is unavailable."""
    import subprocess  # noqa: PLC0415

    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=ROOT, capture_output=True, text=True,
                               check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return h + ("+dirty" if dirty else "")


def warn_once() -> None:
    """Print each distinct warning category once, then stay silent."""
    seen: set[str] = set()
    original = warnings.showwarning

    def _show(message, category, filename, lineno, file=None, line=None):
        key = category.__name__
        if key not in seen:
            seen.add(key)
            original(f"{message} (further {key}s suppressed)",
                     category, filename, lineno, file, line)

    warnings.simplefilter("always")   # route every warning through _show
    warnings.showwarning = _show


class EvalRecorder:
    """Wraps a fitness callable and records every mask that cost a NEW fit.

    Returns exactly what the wrapped fitness returns, so passing it to
    binary_firefly does not change the search.
    """

    def __init__(self, fitness, state):
        self.fitness = fitness
        self.state = state
        self.masks: list[np.ndarray] = []
        self.fits: list[float] = []          # fitness of each recorded mask
        self.calls = 0
        self.cache_hits = 0

    def __call__(self, mask: np.ndarray) -> float:
        before = self.state["evals"]
        f = self.fitness(mask)
        self.calls += 1
        if self.state["evals"] > before:
            self.masks.append(mask.copy())
            self.fits.append(float(f))
        else:
            self.cache_hits += 1
        return f

    def informative_coverage(self, n_informative: int) -> tuple[int, float]:
        """(distinct informative cols ever evaluated, mean informative per eval)."""
        if not self.masks:
            return 0, 0.0
        M = np.asarray(self.masks)[:, :n_informative]
        return int(M.any(axis=0).sum()), float(M.sum(axis=1).mean())


def paired_wilcoxon(a, b) -> dict:
    """Two-sided paired Wilcoxon signed-rank test of a vs b.

    Zero differences are dropped (scipy default 'wilcox'); n_nonzero says how
    many pairs remained.  If every pair ties, p is NaN (test undefined).
    """
    from scipy.stats import wilcoxon  # noqa: PLC0415

    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    n_nonzero = int(np.count_nonzero(diff))
    out = dict(n=len(diff), n_nonzero=n_nonzero,
               mean_diff=float(diff.mean()), stat=np.nan, p=np.nan)
    if n_nonzero > 0:
        res = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        out.update(stat=float(res.statistic), p=float(res.pvalue))
    return out


def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, in the input order.

    NaN p-values (undefined tests, e.g. all pairs tied) stay NaN and do not
    count towards the family size m.  Adjusted values are capped at 1 and
    made monotone in the sorted order.
    """
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    m = len(ok)
    running = 0.0
    for rank, i in enumerate(ok[np.argsort(p[ok], kind="stable")]):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def direction(mean_diff: float) -> str:
    """'higher' / 'lower' / 'equal' for a paired mean difference."""
    return "higher" if mean_diff > 0 else "lower" if mean_diff < 0 else "equal"


def random_search(fitness, state, d: int, k: int, budget: int, seed: int,
                  init_n: int = 20):
    """Random valid k-masks until `budget` NEW evals are used.

    Picks by the same validation fitness as Firefly.  Returns a dict with
    best_mask, best_fit, best_init (best of the first `init_n` draws, the
    counterpart of Firefly's initial swarm), proposals, improvements,
    evals_used, stopped ("budget", "max_proposals" or "cache_hits").

    Loop guards (F4): stop after 100*budget proposals, or after 5*budget
    consecutive cache hits (e.g. budget > C(d, k) distinct masks).  Neither
    guard can trigger before the budget is spent unless new masks become
    rare, so normal runs are unchanged.
    """
    if budget < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")
    rng = np.random.default_rng(seed)
    best_fit, best_mask, best_init = np.inf, None, np.inf
    proposals = improvements = consecutive_hits = 0
    stopped = "budget"
    while state["evals"] < budget:
        if proposals >= 100 * budget:
            stopped = "max_proposals"
            break
        if consecutive_hits >= 5 * budget:
            stopped = "cache_hits"
            break
        mask = np.zeros(d, dtype=bool)
        mask[rng.choice(d, size=k, replace=False)] = True
        before = state["evals"]
        f = fitness(mask)
        consecutive_hits = consecutive_hits + 1 if state["evals"] == before else 0
        proposals += 1
        if f < best_fit:
            best_fit, best_mask = f, mask.copy()
            improvements += 1
        if proposals == init_n:
            best_init = best_fit
    if proposals < init_n:
        best_init = best_fit
    return dict(best_mask=best_mask, best_fit=float(best_fit),
                best_init=float(best_init), proposals=proposals,
                improvements=improvements, evals_used=int(state["evals"]),
                stopped=stopped)
