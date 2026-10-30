"""Takeaway sentences built from dataframes (ANALYSIS.md rule 8).

Every number in a takeaway is formatted from data here; nothing is typed.
Accuracy differences are in percentage points, p-values come with their
Holm-adjusted value and n.
"""

from __future__ import annotations

import math

import pandas as pd

from experiments.analysis import common as C


def pts(diff_fraction: float, digits: int = 1) -> str:
    """0.0123 -> '+1.2 pts' (accuracy difference as percentage points)."""
    return f"{diff_fraction * 100:+.{digits}f} pts"


def acc(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def pval(p: float) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "p = n/a"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def test_phrase(mean_diff: float, p_raw: float, p_holm: float, n: int) -> str:
    """'+1.2 pts (p = 0.012, Holm p = 0.072, n = 10)'."""
    return f"{pts(mean_diff)} ({pval(p_raw)}, Holm {pval(p_holm)}, n = {n})"


def best_mean(df: pd.DataFrame, value: str = "mean", by: str = "method") -> tuple[str, float]:
    """(display name, value) of the highest mean in df."""
    r = df.sort_values(value, ascending=False).iloc[0]
    return C.DISPLAY.get(r[by], r[by]), float(r[value])


def integrity_sentence(checks) -> str:
    """One line for the A0 output, from the check results."""
    real = [c for c in checks if not c.info]
    failed = [c.name for c in real if not c.ok]
    if not failed:
        return f"All {len(real)} integrity checks passed."
    return f"{len(failed)} of {len(real)} integrity checks failed: {', '.join(failed)}."
