"""Formatting for statistics shown in the UI: labels, p-values, points, "never"."""

from __future__ import annotations

import math

from app.lib.theme import LABEL_STYLE


def label_text(label: str) -> str:
    """Pre-registered label as symbol plus word, e.g. "✓ WIN"."""
    style = LABEL_STYLE.get(label)
    return f"{style['symbol']} {label}" if style else label


def format_p(p: float | None) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def format_points(diff_pts: float) -> str:
    """Accuracy difference already in percentage points."""
    return f"{diff_pts:+.2f} pts"


def format_break_even(value: float | str | None) -> str:
    """Break-even sample count, or "never" when there is no saving."""
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "never"
    if not math.isfinite(v) or v <= 0:
        return "never"
    return f"{v:,.0f}"
