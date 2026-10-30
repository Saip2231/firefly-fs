"""Consistent figure style for every analysis stage (ANALYSIS.md section 3).

One colour, marker and line style per method (Okabe-Ito colour-blind-safe
palette).  PCA is dashed and labelled "projection, not a selector"; 'all'
is drawn as a horizontal reference line.  save_fig() writes PNG (200 dpi)
and PDF.

matplotlib is imported only inside the functions that draw or save, so the
style tables can be used (and tested) without it installed.
"""

from __future__ import annotations

from pathlib import Path

from experiments.analysis import common as C

PALETTE = {                      # Okabe-Ito
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
    "grey": "#999999",
}

METHOD_STYLE = {
    "all":            dict(color=PALETTE["black"], marker="", linestyle=":"),
    "random":         dict(color=PALETTE["grey"], marker="x", linestyle="-"),
    "anova":          dict(color=PALETTE["sky"], marker="s", linestyle="-"),
    "mi":             dict(color=PALETTE["green"], marker="^", linestyle="-"),
    "l1":             dict(color=PALETTE["yellow"], marker="v", linestyle="-"),
    "rf":             dict(color=PALETTE["purple"], marker="D", linestyle="-"),
    "pca":            dict(color=PALETTE["black"], marker="o", linestyle="--"),
    "firefly":        dict(color=PALETTE["vermillion"], marker="*", linestyle="-"),
    "pso":            dict(color=PALETTE["blue"], marker="P", linestyle="-"),
    "firefly_seeded": dict(color=PALETTE["vermillion"], marker="*", linestyle="-."),
    "pso_seeded":     dict(color=PALETTE["blue"], marker="P", linestyle="-."),
    "firefly_levy":   dict(color=PALETTE["orange"], marker="*", linestyle=":"),
}
LABEL_COLOURS = {"WIN": PALETTE["green"], "LOSS": PALETTE["vermillion"], "TIE": "#DDDDDD"}
DPI = 200


def style(method: str) -> dict:
    """matplotlib kwargs (color, marker, linestyle, label) for a method."""
    if method not in METHOD_STYLE:
        raise KeyError(f"no style for method {method!r}")
    return {**METHOD_STYLE[method], "label": label(method)}


def label(method: str) -> str:
    return C.DISPLAY.get(method, method)


def save_fig(fig, name: str, directory: Path = C.FIGURES) -> list[Path]:
    """Write <directory>/<name>.png (200 dpi) and .pdf; return both paths."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"{name}.png", directory / f"{name}.pdf"]
    fig.savefig(paths[0], dpi=DPI, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    return paths


def log_x(ax, label_text: str | None = None) -> None:
    """Log-scale x axis with plain-number ticks."""
    from matplotlib.ticker import ScalarFormatter  # noqa: PLC0415

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    if label_text:
        ax.set_xlabel(label_text)


def log_y(ax, label_text: str | None = None) -> None:
    from matplotlib.ticker import ScalarFormatter  # noqa: PLC0415

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ScalarFormatter())
    if label_text:
        ax.set_ylabel(label_text)
