"""Plotly figure builders shared by the pages (data in, figure out).

Every figure uses the shared template from theme.py, method colours from
the tokens, PCA dashed and labelled as a projection, and never encodes a
category by colour alone (labels, symbols or text accompany it).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.lib import theme
from app.lib.methods import display, sort_key
from app.lib.stats_view import format_p

LABEL_Y = "test accuracy (mean ± std over seeds)"


def _fig(**layout) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template=theme.plotly_template(), margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.2), **layout)
    return fig


def acc_vs_k(desc: pd.DataFrame, title: str) -> go.Figure:
    """Lines per method (mean ± std); 'all' as a horizontal reference."""
    fig = _fig(title=title, xaxis_title="k (features kept)", yaxis_title=LABEL_Y)
    for method in sorted(desc["method"].unique(), key=sort_key):
        d = desc[desc["method"] == method].sort_values("k")
        if method == "all":
            fig.add_hline(y=float(d["mean"].iloc[0]), line_dash="dot",
                          line_color=theme.method_colour("all"),
                          annotation_text=f"All features (k={int(d['k'].iloc[0])})")
            continue
        fig.add_trace(go.Scatter(
            x=d["k"], y=d["mean"], error_y=dict(type="data", array=d["std"]),
            mode="lines+markers", name=display(method),
            line=dict(color=theme.method_colour(method),
                      dash=theme.PCA_DASH if method == "pca" else "solid"),
            hovertemplate="k=%{x}<br>accuracy %{y:.3f}<extra>" + display(method) + "</extra>"))
    fig.update_xaxes(type="category")
    return fig


def label_grid(grid: pd.DataFrame) -> go.Figure:
    """Families (rows) x comparators (columns); square markers with the label
    as symbol + word and the mean difference as text.  Clickable points."""
    fig = _fig(title="RQ1: plain Firefly vs each comparator (click a cell)",
               height=110 + 38 * grid["family"].nunique())
    for label, style in theme.LABEL_STYLE.items():
        g = grid[grid["label"] == label]
        if g.empty:
            continue
        fig.add_trace(go.Scatter(
            x=g["comparator_name"], y=g["family"], mode="markers+text",
            marker=dict(symbol="square", size=34, color=style["fill"]),
            text=[f"{style['symbol']} {d:+.1f}" for d in g["mean_diff_pts"]],
            textfont=dict(color=style["text"], size=12),
            customdata=np.stack([g["family"], g["comparator"],
                                 [format_p(p) for p in g["p_holm"]]], axis=1),
            name=f"{style['symbol']} {label}",
            hovertemplate=("%{y} vs %{x}<br>" + label + ", mean difference %{text} pts"
                           "<br>adjusted p %{customdata[2]}<extra></extra>")))
    fig.update_yaxes(autorange="reversed", type="category", categoryorder="array",
                     categoryarray=list(dict.fromkeys(grid["family"])))
    fig.update_xaxes(type="category", side="top", categoryorder="array",
                     categoryarray=list(dict.fromkeys(grid["comparator_name"])))
    return fig


def seed_diffs(diffs: pd.DataFrame, comparator: str) -> go.Figure:
    """Bars of per-seed paired differences (Firefly minus comparator), in points."""
    fig = _fig(title=f"Per-seed difference: Firefly minus {display(comparator)}",
               xaxis_title="seed", yaxis_title="difference (accuracy points)")
    colours = [theme.LABEL_STYLE["WIN"]["fill"] if v > 0 else
               theme.LABEL_STYLE["LOSS"]["fill"] if v < 0 else theme.LABEL_STYLE["TIE"]["fill"]
               for v in diffs["diff_pts"]]
    fig.add_trace(go.Bar(x=diffs["seed"].astype(str), y=diffs["diff_pts"],
                         marker_color=colours, text=[f"{v:+.2f}" for v in diffs["diff_pts"]],
                         textposition="outside", name="difference",
                         hovertemplate="seed %{x}: %{y:+.2f} pts<extra></extra>"))
    fig.add_hline(y=0, line_color=theme.SURFACES["dark"]["muted"])
    return fig


def cost_scatter(cost: pd.DataFrame, title: str) -> go.Figure:
    """Accuracy vs search time (log x), one point per method (and budget)."""
    fig = _fig(title=title, xaxis_title="search time, s (log; measured under 15-way "
               "parallel load)", yaxis_title="test accuracy (mean over seeds)")
    for method in sorted(cost["method"].unique(), key=sort_key):
        d = cost[cost["method"] == method]
        lab = [f"b={int(b)}" if pd.notna(b) else "" for b in d["budget"]]
        fig.add_trace(go.Scatter(
            x=d["search_s_mean"].clip(lower=1e-3), y=d["acc_mean"], mode="markers+text",
            text=lab, textposition="top center", name=display(method),
            marker=dict(size=12, color=theme.method_colour(method),
                        symbol="diamond" if method == "pca" else "circle"),
            hovertemplate="%{x:.2f} s, accuracy %{y:.3f} %{text}<extra>"
                          + display(method) + "</extra>"))
    fig.update_xaxes(type="log")
    return fig


def bars_with_reference(df: pd.DataFrame, value: str, err: str | None, title: str,
                        y_title: str, refs: dict[str, float] | None = None) -> go.Figure:
    """One bar per row (column `name`, colour by `method`), optional reference lines."""
    fig = _fig(title=title, yaxis_title=y_title, showlegend=False)
    fig.add_trace(go.Bar(
        x=df["name"], y=df[value],
        error_y=dict(type="data", array=df[err]) if err else None,
        marker_color=[theme.method_colour(m) for m in df["method"]],
        text=[f"{v:.3f}" for v in df[value]], textposition="outside",
        hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    for i, (name, y) in enumerate((refs or {}).items()):
        fig.add_hline(y=y, line_dash="dash" if i == 0 else "dot",
                      line_color=theme.SURFACES["dark"]["muted"], annotation_text=name)
    return fig


def diff_bars(df: pd.DataFrame, title: str) -> go.Figure:
    """Horizontal bars of mean paired differences (points) with ±std."""
    fig = _fig(title=title, xaxis_title="mean paired difference (accuracy points)",
               showlegend=False, height=120 + 28 * len(df))
    fig.add_trace(go.Bar(
        y=df["name"], x=df["mean_diff_pts"], orientation="h",
        error_x=dict(type="data", array=df["std_diff_pts"]),
        marker_color=[theme.method_colour(m) for m in df["method"]],
        text=[f"{v:+.2f}" for v in df["mean_diff_pts"]], textposition="outside",
        hovertemplate="%{y}: %{x:+.2f} pts<extra></extra>"))
    fig.add_vline(x=0, line_color=theme.SURFACES["dark"]["muted"])
    fig.update_yaxes(autorange="reversed")
    return fig


def pixel_grid(values: np.ndarray, title: str, zmax: float | None = None) -> go.Figure:
    """8 x 8 heatmap (Digits pixels), value printed in every cell."""
    z = np.asarray(values, dtype=float).reshape(8, 8)
    fig = _fig(title=title, height=360, width=360)
    fig.add_trace(go.Heatmap(
        z=z, zmin=0, zmax=zmax if zmax is not None else max(1.0, float(z.max())),
        colorscale=[[0, theme.SURFACES["dark"]["card"]], [1, theme.ACCENT]],
        text=[[f"{v:g}" for v in row] for row in z], texttemplate="%{text}",
        showscale=False, hovertemplate="row %{y}, col %{x}: %{z:g}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False, scaleanchor="x")
    fig.update_xaxes(showgrid=False)
    return fig


def population(pop: np.ndarray, values: np.ndarray, title: str) -> go.Figure:
    """Rows = fireflies / particles, columns = features; a selected cell's
    brightness grows as the member's fitness gets lower (better)."""
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    lo, hi = (v[finite].min(), v[finite].max()) if finite.any() else (0.0, 1.0)
    rank = np.zeros_like(v)
    if hi > lo:
        rank[finite] = (v[finite] - lo) / (hi - lo)
    bright = np.where(finite, 1.0 - rank, 0.0)
    z = pop.astype(float) * (0.35 + 0.65 * bright[:, None])
    fig = _fig(title=title, height=140 + 22 * pop.shape[0],
               xaxis_title="feature index", yaxis_title="member")
    fig.add_trace(go.Heatmap(
        z=z, zmin=0, zmax=1, showscale=False,
        colorscale=[[0, theme.SURFACES["dark"]["background"]], [0.01, "#5A4100"],
                    [1, theme.ACCENT]],
        customdata=np.repeat(v[:, None], pop.shape[1], axis=1),
        hovertemplate="member %{y}, feature %{x}<br>fitness %{customdata:.4f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return fig


def embedding_strip(emb: np.ndarray, idx: np.ndarray | None, title: str) -> go.Figure:
    """All embedding dimensions as bars; the method's chosen subset highlighted
    (colour plus a separate legend entry)."""
    sel = np.zeros(len(emb), dtype=bool)
    if idx is not None:
        sel[np.asarray(idx, dtype=int)] = True
    x = np.arange(len(emb))
    fig = _fig(title=title, xaxis_title="embedding dimension", yaxis_title="activation",
               height=280, bargap=0)
    fig.add_trace(go.Bar(x=x[~sel], y=emb[~sel], name="not used by this method",
                         marker_color=theme.SURFACES["dark"]["muted"],
                         hovertemplate="dim %{x}: %{y:.2f}<extra></extra>"))
    if sel.any():
        fig.add_trace(go.Bar(x=x[sel], y=emb[sel], name="used by this method",
                             marker_color=theme.ACCENT,
                             hovertemplate="dim %{x} (used): %{y:.2f}<extra></extra>"))
    return fig


def fitness_curves(curves: dict[str, pd.DataFrame], marker: tuple[str, float, float] | None,
                   title: str) -> go.Figure:
    """Best validation fitness vs evaluations; optional marker for the current frame."""
    fig = _fig(title=title, xaxis_title="evaluations used",
               yaxis_title="best validation fitness (lower is better)")
    for method, c in curves.items():
        fig.add_trace(go.Scatter(x=c["evals"], y=c["best"], mode="lines", line_shape="hv",
                                 name=display(method),
                                 line=dict(color=theme.method_colour(method))))
    if marker:
        name, x, y = marker
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers", name=f"{name}: current frame",
                                 marker=dict(size=14, symbol="star", color=theme.ACCENT,
                                             line=dict(color="#FFFFFF", width=1))))
    return fig
