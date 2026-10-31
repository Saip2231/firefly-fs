"""Demo traces (B1 item 2): re-run logged searches and record the population.

src/'s trace= records only per-sweep numbers, and src/ must not change
(CLAUDE.md rule 7), so PopulationRecorder reads the population from the
search function's local variables with sys.settrace.  It only copies arrays:
no RNG draw, fitness value or budget count changes, and each run is checked
against the logged `idx` (match OK / FAIL).

Snapshot points (line events, before the line runs):
  `history: list[float] = []`  -> initial population, already scored
  `history.append(...)`        -> end of each sweep / iteration
"""

from __future__ import annotations

import inspect
import sys

import numpy as np

from src.firefly import binary_firefly
from src.pso import binary_pso

from .common import MAX_FRAMES, subsample_frames

# code object -> (population, value, best mask, best value, value label)
_SPECS = {
    binary_firefly.__code__: ("positions", "fits", "best_mask", "best_fit",
                              "current fitness"),
    binary_pso.__code__: ("masks", "pbest_f", "gbest_mask", "gbest_f",
                          "personal-best fitness"),
}


def _marker_lines(func) -> set[int]:
    lines, first = inspect.getsourcelines(func)
    marks = set()
    for text in ("history: list[float] = []", "history.append("):
        hits = [first + i for i, line in enumerate(lines) if line.strip().startswith(text)]
        if len(hits) != 1:
            raise RuntimeError(f"{func.__name__}: expected one line starting with "
                               f"{text!r}, found {len(hits)}; update {__name__}")
        marks.add(hits[0])
    return marks


_MARKS = {binary_firefly.__code__: _marker_lines(binary_firefly),
          binary_pso.__code__: _marker_lines(binary_pso)}


class PopulationRecorder:
    """Context manager; read-only snapshots of binary_firefly / binary_pso."""

    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.value_kind = ""

    def __enter__(self) -> "PopulationRecorder":
        self._previous = sys.gettrace()
        sys.settrace(self._global)
        return self

    def __exit__(self, *exc) -> None:
        sys.settrace(self._previous)

    def _global(self, frame, event, arg):
        return self._local if event == "call" and frame.f_code in _SPECS else None

    def _local(self, frame, event, arg):
        if event == "line" and frame.f_lineno in _MARKS[frame.f_code]:
            self._snapshot(frame)
        return self._local

    def _snapshot(self, frame) -> None:
        pop, val, best_mask, best, kind = _SPECS[frame.f_code]
        loc = frame.f_locals
        self.value_kind = kind
        self.frames.append(dict(
            pop=np.array(loc[pop], dtype=bool, copy=True),
            value=np.array(loc[val], dtype=float, copy=True),
            best_mask=np.array(loc[best_mask], dtype=bool, copy=True),
            best=float(loc[best]),
            evals=int(loc["state"]["evals"])))


def record_search(method, fitness, state, d, k, budget, seed, hp) -> tuple[dict, PopulationRecorder]:
    """run_search under the recorder; returns (run_search output, recorder)."""
    from src.search_registry import run_search  # noqa: PLC0415

    with PopulationRecorder() as rec:
        out = run_search(method, fitness, state, d, k, budget, seed, hp)
    if not rec.frames:
        raise RuntimeError(f"no frames recorded for {method}")
    if not np.array_equal(rec.frames[-1]["best_mask"], out["best_mask"]):
        raise RuntimeError("last frame's best mask differs from the returned mask")
    return out, rec


def trace_arrays(out: dict, rec: PopulationRecorder, cap: int = MAX_FRAMES) -> dict:
    """npz-ready arrays: capped frames plus the full (uncapped) curve."""
    f = rec.frames
    keep = subsample_frames(len(f), cap)
    return dict(
        pop=np.stack([f[i]["pop"] for i in keep]),
        pop_value=np.stack([f[i]["value"] for i in keep]),
        value_kind=np.str_(rec.value_kind),
        frame_index=keep.astype(np.int32),          # 0 = initial, s = after sweep s
        frame_evals=np.array([f[i]["evals"] for i in keep], dtype=np.int32),
        frame_best=np.array([f[i]["best"] for i in keep]),
        frame_best_mask=np.stack([f[i]["best_mask"] for i in keep]),
        frames_total=np.int32(len(f)),
        curve_evals=np.array([fr["evals"] for fr in f], dtype=np.int32),
        curve_best=np.array([fr["best"] for fr in f]),
        diversity=np.asarray(out["diversity"], dtype=float),
        final_mask=np.asarray(out["best_mask"], dtype=bool),
        best_fit=np.float64(out["best_fit"]),
        evals_used=np.int32(out["evals_used"]),
    )


def match_logged(final_mask: np.ndarray, logged_idx: str) -> bool:
    """True when the replayed mask selects exactly the logged features."""
    logged = sorted(int(t) for t in str(logged_idx).split(","))
    return np.flatnonzero(final_mask).tolist() == logged
