"""Stage A8: convergence (ANALYSIS.md A8) -- deferred.

Runs only if results/traces/*.npz exist.  No traces have been recorded yet
(decision 2026-10-30: skip A8 for now, return to it later), so run() reports
the skip and produces no output.  The plotting code will be added together
with the trace-recording script.
"""

from __future__ import annotations

from pathlib import Path

from experiments.analysis import common as C

TRACES = C.RESULTS / "traces"


def traces_present(directory: Path = TRACES) -> bool:
    return directory.is_dir() and any(directory.glob("*.npz"))


def run() -> dict:
    if not traces_present():
        msg = (f"A8 skipped: no {TRACES.relative_to(C.ROOT)}/*.npz traces "
               "(deferred; convergence figure not produced)")
        print(msg)
        return dict(skipped=True, reason=msg)
    raise NotImplementedError("traces found but the A8 plotting code is not written yet")
