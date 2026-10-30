"""Analysis orchestrator (ANALYSIS.md).  Reads result CSVs only.

    python -m experiments.analyze                  # A0..A8 in order, then the manifest
    python -m experiments.analyze --only A3        # one stage (no manifest)
    python -m experiments.analyze --rebuild-check  # A9 final check: move analyze-owned
                                                   # outputs aside, rebuild, compare

Stages live in experiments/analysis/a<n>_*.py, each with run().  A0 must
pass before anything else.  A8 runs only if results/traces/*.npz exist.  A4
(experiments/time_final.py) and A6a (experiments/eval_baseline_transfer.py)
are separate scripts; the manifest lists them with their own commands.
A full run fails if any file in figures/ or results/analysis/ is not claimed
by a stage in experiments/analysis/a9_manifest.REGISTRY.
"""

from __future__ import annotations

import argparse
import sys

from experiments.analysis import (a0_integrity, a1_rq1, a2_figures, a3_cost, a5_stability,
                                  a6_transfer, a7_ablations, a8_convergence, a9_manifest,
                                  a9_rebuild)
from experiments.analysis.takeaway import integrity_sentence

STAGES = {
    "A0": a0_integrity.run,
    "A1": a1_rq1.run,          # confirmatory; re-validates A0 itself
    "A2": a2_figures.run,      # F1, F2 + figures/takeaways.csv
    "A3": a3_cost.run,         # RQ2 cost, exploratory; F4, F5
    "A5": a5_stability.run,    # RQ3 stability, exploratory; F6, F7
    "A6": a6_transfer.run,     # RQ4 transfer, exploratory; needs A6a CSVs; F8
    "A7": a7_ablations.run,    # ablations, exploratory; F3, F9
    "A8": a8_convergence.run,  # convergence; skipped while no traces exist
}


def run_stage(name: str):
    print(f"===== {name} =====", flush=True)
    out = STAGES[name]()
    if name == "A0":
        print(integrity_sentence(out))
    return out


def run_all() -> dict:
    """Every stage in order, then the unregistered-output check and the manifest."""
    notes = {}
    for name in STAGES:
        out = run_stage(name)
        if isinstance(out, dict) and out.get("skipped"):
            notes[name] = out["reason"]
    stray = a9_manifest.unregistered()
    if stray:
        print("Outputs without a manifest entry (add them to a9_manifest.REGISTRY):")
        for f in stray:
            print(f"  {f.relative_to(a9_manifest.C.ROOT).as_posix()}")
        raise SystemExit(1)
    path = a9_manifest.write_manifest(notes=notes)
    print(f"\n===== A9 =====\nWrote {path}")
    return notes


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--only", choices=sorted(STAGES), default=None)
    group.add_argument("--rebuild-check", action="store_true")
    args = ap.parse_args(argv)
    if args.only:
        run_stage(args.only)
    elif args.rebuild_check:
        sys.exit(0 if a9_rebuild.rebuild_check(run_all) else 1)
    else:
        run_all()


if __name__ == "__main__":
    main()
