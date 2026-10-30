"""Analysis orchestrator (ANALYSIS.md).  Reads result CSVs only.

    python -m experiments.analyze              # every stage implemented so far
    python -m experiments.analyze --only A0    # one stage

Stages live in experiments/analysis/a<n>_*.py, each with run().  A0 must
pass before any later stage runs.  Stages are added here as they are built;
the manifest (A9) comes last.
"""

from __future__ import annotations

import argparse

from experiments.analysis import a0_integrity, a1_rq1, a2_figures, a3_cost
from experiments.analysis.takeaway import integrity_sentence

STAGES = {
    "A0": a0_integrity.run,
    "A1": a1_rq1.run,          # confirmatory; re-validates A0 itself
    "A2": a2_figures.run,      # F1, F2 + figures/takeaways.csv
    "A3": a3_cost.run,         # RQ2 cost, exploratory; F4, F5
}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", choices=sorted(STAGES), default=None)
    args = ap.parse_args(argv)
    todo = [args.only] if args.only else list(STAGES)
    for name in todo:
        print(f"===== {name} =====")
        out = STAGES[name]()
        if name == "A0":
            print(integrity_sentence(out))


if __name__ == "__main__":
    main()
