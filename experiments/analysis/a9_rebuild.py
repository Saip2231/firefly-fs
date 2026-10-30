"""Stage A9 (part 2): clean-rebuild check (ANALYSIS.md A9, final check).

ANALYSIS.md says "delete figures/ and results/analysis/, rerun, compare".
results/analysis/ also holds the A4 timing outputs, which `analyze` cannot
regenerate (idle-machine measurements), and CLAUDE.md asks before deleting
results.  So instead of deleting, rebuild_check():
  1. MOVES every file that `analyze` owns (a9_manifest.analyze_owned(): not
     A4, not A6a, not the manifest) into logs/rebuild_snapshot_<time>/,
     keeping relative paths (logs/ is git-ignored; nothing is deleted);
  2. runs every analyze stage from that clean state;
  3. compares each regenerated file with its snapshot copy:
       .csv  exact equality of the dataframes (pandas, check_exact)
       .md   identical text apart from the "- Generated:" timestamp line
       .png / .pdf  must exist; figures are compared through their data
             (figures/data/F1_frames.csv, rq1_tests.csv, rq2_cost.csv,
             rq3_*.csv, rq4_*.csv, ablations.csv); byte equality is
             reported for information only (PDFs embed a creation date)
  4. reports PASS / FAIL per file.  On FAIL the snapshot is kept so the
     originals can be restored by moving them back.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from experiments.analysis import a9_manifest as M
from experiments.analysis import common as C

TIMESTAMP_PREFIXES = ("- Generated:",)


def _strip(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if not ln.startswith(TIMESTAMP_PREFIXES)]


def compare_file(old: Path, new: Path) -> tuple[str, str]:
    """(status, detail) with status PASS / FAIL / INFO."""
    if not new.exists():
        return "FAIL", "not regenerated"
    if old.suffix == ".csv":
        try:
            pd.testing.assert_frame_equal(pd.read_csv(old), pd.read_csv(new), check_exact=True)
            return "PASS", "identical data"
        except AssertionError as e:
            return "FAIL", str(e).splitlines()[0][:200]
    if old.suffix == ".md":
        same = _strip(old.read_text(encoding="utf-8")) == _strip(new.read_text(encoding="utf-8"))
        return ("PASS", "identical text (timestamp ignored)") if same else ("FAIL", "text differs")
    same = old.read_bytes() == new.read_bytes()
    return "PASS", "regenerated; bytes " + ("identical" if same else
                                            "differ (compared via its data CSV)")


def snapshot(files: list[Path], root: Path, snap: Path) -> list[Path]:
    moved = []
    for f in files:
        dest = snap / f.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(f), str(dest))
        moved.append(f)
    return moved


def rebuild_check(run_all, root: Path = C.ROOT) -> bool:
    files = M.analyze_owned(root)
    snap = root / "logs" / f"rebuild_snapshot_{datetime.now():%Y%m%d_%H%M%S_%f}"
    print(f"Moving {len(files)} analyze-owned files to {snap.relative_to(root)} "
          "(A4 / A6a outputs untouched) ...")
    moved = snapshot(files, root, snap)
    run_all()
    results = [(f, *compare_file(snap / f.relative_to(root), f)) for f in moved]
    new_only = sorted(set(M.analyze_owned(root)) - set(moved))
    n_fail = sum(s == "FAIL" for _, s, _ in results)
    print(f"\nREBUILD CHECK: {len(results) - n_fail}/{len(results)} files PASS")
    for f, s, d in results:
        print(f"[{s}] {f.relative_to(root).as_posix()}: {d}")
    for f in new_only:
        print(f"[INFO] new file not in the snapshot: {f.relative_to(root).as_posix()}")
    print(f"\nSnapshot kept at {snap.relative_to(root)} "
          f"({'safe to delete' if n_fail == 0 else 'move files back to restore'}).")
    return n_fail == 0
