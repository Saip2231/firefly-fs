"""Stage A9 (part 1): output registry and manifest (ANALYSIS.md A9).

REGISTRY says, for every stage, which files it writes (glob patterns
relative to the repo root), which CSVs it reads, and the exact command that
regenerates them.  write_manifest() records, for every output file that
exists: its stage, size, SHA-256, the command, and each source CSV with its
SHA-256 and row count, plus the git hash and date.  unregistered() lists
files in figures/ and results/analysis/ that no stage claims; the
orchestrator fails if there are any.

A4 (experiments/time_final.py) and A6a (experiments/eval_baseline_transfer.py)
are separate scripts and are listed with their own commands; A8 is listed as
skipped while no traces exist.
"""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

from experiments.analysis import common as C

ANALYZE = "python -m experiments.analyze"
MAIN_INPUTS = ["results/baselines_digits.csv", "results/baselines_cifar_r18.csv",
               "results/meta_digits.csv", "results/meta_cifar_r18.csv"]

REGISTRY: dict[str, dict] = {
    "A0": dict(outputs=["results/analysis/integrity_report.md"],
               inputs=MAIN_INPUTS + ["results/split_hashes.csv", "PREREG.md"],
               command=f"{ANALYZE} --only A0"),
    "A1": dict(outputs=["results/analysis/rq1_tests.csv", "results/analysis/descriptive.csv",
                        "results/analysis/rq1_summary.md"],
               inputs=MAIN_INPUTS + ["results/split_hashes.csv", "PREREG.md"],
               command=f"{ANALYZE} --only A1"),
    "A2": dict(outputs=["figures/F1_*", "figures/F2_*", "figures/data/F1_frames.csv"],
               inputs=MAIN_INPUTS + ["results/split_hashes.csv"],
               command=f"{ANALYZE} --only A2"),
    "A3": dict(outputs=["results/analysis/rq2_cost.csv", "figures/F4_*", "figures/F5_*"],
               inputs=MAIN_INPUTS, command=f"{ANALYZE} --only A3"),
    "A4": dict(outputs=["results/analysis/timing_idle.csv", "results/analysis/contention.csv",
                        "results/analysis/contention_factor.md"],
               inputs=MAIN_INPUTS + ["config/frozen_config.yaml"],
               command="python experiments/time_final.py  (idle machine; not rerun by analyze)",
               separate=True),
    "A5": dict(outputs=["results/analysis/rq3_stability.csv",
                        "results/analysis/rq3_feature_frequency.csv",
                        "figures/F6_*", "figures/F7_*"],
               inputs=MAIN_INPUTS, command=f"{ANALYZE} --only A5"),
    "A6a": dict(outputs=["results/baselines_transfer_digits.csv",
                         "results/baselines_transfer_cifar_r18.csv"],
                inputs=["results/baselines_digits.csv", "results/baselines_cifar_r18.csv"],
                command="python experiments/eval_baseline_transfer.py  (touches the test "
                        "split via the runner's evaluate_on_test; not rerun by analyze)",
                separate=True),
    "A6": dict(outputs=["results/analysis/rq4_transfer.csv", "results/analysis/rq4_gaps.csv",
                        "figures/F8_*"],
               inputs=["results/baselines_transfer_digits.csv",
                       "results/baselines_transfer_cifar_r18.csv",
                       "results/meta_digits.csv", "results/meta_cifar_r18.csv"],
               command=f"{ANALYZE} --only A6"),
    "A7": dict(outputs=["results/analysis/ablations.csv", "figures/F3_*", "figures/F9_*"],
               inputs=MAIN_INPUTS, command=f"{ANALYZE} --only A7"),
    "A8": dict(outputs=["figures/F10_*"], inputs=["results/traces/*.npz"],
               command=f"{ANALYZE} --only A8  (skipped while no traces exist)"),
    "shared": dict(outputs=["figures/takeaways.csv"], inputs=[],
                   command=f"{ANALYZE}  (rows upserted by A2, A3, A5, A6, A7)"),
    "A9": dict(outputs=["results/analysis/analysis_manifest.md"], inputs=[],
               command=ANALYZE),
}
SCANNED_DIRS = ("figures", "results/analysis")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> int | str:
    if path.suffix == ".csv":
        return len(pd.read_csv(path))
    return "-"


def expand(patterns, root: Path = C.ROOT) -> list[Path]:
    out = []
    for p in patterns:
        out += sorted(f for f in root.glob(p) if f.is_file())
    return out


def owner_map(root: Path = C.ROOT, registry=REGISTRY) -> dict[Path, str]:
    return {f: stage for stage, spec in registry.items() for f in expand(spec["outputs"], root)}


def unregistered(root: Path = C.ROOT, registry=REGISTRY) -> list[Path]:
    """Files under figures/ and results/analysis/ that no stage claims."""
    owned = set(owner_map(root, registry))
    found = []
    for d in SCANNED_DIRS:
        base = root / d
        if base.is_dir():
            found += [f for f in base.rglob("*") if f.is_file()]
    return sorted(f for f in found if f not in owned)


def analyze_owned(root: Path = C.ROOT, registry=REGISTRY) -> list[Path]:
    """Outputs that `python -m experiments.analyze` regenerates (not A4/A6a)."""
    return sorted(f for f, stage in owner_map(root, registry).items()
                  if not registry[stage].get("separate") and stage != "A9")


def git_hash(root: Path = C.ROOT) -> str:
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                           capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        return h + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_manifest(path: Path | None = None, root: Path = C.ROOT, registry=REGISTRY,
                   notes: dict | None = None) -> Path:
    path = path or root / "results" / "analysis" / "analysis_manifest.md"
    notes = notes or {}
    lines = ["# Analysis manifest (Stage A9)", "",
             f"- Date: {datetime.now().isoformat(timespec='seconds')}",
             f"- git: {git_hash(root)}",
             f"- Command (all analyze stages): `{ANALYZE}`", ""]
    for stage, spec in registry.items():
        if stage == "A9":
            continue
        files = expand(spec["outputs"], root)
        lines += [f"## {stage}", "", f"- Command: `{spec['command']}`"]
        if stage in notes:
            lines.append(f"- Note: {notes[stage]}")
        ins = expand(spec["inputs"], root)
        if ins:
            lines += ["- Sources:", "", "| source | rows | sha256 |", "|---|---|---|"]
            lines += [f"| `{f.relative_to(root).as_posix()}` | {rows(f)} | `{sha256(f)}` |"
                      for f in ins]
        lines += ["", "| output | bytes | sha256 |", "|---|---|---|"]
        lines += [f"| `{f.relative_to(root).as_posix()}` | {f.stat().st_size} | `{sha256(f)}` |"
                  for f in files] or ["| (none) | | |"]
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
