"""Day 5 preflight: read-only checks before launching the main runs.

Prints PASS / FAIL / INFO per item and exits non-zero if anything FAILs.
Launches nothing, edits nothing (the only write is a temporary file in
results/ and logs/ to test that they are writable; it is deleted at once).

  1. git: tracked tree clean; tag day5-launch exists, contains PREREG.md and
     config/frozen_config.yaml identical to the working copy (line endings
     normalised); PREREG cites the commit that last changed the frozen
     config; PREREG was committed before any row in results/meta_*.csv.
  2. config_hash: SHA-256 the runner will write == SHA-256 of the file.
  3. results/meta_*.csv: no row with budget outside {800, 3000}.
  4. Day-5 grid from the baseline CSVs: k lists, jobs, rows, comparison
     families (vs the number stated in PREREG.md).
  5. cpu_count, free disk space, results/ and logs/ writable.
  6. Stall-swap share of PSO evaluations in the smoke rows (if logged).

Usage (from firefly-fs/, venv active):
    python experiments/preflight_day5.py
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime

import pandas as pd

from _common import RESULTS_DIR, ROOT

import run_metaheuristics as rm

TAG = "day5-launch"
BUDGETS = {800, 3000}
N_METHODS, N_SEEDS, N_MASK_TYPES = len(rm.DEFAULT_METHODS), 10, 2
PREREG = ROOT / "PREREG.md"
CONFIG = ROOT / "config" / "frozen_config.yaml"
SELF = "experiments/preflight_day5.py"

results: list[tuple[str, str, str]] = []


def report(item: str, status: str, msg: str) -> None:
    results.append((item, status, msg))
    print(f"[{status:<4}] {item}: {msg}")


def git(*args: str) -> tuple[int, str]:
    p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return p.returncode, p.stdout.strip()


def git_bytes(spec: str) -> bytes | None:
    p = subprocess.run(["git", "show", spec], cwd=ROOT, capture_output=True)
    return p.stdout if p.returncode == 0 else None


def norm(b: bytes) -> bytes:
    return b.replace(b"\r\n", b"\n")


# ---------------------------------------------------------------------------
# 1. git state
# ---------------------------------------------------------------------------

def check_git() -> None:
    code, tracked = git("status", "--porcelain", "--untracked-files=no")
    if code != 0:
        report("1 git", "FAIL", "git not available")
        return
    report("1a tracked tree clean", "PASS" if not tracked else "FAIL",
           "no uncommitted changes" if not tracked else f"uncommitted:\n{tracked}")
    _, untracked = git("status", "--porcelain")
    others = [ln for ln in untracked.splitlines()
              if ln.startswith("??") and SELF not in ln.replace("\\", "/")]
    report("1b untracked files", "PASS" if not others else "FAIL",
           "none besides this script" if not others else "\n".join(others))

    code, tag_commit = git("rev-parse", "-q", "--verify", f"refs/tags/{TAG}^{{commit}}")
    if code != 0:
        report("1c tag", "FAIL", f"tag {TAG} does not exist (git tag {TAG})")
        return
    _, head = git("rev-parse", "HEAD")
    report("1c tag", "PASS", f"{TAG} -> {tag_commit[:10]} (HEAD {head[:10]}"
           f"{', same' if head == tag_commit else ', HEAD is later'})")

    for rel, path in (("PREREG.md", PREREG), ("config/frozen_config.yaml", CONFIG)):
        at_tag = git_bytes(f"{TAG}:{rel}")
        if at_tag is None:
            report(f"1d {rel} in tag", "FAIL", f"{rel} is not in {TAG}")
        elif norm(at_tag) != norm(path.read_bytes()):
            report(f"1d {rel} in tag", "FAIL", "working copy differs from the tagged version")
        else:
            report(f"1d {rel} in tag", "PASS", "present and identical to the working copy")

    _, cfg_commit = git("log", "-1", "--format=%H", f"{TAG}", "--", "config/frozen_config.yaml")
    text = PREREG.read_text(encoding="utf-8")
    cited = re.findall(r"\b[0-9a-f]{40}\b", text)
    if "FROZEN_CONFIG_COMMIT" in text:
        report("1e PREREG cites config commit", "FAIL", "placeholder FROZEN_CONFIG_COMMIT still present")
    elif cfg_commit in cited:
        report("1e PREREG cites config commit", "PASS",
               f"cites {cfg_commit[:10]}, the last commit that changed frozen_config.yaml")
    else:
        report("1e PREREG cites config commit", "FAIL",
               f"PREREG cites {[c[:10] for c in cited]}, config last changed in {cfg_commit[:10]}")

    _, prereg_time = git("log", "-1", "--format=%cI", f"{TAG}", "--", "PREREG.md")
    t_prereg = datetime.fromisoformat(prereg_time)
    metas = [p for p in RESULTS_DIR.glob("meta_*.csv")]
    if not metas:
        report("1f PREREG before results", "PASS",
               f"PREREG committed {prereg_time}; no results/meta_*.csv yet")
        return
    early = []
    for p in metas:
        df = pd.read_csv(p, dtype={"idx": str})
        for rid in df["run_id"].dropna().unique() if "run_id" in df else []:
            t = datetime.strptime(str(rid)[:15], "%Y%m%d_%H%M%S").astimezone()
            if t <= t_prereg:
                early.append(f"{p.name}: {rid}")
    report("1f PREREG before results", "PASS" if not early else "FAIL",
           f"PREREG committed {prereg_time}; "
           + ("all meta rows are later" if not early else "earlier runs: " + ", ".join(early)))


# ---------------------------------------------------------------------------
# 2. config hash
# ---------------------------------------------------------------------------

def check_config_hash() -> None:
    _, runner_hash = rm.load_config(CONFIG)
    direct = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    lf = hashlib.sha256(norm(CONFIG.read_bytes())).hexdigest()
    crlf = b"\r\n" in CONFIG.read_bytes()
    report("2 config_hash", "PASS" if runner_hash == direct else "FAIL",
           f"runner {runner_hash[:16]}... == file {direct[:16]}..."
           if runner_hash == direct else f"runner {runner_hash} != file {direct}")
    report("2 line endings", "INFO",
           f"working copy has {'CRLF' if crlf else 'LF'} line endings; the hash is of the "
           f"bytes on disk (LF-normalised it would be {lf[:16]}...). A fresh checkout on "
           f"another OS can give a different config_hash for the same content.")


# ---------------------------------------------------------------------------
# 3. budgets in meta CSVs
# ---------------------------------------------------------------------------

def check_meta_budgets() -> None:
    metas = sorted(RESULTS_DIR.glob("meta_*.csv"))
    if not metas:
        report("3 meta budgets", "PASS", "no results/meta_*.csv yet (smoke rows are in results/smoke/)")
        return
    bad = []
    for p in metas:
        df = pd.read_csv(p, dtype={"idx": str})
        off = df[~df.budget.isin(BUDGETS)]
        for _, r in off.iterrows():
            bad.append(f"{p.name}: {r.method} budget={r.budget} seed={r.seed} k={r.k} {r.mask_type}")
    report("3 meta budgets", "PASS" if not bad else "FAIL",
           f"{len(metas)} file(s), all budgets in {sorted(BUDGETS)}" if not bad
           else f"{len(bad)} offending row(s):\n" + "\n".join(bad))


# ---------------------------------------------------------------------------
# 4. grid size
# ---------------------------------------------------------------------------

def check_grid() -> None:
    total_jobs = total_rows = families = 0
    for ds in ("digits", "cifar_r18"):
        ks, skipped = rm.default_ks(ds)
        jobs = N_METHODS * N_SEEDS * len(BUDGETS) * len(ks)
        rows = jobs * N_MASK_TYPES
        fam = len(ks) * len(BUDGETS)
        total_jobs, total_rows, families = total_jobs + jobs, total_rows + rows, families + fam
        report(f"4 grid {ds}", "INFO",
               f"k={ks} (skipped {skipped}); {jobs} jobs, {rows} rows, {fam} families")
    m = re.search(r"\((\d+) families", PREREG.read_text(encoding="utf-8"))
    stated = int(m.group(1)) if m else None
    report("4 families vs PREREG", "PASS" if stated == families else "FAIL",
           f"computed {families}, PREREG states {stated}; total {total_jobs} jobs, "
           f"{total_rows} rows ({N_METHODS} methods x {N_SEEDS} seeds x "
           f"{len(BUDGETS)} budgets x {N_MASK_TYPES} mask types)")


# ---------------------------------------------------------------------------
# 5. machine
# ---------------------------------------------------------------------------

def check_machine() -> None:
    cpu = os.cpu_count()
    report("5 cpu", "INFO", f"os.cpu_count() = {cpu}; runner default n_jobs = {max(1, (cpu or 2) - 1)}")
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    report("5 disk", "PASS" if free_gb > 2 else "FAIL", f"{free_gb:.1f} GB free on the repo drive")
    for d in (RESULTS_DIR, ROOT / "logs"):
        if not d.is_dir():
            report(f"5 writable {d.name}/", "FAIL", f"{d} does not exist")
            continue
        try:
            with tempfile.NamedTemporaryFile(dir=d, prefix=".preflight_", delete=True) as fh:
                fh.write(b"x")
            report(f"5 writable {d.name}/", "PASS", "temporary file created and removed")
        except OSError as e:
            report(f"5 writable {d.name}/", "FAIL", str(e))


# ---------------------------------------------------------------------------
# 6. stall-swap share
# ---------------------------------------------------------------------------

PROPOSAL = """\
    Proposed minimal additive change (NOT applied; it would change src/ and the
    CSV schema after the day5-launch tag, so it needs your approval and a re-tag):
      src/search_registry.py, in the returned dict:
          "stall_evals": sum(r["new_evals"] - r["move_evals"]
                             for r in trace.get("iters", [])),   # PSO; 0 for Firefly
      experiments/run_metaheuristics.py: add "stall_evals" to SCHEMA and set
          stall_evals=out["stall_evals"] in `common`.
    Behaviour unchanged (trace is already recorded); old CSVs would lack the
    column.  Alternative with no code change: a separate diagnostic script that
    reruns a few PSO jobs with trace={} after Day 5 (exploratory)."""


def check_stall_share() -> None:
    smoke = RESULTS_DIR / "smoke" / "meta_digits.csv"
    cols = list(pd.read_csv(smoke, nrows=0).columns) if smoke.exists() else []
    if "stall_evals" in cols:
        df = pd.read_csv(smoke)
        pso = df[(df.method.str.startswith("pso")) & (df.mask_type == "raw")]
        share = pso.stall_evals.sum() / pso.evals_used.sum()
        report("6 stall-swap share", "INFO", f"{share:.1%} of PSO smoke evaluations")
    else:
        report("6 stall-swap share", "INFO",
               "not logged: the runner CSV has no stall-swap column "
               f"({'smoke file present' if smoke.exists() else 'no smoke file'}).\n" + PROPOSAL)


def main() -> int:
    print(f"Day 5 preflight, {datetime.now().isoformat(timespec='seconds')}, {ROOT}\n")
    for check in (check_git, check_config_hash, check_meta_budgets, check_grid,
                  check_machine, check_stall_share):
        try:
            check()
        except Exception as e:  # noqa: BLE001  (a crashing check is a FAIL, not a stop)
            report(check.__name__, "FAIL", f"{type(e).__name__}: {e}")
    n_fail = sum(s == "FAIL" for _, s, _ in results)
    print(f"\n{'ALL CHECKS PASSED' if n_fail == 0 else f'{n_fail} CHECK(S) FAILED'} "
          f"({len(results)} items)")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
