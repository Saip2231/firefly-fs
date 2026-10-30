"""Stage A0: integrity checks on the result CSVs (ANALYSIS.md section 4, A0).

validate() runs every check, writes results/analysis/integrity_report.md,
then raises AssertionError if any check failed (the report is written first,
so a failure can be inspected).  Each check is a pure function of the loaded
frames, so tests can feed small synthetic frames, good and broken.

Checks
  row_counts      rows per file = expected grid size (info table + check)
  seeds_complete  every (dataset, method, budget, k, mask_type) group and
                  every baseline (method, k) group has seeds 0-9 exactly once
  no_nan_acc      no NaN in acc (baselines) / acc_lr, acc_knn, acc_svm (meta)
  evals_budget    evals_used <= budget
  idx_length      len(idx) == k and all indices in [0, d) for every row with
                  a subset (PCA and 'all' excluded)
  config_hash     a single config_hash across all meta rows
  budgets         budgets in {800, 3000}; mask_type in {raw, rerank}
  seeds_match     baseline and meta files use the same seed set per dataset
  split_hashes    results/split_hashes.csv has non-empty hashes for seeds 0-9
                  of every dataset
  families        dataset x k x budget families in the meta files = the count
                  stated in PREREG.md
  git_hashes      distinct git_hash values (listed; '+dirty' from the
                  runs.json append is expected on CIFAR and is not an error)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from experiments.analysis import common as C


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    info: bool = False          # informational line, never fails


def _fmt_groups(groups, limit=8) -> str:
    groups = list(groups)
    head = "; ".join(str(g) for g in groups[:limit])
    return head + (f"; ... ({len(groups)} total)" if len(groups) > limit else "")


# ---------------------------------------------------------------------------
# Individual checks (pure functions)
# ---------------------------------------------------------------------------

def check_row_counts(baselines: dict, meta: dict) -> Check:
    lines, bad = [], []
    for ds, b in baselines.items():
        n_methods = b.loc[b.method != "all", "method"].nunique()
        n_ks = b.loc[b.method != "all", "k"].nunique()
        expected = len(C.SEEDS) * (1 + n_methods * n_ks)
        lines.append(f"baselines_{ds}.csv: {len(b)} rows (expected {expected})")
        if len(b) != expected:
            bad.append(f"baselines_{ds}")
    for ds, m in meta.items():
        ks = C.selection_ks(baselines[ds]) if ds in baselines else sorted(m.k.unique())
        expected = (m.method.nunique() * m.budget.nunique() * len(C.SEEDS)
                    * len(ks) * len(C.MASK_TYPES))
        lines.append(f"meta_{ds}.csv: {len(m)} rows (expected {expected} = "
                     f"{m.method.nunique()} methods x {m.budget.nunique()} budgets x "
                     f"{len(C.SEEDS)} seeds x {len(ks)} k x {len(C.MASK_TYPES)} mask types)")
        if len(m) != expected:
            bad.append(f"meta_{ds}")
    return Check("row_counts", not bad, "\n".join(lines) + (f"\nMISMATCH: {bad}" if bad else ""))


def check_seeds_complete(baselines: dict, meta: dict) -> Check:
    want = list(C.SEEDS)
    bad = []
    for ds, b in baselines.items():
        for key, g in b.groupby(["method", "k"]):
            if sorted(g.seed.tolist()) != want:
                bad.append((f"baselines_{ds}",) + key + (sorted(g.seed.tolist()),))
    for ds, m in meta.items():
        for key, g in m.groupby(["method", "budget", "k", "mask_type"]):
            if sorted(g.seed.tolist()) != want:
                bad.append((f"meta_{ds}",) + key + (sorted(g.seed.tolist()),))
    n_groups = (sum(b.groupby(["method", "k"]).ngroups for b in baselines.values())
                + sum(m.groupby(["method", "budget", "k", "mask_type"]).ngroups
                      for m in meta.values()))
    return Check("seeds_complete", not bad,
                 f"{n_groups} groups, each with seeds 0-9 exactly once" if not bad
                 else "groups without seeds 0-9 exactly once: " + _fmt_groups(bad))


def check_no_nan_acc(baselines: dict, meta: dict) -> Check:
    bad = []
    for ds, b in baselines.items():
        n = int(b["acc"].isna().sum())
        if n:
            bad.append(f"baselines_{ds}.acc: {n} NaN")
    for ds, m in meta.items():
        for col in C.META_ACC:
            n = int(m[col].isna().sum())
            if n:
                bad.append(f"meta_{ds}.{col}: {n} NaN")
    return Check("no_nan_acc", not bad,
                 "no NaN in acc / acc_lr / acc_knn / acc_svm" if not bad else "; ".join(bad))


def check_evals_budget(meta: dict) -> Check:
    bad = []
    for ds, m in meta.items():
        over = m[m.evals_used > m.budget]
        bad += [(ds, r.method, r.budget, r.seed, r.k, r.evals_used) for r in over.itertuples()]
    at = sum(int((m.evals_used == m.budget).sum()) for m in meta.values())
    total = sum(len(m) for m in meta.values())
    return Check("evals_budget", not bad,
                 f"evals_used <= budget on all {total} rows ({at} use the full budget)"
                 if not bad else "evals_used > budget: " + _fmt_groups(bad))


def check_idx_length(baselines: dict, meta: dict) -> Check:
    bad, n_checked = [], 0
    frames = [(f"baselines_{ds}", b, C.dimension(b)) for ds, b in baselines.items()]
    frames += [(f"meta_{ds}", m, C.dimension(baselines[ds]) if ds in baselines else None)
               for ds, m in meta.items()]
    for name, df, d in frames:
        rows = df[~df.method.isin(C.NO_IDX_METHODS)]
        for r in rows.itertuples():
            n_checked += 1
            if len(r.idx) != r.k or (d is not None and any(i < 0 or i >= d for i in r.idx)):
                bad.append((name, r.method, getattr(r, "budget", "-"), r.seed, r.k, len(r.idx)))
    return Check("idx_length", not bad,
                 f"len(idx) == k and indices in [0, d) on {n_checked} rows "
                 f"(PCA and 'all' excluded)" if not bad
                 else "bad idx (file, method, budget, seed, k, len): " + _fmt_groups(bad))


def check_config_hash(meta: dict) -> Check:
    hashes = sorted({h for m in meta.values() for h in m.config_hash.unique()})
    return Check("config_hash", len(hashes) == 1,
                 f"single config_hash {hashes[0][:16]}..." if len(hashes) == 1
                 else f"{len(hashes)} config hashes: {[h[:12] for h in hashes]}")


def check_budgets(meta: dict) -> Check:
    bad = []
    for ds, m in meta.items():
        off_b = sorted(set(m.budget) - set(C.BUDGETS))
        off_m = sorted(set(m.mask_type) - set(C.MASK_TYPES))
        off_meth = sorted(set(m.method) - set(C.META_METHODS))
        if off_b:
            bad.append(f"meta_{ds}: budgets {off_b}")
        if off_m:
            bad.append(f"meta_{ds}: mask types {off_m}")
        if off_meth:
            bad.append(f"meta_{ds}: unknown methods {off_meth}")
    found = sorted({int(b) for m in meta.values() for b in m.budget.unique()})
    return Check("budgets", not bad,
                 f"budgets {found} within {list(C.BUDGETS)}; mask types and methods known"
                 if not bad else "; ".join(bad))


def check_seeds_match(baselines: dict, meta: dict) -> Check:
    bad, lines = [], []
    for ds in sorted(set(baselines) | set(meta)):
        sb = sorted(int(s) for s in baselines[ds].seed.unique()) if ds in baselines else None
        sm = sorted(int(s) for s in meta[ds].seed.unique()) if ds in meta else None
        lines.append(f"{ds}: baselines {sb}, meta {sm}")
        if sb != sm:
            bad.append(ds)
    return Check("seeds_match", not bad, "\n".join(lines))


def check_split_hashes(split_hashes: pd.DataFrame, datasets) -> Check:
    bad = []
    for ds in datasets:
        g = split_hashes[split_hashes.dataset == ds]
        seeds = sorted(int(s) for s in g.seed)
        empty = g["sha_all"].isna() | (g["sha_all"].astype(str).str.len() == 0)
        if seeds != list(C.SEEDS) or empty.any():
            bad.append(f"{ds}: seeds {seeds}, {int(empty.sum())} empty hash(es)")
    return Check("split_hashes", not bad,
                 f"seeds 0-9 hashed for {list(datasets)}" if not bad else "; ".join(bad))


def check_families(meta: dict, stated: int | None) -> Check:
    fams = sorted({(ds, int(k), int(b)) for ds, m in meta.items()
                   for k, b in m[["k", "budget"]].drop_duplicates().itertuples(index=False)})
    per = {ds: sum(1 for f in fams if f[0] == ds) for ds in meta}
    return Check("families", stated is not None and len(fams) == stated,
                 f"{len(fams)} dataset x k x budget families in the data "
                 f"({', '.join(f'{ds} {n}' for ds, n in per.items())}); PREREG.md states {stated}")


def git_hashes(meta: dict) -> Check:
    lines = []
    for ds, m in meta.items():
        counts = m.git_hash.value_counts()
        lines.append(f"{ds}: " + ", ".join(f"{h} ({n} rows)" for h, n in counts.items()))
    return Check("git_hashes", True,
                 "\n".join(lines) + "\n'+dirty' on CIFAR comes only from the results/runs.json "
                 "append between runs (LOG.md, Day 5); not an error.", info=True)


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------

def run_checks(baselines: dict, meta: dict, split_hashes: pd.DataFrame,
               stated_families: int | None) -> list[Check]:
    return [
        check_row_counts(baselines, meta),
        check_seeds_complete(baselines, meta),
        check_no_nan_acc(baselines, meta),
        check_evals_budget(meta),
        check_idx_length(baselines, meta),
        check_config_hash(meta),
        check_budgets(meta),
        check_seeds_match(baselines, meta),
        check_split_hashes(split_hashes, sorted(set(baselines) | set(meta))),
        check_families(meta, stated_families),
        git_hashes(meta),
    ]


def write_report(checks: list[Check], baselines: dict, meta: dict, path) -> None:
    n_fail = sum(not c.ok for c in checks)
    lines = ["# Integrity report (Stage A0)", "",
             f"- Generated: {datetime.now().isoformat(timespec='seconds')} by "
             "`python -m experiments.analyze --only A0`",
             "- Inputs: " + ", ".join(
                 [f"`results/baselines_{d}.csv`" for d in baselines]
                 + [f"`results/meta_{d}.csv`" for d in meta]
                 + ["`results/split_hashes.csv`", "`PREREG.md`"]),
             f"- Result: **{'ALL CHECKS PASSED' if n_fail == 0 else f'{n_fail} CHECK(S) FAILED'}**",
             "", "| check | status | detail |", "|---|---|---|"]
    for c in checks:
        status = "INFO" if c.info else ("PASS" if c.ok else "FAIL")
        lines.append(f"| {c.name} | {status} | {c.detail.replace(chr(10), '<br>')} |")
    lines += ["", "## k lists (from the baseline files)", ""]
    for ds, b in baselines.items():
        lines.append(f"- {C.DATASET_DISPLAY.get(ds, ds)}: d = {C.dimension(b)}, selection "
                     f"k = {C.selection_ks(b)}; baseline methods "
                     f"{sorted(b.method.unique())}")
    for ds, m in meta.items():
        lines.append(f"- meta_{ds}: methods {sorted(m.method.unique())}, budgets "
                     f"{sorted(int(x) for x in m.budget.unique())}, k {sorted(int(x) for x in m.k.unique())}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


REPORT = C.ANALYSIS_DIR / "integrity_report.md"


def validate(datasets=C.DATASETS, report_path=REPORT, raise_on_fail=True) -> list[Check]:
    baselines = {ds: C.load_baselines(ds) for ds in datasets}
    meta = {ds: C.load_meta(ds) for ds in datasets}
    checks = run_checks(baselines, meta, C.load_split_hashes(), C.prereg_family_count())
    write_report(checks, baselines, meta, report_path)
    failed = [c.name for c in checks if not c.ok]
    if failed and raise_on_fail:
        raise AssertionError(f"integrity checks failed: {failed} (see {report_path})")
    return checks


def run() -> list[Check]:
    checks = validate(raise_on_fail=False)
    for c in checks:
        status = "INFO" if c.info else ("PASS" if c.ok else "FAIL")
        print(f"[{status}] {c.name}: {c.detail}")
    print(f"\nWrote {REPORT}")
    failed = [c.name for c in checks if not c.ok]
    if failed:
        raise AssertionError(f"integrity checks failed: {failed}; later stages must not run")
    return checks
