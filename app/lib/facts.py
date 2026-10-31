"""Facts computed from the data layer (DASHBOARD.md rule 2: nothing typed).

Pure functions take dataframes or text so tests can feed known inputs; the
thin wrappers at the bottom read through app.lib.data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from app.lib import data

LABELS = ("WIN", "LOSS", "TIE")
_OK_STATUSES = {"PASS", "INFO"}


@dataclass(frozen=True)
class IntegrityCheck:
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class IntegrityStatus:
    overall: str                      # PASS, FAIL or MISSING
    result_line: str
    generated: str
    checks: tuple[IntegrityCheck, ...]

    def count(self, status: str) -> int:
        return sum(c.status == status for c in self.checks)

    @property
    def failed(self) -> list[str]:
        return [c.name for c in self.checks if c.status not in _OK_STATUSES]


def parse_integrity(text: str) -> IntegrityStatus:
    """Parse results/analysis/integrity_report.md.

    PASS only if the report states a passing result, has at least one check,
    and every check is PASS or INFO.
    """
    result = re.search(r"^- Result:\s*\*\*(.+?)\*\*", text, re.M)
    generated = re.search(r"^- Generated:\s*(\S+)", text, re.M)
    checks = []
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not line.startswith("|") or len(cells) != 3:
            continue
        if cells[0] == "check" or set(cells[0]) <= {"-", ":"}:
            continue
        checks.append(IntegrityCheck(cells[0], cells[1].upper(),
                                     cells[2].replace("<br>", "; ")))
    result_line = result.group(1) if result else ""
    ok = (bool(checks) and "PASS" in result_line.upper()
          and "FAIL" not in result_line.upper()
          and all(c.status in _OK_STATUSES for c in checks))
    return IntegrityStatus("PASS" if ok else "FAIL", result_line,
                           generated.group(1) if generated else "unknown",
                           tuple(checks))


def rq1_label_counts(rq1: pd.DataFrame) -> dict[str, int]:
    """WIN / LOSS / TIE counts over the pre-registered RQ1 tests."""
    counts = rq1["label"].value_counts()
    return {lab: int(counts.get(lab, 0)) for lab in LABELS}


def rq1_shape(rq1: pd.DataFrame) -> dict[str, int]:
    """Number of tests, families (dataset x k x budget) and comparators."""
    return {"tests": len(rq1),
            "families": int(rq1[["dataset", "k", "budget"]].drop_duplicates().shape[0]),
            "comparators": int(rq1["comparator"].nunique())}


def min_two_sided_p(n: int) -> float:
    """Smallest two-sided exact Wilcoxon signed-rank p with n non-zero pairs."""
    return 2.0 / 2 ** n


def rq1_verdict(rq1: pd.DataFrame) -> list[str]:
    """Verdict sentences, every number taken from rq1_tests.csv."""
    from app.lib.methods import display  # noqa: PLC0415
    from app.lib.stats_view import format_p  # noqa: PLC0415

    c = rq1_label_counts(rq1)
    shape = rq1_shape(rq1)
    n = int(rq1["n"].max())
    lines = [
        f"Of {shape['tests']} pre-registered comparisons (plain Firefly against "
        f"{shape['comparators']} comparators in {shape['families']} dataset x k x budget "
        f"families), {c['WIN']} are WIN, {c['LOSS']} LOSS and {c['TIE']} TIE "
        f"(paired Wilcoxon over n={n} seeds, Holm within each family, adjusted p < 0.05)."]
    wins = rq1[rq1["label"] == "WIN"]
    if len(wins):
        lines.append("WIN: " + "; ".join(
            f"{r.dataset} k={r.k} budget {r.budget} vs {display(r.comparator)} "
            f"({r.mean_diff_pts:+.2f} pts, adjusted p {format_p(r.p_holm)})"
            for r in wins.itertuples()) + ".")
    losses = rq1[rq1["label"] == "LOSS"]["comparator"].value_counts()
    if len(losses):
        lines.append("LOSS by comparator: " + ", ".join(
            f"{display(m)} {v}" for m, v in losses.items()) + ".")
    lines.append(
        f"With n={n} the smallest possible two-sided p is {min_two_sided_p(n):.4f}, so a "
        "label needs nearly every seed to agree; TIE means no detectable difference, "
        "not equivalence.")
    return lines


def top_method_per_k(desc: pd.DataFrame, dataset: str, budget: int) -> pd.DataFrame:
    """Highest-mean method at each k (baselines plus raw meta rows at `budget`),
    excluding 'all' (it has k = d)."""
    d = desc[(desc["dataset"] == dataset) & (desc["method"] != "all")
             & desc["metric"].isin(["acc", "acc_lr"])]
    d = d[d["budget"].isna() | (d["budget"] == budget)]
    d = d[d["source"].str.contains("raw") | d["source"].str.startswith("baseline")]
    idx = d.groupby("k")["mean"].idxmax()
    return d.loc[idx, ["k", "method", "mean", "std"]].sort_values("k")


def integrity_status() -> IntegrityStatus:
    text = data.read_text(data.INTEGRITY)
    if text is None:
        return IntegrityStatus("MISSING", "", "unknown", ())
    return parse_integrity(text)


def asset_facts() -> dict:
    """app/assets/facts.json (built by experiments/build_app_assets.py), or {}."""
    import json  # noqa: PLC0415

    text = data.read_text(data.ASSETS / "facts.json")
    return json.loads(text) if text else {}


def contention_factor() -> float | None:
    """A4's load-inflation factor (logged search time / idle search time)."""
    text = data.read_text(data.ANALYSIS / "contention_factor.md")
    m = re.search(r"contention_factor\s*=\s*([0-9.]+)", text or "")
    return float(m.group(1)) if m else None


def raw_row_counts() -> dict[str, int]:
    """Rows in the raw result files, summed over datasets."""
    return {kind: sum(len(data.load_raw(kind, ds)) for ds in data.DATASETS)
            for kind in ("baselines", "meta")}
