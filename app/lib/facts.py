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


def integrity_status() -> IntegrityStatus:
    text = data.read_text(data.INTEGRITY)
    if text is None:
        return IntegrityStatus("MISSING", "", "unknown", ())
    return parse_integrity(text)


def raw_row_counts() -> dict[str, int]:
    """Rows in the raw result files, summed over datasets."""
    return {kind: sum(len(data.load_raw(kind, ds)) for ds in data.DATASETS)
            for kind in ("baselines", "meta")}
