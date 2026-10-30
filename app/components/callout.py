"""Callouts: integrity banner, stage placeholder, data-error message."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.lib.data import DataError
from app.lib.facts import IntegrityStatus

_SOURCE = "results/analysis/integrity_report.md"


def integrity_banner(status: IntegrityStatus) -> None:
    """PASS / FAIL / MISSING in words (never colour alone), plus the checks."""
    if status.overall == "MISSING":
        st.warning(f"Integrity report not found ({_SOURCE}). "
                   "Run `python -m experiments.analyze` to create it.",
                   icon=":material/help:")
        return
    n_pass, n_info = status.count("PASS"), status.count("INFO")
    summary = f"{n_pass} of {len(status.checks)} checks passed"
    if n_info:
        summary += f", {n_info} informational"
    if status.overall == "PASS":
        st.success(f"Integrity: PASS. {summary} ({_SOURCE}, generated "
                   f"{status.generated}).", icon=":material/verified:")
    else:
        st.error(f"Integrity: FAIL. {summary}; failing: "
                 f"{', '.join(status.failed) or status.result_line or 'no checks found'} "
                 f"({_SOURCE}).", icon=":material/error:")
    with st.expander("Integrity checks"):
        st.dataframe(pd.DataFrame([vars(c) for c in status.checks]), hide_index=True)


def placeholder(stage: str, what: str) -> None:
    st.info(f"Placeholder: this page is built in Stage {stage} ({what}).",
            icon=":material/construction:")


def data_error(err: DataError) -> None:
    """Show a loader problem as one readable line instead of a stack trace."""
    st.error(f"Data problem: {err}", icon=":material/error:")
