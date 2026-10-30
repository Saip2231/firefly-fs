"""Page registry: one entry per file in app/pages/ (order = sidebar order)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PageSpec:
    file: str
    title: str
    icon: str
    section: str
    url_path: str
    stage: str          # stage of DASHBOARD.md that fills the page


PAGE_SPECS: tuple[PageSpec, ...] = (
    PageSpec("01_home.py", "Home", ":material/home:", "Overview", "home", "B2"),
    PageSpec("02_protocol.py", "Protocol", ":material/account_tree:", "Overview",
             "protocol", "B2"),
    PageSpec("03_results.py", "Results explorer (RQ1)", ":material/grid_on:",
             "Results", "results", "B3"),
    PageSpec("04_cost.py", "Cost (RQ2)", ":material/timer:", "Results", "cost", "B4"),
    PageSpec("05_stability.py", "Stability (RQ3)", ":material/stacked_bar_chart:",
             "Results", "stability", "B4"),
    PageSpec("06_transfer_ablations.py", "Transfer and ablations",
             ":material/swap_horiz:", "Results", "transfer-ablations", "B5"),
    PageSpec("07_watch_search.py", "Watch a real search", ":material/play_circle:",
             "Demos", "watch", "B6"),
    PageSpec("08_try_image.py", "Try an image", ":material/image:", "Playground",
             "try-image", "B7"),
    PageSpec("09_digits_live.py", "Digits live and build your own", ":material/tune:",
             "Playground", "digits-live", "B8"),
    PageSpec("10_limits.py", "Limits", ":material/info:", "About", "limits", "B2"),
)
