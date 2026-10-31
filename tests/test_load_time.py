"""Performance budgets (DASHBOARD.md rule 8), measured with Streamlit's AppTest.

cold start: a fresh Python process imports Streamlit, runs the entry script
            and renders Home; must take under 3 s (timed inside the process,
            so interpreter start-up is excluded)
page switch: after every page has been opened once (caches warm), opening
            each page again must take under 1 s
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from app.lib.navigation import PAGE_SPECS  # noqa: E402

COLD_BUDGET_S = 3.0
SWITCH_BUDGET_S = 1.0


def test_cold_start_to_first_page():
    script = textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {str(ROOT)!r})
        t0 = time.perf_counter()
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file({str(APP / 'streamlit_app.py')!r}, default_timeout=60).run()
        dt = time.perf_counter() - t0
        assert not at.exception, [e.value for e in at.exception]
        print(f"{{dt:.3f}}")
    """)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         cwd=ROOT, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    seconds = float(out.stdout.strip().splitlines()[-1])
    print(f"cold start to Home: {seconds:.2f} s (budget {COLD_BUDGET_S} s)")
    assert seconds < COLD_BUDGET_S


def test_page_switch_after_first_load():
    for spec in PAGE_SPECS:                         # first visit: fill the caches
        AppTest.from_file(str(APP / "pages" / spec.file), default_timeout=60).run()
    times = {}
    for spec in PAGE_SPECS:
        t0 = time.perf_counter()
        at = AppTest.from_file(str(APP / "pages" / spec.file), default_timeout=60).run()
        times[spec.file] = time.perf_counter() - t0
        assert not at.exception, spec.file
    print("warm page times: " + ", ".join(f"{k} {v:.2f} s" for k, v in times.items()))
    slow = {k: round(v, 2) for k, v in times.items() if v >= SWITCH_BUDGET_S}
    assert not slow, f"pages over {SWITCH_BUDGET_S} s after first load: {slow}"
