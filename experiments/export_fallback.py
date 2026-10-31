"""DASHBOARD.md B9: static fallback of the key results (no Streamlit needed).

Writes fallback/index.html: one self-contained page (figures embedded as
base64) with the integrity status, the computed RQ1 verdict, and every
analysis figure from figures/ with its computed takeaway from
figures/takeaways.csv.  Reads figures/ and results/ only; writes only
fallback/.

Usage (from firefly-fs/, venv active):
  python experiments/export_fallback.py
"""

from __future__ import annotations

import base64
import html
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.lib import facts  # noqa: E402

FIGURES = ROOT / "figures"
OUT = ROOT / "fallback" / "index.html"

CSS = """
:root { --bg:#0E1117; --card:#161B22; --border:#30363D; --text:#E6EDF3; --muted:#9DA7B3;
        --accent:#E69F00; }
@media (prefers-color-scheme: light) {
  :root { --bg:#FFFFFF; --card:#F6F8FA; --border:#D0D7DE; --text:#1F2328; --muted:#57606A; } }
body { background:var(--bg); color:var(--text); margin:0; font-size:16px; line-height:1.5;
       font-family:system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif; }
main { max-width:1000px; margin:0 auto; padding:24px 16px; }
h1 { font-size:31px; } h2 { font-size:25px; margin-top:40px; }
.card { background:var(--card); border:1px solid var(--border); border-radius:8px;
        padding:16px; margin:16px 0; }
figure { margin:24px 0; } img { max-width:100%; height:auto; border-radius:6px; background:#fff; }
figcaption { color:var(--muted); margin-top:8px; } code { color:var(--accent); }
"""


def figure_html(name: str, takeaway: str) -> str:
    png = FIGURES / f"{name}.png"
    if not png.exists():
        return ""
    b64 = base64.b64encode(png.read_bytes()).decode("ascii")
    alt = html.escape(f"{name}: {takeaway}")
    return (f'<figure><img src="data:image/png;base64,{b64}" alt="{alt}">'
            f"<figcaption><strong>{html.escape(name)}.</strong> "
            f"{html.escape(takeaway)}</figcaption></figure>")


def main() -> int:
    rq1 = pd.read_csv(ROOT / "results" / "analysis" / "rq1_tests.csv")
    takeaways = pd.read_csv(FIGURES / "takeaways.csv").sort_values("figure")
    status = facts.integrity_status()
    verdict = "".join(f"<p>{html.escape(line)}</p>" for line in facts.rq1_verdict(rq1))
    figs = "".join(figure_html(r.figure, r.takeaway) for r in takeaways.itertuples())
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Firefly benchmark results</title><style>{CSS}</style></head>
<body><main>
<h1>Firefly feature selection: benchmark results (static fallback)</h1>
<p>Static copy of the dashboard's key results, generated
{datetime.now().isoformat(timespec="minutes")} by <code>python experiments/export_fallback.py</code>
from <code>results/</code> and <code>figures/</code>. The interactive dashboard is
<code>streamlit run app/streamlit_app.py</code>.</p>
<div class="card"><strong>Integrity: {html.escape(status.overall)}</strong>
({status.count("PASS")} of {len(status.checks)} checks passed, {status.count("INFO")}
informational; results/analysis/integrity_report.md)</div>
<h2>Verdict (RQ1, confirmatory)</h2><div class="card">{verdict}</div>
<h2>Figures</h2>
<p>Exploratory figures (F3 to F9) are not corrected across families. PCA is a projection,
not a selector. Search times were measured under 15-way parallel load.</p>
{figs}
</main></body></html>
"""
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    n = page.count("<figure>")
    print(f"fallback: {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.1f} MB, {n} figures, "
          f"integrity {status.overall})")
    return 0 if n and status.overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
