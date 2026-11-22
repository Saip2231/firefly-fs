"""Animated protocol diagram (Protocol page): every number in the payload
equals a plain pandas recomputation from the result files, the payload is
valid JSON, injection is safe, and the bundle makes no external requests
(DASHBOARD.md rules 2 and 8)."""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("streamlit")
from app.lib import pipeline_view as pv  # noqa: E402

ANALYSIS = ROOT / "results" / "analysis"
FRONTEND = ROOT / "app" / "frontend"


@pytest.fixture(scope="module")
def payload():
    return pv.build_payload()


def _cost():
    c = pd.read_csv(ANALYSIS / "rq2_cost.csv")
    return c[c["mask_type"].isin(["raw", "-"])]


def test_payload_is_strict_json(payload):
    json.dumps(payload, allow_nan=False)
    assert set(payload["datasets"]) == {"digits", "cifar_r18"}


def test_accuracy_time_and_fits_equal_rq2_cost(payload):
    cost = _cost()
    for ds, block in payload["datasets"].items():
        c = cost[cost["dataset"] == ds]
        sel = c[c["method"].isin(pv.BASELINES + pv.METAS)]
        assert block["k_values"] == sorted(int(k) for k in sel["k"].unique())
        assert block["d"] == int(c.loc[c["method"] == "all", "k"].iloc[0])
        assert len(block["results"]) == len(sel)
        for r in sel.itertuples():
            budget = "" if pd.isna(r.budget) else int(r.budget)
            cell = block["results"][f"{r.method}|{int(r.k)}|{budget}"]
            assert cell["n"] == r.n
            assert math.isclose(cell["acc_mean"], r.acc_mean, rel_tol=1e-12)
            assert math.isclose(cell["acc_std"], r.acc_std, rel_tol=1e-12)
            assert math.isclose(cell["search_s_mean"], r.search_s_mean, rel_tol=1e-12)
        allr = c[c["method"] == "all"].iloc[0]
        assert math.isclose(block["all_features"]["acc_mean"], allr["acc_mean"], rel_tol=1e-12)


def test_inner_val_error_equals_pandas(payload):
    w = pv.fitness_weight()
    for ds, block in payload["datasets"].items():
        meta = pd.read_csv(ROOT / "results" / f"meta_{ds}.csv")
        meta = meta[(meta["mask_type"] == "raw") & meta["method"].isin(pv.METAS)]
        d = block["d"]
        for (method, k, budget), g in meta.groupby(["method", "k", "budget"]):
            expected = ((g["val_fitness"] - w * k / d) / (1 - w)).mean()
            cell = block["results"][f"{method}|{int(k)}|{int(budget)}"]
            assert math.isclose(cell["inner_val_error_mean"], expected, rel_tol=1e-9)
            assert 0.0 <= cell["inner_val_error_mean"] <= 1.0


def test_inject_escapes_and_requires_placeholder():
    html = '<script id="pipeline-data" type="application/json">__PIPELINE_DATA__</script>'
    out = pv.inject(html, {"x": "</script><b>"})
    assert "__PIPELINE_DATA__" not in out and "</script><b>" not in out
    assert out.count("</script>") == 1
    with pytest.raises(pv.data.DataError):
        pv.inject("<html></html>", {})


def test_frontend_source_has_placeholder_and_no_external_urls():
    index = (FRONTEND / "index.html").read_text(encoding="utf-8")
    assert pv.PLACEHOLDER in index
    for p in [FRONTEND / "index.html", *sorted((FRONTEND / "src").glob("*.ts*"))]:
        text = p.read_text(encoding="utf-8")
        assert not re.search(r"https?://", text), p
        assert not re.search(r"\b(Xte|yte)\b", text), p


def test_frontend_tokens_cover_both_themes():
    """The visualizer's CSS defines every role variable in light and dark, and
    the surface colours match app/lib/theme.py."""
    from app.lib import theme  # noqa: PLC0415
    css = (FRONTEND / "src" / "index.css").read_text(encoding="utf-8")
    light = css.split(":root {", 1)[1].split("}", 1)[0]
    dark = css.split(".dark {", 1)[1].split("}", 1)[0]
    for var in ("--bg", "--surface", "--text", "--muted", "--accent", "--flow", "--search",
                "--lock", "--on-accent"):
        assert f"{var}:" in light and f"{var}:" in dark, var
    for mode, block in (("light", light), ("dark", dark)):
        t = theme.TOKENS[mode]
        assert f"--bg: {t['background']};" in block
        assert f"--surface: {t['card']};" in block
        assert f"--text: {t['text']};" in block


def test_built_bundle_is_self_contained():
    if not pv.BUNDLE.exists():
        pytest.skip("bundle not built (npm run build in app/frontend)")
    html = pv.BUNDLE.read_text(encoding="utf-8")
    assert pv.PLACEHOLDER in html
    assert not re.search(r"<script[^>]+src=", html), "external script in bundle"
    assert not re.search(r"<link[^>]+href=", html), "external stylesheet in bundle"
    assert pv.BUNDLE.stat().st_size < 2_000_000
