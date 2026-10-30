"""Stage B0 tests for the dashboard (app/, DASHBOARD.md rule 11).

(a) every page opens under Streamlit's AppTest; the entry app runs and Home
    shows the integrity banner with PASS
(b) page registry matches app/pages/
(c) data layer: manifest discovery, schema validation, clear errors
(d) facts equal a plain pandas recomputation
(e) static rules: no test-split names under app/, no file over 400 lines
(f) theme tokens and contrast
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from app.lib import data, facts, stats_view, theme  # noqa: E402
from app.lib.navigation import PAGE_SPECS  # noqa: E402

TIMEOUT = 30
TEXT_SUFFIXES = {".py", ".md", ".toml", ".txt", ".json", ".css", ".html", ".js"}


def _app_text_files():
    return [p for p in APP.rglob("*")
            if p.is_file() and p.suffix in TEXT_SUFFIXES and "__pycache__" not in p.parts]


# (a) -----------------------------------------------------------------------

def test_entry_app_runs_and_home_shows_integrity_pass():
    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=TIMEOUT).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value == "Home"
    assert any(s.value.startswith("Integrity: PASS") for s in at.success)


@pytest.mark.parametrize("spec", PAGE_SPECS, ids=lambda s: s.file)
def test_every_page_opens(spec):
    at = AppTest.from_file(str(APP / "pages" / spec.file), default_timeout=TIMEOUT).run()
    assert not at.exception, [e.value for e in at.exception]
    assert [t.value for t in at.title] == [spec.title]
    assert at.markdown, "page has no description"


# (b) -----------------------------------------------------------------------

def test_page_registry_matches_files():
    files = sorted(p.name for p in (APP / "pages").glob("*.py"))
    assert sorted(s.file for s in PAGE_SPECS) == files
    assert len({s.url_path for s in PAGE_SPECS}) == len(PAGE_SPECS)


# (c) -----------------------------------------------------------------------

def test_manifest_discovery_finds_analysis_tables():
    outputs = data.manifest_outputs()
    assert "results/analysis/rq1_tests.csv" in outputs
    tables = data.discover_tables()
    for name in data.ANALYSIS_SCHEMAS:
        assert name in tables, name


def test_all_schema_tables_load_and_validate():
    for name in data.ANALYSIS_SCHEMAS:
        df = data.load_analysis(name)
        assert len(df) > 0, name


def test_rq1_table_shape():
    rq1 = data.load_analysis("rq1_tests")
    shape = facts.rq1_shape(rq1)
    assert shape["tests"] == 72 == shape["families"] * shape["comparators"]
    assert set(rq1["label"]) <= set(facts.LABELS)


def test_missing_column_gives_clear_error(tmp_path):
    good = data.load_analysis("rq1_tests")
    path = tmp_path / "rq1_tests.csv"
    good.drop(columns=["label", "p_holm"]).to_csv(path, index=False)
    with pytest.raises(data.DataError, match=r"missing required column\(s\) p_holm, label"):
        data.read_csv_checked(path, data.ANALYSIS_SCHEMAS["rq1_tests"])


def test_missing_and_empty_files(tmp_path):
    with pytest.raises(data.DataError, match="file not found"):
        data.read_csv_checked(tmp_path / "nope.csv")
    empty = tmp_path / "empty.csv"
    empty.write_text("a,b\n", encoding="utf-8")
    with pytest.raises(data.DataError, match="no rows"):
        data.read_csv_checked(empty)
    assert data.load_optional("no_such_table") is None
    with pytest.raises(data.DataError, match="not in the analysis manifest"):
        data.load_analysis("no_such_table")


def test_discovery_uses_manifest_and_glob(tmp_path):
    analysis = tmp_path / "results" / "analysis"
    analysis.mkdir(parents=True)
    (analysis / "listed.csv").write_text("x\n1\n", encoding="utf-8")
    (analysis / "globbed.csv").write_text("x\n1\n", encoding="utf-8")
    manifest = analysis / "analysis_manifest.md"
    manifest.write_text("| output | bytes | sha256 |\n|---|---|---|\n"
                        "| `results/analysis/listed.csv` | 4 | `abc` |\n"
                        "| `results/analysis/report.md` | 4 | `def` |\n", encoding="utf-8")
    assert data.manifest_outputs(manifest) == ["results/analysis/listed.csv",
                                               "results/analysis/report.md"]
    tables = data.discover_tables(analysis, manifest, tmp_path)
    assert set(tables) == {"listed", "globbed"}


def test_raw_loaders_validate_and_primary_is_raw_only():
    for ds in data.DATASETS:
        meta = data.load_raw("meta", ds)
        prim = data.primary_meta(ds)
        assert set(prim["mask_type"]) == {"raw"}
        assert len(prim) == (meta["mask_type"] == "raw").sum()
    with pytest.raises(data.DataError, match="unknown dataset"):
        data.load_raw("meta", "mnist")


# (d) -----------------------------------------------------------------------

def test_label_counts_equal_pandas():
    rq1 = pd.read_csv(data.ANALYSIS / "rq1_tests.csv")
    expected = rq1["label"].value_counts().reindex(list(facts.LABELS), fill_value=0)
    counts = facts.rq1_label_counts(data.load_analysis("rq1_tests"))
    assert counts == {k: int(v) for k, v in expected.items()}
    assert sum(counts.values()) == len(rq1)


def test_raw_row_counts_equal_pandas():
    counts = facts.raw_row_counts()
    for kind in ("baselines", "meta"):
        assert counts[kind] == sum(len(pd.read_csv(data.RESULTS / f"{kind}_{ds}.csv"))
                                   for ds in data.DATASETS)


def test_integrity_report_parses_as_pass():
    status = facts.integrity_status()
    assert status.overall == "PASS"
    assert status.checks and not status.failed
    assert status.count("PASS") + status.count("INFO") == len(status.checks)


def test_integrity_parser_detects_failures():
    text = (data.INTEGRITY).read_text(encoding="utf-8")
    broken = text.replace("| seeds_complete | PASS |", "| seeds_complete | FAIL |")
    status = facts.parse_integrity(broken)
    assert status.overall == "FAIL" and status.failed == ["seeds_complete"]
    assert facts.parse_integrity("").overall == "FAIL"
    no_result = re.sub(r"^- Result:.*$", "", text, flags=re.M)
    assert facts.parse_integrity(no_result).overall == "FAIL"


def test_stats_view_formatting():
    assert stats_view.label_text("WIN") == "✓ WIN"
    assert stats_view.label_text("LOSS") == "✗ LOSS"
    assert stats_view.label_text("TIE") == "= TIE"
    assert stats_view.format_p(0.0004) == "<0.001"
    assert stats_view.format_p(0.01171875) == "0.012"
    assert stats_view.format_p(float("nan")) == "n/a"
    assert stats_view.format_points(-1.09) == "-1.09 pts"
    for never in ("never", -3.0, float("inf"), None):
        assert stats_view.format_break_even(never) == "never"
    assert stats_view.format_break_even(36895.1) == "36,895"


# (e) -----------------------------------------------------------------------

def test_no_test_split_names_under_app():
    offenders = [p.relative_to(ROOT).as_posix() for p in _app_text_files()
                 if re.search(r"\b(Xte|yte)\b", p.read_text(encoding="utf-8"))]
    assert not offenders


def test_no_file_over_400_lines():
    long = {p.relative_to(ROOT).as_posix(): n for p in _app_text_files()
            if (n := len(p.read_text(encoding="utf-8").splitlines())) > 400}
    assert not long


def test_no_dataset_loading_under_app():
    for p in _app_text_files():
        text = p.read_text(encoding="utf-8")
        assert "load_dataset" not in text and "src.data" not in text, p


# (f) -----------------------------------------------------------------------

def test_theme_tokens_follow_spec():
    assert theme.METHOD_COLOURS["firefly"] == "#E69F00"
    assert theme.METHOD_COLOURS["pso"] == "#0072B2"
    assert theme.LABEL_STYLE["WIN"]["fill"] == "#009E73"
    assert theme.LABEL_STYLE["LOSS"]["fill"] == "#D55E00"
    assert theme.LABEL_STYLE["TIE"]["fill"] == "#BDBDBD"
    assert theme.method_colour("pca", "light") == theme.SURFACES["light"]["text"]
    assert "16px" in theme.css()


def test_contrast_at_least_4_5():
    assert theme.contrast("#000000", "#FFFFFF") == pytest.approx(21.0)
    for mode, s in theme.SURFACES.items():
        for bg in ("background", "card"):
            assert theme.contrast(s["text"], s[bg]) >= 4.5, (mode, bg)
    for lab, style in theme.LABEL_STYLE.items():
        assert theme.contrast(style["text"], style["fill"]) >= 4.5, lab


def test_plotly_template_builds():
    pytest.importorskip("plotly")
    tpl = theme.plotly_template()
    assert tpl.layout.paper_bgcolor == "rgba(0,0,0,0)"
