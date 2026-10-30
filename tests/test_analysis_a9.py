"""Tests for Stage A9 (a9_manifest.py, a9_rebuild.py, a8 skip) on a temporary
repo tree, plus registry coverage of the real outputs."""

from __future__ import annotations

import hashlib

import pandas as pd

from experiments.analysis import a8_convergence as A8
from experiments.analysis import a9_manifest as M
from experiments.analysis import a9_rebuild as R
from experiments.analysis import common as C

REG = {
    "S1": dict(outputs=["results/analysis/t.csv", "figures/F1_*"],
               inputs=["results/in.csv"], command="cmd S1"),
    "S2": dict(outputs=["results/analysis/sep.csv"], inputs=[], command="cmd S2",
               separate=True),
    "A9": dict(outputs=["results/analysis/analysis_manifest.md"], inputs=[], command="cmd"),
}


def _tree(tmp_path):
    (tmp_path / "results" / "analysis").mkdir(parents=True)
    (tmp_path / "figures").mkdir()
    pd.DataFrame(dict(a=[1, 2, 3])).to_csv(tmp_path / "results" / "in.csv", index=False)
    pd.DataFrame(dict(x=[0.1, 0.2])).to_csv(tmp_path / "results/analysis/t.csv", index=False)
    pd.DataFrame(dict(y=[1])).to_csv(tmp_path / "results/analysis/sep.csv", index=False)
    (tmp_path / "figures" / "F1_a.png").write_bytes(b"png")
    return tmp_path


def test_unregistered_detects_stray_file(tmp_path):
    root = _tree(tmp_path)
    assert M.unregistered(root, REG) == []
    (root / "figures" / "stray.png").write_bytes(b"x")
    assert [f.name for f in M.unregistered(root, REG)] == ["stray.png"]


def test_analyze_owned_excludes_separate_stages(tmp_path):
    root = _tree(tmp_path)
    names = sorted(f.name for f in M.analyze_owned(root, REG))
    assert names == ["F1_a.png", "t.csv"]                     # not sep.csv (separate stage)


def test_manifest_lists_outputs_sources_hashes_rows(tmp_path):
    root = _tree(tmp_path)
    path = M.write_manifest(root=root, registry=REG, notes={"S1": "a note"})
    text = path.read_text(encoding="utf-8")
    src = root / "results" / "in.csv"
    assert f"| `results/in.csv` | 3 | `{hashlib.sha256(src.read_bytes()).hexdigest()}` |" in text
    assert "results/analysis/t.csv" in text and "figures/F1_a.png" in text
    assert "results/analysis/sep.csv" in text and "cmd S2" in text and "a note" in text
    assert "- git:" in text and "- Date:" in text


def test_compare_file(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    pd.DataFrame(dict(x=[1.0, 2.0])).to_csv(a, index=False)
    pd.DataFrame(dict(x=[1.0, 2.0])).to_csv(b, index=False)
    assert R.compare_file(a, b)[0] == "PASS"
    pd.DataFrame(dict(x=[1.0, 2.0000001])).to_csv(b, index=False)
    assert R.compare_file(a, b)[0] == "FAIL"
    assert R.compare_file(a, tmp_path / "missing.csv")[0] == "FAIL"
    m1, m2 = tmp_path / "a.md", tmp_path / "b.md"
    m1.write_text("# R\n- Generated: 2026-10-30T10:00\nbody\n", encoding="utf-8")
    m2.write_text("# R\n- Generated: 2026-10-31T11:00\nbody\n", encoding="utf-8")
    assert R.compare_file(m1, m2)[0] == "PASS"
    m2.write_text("# R\n- Generated: x\nchanged\n", encoding="utf-8")
    assert R.compare_file(m1, m2)[0] == "FAIL"


def test_rebuild_check_moves_not_deletes(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    monkeypatch.setattr(M, "REGISTRY", REG)
    orig = M.analyze_owned
    monkeypatch.setattr(M, "analyze_owned", lambda r=root, reg=REG: orig(r, reg))

    def rebuild():                                            # deterministic "stage"
        pd.DataFrame(dict(x=[0.1, 0.2])).to_csv(root / "results/analysis/t.csv", index=False)
        (root / "figures" / "F1_a.png").write_bytes(b"png")

    assert R.rebuild_check(rebuild, root=root)
    snaps = list((root / "logs").glob("rebuild_snapshot_*"))
    assert len(snaps) == 1 and (snaps[0] / "results/analysis/t.csv").exists()
    assert (root / "results/analysis/sep.csv").exists()       # separate output untouched

    def broken():
        pd.DataFrame(dict(x=[0.1, 0.3])).to_csv(root / "results/analysis/t.csv", index=False)
        (root / "figures" / "F1_a.png").write_bytes(b"png")

    assert not R.rebuild_check(broken, root=root)


def test_a8_skips_without_traces(tmp_path, monkeypatch):
    assert not A8.traces_present(tmp_path)
    (tmp_path / "x.npz").write_bytes(b"")
    assert A8.traces_present(tmp_path)


def test_real_outputs_all_registered():
    """Every file currently in figures/ and results/analysis/ belongs to a stage."""
    stray = M.unregistered(C.ROOT)
    assert stray == [], [str(f) for f in stray]
    separate = {s for s, spec in M.REGISTRY.items() if spec.get("separate")}
    assert separate == {"A4", "A6a"}
