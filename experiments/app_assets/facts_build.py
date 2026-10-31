"""facts.json (B1 item 1): counts and provenance computed from the files.

Everything here is read from results/, LOG.md or the built assets; nothing
is typed in.  Written last so it can record every asset's SHA-256.
"""

from __future__ import annotations

import json
import re
from datetime import datetime

import numpy as np
import pandas as pd

from experiments.analysis import common as C

from .common import (ANALYSIS, ASSETS, ASSETS_MAX_BYTES, RESULTS, ROOT, TRACE_MAX_BYTES,
                     TRACES, raw_features, rel, sha256, train_indices)


def split_rows() -> dict:
    """Train / test rows per dataset (identical for every seed: stratified 80/20)."""
    out = {}
    for ds in C.DATASETS:
        _, y = raw_features(ds)
        n_train = len(train_indices(y, 0))
        out[ds] = dict(total=len(y), train=n_train, test=len(y) - n_train)
    return out

FACTS = ASSETS / "facts.json"


def last_pytest_count(log_text: str) -> int | None:
    """The last 'N passed' in LOG.md (the most recent recorded pytest run)."""
    hits = re.findall(r"(\d+) passed", log_text)
    return int(hits[-1]) if hits else None


def contention_factor(text: str) -> float | None:
    m = re.search(r"contention_factor\s*=\s*([0-9.]+)", text)
    return float(m.group(1)) if m else None


def trace_summaries() -> list[dict]:
    out = []
    for path in sorted(TRACES.glob("*.npz")):
        with np.load(path) as z:
            out.append(dict(
                file=rel(path), dataset=str(z["dataset"]), method=str(z["method"]),
                k=int(z["k"]), budget=int(z["budget"]), seed=int(z["seed"]),
                frames=int(len(z["frame_index"])), frames_total=int(z["frames_total"]),
                evals_used=int(z["evals_used"]), match=bool(z["match"]),
                bytes=path.stat().st_size))
    return out


def _npz_summary(path, fields: tuple[str, ...]) -> dict | None:
    if not path.exists():
        return None
    with np.load(path) as z:
        return {f: z[f].tolist() for f in fields if f in z.files}


def build(build_git: str, config_hash: str, blocked: dict, environment: dict) -> dict:
    meta = {ds: pd.read_csv(RESULTS / f"meta_{ds}.csv", dtype={"idx": str})
            for ds in C.DATASETS}
    base = {ds: C.load_baselines(ds) for ds in C.DATASETS}
    rq1 = pd.read_csv(ANALYSIS / "rq1_tests.csv")
    all_meta = pd.concat(meta.values())
    hashes = sorted(all_meta.config_hash.unique())
    labels = rq1.label.value_counts()

    traces = trace_summaries()
    assets = {rel(p): dict(bytes=p.stat().st_size, sha256=sha256(p))
              for p in sorted(ASSETS.rglob("*")) if p.is_file() and p != FACTS}
    total = sum(a["bytes"] for a in assets.values())
    log_text = (ROOT / "LOG.md").read_text(encoding="utf-8")
    cf_path = ANALYSIS / "contention_factor.md"

    facts = dict(
        generated=datetime.now().isoformat(timespec="seconds"),
        command="python experiments/build_app_assets.py",
        build_git_hash=build_git,
        environment=environment,
        rows=dict(meta={ds: len(df) for ds, df in meta.items()},
                  meta_total=len(all_meta),
                  meta_raw=int((all_meta.mask_type == "raw").sum()),
                  baselines={ds: len(df) for ds, df in base.items()}),
        split_rows=split_rows(),
        seeds=sorted(int(s) for s in all_meta.seed.unique()),
        budgets=sorted(int(b) for b in all_meta.budget.unique()),
        k={ds: C.selection_ks(base[ds]) for ds in C.DATASETS},
        d={ds: C.dimension(base[ds]) for ds in C.DATASETS},
        methods=dict(meta=sorted(all_meta.method.unique()),
                     baselines=sorted(pd.concat(base.values()).method.unique())),
        rq1=dict(tests=len(rq1),
                 families=int(rq1[["dataset", "k", "budget"]].drop_duplicates().shape[0]),
                 comparators=sorted(rq1.comparator.unique()),
                 labels={lab: int(labels.get(lab, 0)) for lab in ("WIN", "LOSS", "TIE")}),
        config_hash=dict(results=hashes, frozen_config=config_hash,
                         consistent=hashes == [config_hash]),
        git_hashes={ds: sorted(df.git_hash.unique()) for ds, df in meta.items()},
        pytest=dict(passed=last_pytest_count(log_text), source="LOG.md (last 'N passed')"),
        load_inflation_factor=dict(
            value=contention_factor(cf_path.read_text(encoding="utf-8"))
            if cf_path.exists() else None,
            source="results/analysis/contention_factor.md (A4)"),
        traces=traces,
        demo_models=_npz_summary(ASSETS / "demo_models.npz",
                                 ("dataset", "seed", "budget", "model_keys")),
        digits=_npz_summary(ASSETS / "digits.npz", ("seed", "holdout_frac")),
        blocked=blocked,
        budgets_check=dict(
            trace_max_bytes=TRACE_MAX_BYTES, assets_max_bytes=ASSETS_MAX_BYTES,
            assets_total_bytes=total,
            ok=total <= ASSETS_MAX_BYTES and all(t["bytes"] <= TRACE_MAX_BYTES for t in traces)),
        assets=assets,
    )
    FACTS.write_text(json.dumps(facts, indent=2), encoding="utf-8")
    return facts
