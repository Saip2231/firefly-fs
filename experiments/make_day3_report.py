"""Stage 5 - Run the full test suite, then write the Day 3 summary and LOG entry.

1. Runs pytest tests/ -v in-process (src.results_store.capture_pytest_results).
   If any test fails or errors, NOTHING is written.
2. results/day3_summary.md - tables built only from the results/*.csv files,
   plus a "What this does and does not show" section whose numbers are also
   computed from those CSVs.
3. LOG.md - appends one dated entry: commands run, git commit (parent of the
   day3 commit; a commit cannot contain its own hash), numpy / scikit-learn
   versions, seeds and runtimes (from results/runs.json), pytest outcome.
   Refuses to append a second day3 entry for the same date.
   Pass --no-log to regenerate day3_summary.md only (used by day3-fixes F5).
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn

from _common import RESULTS_DIR, ROOT, direction, holm, paired_wilcoxon

from src.results_store import capture_pytest_results, load_all_runs

DAY3_EXPERIMENTS = ["diagnose_d500", "validate_clean_signal", "check_d500_seed1",
                    "rescore_clean_signal", "rerank_ablation", "time_fitness"]
COMMANDS = [
    "python -m pytest tests/test_firefly.py -v -k trace",
    "python experiments/diagnose_d500.py",
    "python experiments/validate_clean_signal.py",
    "python experiments/check_d500_seed1.py",
    "python experiments/rescore_clean_signal.py",
    "python -m pytest tests/ -v",
    "python -m pytest tests/test_rerank.py -v",
    "python experiments/rerank_ablation.py",
    "python experiments/time_fitness.py",
    "python experiments/make_day3_report.py   (runs pytest tests/ -v, writes summary + LOG)",
]
COMMIT_MSG = "day3: firefly validated, diagnostics, rerank, timing"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "nan" if np.isnan(v) else floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def pm(g: pd.Series) -> str:
    return f"{g.mean():.4f} ± {g.std():.4f}"


def git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / name)


PRE_F1_HOLDOUT_CLF = "LogisticRegression(max_iter=100, random_state=seed) (pre-F1 code)"


def holdout_clf_used(experiment: str) -> str:
    """Holdout classifier of the latest logged run of `experiment` (runs.json)."""
    rec = None
    for r in load_all_runs():
        if r.get("dataset") == experiment:
            rec = r
    if rec is None:
        return "no run logged"
    return (rec.get("params", {}).get("holdout_clf")
            or rec.get("classifiers", {}).get("final_clf")
            or PRE_F1_HOLDOUT_CLF)


def stage2_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Firefly / firefly_levy vs random_search on every metric, one Holm family."""
    tests = []
    for (d, ns), gg in df.groupby(["d", "n_search"]):
        rs = gg[gg.method == "random_search"].sort_values("seed")
        for m in ("firefly", "firefly_levy"):
            fm = gg[gg.method == m].sort_values("seed")
            for metric in ("holdout_acc", "sep_hits", "hits", "val_fitness"):
                w = paired_wilcoxon(fm[metric].values, rs[metric].values)
                tests.append(dict(d=d, n_search=ns, method=m, metric=metric,
                                  direction=direction(w["mean_diff"]), mean_diff=w["mean_diff"],
                                  n=w["n"], n_nonzero=w["n_nonzero"], p=w["p"]))
    t = pd.DataFrame(tests)
    t["p_holm"] = holm(t.p.values)
    return t


def stage3_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d, g in df.groupby("d"):
        raw = g[g.variant == "raw"].sort_values("seed")
        rr = g[g.variant == "reranked"].sort_values("seed")
        for metric in ("holdout_acc", "sep_hits", "hits"):
            w = paired_wilcoxon(rr[metric].values, raw[metric].values)
            rows.append(dict(d=d, metric=metric, raw=raw[metric].mean(), reranked=rr[metric].mean(),
                             direction=direction(w["mean_diff"]), mean_diff=w["mean_diff"],
                             n=w["n"], n_nonzero=w["n_nonzero"], p=w["p"],
                             n_changed=int(rr.changed.sum())))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p.values)
    return t


# ---------------------------------------------------------------------------
# summary sections (CSV numbers only)
# ---------------------------------------------------------------------------

def section_stage1():
    df = read("diagnostics_d500.csv")
    g = df.groupby(["config", "method"], sort=False)
    t = g.agg(n=("seed", "count"), evals_used=("evals_used", "mean"), sweeps=("sweeps", "mean"),
              best_init=("best_init", "mean"), best_end=("best_end", "mean"),
              hits=("hits_informative", "mean"), inf_cols_tried=("inf_cols_tried", "mean"),
              mean_inf_per_eval=("mean_inf_per_eval", "mean"),
              diversity_last=("diversity_last", "mean")).reset_index()
    v = (df.groupby(["config", "method"], sort=False)["verdict"]
           .apply(lambda s: ", ".join(f"{k} x{n}" for k, n in s.value_counts().items()))
           .reset_index())
    return ("## Stage 1 - d=500 diagnostics (`diagnostics_d500.csv`, seeds 0-4, means)\n\n"
            + md_table(t) + "\n\nVerdicts:\n\n" + md_table(v))


def section_stage2():
    df = read("validate_clean_signal_rescored.csv")
    t = pd.DataFrame([
        dict(d=d, n_search=ns, method=m, n=len(x), hits=f"{x.hits.mean():.2f}",
             sep_hits=f"{x.sep_hits.mean():.2f}", n_sep=f"{x.n_sep.mean():.2f}",
             sep_recall=pm(x.sep_recall), val_fitness=pm(x.val_fitness),
             holdout_acc=pm(x.holdout_acc), evals_used=f"{x.evals_used.mean():.1f}")
        for (d, ns, m), x in df.groupby(["d", "n_search", "method"])
    ])
    return ("## Stage 2 - clean signal (`validate_clean_signal_rescored.csv`, seeds 0-9)\n\n"
            "class_sep=2.0, 1 cluster/class, k=5, budget=600, 5000-row independent holdout. "
            "`sep_*` count only informative columns whose class means differ "
            "(`separating_columns_clean_signal.csv`). Primary metrics: holdout_acc and sep_hits; "
            "hits and val_fitness are secondary (val_fitness is in-sample: it is the objective "
            "the search minimised on the rows that chose the mask). "
            f"Holdout classifier: {holdout_clf_used('validate_clean_signal')}.\n\n" + md_table(t)
            + "\n\nPaired two-sided Wilcoxon vs random_search (mean_diff = method - random; "
              "zero differences dropped). p_holm: Holm correction across all rows of this table. "
              "Primary test: holdout_acc, firefly, d=500, n_search=2000.\n\n"
            + md_table(stage2_tests(df)))


def section_seed1():
    df = read("check_d500_seed1_freq.csv")
    t = df.groupby("method").agg(
        runs=("algo_seed", "count"), zero_hit_runs=("hits", lambda s: int((s == 0).sum())),
        mean_hits=("hits", "mean"), mean_val_fitness=("val_fitness", "mean"),
        mean_masks_with_inf=("masks_with_inf", "mean"),
        mean_inf_cols_tried=("inf_cols_tried", "mean"),
        runs_not_all_5_tried=("inf_cols_tried", lambda s: int((s < 5).sum()))).reset_index()
    return ("## Stage 2 follow-up - d=500, n_search=1000, data seed 1 "
            "(`check_d500_seed1_freq.csv`, algorithm seeds 0-29)\n\n" + md_table(t))


def section_stage3():
    df = read("rerank_ablation.csv")
    tests = stage3_tests(df)
    # Pre-F2 CSVs name the column search_evals and give rerank_fits on both
    # variants; post-F2 rows give it on the reranked variant only.
    df = df.rename(columns={"search_evals": "evals_used"})
    rr_rows = df[df.variant == "reranked"]
    cost = rr_rows.groupby("d").agg(evals_used=("evals_used", "mean"),
                                    rerank_fits=("rerank_fits", "mean")).reset_index()
    cost["total_fits"] = cost.evals_used + cost.rerank_fits
    if "rerank_s" in rr_rows:
        cost["rerank_s"] = rr_rows.groupby("d")["rerank_s"].mean().values
    return ("## Stage 3 - re-ranking ablation (`rerank_ablation.csv`, seeds 0-9)\n\n"
            "Noisy toy (class_sep=1.0, 2 clusters/class), 1000 search rows, 5000-row holdout, "
            "k=5, budget=800, top_m=10, 5-fold. Ablation: reranked vs raw mask of the SAME "
            "search (never against another method). mean_diff = reranked - raw; p_holm across "
            f"the rows of this table. Holdout classifier: {holdout_clf_used('rerank_ablation')}.\n\n"
            + md_table(tests) + "\n\nCost per run (reranked variant; raw uses evals_used only):\n\n"
            + md_table(cost))


def section_stage4():
    df = read("time_fitness.csv")
    t = df.groupby(["dataset", "k"], sort=False)["seconds"].agg(
        n="count", mean="mean", median="median",
        p95=lambda s: float(np.percentile(s, 95))).reset_index()
    hit = df.groupby(["dataset", "k"], sort=False)["hit_max_iter"].mean().values
    t["share_hit_max_iter"] = hit
    return ("## Stage 4 - fitness timing (`time_fitness.csv`, seconds per NEW evaluation, "
            "single thread, max_rows=2000)\n\n" + md_table(t)
            + "\n\nRe-rank times and wall-time estimates are not stored in a CSV; "
              "see `results/time_fitness.md`.")


def section_does_not_show():
    s2 = read("validate_clean_signal_rescored.csv")
    fq = read("check_d500_seed1_freq.csv")
    ra = read("rerank_ablation.csv")

    t2, t3 = stage2_tests(s2), stage3_tests(ra)

    def recall(d, ns, m):
        return s2[(s2.d == d) & (s2.n_search == ns) & (s2.method == m)].sep_recall.mean()

    def row(t, **kw):
        sel = np.logical_and.reduce([t[c] == v for c, v in kw.items()])
        return t[sel].iloc[0]

    def stat(r):
        return (f"{r.direction} (mean diff {r.mean_diff:+.4f}, p={r.p:.4f}, "
                f"Holm-adjusted p={r.p_holm:.4f}, n={r.n})")

    prim = row(t2, d=500, n_search=2000, method="firefly", metric="holdout_acc")
    vf1 = row(t2, d=500, n_search=1000, method="firefly", metric="val_fitness")
    vf2 = row(t2, d=500, n_search=2000, method="firefly", metric="val_fitness")
    sh50 = row(t2, d=50, n_search=1000, method="firefly", metric="sep_hits")
    zf = int((fq[fq.method == "firefly"].hits == 0).sum())
    zr = int((fq[fq.method == "random_search"].hits == 0).sum())
    h50 = row(t3, d=50, metric="holdout_acc")
    h500 = row(t3, d=500, metric="holdout_acc")
    n_changed = int(ra[ra.variant == "reranked"].changed.sum())
    n_runs = int((ra.variant == "reranked").sum())
    n_sig = int((t2.p_holm < 0.05).sum())
    return "\n".join([
        "## What this does and does not show", "",
        "No result below is described as \"beats\" or \"better\": each comparison is given as "
        "higher/lower with its raw and Holm-adjusted p. With n=10 seeds the smallest two-sided "
        "p is about 0.002, so power is limited and a non-significant result is not evidence of "
        f"equality. In the Stage 2 family, {n_sig} of {len(t2)} tests have Holm-adjusted p < 0.05.",
        "",
        f"- Primary test (clean signal, d=500, 2000 search rows): Firefly's holdout accuracy vs "
        f"random search was {stat(prim)}.",
        f"- Clean signal, d=50, 1000 search rows: mean recall of separating columns was "
        f"{recall(50, 1000, 'firefly'):.3f} (Firefly) vs {recall(50, 1000, 'random_search'):.3f} "
        f"(random search); Firefly sep_hits was {stat(sh50)}. It does not show recovery of all 5 "
        f"informative columns: only columns whose class means differ carry linear signal.",
        f"- Validation fitness at d=500 (secondary, in-sample): with 2000 search rows "
        f"Firefly's was {stat(vf2)}; with 1000 rows it was {stat(vf1)}. Lower validation "
        f"fitness on the rows that selected the mask is not evidence of better generalisation.",
        f"- On one dataset (d=500, data seed 1) over {len(fq) // 2} algorithm seeds Firefly "
        f"ended with 0 informative hits in {zf} runs vs {zr} for random search (descriptive, "
        f"no test).",
        f"- Re-ranking (ablation, same search) changed the selected mask in {n_changed}/{n_runs} "
        f"runs; holdout accuracy was {h50.direction} at d=50 (mean diff {h50.mean_diff:+.4f}, "
        f"p={h50.p:.4f}, Holm p={h50.p_holm:.4f}) and {h500.direction} at d=500 (mean diff "
        f"{h500.mean_diff:+.4f}, p={h500.p:.4f}, Holm p={h500.p_holm:.4f}), n=10 each; re-ranking "
        f"also costs extra fits (see cost table).",
        "- Operator caveat: every Day-3 synthetic run used k=5 or 10 at alpha=0.02, i.e. exactly "
        "one swap per move (1-3 with long-jump); alpha decay had no effect. Results describe that "
        "operator only.",
        f"- Holdout classifiers used by the CSVs above: Stage 2 "
        f"{holdout_clf_used('validate_clean_signal')}; Stage 3 "
        f"{holdout_clf_used('rerank_ablation')}. All of this is synthetic data; it shows nothing "
        f"yet about the image embeddings. Timing measures cost per evaluation only.",
    ])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("Running pytest tests/ -v ...\n", flush=True)
    tr = capture_pytest_results([str(ROOT / "tests"), "-v"])
    print(f"\npytest: exit_code={tr['exit_code']} passed={tr['passed']} failed={tr['failed']} "
          f"errors={tr['errors']} duration={tr['duration_s']} s")
    if tr["exit_code"] != 0 or tr["failed"] or tr["errors"]:
        print("Tests did not all pass -> NOT writing day3_summary.md or LOG.md.")
        for t in tr["tests"]:
            if t["outcome"] != "passed":
                print(f"  {t['outcome'].upper()}: {t['nodeid']}")
        sys.exit(1)

    # -- day3_summary.md -------------------------------------------------------
    parts = ["# Day 3 summary", "",
             "All numbers below are computed from the CSV files in `results/`.", "",
             section_stage1(), "", section_stage2(), "", section_seed1(), "",
             section_stage3(), "", section_stage4(), "", section_does_not_show(), ""]
    summary_path = RESULTS_DIR / "day3_summary.md"
    summary_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {summary_path}")
    if "--no-log" in sys.argv[1:]:
        print("--no-log: LOG.md not touched.")
        return

    # -- LOG.md ----------------------------------------------------------------
    today = datetime.now().strftime("%Y-%m-%d")
    marker = f"## {today} - day3"
    log_path = ROOT / "LOG.md"
    existing = log_path.read_text(encoding="utf-8") if log_path.exists() else "# LOG\n"
    if marker in existing:
        print(f"LOG.md already has an entry '{marker}' -> not appending again.")
        return

    runs = load_all_runs()
    latest = {}
    for r in runs:
        if r.get("dataset") in DAY3_EXPERIMENTS:
            latest[r["dataset"]] = r            # runs.json is append-only -> last wins
    exp_rows = [f"| {n} | {latest[n]['run_id']} | {latest[n].get('seeds', '-')} | "
                f"{latest[n].get('extra', {}).get('runtime_s', '-')} |"
                for n in DAY3_EXPERIMENTS if n in latest]

    head = git("rev-parse", "HEAD")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    entry = "\n".join([
        "", f"{marker}: {COMMIT_MSG.split(': ', 1)[1]}", "",
        f"- Git: branch `{branch}`, parent commit `{head}`. This entry is part of the commit "
        f"\"{COMMIT_MSG}\" (a commit cannot record its own hash; see `git log`).",
        f"- Environment: Python {sys.version.split()[0]}, numpy {np.__version__}, "
        f"scikit-learn {sklearn.__version__}, pandas {pd.__version__}, "
        f"{platform.platform(terse=True)}, os.cpu_count()={os.cpu_count()}",
        f"- Tests: pytest tests/ -v -> {tr['passed']} passed, {tr['failed']} failed, "
        f"{tr['errors']} errors in {tr['duration_s']} s",
        "- Commands run (from firefly-fs/, venv active):", "",
        *[f"  - `{c}`" for c in COMMANDS], "",
        "- Experiments (seeds and runtimes from results/runs.json):", "",
        "| experiment | run_id | seeds | runtime_s |", "|---|---|---|---|", *exp_rows, "",
        "- Summary: results/day3_summary.md", "",
    ])
    log_path.write_text(existing.rstrip("\n") + "\n" + entry, encoding="utf-8")
    print(f"Appended '{marker}' to {log_path}")


if __name__ == "__main__":
    main()
