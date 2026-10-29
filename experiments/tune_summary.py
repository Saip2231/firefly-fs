"""Day 4 Stage 4: summarise results/tuning_runs.csv and apply the decision rule.

Decision rule (CLAUDE.md section 8, fixed before tuning; applied mechanically)
  Per method, the score of a config is its mean tuning_acc over all
  3 conditions x 3 seeds = 9 runs.  winner = highest mean; if the winner is
  not the default and beats the default by less than 0.003 (mean difference
  < 0.003), keep the default.  Ties in the mean go to the default, then to
  the earlier config in experiments/tune.CONFIGS.  No combinations of
  variants, no second round.
  The grid must be complete (9 configs x 9 runs per method); otherwise the
  script stops without a decision.

Outputs
  results/tuning_summary.csv   mean/std/n of tuning_acc per (method, config,
                               condition) and overall (condition="all")
  results/tuning_decision.md   tables and the decision per method
  entry in results/runs.json

Usage (from firefly-fs/, venv active):
    python experiments/tune_summary.py
"""

from __future__ import annotations

import json

import pandas as pd

from _common import RESULTS_DIR, git_hash
from tune import BUDGET, CONDITIONS, CONFIGS, RUNS_CSV, TUNING_SEEDS

from src.results_store import log_experiment

MARGIN = 0.003
SUMMARY_CSV = RESULTS_DIR / "tuning_summary.csv"
DECISION_MD = RESULTS_DIR / "tuning_decision.md"
N_PER_CONFIG = len(CONDITIONS) * len(TUNING_SEEDS)


def decide(means: dict[str, float], order: list[str], margin: float = MARGIN) -> dict:
    """Apply the decision rule to {config_id: mean tuning_acc}.

    order lists config ids with "default" first; it breaks ties.
    Returns dict(best, chosen, diff = mean[best] - mean[default], reason).
    """
    if order[0] != "default":
        raise ValueError("order must start with 'default'")
    best = max(order, key=lambda c: (means[c], -order.index(c)))
    diff = means[best] - means["default"]
    if best == "default":
        return dict(best=best, chosen="default", diff=0.0,
                    reason="default has the highest mean")
    if diff < margin:
        return dict(best=best, chosen="default", diff=diff,
                    reason=f"{best} beats default by {diff:.4f} < {margin}; keep default")
    return dict(best=best, chosen=best, diff=diff,
                reason=f"{best} beats default by {diff:.4f} >= {margin}")


def check_complete(df: pd.DataFrame) -> None:
    """Every method x config must have exactly the 9 expected runs, no duplicates."""
    problems = []
    for method, configs in CONFIGS.items():
        for cid, _ in configs:
            g = df[(df.method == method) & (df.config_id == cid)]
            keys = set(zip(g.dataset, g.k, g.seed))
            want = {(ds, k, s) for ds, k in CONDITIONS for s in TUNING_SEEDS}
            if len(g) != N_PER_CONFIG or keys != want:
                problems.append(f"{method}/{cid}: {len(g)} rows")
    extra = set(df.seed) - set(TUNING_SEEDS)
    if extra:
        problems.append(f"non-tuning seeds present: {sorted(extra)}")
    if problems:
        raise SystemExit("Tuning grid incomplete or inconsistent; no decision made:\n  "
                         + "\n  ".join(problems))


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, configs in CONFIGS.items():
        for cid, hp in configs:
            g = df[(df.method == method) & (df.config_id == cid)]
            parts = [("all", g)] + [(f"{ds}/k={k}", g[(g.dataset == ds) & (g.k == k)])
                                    for ds, k in CONDITIONS]
            for cond, gg in parts:
                rows.append(dict(method=method, config_id=cid, condition=cond,
                                 mean_acc=gg.tuning_acc.mean(), std_acc=gg.tuning_acc.std(),
                                 n=len(gg), mean_val_fitness=gg.val_fitness.mean(),
                                 mean_search_s=gg.search_s.mean(),
                                 config_json=json.dumps(hp, sort_keys=True)))
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(RUNS_CSV)
    check_complete(df)
    summ = summarise(df)
    summ.to_csv(SUMMARY_CSV, index=False)

    conds = ["all"] + [f"{ds}/k={k}" for ds, k in CONDITIONS]
    lines = ["# Tuning decision (Day 4 Stage 4)", "",
             f"- Source: `{RUNS_CSV.relative_to(RESULTS_DIR.parent)}` ({len(df)} runs), "
             f"tuning seeds {list(TUNING_SEEDS)}, budget {BUDGET}, "
             f"conditions {[f'{ds}/k={k}' for ds, k in CONDITIONS]}.",
             "- Score: tuning_acc = final_clf() accuracy on the 25 % tuning holdout of Xtr "
             "(the test split is never loaded).",
             f"- Rule: highest mean tuning_acc over {N_PER_CONFIG} runs; if it beats the "
             f"default by < {MARGIN}, keep the default.  No combinations, no second round.",
             f"- git (summary): {git_hash()}; git (runs): "
             f"{', '.join(sorted(df.git_hash.astype(str).unique()))}", ""]
    decisions = {}
    for method, configs in CONFIGS.items():
        order = [cid for cid, _ in configs]
        s = summ[summ.method == method]
        means = dict(zip(s[s.condition == "all"].config_id, s[s.condition == "all"].mean_acc))
        dec = decide(means, order)
        chosen_hp = dict(configs)[dec["chosen"]]
        decisions[method] = dict(**dec, hp=chosen_hp)

        table = s.pivot(index="config_id", columns="condition", values="mean_acc").loc[order, conds]
        stds = s.pivot(index="config_id", columns="condition", values="std_acc").loc[order, conds]
        lines += [f"## {method}", "",
                  "| config | " + " | ".join(conds) + " |",
                  "|---|" + "---|" * len(conds)]
        for cid in order:
            cells = [f"{table.loc[cid, c]:.4f} ± {stds.loc[cid, c]:.4f}" for c in conds]
            lines.append(f"| {cid} | " + " | ".join(cells) + " |")
        lines += ["", f"**Decision: `{dec['chosen']}`** ({dec['reason']}).",
                  f"Frozen hyperparameters: `{json.dumps(chosen_hp, sort_keys=True)}`", ""]

    n_ff = decisions["firefly"]["hp"]["n_fireflies"]
    n_ps = decisions["pso"]["hp"]["n_particles"]
    if n_ff != n_ps:
        lines += [f"**Note:** population sizes differ (firefly {n_ff}, pso {n_ps}); "
                  f"their initial populations are then shared only for the first "
                  f"{min(n_ff, n_ps)} members (fairness rule 5).  Record this in PREREG.", ""]
    lines += ["The frozen values apply to firefly, firefly_levy and firefly_seeded "
              "(Firefly) and to pso and pso_seeded (PSO); frac_seeded stays 0.5.", ""]
    DECISION_MD.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"Saved {SUMMARY_CSV}\nSaved {DECISION_MD}")
    log_experiment(
        name="tune_summary",
        params=dict(margin=MARGIN, tuning_seeds=list(TUNING_SEEDS), budget=BUDGET,
                    ks=sorted({k for _, k in CONDITIONS}),
                    decisions={m: {k: v for k, v in d.items()} for m, d in decisions.items()}),
        seeds=list(TUNING_SEEDS),
        csv_paths=[str(p.relative_to(RESULTS_DIR.parent)) for p in (SUMMARY_CSV, DECISION_MD)],
        extra=dict(git_hash=git_hash()),
    )


if __name__ == "__main__":
    main()
