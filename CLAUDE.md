# CLAUDE.md: Firefly feature-selection benchmark (Day 4 and Day 5 operations)

This file is loaded at the start of every session. Follow it exactly. If an instruction in a task conflicts with a rule here, stop and ask before acting.

@ANALYSIS.md
@DASHBOARD.md

## 1. Project in one paragraph

Graded college project (100 marks, 5-day timeline). We benchmark binary Firefly feature selection against standard baselines on pretrained image embeddings, with leakage-safe protocol, 10 seeds, matched search budgets, stability and cost analysis, and an interactive Streamlit dashboard. The goal is an honest empirical study, not a new algorithm. A tie or loss for Firefly, measured properly, is a valid result.

Research questions: RQ1 accuracy at fixed k; RQ2 search cost and break-even; RQ3 subset stability across seeds; RQ4 transfer of subsets to KNN and linear SVM.

## 2. Status (update this section at the end of every session)

- [x] Day 1: data, splits, ResNet-18 embeddings
- [x] Day 2: baselines (10 seeds, both datasets), leakage tests (14)
- [x] Day 3: Firefly validated, diagnostics, rerank, timing (33 tests pass)
- [x] Day 4: init_pop, PSO, registry, tuning, frozen config (7b985ca), runner, PREREG, compute plan (127 tests pass)
- [x] Day 5: main runs (960 rows), analysis A0-A7 + A9 (A8 convergence deferred: no traces yet), 231 tests pass; dashboard skeleton (app/) not yet wired to the meta results
- [ ] Day 5+: A8 traces, dashboard wiring (break-even slider, stability tab), cost/stability/transfer write-up, report, slides

Current stage: analysis complete except A8; `python -m experiments.analyze --rebuild-check` -> 58/58 PASS. Last commit: see `git log` (frozen config = 7b985ca, tag day5-launch).

## 3. Environment and commands (Windows, PowerShell)

- Python 3.11, venv at `.venv`. Activate: `.venv\Scripts\Activate.ps1`
- Tests: `pytest tests/ -q` (all must pass before any commit)
- Use `pathlib`. No hard-coded absolute paths. Never write outside the repo.
- Logs go to `logs/`. Run long jobs like: `python -m experiments.run_metaheuristics --dataset digits *> logs\digits.log`
- The repo lives inside OneDrive. Pause OneDrive sync during long runs, and push to GitHub regularly (GitHub is the real backup).

## 4. Repo map

| Path | Purpose |
|---|---|
| `src/data.py` | `load_dataset(name, seed)` returns `Xtr, Xte, ytr, yte`; scaler fitted on train only |
| `src/baselines.py`, `src/evaluate.py` | baseline selectors, `evaluate()` harness |
| `src/firefly.py` | `make_fitness`, `repair`, `binary_firefly`, `rerank`, `run_firefly` |
| `src/init_pop.py` | `init_population(kind, ...)`: random or filter-seeded (Day 4) |
| `src/pso.py` | PSO with random-key (top-k) decoding (Day 4) |
| `src/search_registry.py` | `run_search(method, ...)` dispatcher (Day 4) |
| `src/synthetic.py` | `separating_columns()`: ground truth for synthetic tests |
| `src/results_store.py` | `log_experiment()`, append-only `results/runs.json` |
| `experiments/` | scripts: tune, runner, analysis, diagnostics |
| `config/` | `frozen_config.yaml` (Day 4), `smoke_config.yaml` |
| `results/` | CSVs, summaries, `PREREG.md` lives at repo root |
| `tests/` | pytest suites |
| `app/` | Streamlit dashboard |

Datasets: `digits` (64-d) and `cifar_r18` = a **10,000-image CIFAR-10 subset embedded with pretrained ResNet-18 (512-d)**; train 8000 x 512, test 2000 x 512. Never write "CIFAR-18".

## 5. Non-negotiable rules

1. **Test split discipline.** `Xte`/`yte` appear only in the final evaluation call of a runner. Never in search, tuning, re-ranking, filter scoring, config choice, or exploratory plots.
2. **Do not run any command on the terminal** all commands will be run only by the user just provide the required commands and the output will be returned after running
3. **Inner-train filter scores.** Seeding filter scores are computed on the fitness function's inner-train rows only, so inner validation stays unbiased.
4. **Budget definition.** Budget = number of NEW (uncached) classifier fits inside `fitness()`. Re-ranking fits are counted and reported separately. The budget check happens BEFORE each fitness call, so `evals <= budget` always holds.
5. **Fairness.** Methods compared in one experiment share: the same inner split (same seed), the same initial population for the same seed, the same budget, and equal tuning effort.
6. **Exactly k.** Every candidate mask has exactly k True bits (swap, repair, or top-k decoding).
7. **Do not change search logic** in `src/firefly.py` (movement, repair, swap mutation, acceptance, budget guard) without explicit approval. New behaviour is additive, with defaults that leave old results identical, plus a test proving it.
8. **Determinism.** `np.random.default_rng(seed)` only. No global RNG. Same seed gives identical masks.
9. **Never edit a test threshold or tolerance to make a test pass.** Explain the failure and ask.
10. **No tuning to hit a target.** Hyperparameters are chosen by the protocol in section 8, then frozen.
11. **Log everything.** Every experiment writes raw per-run rows to `results/*.csv` and a `log_experiment()` entry (run_id, timestamp, parameters, git hash).
12. **Thread limits.** In any parallel job set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` to `1` before importing numpy.
13. **No invented numbers or citations.** If unsure, say so.
14. **Suppress repeated sklearn warnings**, but print each distinct warning type once.

## 6. Working protocol (stage gates)

For every stage:

1. Read the relevant files first.
2. Explain design choices in a few lines BEFORE writing code. Keep diffs minimal. Do not refactor unrelated files.
3. Implement, then run the code and `pytest tests/ -q`.
4. Show the exact output, give a 5-line summary, and **STOP**. Wait for the user to say "continue".
5. The user runs a fresh-session review (section 12) before the commit.

If a result looks surprising (too good, or too bad), investigate for a bug before reporting it as a finding.

## 7. Terminology and wording (code comments, reports, dashboard)

- "long-jump mutation" (with probability 0.1, 3x the usual swaps), never "Levy flight" and never "heavy-tailed".
- "PSO with random-key (top-k) decoding", not classic sigmoid binary PSO.
- Filter-seeded variants are **ablations**. Plain Firefly is the method under test.
- PCA is a projection, not a selection method. It needs the full 512-d embedding plus a projection matrix at inference. Say so wherever PCA appears.
- The `w * k/d` term in fitness is a constant for valid masks. Never claim it breaks ties.
- In `make_classification`, "informative" columns are not necessarily separating. Score synthetic hits with `src/synthetic.separating_columns`.
- Do not write "beats" or "better" without a test and an effect size. With 10 seeds, power is limited, and the text should say so.

## 8. Day 4 specification

Deliverables D1 to D7. Each stage ends with a STOP.

**Stage 0 (pre-flight).** `experiments/check_reproducibility.py` hashes `load_dataset` outputs for seeds 0 to 9 across two separate processes and saves `results/split_hashes.csv`. Record shapes in `LOG.md`. Read the `k` lists from `results/baselines_<dataset>.csv` (Digits has three k values). List occurrences of "CIFAR-18" and fix them.

**Stage 1 (`src/init_pop.py`).** `init_population(kind, n, d, k, rng, filter_scores=None, frac_seeded=0.5)`.
- `random`: current behaviour (uniform random k-masks).
- `seeded`: member 0 is the exact top-k by filter score; about `frac_seeded * n` members are score-biased (sample k features without replacement, p proportional to `(1 - rank/d)^3`); the remainder are uniform random.
- `make_fitness` exposes `state["filter_scores"]` (ANOVA F via `f_classif`, `nan_to_num`) from inner-train rows only.
- `binary_firefly(init_masks=None)`: when None, behaviour is byte-for-byte unchanged.
- Tests: exactly k bits; seeded population contains the exact top-k mask; same rng gives same population; scores equal `f_classif` on inner-train rows and differ from full-train scores; `init_masks=None` reproduces old output; seeded best fitness at sweep 0 is at most the ANOVA mask's fitness.

**Stage 2 (`src/pso.py`).** `binary_pso(fitness, state, d, k, budget=800, n_particles=20, w_start=0.9, w_end=0.4, c1=1.5, c2=1.5, vmax=0.5, init_masks=None, seed=0)` returns `(best_mask, best_fit, history)`, the same interface as `binary_firefly`.
- Random-key encoding: real position `x`, mask = top-k of `x`. `v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)`, clip `|v| <= vmax`, `x += v`.
- Inertia decays linearly over `T = budget // n_particles`; cap total iterations at `10 * T`.
- Initial `x = mask + 0.1 * N(0,1)`. Same initial population as Firefly for the same seed.
- Tests: exactly k; `evals <= budget`; deterministic; history non-increasing; identical init across methods; `init_masks` respected.
- Sanity script: clean-signal toy (d=50 and 500, seeds 0 to 9, budget 600): PSO validation fitness should be at or below random search.

**Stage 3 (`src/search_registry.py`).** `run_search(method, fitness, state, d, k, budget, seed, hp)` with methods `firefly`, `firefly_levy`, `pso`, `firefly_seeded`, `pso_seeded`. Returns dict: `best_mask, best_fit, history, diversity, evals_used`. Tests: all methods exactly k and within budget; shared init within pairs (firefly/pso, firefly_seeded/pso_seeded); unknown method raises `ValueError`. Sanity: seeded methods start at or below the ANOVA mask's validation fitness. Do not print test accuracy.

**Stage 4 (`experiments/tune.py`, `tune_summary.py`).**
- Tuning seeds 100, 101, 102 only. Take `Xtr, ytr` from `load_dataset`; never touch `Xte`. Split stratified 75/25. Search on 75%; score = LogisticRegression (max_iter=500) accuracy on the 25% tuning holdout.
- Conditions: (cifar_r18, k=16), (cifar_r18, k=64), (digits, k=16); budget 800; 9 runs per config.
- One-factor-at-a-time, 9 configs per method:
  - Firefly defaults n=20, beta0=0.4, gamma=2, alpha=0.02; variants n {10,40}, beta0 {0.2,0.6}, gamma {1,4}, alpha {0.1,0.25}.
  - PSO defaults n=20, w_start=0.9, c1=c2=1.5, vmax=0.5; variants n {10,40}, c1=c2 {1.0,2.0}, vmax {0.25,1.0}, w_start {0.7,1.0}.
- Decision rule: highest mean tuning accuracy; if it beats the default by less than 0.003, keep the default. No combinations, no second round.
- Resumable: append every finished run to `results/tuning_runs.csv` immediately. Outputs: `results/tuning_summary.csv`, `results/tuning_decision.md`.
- Dry run (1 config, 1 seed) before the full grid.

**Stage 5 (`experiments/run_metaheuristics.py`).** Resumable runner; see section 9 for the schema and Day 5 usage. Smoke test: Digits, seeds 0 to 1, budget 150, two k values.
- Tests: expected row count and columns; rerun adds zero rows; `--force` reproduces identical `idx` and `acc_lr`; `evals_used <= budget`; static check that `Xte`/`yte` appear only in the final evaluation function.

**Stage 6 (freeze, PREREG, compute plan).**
- Apply the decision rule mechanically, then write `config/frozen_config.yaml` (hyperparameters for both methods, `frac_seeded: 0.5`, `top_m: 10`, `n_splits: 5`, tuning run ID, git hash). Commit `day4: frozen config`. **No hyperparameter changes after this commit.**
- `PREREG.md` at repo root, committed before any Day 5 job (template in the Day 4 playbook): primary comparison is plain Firefly vs each of ANOVA, MI, L1, RF, PCA, PSO, for each dataset x k x budget, paired two-sided Wilcoxon by seed, Holm correction across the 6 comparisons; WIN means adjusted p < 0.05 and positive mean difference, LOSS the reverse, TIE otherwise.
- `experiments/plan_compute.py` reads `results/time_fitness.md` and writes `results/compute_plan.md` (+20% margin). If any total exceeds 8 hours with available workers, reduce in this order: fewer budgets, drop `firefly_levy`, fewer seeds, fewer k values.

## 9. Day 5 specification (main runs)

**Preconditions (all must be true before launching):** tests green; `config/frozen_config.yaml` and `PREREG.md` committed; smoke test passed with resume and determinism verified; `compute_plan.md` reviewed; OneDrive sync paused; sleep disabled; `git tag day5-launch` created.

**Grid:** methods `firefly pso firefly_seeded pso_seeded` (plus `firefly_levy` only at budget 800 if time allows); budgets 800 and 3000; seeds 0 to 9; k values from the baseline CSVs; datasets `digits` and `cifar_r18`.

**Launch order (resumable, each in its own log):**

```powershell
python -m experiments.run_metaheuristics --dataset digits    --budgets 800 3000 *> logs\digits.log
python -m experiments.run_metaheuristics --dataset cifar_r18 --budgets 800      *> logs\cifar_800.log
python -m experiments.run_metaheuristics --dataset cifar_r18 --budgets 3000     *> logs\cifar_3000.log
```

If a run is interrupted, rerun the same command. Completed job keys `(dataset, method, budget, seed, k)` are skipped.

**Output schema** (`results/meta_<dataset>.csv`, one row per mask type):
`dataset, method, budget, seed, k, mask_type (raw|rerank), acc_lr, acc_knn, acc_svm, n_feat, search_s, rerank_s, infer_s, evals_used, rerank_fits, val_fitness, diversity_last, idx, config_hash, git_hash, run_id`

**During the run:**
- Do not edit anything in `src/` or `config/`. Analysis scripts and dashboard work may proceed in separate files.
- Every 30 to 60 minutes: check `logs/`, row counts, and failed-job messages. Copy `results/` to a second location (Drive) after each dataset completes. Push to GitHub.
- If a bug is found: stop, fix, add a regression test, rerun only the affected jobs (`--force` with the relevant filters), and record the change in `LOG.md` with the reason. Never silently patch results.

**Analysis scripts to write while jobs run** (read CSVs only; reproducible with one command, `python -m experiments.analyze`):

| Output | Content |
|---|---|
| `figures/acc_vs_k_<dataset>_<budget>.png` | Accuracy vs k for all methods (baselines plus metaheuristics), mean and std over seeds; PCA marked as a projection |
| `results/rq1_tests.csv` | Paired Wilcoxon (Firefly vs each of 6 methods) per dataset x k x budget, Holm-adjusted p, mean difference, WIN/LOSS/TIE label (handle all-zero differences as TIE) |
| `figures/budget_curve_<dataset>.png` | Accuracy vs search budget (800 vs 3000), per k |
| `results/rq2_cost.csv` and figure | `search_s`, `rerank_s`, inference time (all features vs selected), break-even `N = search_s / (infer_all - infer_selected)`; state clearly when the saving is tiny (linear models) or negative; also report KNN, where feature count matters more |
| `figures/stability_<dataset>.png` | Mean pairwise Jaccard of selected sets across seeds, per method and k, with random as reference |
| `results/rq4_transfer.csv` | KNN and linear SVM accuracy of subsets selected with logistic regression, vs ANOVA/MI subsets |
| `figures/ablations_*.png` | Seeded vs plain; rerank vs raw (with cost in seconds); long-jump vs plain if run |
| `figures/convergence_*.png` | Best fitness vs evaluations for Firefly and PSO (needs history; if not stored in CSV, store separately in `results/histories/`) |

Every figure and table must be regenerated from CSVs by script, not by hand. Under each figure in the report or dashboard, write one plain sentence of takeaway.

**Dashboard wiring (parallel, teammate):** `app/streamlit_app.py` reads `results/baselines_*.csv` and `results/meta_*.csv` (tolerate the second being absent or partial), `st.cache_data` for loads, a budget-capped live Firefly playground (at most 300 evaluations), a break-even calculator slider, and a stability tab. The dashboard never loads test data except through `evaluate()`.

## 10. Statistics protocol

- Paired by seed (splits are identical across methods because `load_dataset(name, seed)` is deterministic; verify with `results/split_hashes.csv`).
- Two-sided Wilcoxon signed-rank; Holm correction within each comparison family defined in `PREREG.md`.
- Always report mean difference in accuracy points and n=10. With n=10 the smallest possible two-sided p is about 0.002, so say that power is limited.
- Never report the best seed. Report mean and std over seeds.
- Secondary comparisons (seeded, rerank, budget, transfer, stability) are labelled exploratory, not corrected across families.

## 11. Git and logging conventions

- Commit after every green stage: `day4: <what>`, `day5: <what>`. Never commit with failing tests.
- `LOG.md` entry per session: date, commands run, git hash, scikit-learn and numpy versions, seeds, runtimes, anything unexpected.
- Large embeddings in `data/` stay out of git if over 50 MB; back them up to Drive. Results CSVs are committed.
- Never rewrite history on pushed commits.

## 12. Review routine (fresh session, read-only)

After each stage the user opens a fresh session and asks for an adversarial review that does NOT modify files. Check: (1) leakage of test data or inner-validation labels into search, filter scores, tuning, re-ranking, or config choice; (2) budget accounting and cache handling; (3) fairness across methods (inner split, initial population, budget, tuning effort); (4) determinism (global RNG, ordering, thread effects); (5) correctness (exactly k, loop termination when everything is cached); (6) overclaiming in comments and text. Report severity (BLOCKER, MAJOR, MINOR), file:line, why, minimal fix, and the three most dangerous issues.

## 13. Ask before doing any of these

- Changing anything in `config/frozen_config.yaml` after the freeze.
- Changing search logic in `src/firefly.py`.
- Adding datasets, backbones, seeds beyond 0 to 9, or budgets beyond 800 and 3000.
- Deleting or overwriting anything in `results/`.
- Installing new packages (propose, then update `requirements.txt`).
- Deploying anything or pushing to a remote.

## 14. Known facts to keep honest in the report

- With 800 evaluations and 20 fireflies the search gets about 4 to 5 sweeps, so it behaves like 20 parallel hill climbers; diversity at the end stays high.
- Wrapper fitness on 250 to 500 validation rows has a standard error of about 2 to 3 points; found subsets can beat the true subset on validation and lose on holdout.
- Re-ranking costs 50 extra fits (5-fold, top 10) and several seconds on CIFAR; it has not shown a significant benefit on synthetic data.
- On CIFAR baselines, PCA is the strongest method at every k; selectors are close to all-feature accuracy only at k=128.
- If Firefly ties or loses, the contribution is the benchmark: leakage-safe protocol, matched budgets, stability, cost, and honest negative results.