# LOG

## 2026-10-29 - day3: firefly validated, diagnostics, rerank, timing

- Git: branch `master`, parent commit `28a1f52afe2d53221d1e8f644e53066bde568a23`. This entry is part of the commit "day3: firefly validated, diagnostics, rerank, timing" (a commit cannot record its own hash; see `git log`).
- Environment: Python 3.11.4, numpy 2.4.6, scikit-learn 1.9.1, pandas 3.1.0, Windows-10, os.cpu_count()=16
- Tests: pytest tests/ -v -> 33 passed, 0 failed, 0 errors in 98.525 s
- Commands run (from firefly-fs/, venv active):

  - `python -m pytest tests/test_firefly.py -v -k trace`
  - `python experiments/diagnose_d500.py`
  - `python experiments/validate_clean_signal.py`
  - `python experiments/check_d500_seed1.py`
  - `python experiments/rescore_clean_signal.py`
  - `python -m pytest tests/ -v`
  - `python -m pytest tests/test_rerank.py -v`
  - `python experiments/rerank_ablation.py`
  - `python experiments/time_fitness.py`
  - `python experiments/make_day3_report.py   (runs pytest tests/ -v, writes summary + LOG)`

- Experiments (seeds and runtimes from results/runs.json):

| experiment | run_id | seeds | runtime_s |
|---|---|---|---|
| diagnose_d500 | 20261028_225402_diagnose_d500 | [0, 1, 2, 3, 4] | 157.6 |
| validate_clean_signal | 20261028_231601_validate_clean_signal | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] | 492.0 |
| check_d500_seed1 | 20261028_233107_check_d500_seed1 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29] | 224.3 |
| rescore_clean_signal | 20261028_233923_rescore_clean_signal | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] | 8.3 |
| rerank_ablation | 20261028_235229_rerank_ablation | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] | 120.2 |
| time_fitness | 20261029_000149_time_fitness | [0] | 55.8 |

- Summary: results/day3_summary.md

## 2026-10-29 - day3-fixes: shared clf, budget accounting, tie-stall, guards, docs

- Why: adversarial review of Day 3 (read-only session) before Day 4. Findings and their status are in the table below.
- Git: branch `master`, parent commit `fb8ed820a930052312a07c91a74869b3a8e362a2` (F5). This entry is part of the commit "day3-fixes: shared clf, budget accounting, tie-stall, guards, docs" (a commit cannot record its own hash; see `git log`). Stage commits since `c73b218` (day3):
  - `bbbcab3` day3-fixes: golden outputs (F0 only)
  - CORRECTION (2026-10-29, Day 4 Stage 0): this entry originally said the F1 changes were in `bbbcab3`. They were not committed at all: `git status` showed src/classifiers.py and tests/test_classifiers.py untracked and the F1 edits to run_baselines.py, run_and_store.py, diagnose_planted.py, toy_check.py and src/results_store.py uncommitted, while the committed F2-F5 files already import src.classifiers. F1 is committed separately at Day 4 Stage 0 (see that entry).
  - `35e45b1` day3-fixes: budget accounting and rerank fairness (F2)
  - `79096a0` day3-fixes: tie-stall fallback in binary_firefly (approved) (F3)
  - `c57660b` day3-fixes: loop guards, budget/k validation, 2*min(k,d-k) normalisation (F4)
  - `fb8ed82` day3-fixes: claims and docs (swap docstring, primary test, Holm) (F5)
- Environment: same venv as the day3 entry (Python 3.11.4, numpy 2.4.6, scikit-learn 1.9.1); not re-printed in this session.
- Tests: pytest tests/ -q -> 53 passed (was 33). New suites: test_classifiers, test_tie_stall, test_guards, test_stats_helpers; new tests in test_rerank; test_firefly_beats_random renamed test_firefly_not_worse_than_random (threshold unchanged).
- Regression: `python experiments/golden_day3.py --compare` -> 33/33 runs identical to results/golden_day3.json (masks, best_fit, history, evals_used, sweeps) after F1, F2, F3, F4 and F5. No golden run stalled (0 stopped before budget), so the F3 tie-stall fallback changed no recorded run.
- Commands run (from firefly-fs/, venv active): `python experiments/golden_day3.py` (F0), then after every stage `pytest tests/ -q` and `python experiments/golden_day3.py --compare`; F5: `python experiments/make_day3_report.py --no-log` (regenerated results/day3_summary.md from the existing Day-3 CSVs).
- Not rerun: diagnose_d500.py and validate_clean_signal.py (rule: rerun only if golden outputs changed; they did not).
- Open: the holdout accuracies in validate_clean_signal.csv, validate_clean_signal_rescored.csv and rerank_ablation.csv were computed with the pre-F1 holdout classifier LogisticRegression(max_iter=100, random_state=seed); day3_summary.md states this. Rerunning validate_clean_signal -> rescore_clean_signal -> rerank_ablation would put them on final_clf() (max_iter=500) and add sep_hits / total_fits columns to the CSVs. Baseline CSVs already used max_iter=500 and are unaffected.
- Day-3 results under the F5 reporting rules: primary test (holdout_acc, firefly vs random search, d=500, 2000 search rows) higher, mean diff +0.0145, p=0.0371, Holm-adjusted p=0.9062, n=10; 0 of 32 Stage-2 tests have Holm-adjusted p < 0.05.

| # | finding (Day-3 review) | sev | status | stage |
|---|---|---|---|---|
| 1 | rerank fits uncounted in budget, use more rows | MAJOR | fixed: evals_used / rerank_fits / total_fits / rerank_s reported; rerank only compared within the same search | F2 |
| 2 | "beats random" judged on in-sample val fitness, no correction | MAJOR | fixed: primary test on holdout_acc named, Holm over 32 tests, no "BEATS" label | F5 |
| 3 | alpha has no effect at small k; swap docstring wrong | MAJOR | docs fixed (max_swaps=1 for k<=74 at alpha=0.02; mean (1+max)/2); behaviour unchanged by design | F5 |
| 4 | random_search can loop forever | MAJOR | fixed in experiments/_common.py (100*budget proposals / 5*budget consecutive cache hits); toy_check.py and the test helper in test_firefly.py still unguarded | F4 |
| 5 | tie-stall stops search with budget unspent | MAJOR | fixed: one random swap per firefly on a stall (approved change); golden unchanged | F3 |
| 6 | final classifier differs across methods | MAJOR | fixed: src/classifiers.final_clf() everywhere + AST test; Day-3 holdout CSVs still pre-F1 (open, above) | F1 |
| 7 | rerank CV includes selecting rows | MINOR | documented in rerank docstring (selection bias, not test leakage) | F2 |
| 8 | 2k normaliser wrong for k > d/2; k >= d | MINOR | fixed: 2*min(k, d-k); ValueError unless 1 <= k < d | F4 |
| 9 | test_firefly_beats_random overclaims | MINOR | fixed: renamed, no-claim docstring, threshold unchanged | F5 |
| 10 | budget=0 returns unevaluated mask | MINOR | fixed: ValueError for budget < 1 (firefly, random_search) | F4 |
| 11 | BLAS thread effects on ties | MINOR | unchanged; golden_day3.py pins thread env vars; CLAUDE.md rule 12 covers parallel jobs | - |

## 2026-10-29 - day4 stage 0: pre-flight

- Split reproducibility: `python experiments/check_reproducibility.py` -> two separate processes produced byte-identical hashes (20 rows = 2 datasets x seeds 0-9); saved results/split_hashes.csv. Distinct splits across seeds: 10/10 for each dataset.
- Shapes (identical for all 10 seeds):

| dataset | Xtr | Xte |
|---|---|---|
| digits | 1437 x 64 | 360 x 64 |
| cifar_r18 | 8000 x 512 | 2000 x 512 |

- Baseline CSVs (`python experiments/inspect_baselines.py`): baselines_digits.csv 190 rows, baselines_cifar_r18.csv 250 rows; seeds 0-9 in both; 10 rows per (method, k). k lists: digits [16, 32, 64] (All Features = 64), cifar_r18 [16, 32, 64, 128] (All Features = 512).
  - Note: on digits k=64 equals d=64, so a k=64 "selection" is all features; binary_firefly raises ValueError for k >= d (day3-fixes F4). The Day-4/5 metaheuristic grid must handle digits k=64 explicitly (decision pending).
- "CIFAR-18" search: no occurrence in labels, docs or comments. Only an accepted alias name "cifar_18" in src/data.py:36 and tests/test_leakage.py:39 (not edited yet).
- experiments/toy_check.py and the random-search helper in tests/test_firefly.py now call the guarded experiments/_common.random_search. Check: toy_check.py output before (git stash) and after compared with `fc.exe` -> no differences.
- F1 catch-up: the day3-fixes F1 files (see correction above) committed as "day3-fixes: shared final classifier (F1, missed in earlier commits)" before the reruns below.
- Reruns on final_clf() (closes the "Open" item of the day3-fixes entry). Commands, in order:
  - `python experiments/compare_day3_pvalues.py --snapshot` (38 tests saved to results/day3_pvalues_before.csv)
  - `python experiments/validate_clean_signal.py` (219.4 s, run_id 20261029_122757_validate_clean_signal)
  - `python experiments/rescore_clean_signal.py` (3.3 s, run_id 20261029_123743_rescore_clean_signal)
  - `python experiments/rerank_ablation.py` (51.9 s)
  - `python experiments/make_day3_report.py --no-log` (53 passed; results/day3_summary.md regenerated)
  - `python experiments/compare_day3_pvalues.py --compare` -> 0 of 38 tests changed p or Holm-adjusted p; Holm p < 0.05: 0/32 (stage 2) and 0/6 (stage 3), before and after.
- Interpretation: holdout accuracies are identical under LogisticRegression(max_iter=100, random_state=seed) and final_clf() (max_iter=500, random_state=0). With k=5 features lbfgs converges well before 100 iterations (no ConvergenceWarning was printed), and lbfgs ignores random_state, so both give the same fit. Search outputs were unchanged (all runs used the full budget; the d=500/n=1000/seed-1 outlier reproduces: val 0.3723, 0 hits, holdout 0.494). The Day-3 conclusions do not depend on the holdout classifier's iteration cap.
- Tests: pytest tests/ -q -> 53 passed.
- git status before the stage commit: only Stage 0 work uncommitted (scripts, results, LOG.md, toy_check.py, tests/test_firefly.py); cluade.md still untracked.

## 2026-10-29 - day4 stage 1: init_pop, inner-train filter scores, init_masks

- Read-only review of the Day-3/Stage-0 code before Stage 1: 0 BLOCKER, 3 MAJOR, 7 MINOR. MAJORs deferred to their stages: (1) reused make_fitness state silently shrinks the budget -> assert evals == 0 in run_search (Stage 3); (2) BLAS threads + discrete fitness ties -> thread env vars in every runner (Stage 5); (3) baselines fit on all 8000 rows, wrapper search on <= 2000 -> state in PREREG (Stage 6).
- New: src/init_pop.py (init_population random/seeded, top_k_mask, check_init_masks); make_fitness exposes state["filter_scores"] = nan_to_num(f_classif) on inner-train rows only; binary_firefly(init_masks=None).
- Tests: tests/test_init_pop.py 26 passed; pytest tests/ -q -> 79 passed. `python experiments/golden_day3.py --compare` -> 33 identical, 0 changed.
- Expected warnings on digits: f_classif reports constant columns [0 24 32 39] on the inner-train rows (NaN F -> 0), printed once per type.
- Committed as "day4: init_pop, inner-train filter scores, init_masks".

## 2026-10-29 - day4 stage 2: PSO with random-key (top-k) decoding (in progress)

- New: src/pso.py binary_pso (same interface as binary_firefly; same initial population as Firefly for the same seed), tests/test_pso.py, experiments/sanity_pso.py (clean-signal toy from validate_clean_signal.py, d 50/500, n_search 1000/2000, seeds 0-9, budget 600).
- v0, plain PSO as specified (results/sanity_pso_v0_noreset.csv, sanity_pso_tests_v0_noreset.csv): tests 90 passed, but PSO val_fitness above random search in 4/4 settings; every PSO run hit the 10*T iteration cap with budget unspent (mean evals_used 210 at d=50/n_search=2000, 477 and 461 at d=500). Cause: swarm collapses onto the gbest mask, every decoded mask is then a cache hit. Not a coding bug.
- v1, velocity redraw U(-vmax, vmax) after an iteration with 0 new evals (run 20261029_133804_sanity_pso, results/sanity_pso_v1_velreset.csv): tests 92 passed; resets fired (mean 172-263 per run) but only 2/40 PSO runs spent the budget (min evals_used 89); val_fitness still above random search in 4/4. Cause: kicks of <= w*vmax are pulled back before any key crosses the top-k threshold.
- v2 (approved): replaced the velocity redraw with a Firefly-style stall swap (one random key swap per particle when the moves of an iteration give 0 new evals; same rule as Firefly F3). Acceptance fixed before the run, on budget use only: >= 38/40 PSO runs spend the full budget.
- v2 result (run 20261029_135554_sanity_pso, results/sanity_pso.csv, sanity_pso_tests.csv): tests 15 PSO / 94 total passed. Acceptance PASS: 40/40 PSO runs used exactly 600 evals (stall fallback fired in a mean of 3.6-8.6 iterations per run). Stage-2 expectation (PSO mean val_fitness <= random search) met in 2/4 settings: d=50 lower (-0.0016, p=0.66; -0.0026, p=0.0625), d=500 higher (+0.0408, p=0.47; +0.0687, p=0.73); every Holm-adjusted p >= 0.5, so no setting differs detectably (n=10, exploratory).
- d=500 gap = a few stuck runs, not a shift: PSO val_fitness ~0.38-0.41 with 0 informative columns at (n_search=1000, seed 1), (2000, seed 1), (2000, seed 2); Firefly is also stuck at (1000, seed 1), the known Day-3 d=500 seed-1 outlier. With 5 informative columns out of 500, a random 5-mask contains one with probability ~5%, so a local search started from 20 masks can begin with no signal, while random search samples 600 masks globally. Not a bug; PSO hyperparameters were not changed (rule 10).
- Committed as "day4: PSO with random-key (top-k) decoding + stall swap".

## 2026-10-29 - day4 stage 3: search registry, seeded variants, real-data sanity

- New: src/search_registry.py run_search(method, fitness, state, d, k, budget, seed, hp) for firefly, firefly_levy, pso, firefly_seeded, pso_seeded. Plain methods pass init_masks=None (both searches then draw init_population("random", default_rng(seed)); keeps plain Firefly byte-identical to Day 3); seeded methods get init_population("seeded") from state["filter_scores"]. Closes review MAJOR 1: run_search raises ValueError on a reused state (evals != 0 or a diversity log present). Unknown hp keys raise.
- Tests: tests/test_registry.py 12 passed; pytest tests/ -q -> 106 passed.
- `python experiments/sanity_real.py` (run 20261029_140724_sanity_real, results/sanity_real.csv): digits, data seed 0, k=16, budget 300, default hp, validation fitness only (no test accuracy). ANOVA top-k (inner-train) 0.0658. init -> final: firefly 0.1400 -> 0.0987 (2 sweeps), firefly_levy 0.1400 -> 0.0823 (2), pso 0.1400 -> 0.0603 (33 iters), firefly_seeded 0.0658 -> 0.0658 (2), pso_seeded 0.0658 -> 0.0520 (63). All 300 evals used; seeded methods start at the ANOVA mask (assert OK). Warnings: the known digits constant columns [0 24 32 39] in f_classif.
- Notes: Firefly gets only 2 sweeps at budget 300 with 20 fireflies (up to n(n-1)/2 proposals per sweep), so firefly_seeded did not improve on its seeded start; PSO's final diversity is ~0.03-0.06 (collapsed) vs Firefly ~0.53-0.66. One seed, in-sample: no comparison is claimed.
- Committed as "day4: search registry, seeded variants, real-data sanity".

## 2026-10-29 - day4 stage 4: tuning

- New: experiments/tune.py (162-job one-factor grid on tuning seeds 100-102, 75/25 split of Xtr, score = final_clf on the 25 % holdout, resumable CSV, joblib loky with 1 thread per worker), experiments/tune_summary.py (decision rule, margin 0.003), tests/test_tuning.py, _common.git_hash().
- First dry run crashed: BrokenProcessPool / AttributeError "Can't get attribute '_tuning_data' on <module '__main__'>" (run_job pickled by value from __main__, its lru_cache'd helper by reference). Fix: main() submits tune.run_job imported by module name; regression test added. No rows were written by the crashed run.
- Dry run 2 (run 20261029_142831_tune_dryrun, results/tuning_runs_dryrun.csv): 6/6 jobs, evals 800 each, 30.7 s wall with 15 workers, mean search 17.6 s per run. tuning_acc (seed 100, default configs): digits k=16 firefly 0.8944 / pso 0.8917; cifar_r18 k=16 0.5770 / 0.6330; cifar_r18 k=64 0.7525 / 0.7405. One seed, no comparison claimed. git was 92d0964+dirty (Stage 4 files uncommitted).
- After dry run 2: worker warnings are now returned to the parent and printed once per category for the whole grid (they were printed once per worker process).
- Tests: pytest tests/ -q -> 116 passed before the warning change (+1 test for it).
- Committed as "day4: tuning protocol (tune.py, tune_summary.py) + dry run" (831fd93); 117 passed.
- Full grid: `python experiments/tune.py *> logs\tune.log`, started 14:33; 162/162 rows in results/tuning_runs.csv, all evals_used = 800, all git_hash = 831fd93 (clean). Search time per run ~21-69 s with 15 workers.
- `python experiments/tune_summary.py` (run 20261029_144439_tune_summary; results/tuning_summary.csv, results/tuning_decision.md). Its "831fd93+dirty" is the runs.json entry tune.py had just appended.
  - Firefly: n_fireflies=10 chosen, mean 0.7487 vs default 0.7430 (+0.0056 >= 0.003). Per condition vs default: cifar k=16 +0.0173, cifar k=64 +0.0088, digits k=16 -0.0093.
  - PSO: w_start=1.0 chosen, mean 0.7462 vs default 0.7431. Borderline, so recomputed exactly from the CSV (holdout accuracies are multiples of 1/2000 and 1/360): mean difference = 0.0030309 >= 0.003, a genuine pass, not a rounding artefact. Per condition: cifar k=16 -0.0190, cifar k=64 +0.0022, digits k=16 +0.0259 (the gain comes from digits only).
  - Both margins are small next to the per-condition stds (0.003-0.03, n=3 each): the decision is the output of the fixed protocol, not evidence that one setting is better. Report it that way.
  - Consequence: population sizes now differ (Firefly 10, PSO 20). Initial populations are shared only for the first 10 members; firefly_seeded seeds 5 members, pso_seeded 10. To be settled and recorded in PREREG at Stage 6.
- Committed as "day4: tuning grid results and decision".

## 2026-10-29 - day4 stage 5: resumable runner

- New: experiments/run_metaheuristics.py (thread env vars forced to 1 before numpy; job key (dataset, method, budget, seed, k); main process appends + fsyncs rows per finished job; --force moves old rows to meta_<dataset>_replaced.csv; test split used only in evaluate_on_test), config/smoke_config.yaml (default hp, smoke only, JSON-flow YAML because PyYAML is not installed), tests/test_runner.py; src/classifiers.py gains knn_clf() (k=5) and svm_clf() (LinearSVC, max_iter=5000, random_state=0); logs/ added to .gitignore.
- Decision: digits k=64 (= d) is skipped automatically by the runner (Stage 0 open item).
- Smoke: `python experiments/run_metaheuristics.py --smoke` (run 20261029_224452_meta_digits, results/smoke/meta_digits.csv): 16 jobs, 32 rows, 29.3 s with 15 workers, every job evals = 150, rerank 1.7-3.2 s. git was b006f11+dirty (Stage 5 files uncommitted; the Day 5 runs must start from a clean commit).
- Tests: tests/test_runner.py 6 passed (84.7 s, runs the smoke grid itself); pytest tests/ -q -> 123 passed in 252 s (was 69 s before the runner tests).
- Closes review MAJOR 2 for the runner (thread limits).

