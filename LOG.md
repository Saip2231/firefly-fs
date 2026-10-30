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
- Committed as 7a23b78 "day4: resumable metaheuristic runner + smoke test".

## 2026-10-29 - day4 stage 6: freeze, PREREG, compute plan

- Decision (user): keep the population sizes exactly as tuned (Firefly 10, PSO 20) and disclose the mismatch.
- Correction to the Stage-4 note and to results/tuning_decision.md (not edited, it is a committed result): plain populations share Firefly's 10 members, but seeded populations share only their first 5 (the number of score-biased members scales with n). Test added in tests/test_registry.py; PREREG.md states it.
- New: config/frozen_config.yaml (values from results/tuning_decision.md; test checks they match), PREREG.md (user draft + raw mask as primary, frozen hp, exact k lists and 12 families, tie/NaN rule, disclosed asymmetries incl. review MAJOR 3), experiments/plan_compute.py. experiments/_common.holm gains nan_policy="as_one" (default "exclude" unchanged, so Day-3 outputs are unchanged).
- Order: commit frozen config -> fill its hash into PREREG.md -> commit PREREG -> tag day5-launch.
- Tests: pytest tests/ -q -> 127 passed (258.9 s).
- `python experiments/plan_compute.py` (run 20261029_231932_plan_compute, results/compute_plan.md): 16 CPUs, 15 workers; contention factor cifar_r18 1.23, digits 1.00; 480 runs; 20.15 h on 1 worker, 1.34 h on 15 workers (ideal packing, 20 % margin). Per dataset x budget (15 workers): digits 800 0.06 h, digits 3000 0.21 h, cifar_r18 800 0.25 h, cifar_r18 3000 0.83 h; longest single run ~488 s (cifar_r18 k=128, budget 3000). Optional firefly_levy +0.08 h. No reduction needed.
- Frozen config committed: 7b985ca49205f025700593e9eba38a5268156b16 ("day4: frozen config"); hash filled into PREREG.md.

## 2026-10-29 - dashboard skeleton (app/)

- New: app/streamlit_app.py (tabs Overview, Benchmark explorer, Live playground; plotly; st.cache_data), app/dashboard_data.py (data logic, no Streamlit import; held-out split only in _score_on_test via src.evaluate.evaluate; playground hard cap 300 evaluations), requirements-app.txt (core libs pinned to the benchmark versions; streamlit/plotly lower bounds), tests/test_app.py.
- `pip install -r requirements-app.txt`; versions picked by pip: (fill in from `pip show streamlit plotly`).
- Tests: tests/test_app.py 8 passed; pytest tests/ -q -> 135 passed (237.9 s).
- `streamlit run app/streamlit_app.py`: Overview, explorer (meta_*.csv absent -> "Showing baselines only") and playground all render. Not deployed.
- Process note: one read-only `grep` was run through the shell by Claude while editing (CLAUDE.md rule 2); nothing changed.

## 2026-10-30 - day5: preflight and main runs

- experiments/preflight_day5.py (read-only): first run 1 FAIL (untracked .agents/ and .claude/, local tool folders) -> added to .gitignore, committed with the script ("day5: preflight script; ignore local tool folders", dd8c6dc). Second run: ALL CHECKS PASSED (18 items): tree clean; tag day5-launch -> b8c3209 holds PREREG.md and frozen_config.yaml identical to the working copy; PREREG cites 7b985ca (last change of the config) and was committed 2026-10-29T23:22:44+05:30, before any Day-5 row; config_hash b14f6b19...; grid 480 jobs, 960 rows, 12 families = PREREG; 16 CPUs, 34.8 GB free. Stall-swap share is not logged by the runner; a post-hoc diagnostic was preferred over changing src/ after the tag.
- OneDrive paused, sleep off, n_jobs = 15. Runs (no files edited between them):
  | run | command | row run_id | runs.json id | rows | runtime |
  |---|---|---|---|---|---|
  | 1 | `--dataset digits --budgets 800 3000` | 20261030_091644_meta_digits | 20261030_093218_meta_digits | 320 | 934.4 s |
  | 2 | `--dataset cifar_r18 --budgets 800` | 20261030_093755_meta_cifar_r18 | 20261030_095505_meta_cifar_r18 | 320 | 1029.9 s |
  | 3 | `--dataset cifar_r18 --budgets 3000` | (see CSV) | 20261030_105128_meta_cifar_r18 | 320 | 3167.4 s |
- Checks on the CSVs: results/meta_digits.csv 320 rows, results/meta_cifar_r18.csv 640 rows; every row has evals_used == budget and rerank_fits 0 (raw) / 50 (rerank); config_hash b14f6b19... on all rows.
- git_hash on the rows: digits `dd8c6dc`; both CIFAR runs `dd8c6dc+dirty`. Cause checked with `git status --short` during run 2: the only modified tracked file was results/runs.json (run 1's log_experiment append); results/meta_*.csv were untracked. Same code as dd8c6dc. Follow-up after Day 5: git_hash() should ignore changes under results/.
- Backups of meta_digits.csv and meta_cifar_r18.csv (after run 2) in ../backup_day5/ (outside the repo).

## 2026-10-30 - analysis A0: loader, integrity checks, style

- New: experiments/analysis/{__init__,common,a0_integrity,style,takeaway}.py, experiments/analyze.py (`--only A0`), tests/test_analysis_a0.py (21 tests: one deliberately broken frame per check, real results must pass).
- `python -m experiments.analyze --only A0` -> results/analysis/integrity_report.md: ALL CHECKS PASSED. Rows: baselines digits 190, cifar_r18 250; meta digits 320, cifar_r18 640. 140 groups with seeds 0-9 exactly once; no NaN accuracies; evals_used = budget on all 960 meta rows; idx length = k on 1310 rows (PCA and all excluded); single config_hash b14f6b19...; budgets {800, 3000}; seeds match; split hashes for seeds 0-9; 12 families = PREREG. git: digits dd8c6dc, cifar_r18 dd8c6dc+dirty (runs.json append, explained in the Day-5 entry).
- Tests: tests/test_analysis_a0.py 20 passed, 1 skipped (save_fig, matplotlib not installed); pytest tests/ -q -> 155 passed, 1 skipped.
- After the run: seeds in the seeds_match / split_hashes details now print as plain ints (numpy 2 printed np.int64(...)); report text only.
- Dependencies approved: matplotlib and statsmodels added to requirements.txt; installed and pinned: matplotlib 3.11.2, statsmodels 0.15.0. After install: pytest tests/ -q -> 156 passed. Committed as 0cf44ee "analysis: A0 loader, integrity checks, style".

## 2026-10-30 - analysis A1: RQ1 confirmatory tests

- New: experiments/analysis/a1_rq1.py (PREREG primary analysis: raw masks, acc_lr; paired two-sided Wilcoxon by seed via experiments._common.paired_wilcoxon, zero differences dropped, all-zero = TIE with p = 1; Holm within each family of 6 via holm(nan_policy="as_one"); seed and split-hash pairing asserted, NaN raises), tests/test_analysis_a1.py (16 tests: known WIN/LOSS/TIE, all-zero, undefined test keeps m = 6, raw p < 0.05 but Holm >= 0.05 -> TIE, guards; cross-checks vs scipy.stats.wilcoxon, statsmodels multipletests(method="holm"), pandas means).
- Tests: tests/test_analysis_a1.py 16 passed; pytest tests/ -q -> 172 passed.
- `python -m experiments.analyze --only A1` -> results/analysis/rq1_tests.csv (72 rows), descriptive.csv, rq1_summary.md. Result: 5 WIN, 24 LOSS, 43 TIE. Per comparator: anova 1W/3L/8T, mi 2W/2L/8T, l1 1W/3L/8T, rf 0W/8L/4T, pca 0W/8L/4T, pso 1W/0L/11T. Mean differences from -16.6 pts (vs PCA, cifar_r18 k=16 b=800) to +7.5 pts (vs MI, cifar_r18 k=16 b=3000). Digits: 23 TIE, 1 LOSS (l1, k=32, b=800). All 5 WINs are at cifar_r18 k=16 (MI at b=800; ANOVA, MI, L1, PSO at b=3000).
- Spot check (rule 12) of cifar_r18 k=16: means in rq1_tests.csv equal descriptive.csv (anova 0.6063, rf 0.6584, pca 0.7618); Holm values recomputed by hand from the raw p (smallest p 2/1024 x 6 = 0.0117). Firefly raw mean rises 0.5954 -> 0.6334 from b=800 to 3000 while PSO stays 0.5839 -> 0.5833 (consistent with PSO's collapsed final diversity; to examine in A7, exploratory). No sign of a bug.

## 2026-10-30 - analysis A2: figures F1 and F2

- New: experiments/analysis/a2_figures.py (F1 accuracy vs k per dataset x budget, mean +/- 1 std, all features as a line, PCA dashed "projection, not a selector", no seeded variants; F2 12 x 6 RQ1 grid coloured WIN/LOSS/TIE with mean difference and Holm p, tests recomputed by a1_rq1.run_rq1), takeaway.record() (upserts figures/takeaways.csv), tests/test_analysis_a2.py (8 tests).
- Tests: 8 passed; pytest tests/ -q -> 180 passed. `python -m experiments.analyze --only A2` wrote figures/F1_acc_vs_k_{digits,cifar_r18}_b{800,3000}.{png,pdf}, figures/F2_rq1_grid.{png,pdf}, figures/takeaways.csv (5 rows).
- Takeaways (computed): digits b800 Firefly 0.927 at k=16 (rank 2 of 8), 0.950 at k=32 (rank 6); cifar_r18 b800 0.595 at k=16 (rank 4), 0.789 at k=128 (rank 8); cifar_r18 b3000 0.633 at k=16 (rank 3), 0.797 at k=128 (rank 6); highest mean PCA at every CIFAR k shown; all features 0.971 (digits), 0.804 (cifar_r18).
- Visual check of F1 cifar_r18 b800 and F2 by Claude: consistent with rq1_tests.csv. Cosmetic: the L1 yellow is faint on white (not changed).

## 2026-10-30 - analysis A3: RQ2 cost (exploratory)

- New: experiments/analysis/a3_cost.py (rq2_cost.csv: mean/std of search_s, rerank_s, infer_s, evals_used, rerank_fits, total_fits, total_s per dataset x method x k x budget x mask_type; break-even N = total_s / (infer_all - infer_selected), "never" if <= 0; extra seconds per accuracy point over ANOVA; F4 accuracy vs search time, F5 break-even bars), tests/test_analysis_a3.py.
- Caveats written into outputs: search_s under parallel load (meta 15 workers, baselines joblib n_jobs=-1); infer_s = batch predict time / test rows, excluding the column slice and the PCA projection (checked in src/evaluate.py); indicative only, see A4.
- First run: 8 tests passed, 188 total; 140 rows. Review of the printed table found two presentation problems, fixed before commit: (1) digits k=64 (= d) baseline rows were included (all features, not a selection) -> now excluded as in the runner and F1; (2) a method faster AND more accurate than ANOVA got a negative "extra cost" (digits PCA k=32: -0.0321 s per pt) -> now "no extra time (faster than ANOVA)". Tests added for both. Numbers otherwise unchanged; rerun pending.
- Rerun after the fixes: 130 rows, digits k=64 gone, digits PCA k=16/32 "no extra time (faster than ANOVA)"; pytest tests/ -q -> 189 passed. Committed "analysis: A3 RQ2 cost (exploratory)".
- Observations (exploratory): with logistic regression all CIFAR break-even N are finite but large (Firefly 1.9e7 to 4.1e8 test samples); on digits most bars are "never" (inference times ~2e-6 s per sample, differences within timing noise). L1 is the slowest baseline on CIFAR (~840-860 s, saga under parallel load).

## 2026-10-30 - analysis A4: idle-machine timing (in progress)

- New: experiments/time_final.py (single-sample prediction timing on TRAINING rows for every seed-0 subset, LR and KNN, incl. column slice / PCA projection; contention check re-running 10 searches sequentially), tests/test_time_final.py.
- First run (run 20261030_224915_time_final, 288.6 s): determinism confirmed, idx matches the logged idx in 10/10 contention jobs (cifar_r18 k=16 b=800, firefly and pso, seeds 0-4); idle search 9.3-11.3 s vs logged 26.3-35.8 s -> contention_factor = 2.95. Note: time_fitness.md predicted ~23 s for these jobs (800 x 0.0285 s/eval), so that file's per-eval times are also ~2x the idle figure.
- First-run inference timings rejected (rule 12): timing_idle.csv shows a step change half-way through cifar_r18 (LR ~300-600 us with IQR 30-310 us up to anova k=128, then ~136 us with IQR 2-5 us for every later row; KNN likewise). Same code path, so the machine changed state during the run (background load or CPU power state); subsets timed early (digits, cifar 'all' and most baselines) look slower than those timed late (all meta masks), and every saving vs a single 'all' measurement is biased. Fix: 20 s CPU burn-in before timing; every subset bracketed by 100 all-features calls before and after with the same classifier, saving measured against those; rows flagged unstable if IQR > 25 % of median or reference drift > 20 %. Timing part to be rerun with --skip-contention (contention result kept).
- Rerun `python experiments/time_final.py --skip-contention` (run 20261030_230338_time_final, 228.0 s, burn-in 20 s, bracketing references): the step change is gone. 162/172 rows stable; 10 flagged (8 by reference drift up to 84.6 % in a 100-call reference block, 2 by IQR), kept in the CSV with stable=False; their medians are mostly in line with stable rows.
- Findings (medians over k and budget, single-sample prediction on training rows, idle machine):
  - Logistic regression: ~131-139 us for every subset AND for all features (cifar 131.9, digits 131.4); single-sample LR time is dominated by call overhead, so selecting features saves nothing (saving -2.7 to -4.8 us: the column slice costs a few us). PCA is slower than all features (~240-250 us; the projection adds ~110 us).
  - KNN (k=5): cifar_r18 all features 7414 us vs ~963 us for every selected subset -> ~6.4 ms saved per prediction; digits 315 vs ~293 us (~24 us saved). PCA on cifar ~1096 us.
  - Tests: tests/test_time_final.py 8 passed (user confirmed). Committed "analysis: A4 idle-machine timing and contention check".
  - Consequence for RQ2 (to use in the write-up, exploratory): with logistic regression, selection gives no single-sample inference saving, so break-even is "never"; with KNN on cifar_r18 the saving is ~6.4 ms per prediction. With the contention factor 2.95, an idle Firefly search at budget 800, k=16 takes ~10.5 s, i.e. a break-even of roughly 10.5 / 0.0064 ~ 1,600 KNN predictions (order of magnitude; to be computed by script, not by hand, if reported).

## 2026-10-30 - analysis A5: RQ3 stability (exploratory)

- New: experiments/analysis/a5_stability.py (mean pairwise Jaccard over 45 seed pairs per dataset x method x k [x budget x mask_type]; random references empirical and analytic (k^2/d)/(2k - k^2/d); feature-frequency counts; F6 stability bars, F7 feature-frequency curves; split-resampling caveat in every output), tests/test_analysis_a5.py (8 tests).
- Tests: 8 passed; pytest tests/ -q -> 205 passed. `python -m experiments.analyze --only A5` wrote results/analysis/rq3_stability.csv, rq3_feature_frequency.csv, figures F6 (2) and F7 (6).
- Sanity: random empirical vs analytic Jaccard agree (ratio 0.87-1.16; cifar_r18 k=128 0.144 vs 0.143; digits k=32 0.336 vs 0.333), so idx parsing and the computation are consistent.
- Results (raw masks, cifar_r18): ANOVA 0.92-0.98, MI 0.81-0.91, RF 0.71-0.84, L1 0.26-0.41; Firefly 0.05-0.16 and PSO 0.035-0.145 (random 0.014-0.144); seeded variants 0.07-0.27. Digits: Firefly 0.33-0.42 vs random 0.14-0.34; ANOVA reaches 1.00 at k=32. Features chosen in >= 5 of 10 seeds, cifar_r18 k=16: Firefly 1 (b800) / 5 (b3000), PSO 0 / 0, ANOVA 16, RF 16. The wrapper searches pick largely different subsets per seed (consistent with noisy validation fitness and ~20 independent hill climbers); filter baselines are far more stable.

## 2026-10-30 - analysis A6a: baseline transfer evaluation (touches the test split)

- New: experiments/eval_baseline_transfer.py (every baseline subset with k < d scored by run_metaheuristics.evaluate_on_test; the load_dataset tuple is passed unopened; no search, tuning or edits), tests/test_eval_baseline_transfer.py (3 tests incl. a static no-Xte/yte check and one real job). common.load_baseline_transfer() added.
- Tests: 3 passed; pytest tests/ -q -> 208 passed.
- `python experiments/eval_baseline_transfer.py` (run 20261030_233155_eval_baseline_transfer, 66.9 s, 15 workers): results/baselines_transfer_digits.csv (100 rows), results/baselines_transfer_cifar_r18.csv (200 rows); no warnings. Cross-check: acc_lr from evaluate_on_test equals the baseline CSV's acc in 100/100 and 200/200 rows.
- Pattern (descriptive): digits KNN above LR (e.g. ANOVA k=16 0.951 vs 0.922); cifar_r18 KNN below LR (ANOVA k=16 0.545 vs 0.606); linear SVM close to LR on both.
- Committed "analysis: A6a baseline transfer evaluation".

## 2026-10-30 - analysis A6: RQ4 transfer (exploratory, descriptive)

- New: experiments/analysis/a6_transfer.py (rq4_transfer.csv: firefly/pso vs anova, mi, l1, rf per family and classifier, paired mean differences in points, seeds higher/lower, no p-values; rq4_gaps.csv: LR-KNN and LR-SVM gaps; F8), tests/test_analysis_a6.py (7 tests, incl. LR reference rows = A1 numbers).
- Tests: 7 passed; pytest tests/ -q -> 215 passed. Outputs: rq4_transfer.csv 288 rows, rq4_gaps.csv 78 rows, F8 x 2.
- Results (descriptive): under KNN and SVM the Firefly/PSO subsets trail RF on cifar_r18 (about -2 to -10 pts) and are mixed vs ANOVA/MI/L1; the largest positive cells are again cifar_r18 k=16 b=3000 (+2.5 to +8.1). Firefly's mean is above the comparator's in 7/32 (KNN) and 8/32 (SVM) cifar_r18 cells, 3/16 and 6/16 on digits. LR-minus-KNN gaps on cifar_r18: Firefly 6.6-7.6 pts vs ANOVA 5.7-6.1 (LR-tailored subsets transfer slightly worse to KNN); digits gaps negative (KNN above LR).

## 2026-10-30 - analysis A7: ablations (exploratory)

- New: experiments/analysis/a7_ablations.py (seeded vs plain, rerank vs raw with total_fits / rerank_s / masks changed, budget 3000 vs 800; Wilcoxon p labelled "exploratory, not corrected across families"; firefly_levy reported as not run; F3, F9), tests/test_analysis_a7.py (9 tests).
- First test run: 8 of 9 failed/errored. Cause: _pair(variant, reference, **info) while every caller also passed variant=... and reference=... labels in **info -> TypeError "got multiple values for argument 'variant'". Fix: parameters renamed (v_df, r_df). No change to any computation.
- Rerun: tests/test_analysis_a7.py 9 passed; pytest tests/ -q -> 224 passed. `python -m experiments.analyze --only A7` -> results/analysis/ablations.csv (96 rows: 24 seeded_vs_plain, 48 rerank_vs_raw, 24 budget_3000_vs_800), F3 and F9 for both datasets. firefly_levy: not run (reported).
- Results (exploratory, uncorrected): budget 800 -> 3000 changes Firefly's mean acc by +0.8 to +3.8 pts on cifar_r18 (k=16 +3.8, consistent with A1: 0.5954 -> 0.6334) and -0.4 to +0.2 on digits; PSO -0.1 to +0.9 (cifar_r18) and +0.8 to +1.4 (digits), for 3.75x the evaluations. Filter seeding: -0.4 to +4.5 pts on cifar_r18 (largest for PSO at k=16: +3.9 b800, +4.5 b3000, 10/10 seeds) and -0.5 to +1.7 on digits. Re-ranking: -0.2 to +1.9 pts (cifar_r18), -0.6 to +0.7 (digits) at +50 fits (mean rerank_s 13.9 s cifar_r18, 2.0 s digits, under parallel load); it picked a different mask than the raw search in 285/320 (cifar_r18) and 141/160 (digits) runs. Raw p < 0.05 in 27/64 (cifar_r18) and 3/32 (digits) comparisons, uncorrected.

## 2026-10-30 - analysis A8 deferred, A9 manifest and rebuild check

- A8 (convergence): no results/traces/*.npz exist; user decision: skip for now, return later. experiments/analysis/a8_convergence.py reports the skip; the manifest records it.
- A9: experiments/analysis/a9_manifest.py (REGISTRY of every stage's outputs, inputs and command; manifest with SHA-256 and row counts of sources and outputs, git hash, date; unregistered-output check), experiments/analysis/a9_rebuild.py, experiments/analyze.py (full run A0..A8 + manifest; --only; --rebuild-check). A2 now also writes figures/data/F1_frames.csv (F1's plotted numbers) so figures can be compared by data.
- Deviation from ANALYSIS.md, stated: the final check does not DELETE figures/ and results/analysis/, because results/analysis/ also holds the A4 idle-machine timings that analyze cannot regenerate (and CLAUDE.md asks before deleting results). Instead it moves every analyze-owned file into logs/rebuild_snapshot_<time>/ (git-ignored), rebuilds, and compares (CSV exact, markdown ignoring the timestamp line, figures via their data CSVs). A4 and A6a outputs are never touched.

