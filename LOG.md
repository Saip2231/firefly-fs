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
