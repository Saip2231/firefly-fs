# ANALYSIS.md: rules and stage specs for the analysis workstream (A0 to A9)

Import this file from CLAUDE.md with the line `@ANALYSIS.md`. It is the specification. Each session you are told "Run Stage A<n> of ANALYSIS.md" and nothing else. Follow CLAUDE.md first; this file adds analysis-specific rules and never overrides them.

## 1. Analysis-mode rules (always on while doing analysis)

1. **Read-only on experiments.** Do not edit anything in `src/`, `config/`, `PREREG.md`, `results/meta_*.csv`, or `results/baselines_*.csv`. Do not rerun the main experiments. Outputs go to `results/analysis/`, `figures/`, and `experiments/analysis/`.
2. **PREREG.md is authoritative** for the primary test, the 12 families, Holm scope, tie/NaN handling, and the WIN/LOSS/TIE labels. Read it first. If this file and PREREG.md disagree, stop and ask. Never invent a rule PREREG does not state.
3. **Primary results use `mask_type == "raw"` only.** Re-ranked rows appear only in the ablation (A7), always with `total_fits` and `rerank_s`.
4. **Pairing:** paired by seed (0 to 9). Before any paired test, assert that both sides have exactly the same seeds, and that split hashes agree (`results/split_hashes.csv`).
5. **Units:** accuracy differences are reported in accuracy points (percentage points), p-values with their Holm-adjusted versions, n always stated. Never report a p-value without the effect size next to it.
6. **Never** plot or report a best seed. Report mean and std over seeds. Say that n=10 limits power wherever a test appears.
7. **Exploratory vs confirmatory:** only A1 is confirmatory. A3 to A7 are exploratory and must be labelled "exploratory" in the output file, figure title or caption.
8. **Takeaways are computed from data:** every figure gets a one-sentence takeaway produced by code from the same dataframe (numbers formatted from data, never typed). Figure titles describe, they never claim ("Accuracy vs k", not "Firefly wins").
9. **Wording:** neutral language. No "beats" or "better" without a test result and effect size. Say "long-jump mutation", "PSO with random-key (top-k) decoding and stall-swap fallback", "CIFAR-10 subset with ResNet-18 embeddings". PCA is labelled "projection, not a selector" wherever it appears.
10. **Determinism:** no randomness except bootstrap resampling with a fixed seed (state it). Every output is reproducible by `python -m experiments.analyze`.
11. **Cross-check:** every statistical routine is verified against an independent implementation (scipy `wilcoxon`, statsmodels `multipletests(method="holm")`, plain pandas for means) in tests using synthetic data with known answers.
12. **Surprises:** if a number looks too good, too bad, or inconsistent with earlier tables, investigate for a bug before presenting it.
13. **Stage gates:** explain the plan in a few lines, implement, run tests and the stage, show exact output, summarise in 5 lines, then **STOP**. Wait for "continue".
14. **Test data:** analysis scripts read CSVs only. The only scripts allowed to touch the test split are A6a (reusing the runner's single evaluation function) and none else. A4 times inference on training rows, never test rows.

## 2. Data contracts

- `results/baselines_<dataset>.csv`: one row per (method, seed, k). Columns include `dataset, method, seed, k, acc, n_feat, fit_s, infer_s, search_s, idx`. Methods include `all, random, anova, mi, l1, rf, pca`. `acc` is logistic regression via `final_clf()`. Inspect the real method labels and put the mapping in ONE place (`experiments/analysis/common.py`).
- `results/meta_<dataset>.csv`: one row per (method, budget, seed, k, mask_type). Columns: `dataset, method, budget, seed, k, mask_type, acc_lr, acc_knn, acc_svm, n_feat, search_s, rerank_s, infer_s, evals_used, rerank_fits, val_fitness, diversity_last, idx, config_hash, git_hash, run_id`. Methods: `firefly, pso, firefly_seeded, pso_seeded` (and `firefly_levy` if it was run).
- Baselines have no budget. For budget b, compare meta rows at budget b against the same baseline rows.
- Digits and CIFAR have different k lists. Read them from the files, never hard-code.
- `search_s` in the meta files was measured under 15-way parallel load. Always label it so.
- CIFAR rows carry a `+dirty` git hash caused only by the log file. Do not treat that as an error, but list distinct `git_hash` values in the integrity report.

## 3. Shared code (Stage A0 builds it)

`experiments/analysis/common.py` (method names, family definitions, loaders), `loader checks`, `style.py` (consistent colour and marker per method, PCA dashed and labelled projection, log-scale helpers, `save_fig()` writing PNG at 200 dpi and PDF), `takeaway.py` (functions that build sentences from dataframes). The orchestrator is `experiments/analyze.py` with `--only A3` support; stages live in `experiments/analysis/a<n>_*.py`, each with `run()`.

## 4. Stage specs

### A0: loader, integrity checks, style
- `load_baselines(dataset)`, `load_meta(dataset)` return tidy dataframes with a parsed `idx` column (set of ints).
- `validate()` asserts and writes `results/analysis/integrity_report.md`: row counts per file; every (dataset, method, budget, k, mask_type) has seeds 0 to 9 exactly once; no NaN in accuracy columns; `evals_used <= budget`; selectors' `idx` length equals k (PCA and `all` excluded); a single `config_hash`; budgets in {800, 3000}; baseline and meta seeds match; split hashes exist for seeds 0 to 9; list distinct git hashes.
- Tests with small synthetic frames, including a deliberately broken frame for each check.
- Show the integrity report. STOP.

### A1: RQ1 confirmatory tests
- Primary metric: `acc_lr`, `mask_type=raw`. For each of the 12 families (dataset x k x budget), plain `firefly` vs each of `anova, mi, l1, rf, pca, pso`.
- Paired two-sided Wilcoxon over seeds 0 to 9, using exactly the zero-difference and NaN handling stated in PREREG.md. Holm correction within each family of 6.
- Output `results/analysis/rq1_tests.csv`: `dataset, k, budget, comparator, n, mean_firefly, mean_comparator, mean_diff_pts, std_diff_pts, nonzero_pairs, p_raw, p_holm, label` (label WIN, LOSS, or TIE per PREREG). Also `results/analysis/descriptive.csv`: mean and std for every method including `all` and `random`, with `acc` or `acc_lr` as appropriate.
- Print a pivot (families x comparators) of labels and mean differences.
- Tests: known win, known loss, known tie, all-zero differences, one NaN, mismatched seeds (must raise), and Holm checked against statsmodels.
- Show output. STOP.

### A2: figures F1 and F2
- F1 `figures/F1_acc_vs_k_<dataset>_b<budget>.png`: accuracy vs k for `all` (horizontal line), `random, anova, mi, l1, rf, pca, firefly, pso`, mean with std bands. Seeded variants do not appear here.
- F2 `figures/F2_rq1_grid.png`: grid of 12 families (rows) by 6 comparators (columns), colour by WIN/LOSS/TIE, annotated with mean difference in points and the adjusted p. Add a legend line "paired Wilcoxon, Holm within family, n=10".
- Each figure gets a computed takeaway saved beside it in `figures/takeaways.csv`.
- Show the figures (describe or open them). STOP.

### A3: RQ2 cost (exploratory)
- `results/analysis/rq2_cost.csv` per (dataset, method, k, budget): mean and std of `search_s`, `rerank_s`, `infer_s`, `evals_used`, `rerank_fits`, `total_fits`. Baselines from `baselines_*.csv` (search_s, infer_s).
- Break-even `N = search_s / (infer_all - infer_selected)` where `infer_all` comes from the `all` rows. If the denominator is zero or negative, write "never". Mark inference times "indicative only, see A4".
- Extra cost over ANOVA: `(search_s_method - search_s_anova) / mean_diff_pts` when the method is more accurate than ANOVA by the mean difference; otherwise "no accuracy gain". Report the mean difference next to it.
- Figures: F4 accuracy vs `search_s` (log x) per dataset and k, all methods; F5 break-even bars (log y, "never" shown as a hatched bar).
- Label every `search_s` "measured under 15-way parallel load". STOP.

### A4: idle-machine timing (computes, not CSV-only)
- `experiments/time_final.py`. Preconditions printed at start: plugged in, other apps closed, `OMP_NUM_THREADS=1`.
- For stored raw masks (seed 0, each method and k): fit `final_clf()` and KNN (k=5) on **training rows** with the selected columns and on all columns, then time single-sample prediction on **training rows** (never test rows), 1,000 repeats after a 50-repeat warm-up; report median and IQR, in microseconds.
- Contention check: re-run about 10 search jobs sequentially (Firefly and PSO, CIFAR k=16, budget 800, seeds 0 to 4, same frozen config) with `time.perf_counter`, and compare `search_s` with the same jobs in `meta_cifar_r18.csv`. Also assert that `idx` equals the logged `idx` (determinism).
- Output `results/analysis/timing_idle.csv` and a short `contention_factor` line. STOP.

### A5: RQ3 stability (exploratory)
- Mean pairwise Jaccard of `idx` sets across seeds 0 to 9, per dataset, method, k, and budget (meta methods) or per k (baselines). Exclude PCA and `all`.
- Reference values: the empirical Jaccard of the `random` baseline, and the analytic expectation `(k^2/d) / (2k - k^2/d)`.
- Note in the output: baseline stability includes variation from different training splits per seed, so it measures sensitivity to resampling, not search noise alone.
- F6 `figures/F6_stability_<dataset>.png` (bars with random reference), F7 `figures/F7_feature_frequency_<dataset>_k<k>.png` (how often each feature index was chosen across seeds, sorted; print how many features were chosen in at least 5 of 10 seeds). STOP.

### A6a: baseline transfer evaluation (this stage touches the test split)
- The baseline CSV has accuracy for logistic regression only, so ANOVA/MI/L1/RF subsets have no KNN or SVM numbers. Write `experiments/eval_baseline_transfer.py` that, for every baseline row with an `idx`, loads `load_dataset(dataset, seed)` and evaluates KNN (k=5) and the same linear SVM used in the runner, **by importing the runner's existing single evaluation function** (the only place test data is used). No search, no tuning, no changes to existing files.
- Output `results/baselines_transfer_<dataset>.csv` (`dataset, method, seed, k, acc_knn, acc_svm, idx`). Log with `log_experiment()`. Add a test that the script contains no direct reference to `Xte` or `yte` except through the imported function. STOP.

### A6: RQ4 transfer (exploratory)
- For each family, compare `acc_knn` and `acc_svm` of Firefly and PSO subsets (selected under logistic regression) with the ANOVA, MI, L1, and RF subsets from A6a. Report mean, std, and the paired mean difference in points (no p-values or an explicit exploratory label).
- Also report the within-method gap `acc_lr - acc_knn` and `acc_lr - acc_svm` to show how subset quality moves across classifiers.
- F8 `figures/F8_transfer_<dataset>.png`. STOP.

### A7: ablations (exploratory)
- Seeded vs plain (Firefly and PSO), re-ranked vs raw (each row reports `total_fits` and `rerank_s`; the comparison is paired by seed), budget 3000 vs 800 (F3 `figures/F3_budget_curve_<dataset>.png`). `firefly_levy` only if present.
- Paired mean differences in points with Wilcoxon p-values labelled "exploratory, not corrected across families". F9 `figures/F9_ablations_<dataset>.png`. STOP.

### A8: convergence (needs demo traces)
- Runs only if `results/traces/*.npz` exist. Plot best validation fitness vs evaluations for Firefly and PSO from the traces.
- Label as "single logged run (seed 0)" unless traces for several seeds exist, in which case plot mean and std. Optional: record traces for seeds 0 to 9 for CIFAR k=16 and 64, budget 800, so the curve is a mean over seeds. STOP.

### A9: manifest and orchestrator
- `experiments/analyze.py` runs A0, A1, A2, A3, A5, A6, A7, A8 (if traces) in order and writes `results/analysis/analysis_manifest.md`: every output file, its source CSVs with SHA-256 and row counts, git hash, date, and the exact command. The run fails if any output exists without a manifest entry. A4 and A6a are separate scripts and appear in the manifest with their own commands.
- Final check: delete `figures/` and `results/analysis/`, run `python -m experiments.analyze`, and confirm every file regenerates identically (figures are compared by their underlying dataframes). STOP.

## 5. Order and dependencies

A0, A1, A2 first (confirmatory results). Then A3, A5, A7. A4 and A6a can run in parallel on an idle machine. A6 after A6a. A8 after the dashboard team records traces. A9 last, then rerun everything from a clean state.

## 6. Reviewer pass (fresh session, read-only, after A1 and again after A9)

Check: pairing and seed alignment; family count equals PREREG; Holm applied within the right family; zero-difference handling matches PREREG; baseline rows reused across both budgets correctly; raw vs re-ranked rows never mixed in primary results; units (points vs fraction); PCA labelled projection everywhere; any "best seed" or unlabelled exploratory result; figure axes, units, legends; any hard-coded numbers in takeaways; test cross-checks against scipy and statsmodels. Report severity, file:line, why, minimal fix.

## 7. Stop and ask if

- PREREG.md and this file disagree.
- Any integrity check fails or a CSV looks different from section 2.
- A result contradicts an earlier table.
- A stage needs a change to `src/`, `config/`, or an existing CSV.
- A new dependency is needed.

## 8. Definition of done (per stage)

Tests pass (`pytest tests/ -q`), outputs exist with the names above, takeaways computed from data, exploratory labels present, manifest updated, reviewer findings of severity BLOCKER or MAJOR resolved, commit made with message `analysis: A<n> <what>`.