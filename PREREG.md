# Pre-registration (written before the real-data metaheuristic runs)

- Date: 29-10-2026
- Frozen config: `config/frozen_config.yaml`, committed as "day4: frozen config" at commit `7b985ca49205f025700593e9eba38a5268156b16`.
- Code state for the analysis: the commit that adds this file, tagged `day5-launch`.

## Questions
- RQ1 accuracy at fixed k
- RQ2 search cost and break-even
- RQ3 subset stability across seeds
- RQ4 transfer of the selected subsets to KNN and linear SVM

## Design fixed before data
- Datasets: `digits` (64-d) and `cifar_r18` (10,000-image CIFAR-10 subset embedded with pretrained ResNet-18, 512-d; train 8000, test 2000).
- Splits: `load_dataset(name, seed)`, seeds 0-9, stratified 80/20, scaler fitted on train only. Splits are identical across methods for a seed (verified in `results/split_hashes.csv`).
- k values: digits {16, 32}; cifar_r18 {16, 32, 64, 128}. Digits k = 64 equals d (all features) and is excluded.
- Budgets: 800 and 3000 new (uncached) fitness evaluations; the budget is checked before every fitness call.
- Methods run on Day 5: `firefly` (method under test), `pso` (PSO with random-key (top-k) decoding), `firefly_seeded` and `pso_seeded` (filter-seeded ablations). `firefly_levy` ("long-jump mutation", not a Levy flight) only at budget 800 and only if time allows; exploratory.
- Fitness: logistic regression (max_iter=100) validation error on an inner stratified 75/25 split of at most 2000 training rows (`make_fitness`, seed = run seed); identical inner split for every method of a seed.
- Final classifier for every method: logistic regression (max_iter=500, random_state=0) trained on the full training split, scored once on the test split. Transfer: KNN (k=5) and LinearSVC (max_iter=5000, random_state=0).
- Frozen hyperparameters (from the one-factor tuning on seeds 100-102, `results/tuning_decision.md`):
  - Firefly: n_fireflies 10, beta0 0.4, gamma 2.0, alpha 0.02
  - PSO: n_particles 20, w_start 1.0, w_end 0.4, c1 = c2 = 1.5, vmax 0.5, stall swap on
  - frac_seeded 0.5; re-rank top_m 10, 5-fold CV on the training split
- Baselines: `results/baselines_<dataset>.csv` (ANOVA F, Mutual Info, L1 Logistic, RF Importance, PCA, Random, All Features), same seeds and splits, same final classifier. Baselines do not depend on the search budget: the same baseline rows are compared at budget 800 and at 3000.

## Primary comparison (RQ1)
- Metric: test accuracy of the final logistic regression (`acc_lr`).
- Firefly mask: the **raw** best mask (`mask_type = raw`), the only one with a matched budget. The re-ranked mask costs 50 extra fits and sees more rows; it is a secondary comparison.
- For each dataset x k x budget (12 families: digits 2 k x 2 budgets, cifar_r18 4 k x 2 budgets): plain Firefly vs each of {ANOVA F, Mutual Info, L1 Logistic, RF Importance, PCA, PSO (raw mask)}.
- Test: paired two-sided Wilcoxon signed-rank by seed (n = 10). Zero differences are dropped (zero_method "wilcox"); if all 10 differences are zero the comparison is TIE with p = 1.
- Multiple testing: Holm correction across the 6 comparisons within each family, family size fixed at 6 (an undefined test counts as p = 1: `holm(..., nan_policy="as_one")`). No correction across the 12 families.
- Labels: WIN = Holm-adjusted p < 0.05 and mean difference > 0; LOSS = Holm-adjusted p < 0.05 and mean difference < 0; TIE otherwise.
- Reported with every label: mean difference in accuracy points, std of the paired differences, number of non-zero pairs, raw and adjusted p, n = 10.
- PCA is a projection, not a selection method: it needs the full embedding plus a projection matrix at inference. It is compared and reported as such.

## Secondary comparisons (exploratory, no correction)
- filter-seeded vs plain (Firefly and PSO), raw masks
- re-ranked vs raw mask, with the extra cost (rerank_fits, rerank_s)
- budget 3000 vs 800
- RQ2 cost: search_s, rerank_s, inference time per sample with all features vs the selected k; break-even N = search_s / (infer_all - infer_selected), for logistic regression and KNN; stated plainly when the saving is tiny or negative
- RQ3 stability: mean pairwise Jaccard of the selected sets across seeds, per method and k, with the Random baseline as reference
- RQ4 transfer: KNN and linear SVM accuracy of the logistic-regression-selected subsets vs the ANOVA and MI subsets
- convergence: best validation fitness vs evaluations (Firefly vs PSO)

## Disclosed asymmetries
- Data seen during selection: baseline selectors fit on all training rows (8000 on cifar_r18, 1437 on digits); the wrapper fitness sees at most 1500 training and 500 validation rows. Every final classifier is trained on the full training split.
- Population sizes differ as tuned (Firefly 10, PSO 20). Plain methods: PSO's first 10 particles are exactly Firefly's 10-member swarm. Seeded methods: score-biased members are about frac_seeded x n (5 of 10 for firefly_seeded, 10 of 20 for pso_seeded), so the two seeded populations share only their first 5 members (the ANOVA top-k mask and 4 score-biased masks); from member 5 on, firefly_seeded draws uniform masks and pso_seeded score-biased ones. Kept as tuned (no post-hoc change).
- PSO uses a stall swap (one random key swap per particle when an iteration's moves give no new evaluation), the same rule as Firefly's tie-stall fallback, added so PSO spends its budget (`LOG.md`, Stage 2).
- The tuning margins were small (Firefly +0.0056, PSO +0.0030 over the default) relative to the per-condition spread; the frozen values are the output of a fixed rule, not evidence that they are better.

## Limitations accepted in advance
- n = 10 seeds limits power: the smallest two-sided Wilcoxon p is about 0.002, so a Holm-adjusted WIN or LOSS needs nearly all seeds to agree.
- One backbone (ResNet-18), two datasets.
- Wrapper fitness is noisy: validation standard error about 2-3 accuracy points (500 validation rows on cifar_r18, 360 on digits); selected subsets can beat the true subset on validation and lose on test.
- Timing is measured on one machine (16 logical CPUs, single-thread BLAS per worker, parallel jobs); absolute seconds are machine-specific.

## What we will do if Firefly does not win
Report it. A tie or loss with measured cost is a valid finding; the contribution is then the benchmark (leakage-safe protocol, matched budgets, stability, cost).
