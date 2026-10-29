# Day 3 summary

All numbers below are computed from the CSV files in `results/`.

## Stage 1 - d=500 diagnostics (`diagnostics_d500.csv`, seeds 0-4, means)

| config | method | n | evals_used | sweeps | best_init | best_end | hits | inf_cols_tried | mean_inf_per_eval | diversity_last |
|---|---|---|---|---|---|---|---|---|---|---|
| d500_ni5_k5_n1000_b600 | firefly | 5 | 600.0000 | 4.0000 | 0.3953 | 0.2036 | 1.8000 | 4.4000 | 0.1150 | 0.9459 |
| d500_ni5_k5_n1000_b600 | firefly_levy | 5 | 600.0000 | 4.0000 | 0.3953 | 0.2718 | 1.0000 | 3.4000 | 0.0590 | 0.9556 |
| d500_ni5_k5_n1000_b600 | random_search | 5 | 600.0000 | 0.0000 | 0.3953 | 0.2329 | 1.0000 | 5.0000 | 0.0477 | nan |
| d500_ni10_k10_n2000_b800 | firefly | 5 | 800.0000 | 5.0000 | 0.3372 | 0.1800 | 3.4000 | 9.2000 | 0.8280 | 0.8342 |
| d500_ni10_k10_n2000_b800 | firefly_levy | 5 | 800.0000 | 5.0000 | 0.3372 | 0.1871 | 2.8000 | 9.8000 | 0.7678 | 0.8724 |
| d500_ni10_k10_n2000_b800 | random_search | 5 | 800.0000 | 0.0000 | 0.3372 | 0.2485 | 1.8000 | 10.0000 | 0.1935 | nan |

Verdicts:

| config | method | verdict |
|---|---|---|
| d500_ni5_k5_n1000_b600 | firefly | OK x5 |
| d500_ni5_k5_n1000_b600 | firefly_levy | OK x4, NO IMPROVEMENT x1 |
| d500_ni5_k5_n1000_b600 | random_search | OK x4, NO IMPROVEMENT x1 |
| d500_ni10_k10_n2000_b800 | firefly | OK x5 |
| d500_ni10_k10_n2000_b800 | firefly_levy | OK x5 |
| d500_ni10_k10_n2000_b800 | random_search | OK x5 |

## Stage 2 - clean signal (`validate_clean_signal_rescored.csv`, seeds 0-9)

class_sep=2.0, 1 cluster/class, k=5, budget=600, 5000-row independent holdout. `sep_*` count only informative columns whose class means differ (`separating_columns_clean_signal.csv`). Primary metrics: holdout_acc and sep_hits; hits and val_fitness are secondary (val_fitness is in-sample: it is the objective the search minimised on the rows that chose the mask). Holdout classifier: LogisticRegression(max_iter=500, random_state=0).

| d | n_search | method | n | hits | sep_hits | n_sep | sep_recall | val_fitness | holdout_acc | evals_used |
|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 1000 | firefly | 10 | 2.80 | 2.10 | 3.20 | 0.7250 ± 0.2486 | 0.0061 ± 0.0019 | 0.9899 ± 0.0057 | 600.0 |
| 50 | 1000 | firefly_levy | 10 | 2.60 | 2.20 | 3.20 | 0.7500 ± 0.2357 | 0.0073 ± 0.0033 | 0.9908 ± 0.0036 | 600.0 |
| 50 | 1000 | random_search | 10 | 2.70 | 2.20 | 3.20 | 0.7500 ± 0.2357 | 0.0097 ± 0.0041 | 0.9899 ± 0.0046 | 600.0 |
| 50 | 1000 | true_mask | 10 | 5.00 | 3.20 | 3.20 | 1.0000 ± 0.0000 | 0.0077 ± 0.0033 | 0.9923 ± 0.0031 | 1.0 |
| 50 | 2000 | firefly | 10 | 2.60 | 2.20 | 3.20 | 0.7500 ± 0.2357 | 0.0075 ± 0.0057 | 0.9912 ± 0.0032 | 600.0 |
| 50 | 2000 | firefly_levy | 10 | 2.90 | 2.30 | 3.20 | 0.7750 ± 0.2189 | 0.0069 ± 0.0045 | 0.9908 ± 0.0034 | 600.0 |
| 50 | 2000 | random_search | 10 | 2.50 | 2.20 | 3.20 | 0.7500 ± 0.2357 | 0.0095 ± 0.0073 | 0.9899 ± 0.0042 | 600.0 |
| 50 | 2000 | true_mask | 10 | 5.00 | 3.20 | 3.20 | 1.0000 ± 0.0000 | 0.0067 ± 0.0041 | 0.9923 ± 0.0031 | 1.0 |
| 500 | 1000 | firefly | 10 | 1.20 | 1.20 | 3.20 | 0.3750 ± 0.1768 | 0.0678 ± 0.1082 | 0.9102 ± 0.1485 | 600.0 |
| 500 | 1000 | firefly_levy | 10 | 1.40 | 1.40 | 3.20 | 0.4750 ± 0.2189 | 0.0456 ± 0.0295 | 0.9522 ± 0.0276 | 600.0 |
| 500 | 1000 | random_search | 10 | 1.10 | 1.10 | 3.20 | 0.3750 ± 0.1318 | 0.0373 ± 0.0183 | 0.9512 ± 0.0272 | 600.0 |
| 500 | 1000 | true_mask | 10 | 5.00 | 3.20 | 3.20 | 1.0000 ± 0.0000 | 0.0108 ± 0.0075 | 0.9921 ± 0.0032 | 1.0 |
| 500 | 2000 | firefly | 10 | 1.70 | 1.70 | 3.20 | 0.5500 ± 0.1581 | 0.0302 ± 0.0317 | 0.9675 ± 0.0333 | 600.0 |
| 500 | 2000 | firefly_levy | 10 | 1.80 | 1.80 | 3.20 | 0.6000 ± 0.2415 | 0.0298 ± 0.0239 | 0.9714 ± 0.0205 | 600.0 |
| 500 | 2000 | random_search | 10 | 1.10 | 1.10 | 3.20 | 0.3750 ± 0.1318 | 0.0458 ± 0.0304 | 0.9530 ± 0.0244 | 600.0 |
| 500 | 2000 | true_mask | 10 | 5.00 | 3.20 | 3.20 | 1.0000 ± 0.0000 | 0.0086 ± 0.0042 | 0.9923 ± 0.0030 | 1.0 |

Paired two-sided Wilcoxon vs random_search (mean_diff = method - random; zero differences dropped). p_holm: Holm correction across all rows of this table. Primary test: holdout_acc, firefly, d=500, n_search=2000.

| d | n_search | method | metric | direction | mean_diff | n | n_nonzero | p | p_holm |
|---|---|---|---|---|---|---|---|---|---|
| 50 | 1000 | firefly | holdout_acc | lower | -0.0000 | 10 | 9 | 0.6680 | 1.0000 |
| 50 | 1000 | firefly | sep_hits | lower | -0.1000 | 10 | 1 | 1.0000 | 1.0000 |
| 50 | 1000 | firefly | hits | higher | 0.1000 | 10 | 6 | 1.0000 | 1.0000 |
| 50 | 1000 | firefly | val_fitness | lower | -0.0036 | 10 | 6 | 0.0312 | 0.9062 |
| 50 | 1000 | firefly_levy | holdout_acc | higher | 0.0009 | 10 | 8 | 0.9844 | 1.0000 |
| 50 | 1000 | firefly_levy | sep_hits | equal | 0.0000 | 10 | 0 | nan | nan |
| 50 | 1000 | firefly_levy | hits | lower | -0.1000 | 10 | 3 | 1.0000 | 1.0000 |
| 50 | 1000 | firefly_levy | val_fitness | lower | -0.0024 | 10 | 4 | 0.1250 | 1.0000 |
| 50 | 2000 | firefly | holdout_acc | higher | 0.0014 | 10 | 7 | 0.0469 | 1.0000 |
| 50 | 2000 | firefly | sep_hits | equal | 0.0000 | 10 | 2 | 1.0000 | 1.0000 |
| 50 | 2000 | firefly | hits | higher | 0.1000 | 10 | 5 | 1.0000 | 1.0000 |
| 50 | 2000 | firefly | val_fitness | lower | -0.0020 | 10 | 4 | 0.1250 | 1.0000 |
| 50 | 2000 | firefly_levy | holdout_acc | higher | 0.0009 | 10 | 9 | 0.4336 | 1.0000 |
| 50 | 2000 | firefly_levy | sep_hits | higher | 0.1000 | 10 | 3 | 1.0000 | 1.0000 |
| 50 | 2000 | firefly_levy | hits | higher | 0.4000 | 10 | 6 | 0.2188 | 1.0000 |
| 50 | 2000 | firefly_levy | val_fitness | lower | -0.0026 | 10 | 5 | 0.0625 | 1.0000 |
| 500 | 1000 | firefly | holdout_acc | lower | -0.0410 | 10 | 8 | 0.1797 | 1.0000 |
| 500 | 1000 | firefly | sep_hits | higher | 0.1000 | 10 | 3 | 1.0000 | 1.0000 |
| 500 | 1000 | firefly | hits | higher | 0.1000 | 10 | 3 | 1.0000 | 1.0000 |
| 500 | 1000 | firefly | val_fitness | higher | 0.0305 | 10 | 6 | 1.0000 | 1.0000 |
| 500 | 1000 | firefly_levy | holdout_acc | higher | 0.0010 | 10 | 9 | 0.9766 | 1.0000 |
| 500 | 1000 | firefly_levy | sep_hits | higher | 0.3000 | 10 | 5 | 0.3750 | 1.0000 |
| 500 | 1000 | firefly_levy | hits | higher | 0.3000 | 10 | 5 | 0.3750 | 1.0000 |
| 500 | 1000 | firefly_levy | val_fitness | higher | 0.0083 | 10 | 9 | 0.4961 | 1.0000 |
| 500 | 2000 | firefly | holdout_acc | higher | 0.0145 | 10 | 10 | 0.0371 | 0.9062 |
| 500 | 2000 | firefly | sep_hits | higher | 0.6000 | 10 | 6 | 0.0312 | 0.9062 |
| 500 | 2000 | firefly | hits | higher | 0.6000 | 10 | 6 | 0.0312 | 0.9062 |
| 500 | 2000 | firefly | val_fitness | lower | -0.0156 | 10 | 10 | 0.0195 | 0.5859 |
| 500 | 2000 | firefly_levy | holdout_acc | higher | 0.0185 | 10 | 10 | 0.0039 | 0.1211 |
| 500 | 2000 | firefly_levy | sep_hits | higher | 0.7000 | 10 | 6 | 0.0312 | 0.9062 |
| 500 | 2000 | firefly_levy | hits | higher | 0.7000 | 10 | 6 | 0.0312 | 0.9062 |
| 500 | 2000 | firefly_levy | val_fitness | lower | -0.0160 | 10 | 9 | 0.0508 | 1.0000 |

## Stage 2 follow-up - d=500, n_search=1000, data seed 1 (`check_d500_seed1_freq.csv`, algorithm seeds 0-29)

| method | runs | zero_hit_runs | mean_hits | mean_val_fitness | mean_masks_with_inf | mean_inf_cols_tried | runs_not_all_5_tried |
|---|---|---|---|---|---|---|---|
| firefly | 30 | 4 | 1.1000 | 0.0926 | 53.9000 | 3.7000 | 23 |
| random_search | 30 | 0 | 1.1333 | 0.0462 | 27.8667 | 4.9333 | 2 |

## Stage 3 - re-ranking ablation (`rerank_ablation.csv`, seeds 0-9)

Noisy toy (class_sep=1.0, 2 clusters/class), 1000 search rows, 5000-row holdout, k=5, budget=800, top_m=10, 5-fold. Ablation: reranked vs raw mask of the SAME search (never against another method). mean_diff = reranked - raw; p_holm across the rows of this table. Holdout classifier: LogisticRegression(max_iter=500, random_state=0).

| d | metric | raw | reranked | direction | mean_diff | n | n_nonzero | p | n_changed | p_holm |
|---|---|---|---|---|---|---|---|---|---|---|
| 50 | holdout_acc | 0.8204 | 0.8223 | higher | 0.0019 | 10 | 8 | 1.0000 | 8 | 1.0000 |
| 50 | sep_hits | 2.6000 | 2.9000 | higher | 0.3000 | 10 | 3 | 0.2500 | 8 | 1.0000 |
| 50 | hits | 2.9000 | 3.2000 | higher | 0.3000 | 10 | 3 | 0.2500 | 8 | 1.0000 |
| 500 | holdout_acc | 0.7603 | 0.7660 | higher | 0.0058 | 10 | 6 | 0.1562 | 6 | 0.9375 |
| 500 | sep_hits | 1.5000 | 1.7000 | higher | 0.2000 | 10 | 2 | 0.5000 | 6 | 1.0000 |
| 500 | hits | 1.5000 | 1.7000 | higher | 0.2000 | 10 | 2 | 0.5000 | 6 | 1.0000 |

Cost per run (reranked variant; raw uses evals_used only):

| d | evals_used | rerank_fits | total_fits | rerank_s |
|---|---|---|---|---|
| 50 | 800.0000 | 50.0000 | 850.0000 | 0.1856 |
| 500 | 800.0000 | 50.0000 | 850.0000 | 0.1550 |

## Stage 4 - fitness timing (`time_fitness.csv`, seconds per NEW evaluation, single thread, max_rows=2000)

| dataset | k | n | mean | median | p95 | share_hit_max_iter |
|---|---|---|---|---|---|---|
| cifar_r18 | 16 | 30 | 0.0285 | 0.0284 | 0.0363 | 0.0000 |
| cifar_r18 | 32 | 30 | 0.0396 | 0.0382 | 0.0487 | 0.0000 |
| cifar_r18 | 64 | 30 | 0.0671 | 0.0669 | 0.0760 | 0.0000 |
| cifar_r18 | 128 | 30 | 0.1038 | 0.1008 | 0.1262 | 0.0000 |
| digits | 8 | 30 | 0.0322 | 0.0321 | 0.0398 | 0.0000 |
| digits | 16 | 30 | 0.0373 | 0.0363 | 0.0448 | 0.0000 |
| digits | 32 | 30 | 0.0393 | 0.0382 | 0.0468 | 0.0000 |

Re-rank times and wall-time estimates are not stored in a CSV; see `results/time_fitness.md`.

## What this does and does not show

No result below is described as "beats" or "better": each comparison is given as higher/lower with its raw and Holm-adjusted p. With n=10 seeds the smallest two-sided p is about 0.002, so power is limited and a non-significant result is not evidence of equality. In the Stage 2 family, 0 of 32 tests have Holm-adjusted p < 0.05.

- Primary test (clean signal, d=500, 2000 search rows): Firefly's holdout accuracy vs random search was higher (mean diff +0.0145, p=0.0371, Holm-adjusted p=0.9062, n=10).
- Clean signal, d=50, 1000 search rows: mean recall of separating columns was 0.725 (Firefly) vs 0.750 (random search); Firefly sep_hits was lower (mean diff -0.1000, p=1.0000, Holm-adjusted p=1.0000, n=10). It does not show recovery of all 5 informative columns: only columns whose class means differ carry linear signal.
- Validation fitness at d=500 (secondary, in-sample): with 2000 search rows Firefly's was lower (mean diff -0.0156, p=0.0195, Holm-adjusted p=0.5859, n=10); with 1000 rows it was higher (mean diff +0.0305, p=1.0000, Holm-adjusted p=1.0000, n=10). Lower validation fitness on the rows that selected the mask is not evidence of better generalisation.
- On one dataset (d=500, data seed 1) over 30 algorithm seeds Firefly ended with 0 informative hits in 4 runs vs 0 for random search (descriptive, no test).
- Re-ranking (ablation, same search) changed the selected mask in 14/20 runs; holdout accuracy was higher at d=50 (mean diff +0.0019, p=1.0000, Holm p=1.0000) and higher at d=500 (mean diff +0.0058, p=0.1562, Holm p=0.9375), n=10 each; re-ranking also costs extra fits (see cost table).
- Operator caveat: every Day-3 synthetic run used k=5 or 10 at alpha=0.02, i.e. exactly one swap per move (1-3 with long-jump); alpha decay had no effect. Results describe that operator only.
- Holdout classifiers used by the CSVs above: Stage 2 LogisticRegression(max_iter=500, random_state=0); Stage 3 LogisticRegression(max_iter=500, random_state=0). All of this is synthetic data; it shows nothing yet about the image embeddings. Timing measures cost per evaluation only.
