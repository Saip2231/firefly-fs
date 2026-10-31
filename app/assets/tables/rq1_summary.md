# RQ1 confirmatory tests (Stage A1)

- Pre-registered in PREREG.md: raw Firefly mask, `acc_lr`; paired two-sided Wilcoxon by seed (zero differences dropped, all-zero = TIE with p = 1); Holm within each family of 6 (undefined test counts as p = 1); no correction across the 12 families.
- scipy 1.17.1 `wilcoxon(zero_method='wilcox', alternative='two-sided')`, default exact/approximate method.
- n = 10 seeds limits power: the smallest two-sided Wilcoxon p is about 0.002, so a Holm-adjusted WIN or LOSS needs nearly all seeds to agree.
- PCA is a projection, not a selector (full embedding + projection matrix at inference). Baselines have no search budget; the same baseline rows are used at budget 800 and 3000.

**Takeaway:** Across 12 families x 6 comparators (72 tests), plain Firefly has 5 WIN, 24 LOSS and 43 TIE (Holm within family, alpha 0.05); per comparator: anova 1W/3L/8T, mi 2W/2L/8T, l1 1W/3L/8T, rf 0W/8L/4T, pca 0W/8L/4T, pso 1W/0L/11T. Mean differences range from -16.6 pts to +7.5 pts.

## Labels and mean difference (Firefly minus comparator, accuracy points)

| family | anova | mi | l1 | rf | pca | pso |
|---|---|---|---|---|---|---|
| cifar_r18 k=16 b=800 | TIE -1.1 | WIN +3.7 | TIE +1.0 | LOSS -6.3 | LOSS -16.6 | TIE +1.2 |
| cifar_r18 k=16 b=3000 | WIN +2.7 | WIN +7.5 | WIN +4.8 | LOSS -2.5 | LOSS -12.8 | WIN +5.0 |
| cifar_r18 k=32 b=800 | TIE -1.4 | TIE -0.8 | TIE -1.2 | LOSS -5.3 | LOSS -12.7 | TIE +0.1 |
| cifar_r18 k=32 b=3000 | TIE +0.5 | TIE +1.1 | TIE +0.7 | LOSS -3.4 | LOSS -10.8 | TIE +2.1 |
| cifar_r18 k=64 b=800 | LOSS -1.6 | LOSS -0.9 | LOSS -1.8 | LOSS -3.1 | LOSS -8.6 | TIE +0.4 |
| cifar_r18 k=64 b=3000 | TIE -0.7 | TIE +0.1 | TIE -0.9 | LOSS -2.2 | LOSS -7.6 | TIE +0.4 |
| cifar_r18 k=128 b=800 | LOSS -2.0 | LOSS -1.4 | LOSS -1.1 | LOSS -2.0 | LOSS -4.0 | TIE -0.6 |
| cifar_r18 k=128 b=3000 | LOSS -1.2 | TIE -0.6 | TIE -0.3 | LOSS -1.2 | LOSS -3.2 | TIE +0.2 |
| digits k=16 b=800 | TIE +0.5 | TIE +1.4 | TIE +0.4 | TIE +0.5 | TIE -1.2 | TIE +1.8 |
| digits k=16 b=3000 | TIE +0.1 | TIE +1.0 | TIE +0.0 | TIE +0.1 | TIE -1.6 | TIE +0.1 |
| digits k=32 b=800 | TIE -0.7 | TIE -0.7 | LOSS -1.4 | TIE -1.0 | TIE -0.8 | TIE +1.2 |
| digits k=32 b=3000 | TIE -0.6 | TIE -0.5 | TIE -1.2 | TIE -0.8 | TIE -0.6 | TIE +0.6 |

Full numbers (means, std of paired differences, non-zero pairs, raw and Holm p, n): `results/analysis/rq1_tests.csv`.
