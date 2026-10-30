# Integrity report (Stage A0)

- Generated: 2026-10-30T22:24:11 by `python -m experiments.analyze --only A0`
- Inputs: `results/baselines_digits.csv`, `results/baselines_cifar_r18.csv`, `results/meta_digits.csv`, `results/meta_cifar_r18.csv`, `results/split_hashes.csv`, `PREREG.md`
- Result: **ALL CHECKS PASSED**

| check | status | detail |
|---|---|---|
| row_counts | PASS | baselines_digits.csv: 190 rows (expected 190)<br>baselines_cifar_r18.csv: 250 rows (expected 250)<br>meta_digits.csv: 320 rows (expected 320 = 4 methods x 2 budgets x 10 seeds x 2 k x 2 mask types)<br>meta_cifar_r18.csv: 640 rows (expected 640 = 4 methods x 2 budgets x 10 seeds x 4 k x 2 mask types) |
| seeds_complete | PASS | 140 groups, each with seeds 0-9 exactly once |
| no_nan_acc | PASS | no NaN in acc / acc_lr / acc_knn / acc_svm |
| evals_budget | PASS | evals_used <= budget on all 960 rows (960 use the full budget) |
| idx_length | PASS | len(idx) == k and indices in [0, d) on 1310 rows (PCA and 'all' excluded) |
| config_hash | PASS | single config_hash b14f6b194e4f7b9d... |
| budgets | PASS | budgets [800, 3000] within [800, 3000]; mask types and methods known |
| seeds_match | PASS | cifar_r18: baselines [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], meta [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]<br>digits: baselines [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], meta [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] |
| split_hashes | PASS | seeds 0-9 hashed for ['cifar_r18', 'digits'] |
| families | PASS | 12 dataset x k x budget families in the data (digits 4, cifar_r18 8); PREREG.md states 12 |
| git_hashes | INFO | digits: dd8c6dc (320 rows)<br>cifar_r18: dd8c6dc+dirty (640 rows)<br>'+dirty' on CIFAR comes only from the results/runs.json append between runs (LOG.md, Day 5); not an error. |

## k lists (from the baseline files)

- Digits (64-d): d = 64, selection k = [16, 32]; baseline methods ['all', 'anova', 'l1', 'mi', 'pca', 'random', 'rf']
- CIFAR-10 subset, ResNet-18 embeddings (512-d): d = 512, selection k = [16, 32, 64, 128]; baseline methods ['all', 'anova', 'l1', 'mi', 'pca', 'random', 'rf']
- meta_digits: methods ['firefly', 'firefly_seeded', 'pso', 'pso_seeded'], budgets [800, 3000], k [16, 32]
- meta_cifar_r18: methods ['firefly', 'firefly_seeded', 'pso', 'pso_seeded'], budgets [800, 3000], k [16, 32, 64, 128]
