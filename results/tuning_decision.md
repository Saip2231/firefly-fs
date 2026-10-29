# Tuning decision (Day 4 Stage 4)

- Source: `results\tuning_runs.csv` (162 runs), tuning seeds [100, 101, 102], budget 800, conditions ['cifar_r18/k=16', 'cifar_r18/k=64', 'digits/k=16'].
- Score: tuning_acc = final_clf() accuracy on the 25 % tuning holdout of Xtr (the test split is never loaded).
- Rule: highest mean tuning_acc over 9 runs; if it beats the default by < 0.003, keep the default.  No combinations, no second round.
- git (summary): 831fd93+dirty; git (runs): 831fd93

## firefly

| config | all | cifar_r18/k=16 | cifar_r18/k=64 | digits/k=16 |
|---|---|---|---|---|
| default | 0.7430 ± 0.1454 | 0.5782 ± 0.0044 | 0.7380 ± 0.0136 | 0.9130 ± 0.0163 |
| n_fireflies=10 | 0.7487 ± 0.1340 | 0.5955 ± 0.0208 | 0.7468 ± 0.0070 | 0.9037 ± 0.0112 |
| n_fireflies=40 | 0.7336 ± 0.1484 | 0.5612 ± 0.0177 | 0.7368 ± 0.0095 | 0.9028 ± 0.0127 |
| beta0=0.2 | 0.7245 ± 0.1518 | 0.5457 ± 0.0078 | 0.7325 ± 0.0055 | 0.8954 ± 0.0140 |
| beta0=0.6 | 0.7398 ± 0.1345 | 0.5848 ± 0.0131 | 0.7410 ± 0.0105 | 0.8935 ± 0.0239 |
| gamma=1 | 0.7398 ± 0.1350 | 0.5830 ± 0.0005 | 0.7420 ± 0.0043 | 0.8944 ± 0.0127 |
| gamma=4 | 0.7274 ± 0.1525 | 0.5502 ± 0.0224 | 0.7310 ± 0.0087 | 0.9009 ± 0.0140 |
| alpha=0.1 | 0.7352 ± 0.1447 | 0.5668 ± 0.0166 | 0.7387 ± 0.0098 | 0.9000 ± 0.0083 |
| alpha=0.25 | 0.7406 ± 0.1342 | 0.5858 ± 0.0081 | 0.7405 ± 0.0048 | 0.8954 ± 0.0105 |

**Decision: `n_fireflies=10`** (n_fireflies=10 beats default by 0.0056 >= 0.003).
Frozen hyperparameters: `{"alpha": 0.02, "beta0": 0.4, "gamma": 2.0, "n_fireflies": 10}`

## pso

| config | all | cifar_r18/k=16 | cifar_r18/k=64 | digits/k=16 |
|---|---|---|---|---|
| default | 0.7431 ± 0.1274 | 0.5990 ± 0.0298 | 0.7397 ± 0.0063 | 0.8907 ± 0.0098 |
| n_particles=10 | 0.7295 ± 0.1415 | 0.5620 ± 0.0230 | 0.7403 ± 0.0187 | 0.8861 ± 0.0144 |
| n_particles=40 | 0.7384 ± 0.1339 | 0.5903 ± 0.0120 | 0.7267 ± 0.0137 | 0.8981 ± 0.0032 |
| c1=c2=1.0 | 0.7394 ± 0.1397 | 0.5808 ± 0.0125 | 0.7347 ± 0.0040 | 0.9028 ± 0.0121 |
| c1=c2=2.0 | 0.7410 ± 0.1233 | 0.5968 ± 0.0137 | 0.7457 ± 0.0048 | 0.8806 ± 0.0139 |
| vmax=0.25 | 0.7362 ± 0.1483 | 0.5658 ± 0.0313 | 0.7373 ± 0.0155 | 0.9056 ± 0.0147 |
| vmax=1.0 | 0.7378 ± 0.1325 | 0.5878 ± 0.0188 | 0.7338 ± 0.0153 | 0.8917 ± 0.0192 |
| w_start=0.7 | 0.7322 ± 0.1406 | 0.5653 ± 0.0028 | 0.7423 ± 0.0134 | 0.8889 ± 0.0139 |
| w_start=1.0 | 0.7462 ± 0.1460 | 0.5800 ± 0.0090 | 0.7418 ± 0.0074 | 0.9167 ± 0.0100 |

**Decision: `w_start=1.0`** (w_start=1.0 beats default by 0.0030 >= 0.003).
Frozen hyperparameters: `{"c1": 1.5, "c2": 1.5, "n_particles": 20, "vmax": 0.5, "w_start": 1.0}`

**Note:** population sizes differ (firefly 10, pso 20); their initial populations are then shared only for the first 10 members (fairness rule 5).  Record this in PREREG.

The frozen values apply to firefly, firefly_levy and firefly_seeded (Firefly) and to pso and pso_seeded (PSO); frac_seeded stays 0.5.
