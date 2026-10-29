# Compute plan (Day 4 Stage 6)

- os.cpu_count() = 16; workers = cpu_count - 1 = 15; single-thread BLAS per worker.
- Per run = (budget x s/eval + setup + rerank) x contention x 1.2 (20 % margin); timing from results/time_fitness.md.
- Contention factor (observed / predicted search time of the budget-800 tuning runs with 15 workers, floored at 1.0): cifar_r18 1.23, digits 1.00.
- Not measured: final test evaluation (3 classifiers x 2 masks); assumed covered by the margin. Hours with workers assume ideal packing (lower bound).
- Grid per dataset x budget: 4 methods (firefly, pso, firefly_seeded, pso_seeded) x 10 seeds x k list.

| dataset | budget | k values | runs | s/run min | s/run mean | s/run max | 1-worker h | 15-worker h |
|---|---|---|---|---|---|---|---|---|
| digits | 800 | 16, 32 | 80 | 38 | 39 | 40 | 0.87 | 0.06 |
| digits | 3000 | 16, 32 | 80 | 136 | 140 | 144 | 3.11 | 0.21 |
| cifar_r18 | 800 | 16, 32, 64, 128 | 160 | 40 | 85 | 151 | 3.78 | 0.25 |
| cifar_r18 | 3000 | 16, 32, 64, 128 | 160 | 133 | 279 | 488 | 12.40 | 0.83 |
| **total** | | | 480 | | | | **20.15** | **1.34** |

Optional firefly_levy (budget 800 only): +1.16 h on 1 worker, +0.08 h on 15 workers.

No total exceeds 8 h with 15 workers; no reduction needed.
