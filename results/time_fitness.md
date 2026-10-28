# Fitness timing (Stage 4)

- os.cpu_count() = 16; parallel column uses 15 workers (ideal scaling, lower bound)
- numpy 2.4.6, scikit-learn 1.9.1; single-thread BLAS
- make_fitness(max_rows=2000), LogisticRegression(max_iter=100); 30 distinct random masks per k, 1 warm-up excluded
- rerank: top_m=10, 5-fold CV on the full training set
- per run = 800 x mean eval + setup + rerank; 10 seeds x 3 metaheuristics per k (PSO assumed same per-eval cost)

## Per k

| dataset | k | mean s/eval | median | p95 | evals hitting max_iter | setup s | rerank s | s/run | 30 runs, 1 worker |
|---|---|---|---|---|---|---|---|---|---|
| cifar_r18 | 16 | 0.0285 | 0.0284 | 0.0363 | 0% | 0.01 | 4.4 | 27 | 13.6 min |
| cifar_r18 | 32 | 0.0396 | 0.0382 | 0.0487 | 0% | 0.01 | 5.8 | 37 | 18.7 min |
| cifar_r18 | 64 | 0.0671 | 0.0669 | 0.0760 | 0% | 0.01 | 9.7 | 63 | 31.7 min |
| cifar_r18 | 128 | 0.1038 | 0.1008 | 0.1262 | 0% | 0.01 | 19.2 | 102 | 51.1 min |
| digits | 8 | 0.0322 | 0.0321 | 0.0398 | 0% | 0.00 | 1.7 | 27 | 13.7 min |
| digits | 16 | 0.0373 | 0.0363 | 0.0448 | 0% | 0.00 | 1.8 | 32 | 15.8 min |
| digits | 32 | 0.0393 | 0.0382 | 0.0468 | 0% | 0.00 | 1.9 | 33 | 16.7 min |

## Total wall time per dataset

| dataset | runs | 1 worker | 15 workers |
|---|---|---|---|
| cifar_r18 | 120 | 1.92 h | 7.7 min |
| digits | 90 | 46.2 min | 3.1 min |

## Reductions (suggested only, not applied; limit 6 h)

No estimate exceeds 6 h; no reduction needed.
