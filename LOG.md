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
