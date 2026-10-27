import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from src.firefly import make_fitness, binary_firefly

D, NI, K, BUDGET = 50, 5, 5, 600
# Same generator settings as _synthetic in tests/test_firefly.py. Edit to match.
X, y = make_classification(6000, D, n_informative=NI, n_redundant=0, n_repeated=0,
                           shuffle=False, random_state=0)
# 1000 rows for the search, 5000 as an independent holdout
Xs, Xh, ys, yh = train_test_split(X, y, train_size=1000, stratify=y, random_state=0)

true_mask = np.zeros(D, bool); true_mask[:NI] = True

def holdout_acc(mask):
    clf = LogisticRegression(max_iter=500).fit(Xs[:, mask], ys)
    return clf.score(Xh[:, mask], yh)

for seed in range(5):
    fit, st = make_fitness(Xs, ys, K, seed=seed)
    m, f, _ = binary_firefly(fit, st, D, K, budget=BUDGET, seed=seed)
    fit2, _ = make_fitness(Xs, ys, K, seed=seed)       # same inner split, fresh cache
    print(f"seed {seed}: hits {int(m[:NI].sum())}/{NI} | "
          f"val-fitness found {f:.4f} vs true {fit2(true_mask):.4f} | "
          f"holdout acc found {holdout_acc(m):.4f} vs true {holdout_acc(true_mask):.4f}")