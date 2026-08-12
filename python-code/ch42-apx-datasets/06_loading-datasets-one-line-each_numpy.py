# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 42: Appendix H · Datasets & further reading
# Section: Loading datasets: one line each
# Code example 6 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-datasets"
# ====================================================================

import numpy as np
np.random.seed(0)                          # determinism -> reproducible split

n = 200
X = np.random.normal(size=(n, 2))          # a synthetic 2-feature dataset
y = (X[:, 0] + X[:, 1] > 0).astype(int)    # a simple linear labelling rule

idx = np.random.permutation(n)             # shuffle the ROW INDICES, once
tr, va = int(0.70 * n), int(0.85 * n)      # 70 / 15 / 15 boundaries
train, val, test = idx[:tr], idx[tr:va], idx[va:]

print("sizes:", train.size, val.size, test.size)      # expected: 140 30 30
print("train balance:", round(float(y[train].mean()), 3))
print("splits disjoint:", len(set(train) & set(test)) == 0)   # expected: True
