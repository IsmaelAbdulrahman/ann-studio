# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 25: Training in practice: the complete recipe
# Section: K-fold cross-validation
# Code example 2 of 8 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "practice"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)

# toy 1-D binary data with 15% label noise
X = rng.normal(0, 1, 200)
y = (X > 0).astype(int)
flip = rng.rand(200) < 0.15
y = np.where(flip, 1 - y, y)

k = 5
idx = rng.permutation(200)
folds = np.array_split(idx, k)
accs = []
for i in range(k):
    val = folds[i]
    train = np.concatenate([folds[j] for j in range(k) if j != i])
    thr = X[train].mean()                       # 'fit' a threshold on train
    pred = (X[val] > thr).astype(int)
    accs.append((pred == y[val]).mean())        # score on the held-out fold

accs = np.array(accs)
print("per-fold accuracy:", accs.round(3))
print(f"CV accuracy: {accs.mean():.3f} +/- {accs.std():.3f}")
