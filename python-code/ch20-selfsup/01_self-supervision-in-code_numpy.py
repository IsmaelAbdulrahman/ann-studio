# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 20: Self-supervised & contrastive learning
# Section: Self-supervision in code
# Code example 1 of 4 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "selfsup"
# ====================================================================

import numpy as np
np.random.seed(0)
np.set_printoptions(precision=3, suppress=True)

N, d, tau = 4, 8, 0.5              # 4 examples, 2 views each -> 2N = 8 embeddings
base = np.random.randn(N, d)       # one hidden factor per example
v1 = base + 0.4 * np.random.randn(N, d)    # view 1 = example + augmentation noise
v2 = base + 0.4 * np.random.randn(N, d)    # view 2 (a different augmentation)
Z  = np.vstack([v1, v2])           # (2N, d); row i is paired with row i+N
Z  = Z / np.linalg.norm(Z, axis=1, keepdims=True)   # L2-normalize -> cosine

S = (Z @ Z.T) / tau                # (2N, 2N) scaled cosine-similarity matrix
np.fill_diagonal(S, -1e9)          # an example is never its own positive

pos = (np.arange(2*N) + N) % (2*N)         # positive of row i is its other view
S  -= S.max(axis=1, keepdims=True)         # softmax stabilization (subtract row max)
P   = np.exp(S) / np.exp(S).sum(axis=1, keepdims=True)
loss = -np.log(P[np.arange(2*N), pos]).mean()      # InfoNCE = pick-the-positive CE

print("mean positive cos-sim:", round(float((Z[:N] * Z[N:]).sum(1).mean()), 3))
print("NT-Xent loss         :", round(float(loss), 4))   # expected: 0.9685
