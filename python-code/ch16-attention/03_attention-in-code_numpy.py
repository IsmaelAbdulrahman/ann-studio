# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 16: Attention & the transformer
# Section: Attention in code
# Code example 3 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "attention"
# ====================================================================

import numpy as np
np.set_printoptions(precision=3, suppress=True)
rng = np.random.RandomState(0)

T, dk = 4, 8                       # 4 tokens, dimension 8
Q = rng.randn(T, dk)
K = rng.randn(T, dk)
V = rng.randn(T, dk)

scores = (Q @ K.T) / np.sqrt(dk)   # (T, T) all pairwise scores
mask = np.triu(np.ones((T, T)), k=1)   # 1s strictly above diagonal = future
scores = np.where(mask == 1, -1e9, scores)   # forbid looking ahead

scores = scores - scores.max(axis=1, keepdims=True)
w = np.exp(scores)
w = w / w.sum(axis=1, keepdims=True)   # row-wise softmax

print("causal attention weights (lower-triangular):")
print(w)
print("row sums:", w.sum(axis=1))
