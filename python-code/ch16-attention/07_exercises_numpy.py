# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 16: Attention & the transformer
# Section: Exercises
# Code example 7 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "attention"
# ====================================================================

import numpy as np
np.set_printoptions(precision=4, suppress=True)

X = np.array([[1., 0.],           # 3 tokens, d_k = 2
              [0., 1.],
              [1., 1.]])
Q = K = V = X                     # identity projections: self-attention
dk = X.shape[1]

S = (Q @ K.T) / np.sqrt(dk)       # 3x3 scaled scores
S = S - S.max(axis=1, keepdims=True)   # row-stable softmax
A = np.exp(S)
A = A / A.sum(axis=1, keepdims=True)   # each row is a distribution
O = A @ V                         # blended outputs

print("attention matrix A:\n", A)
print("row sums:", A.sum(axis=1))     # [1. 1. 1.]
print("outputs O:\n", O)              # [0.8022 0.5989] [0.5989 0.8022] [0.7517 0.7517]
