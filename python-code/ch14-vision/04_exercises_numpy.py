# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 14: Modern CNN architectures & computer vision
# Section: Exercises
# Code example 4 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "vision"
# ====================================================================

import numpy as np
np.random.seed(0)
n, L = 16, 60
Ws = [0.10 * np.random.randn(n, n) / np.sqrt(n) for _ in range(L)]  # small init weights
g  = np.random.randn(n)                       # gradient seeded at the top layer
gp = g.copy(); gr = g.copy()
for W in Ws:                                   # one backward step per layer
    gp = W.T @ gp                              # plain net:    dy/dx = W^T
    gr = gr + W.T @ gr                         # residual net: dy/dx = I + W^T
print("plain    ||grad|| after 60 layers =", np.linalg.norm(gp))   # ~1e-61: vanished
print("residual ||grad|| after 60 layers =", np.linalg.norm(gr))   # ~O(1): survives
