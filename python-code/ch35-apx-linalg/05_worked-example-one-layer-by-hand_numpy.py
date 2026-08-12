# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 35: Appendix A · Linear algebra refresher
# Section: A whole layer, and the batch-as-rows convention → Worked example — one layer by hand
# Code example 5 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-linalg"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)
X = rng.randn(4, 3)                  # batch of 4 samples, 3 features each
W = rng.randn(3, 2)                  # 3 inputs -> 2 neurons (one column each)
b = np.array([0.5, -1.0])           # one bias per neuron
Z = X @ W + b                       # (4,3) @ (3,2) + (2,) -> (4,2)
print("Z shape:", Z.shape)
print("Z =\n", np.round(Z, 3))
A = np.maximum(0.0, Z)              # ReLU, elementwise -> same shape
print("after ReLU:\n", np.round(A, 3))
