# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 35: Appendix A · Linear algebra refresher
# Section: Exercises
# Code example 7 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-linalg"
# ====================================================================

import numpy as np
rng = np.random.RandomState(1)
X = rng.randn(5, 3)             # (5, 3)
W = rng.randn(3, 4)            # (3, 4)
b = rng.randn(4)              # (4,)
Z = X @ W + b
print("output shape:", Z.shape)   # (5, 4)
print(np.round(Z, 2))
