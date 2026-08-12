# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 39: Appendix E · Python & NumPy primer
# Section: The operations you'll use everywhere
# Code example 10 of 22 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-python"
# ====================================================================

import numpy as np

X = np.arange(6).reshape(2, 3).astype(float)   # (2,3)
W = np.ones((3, 4))                             # (3,4)
print("X @ W shape :", (X @ W).shape)           # (2,4): inner 3 summed away
print("elementwise :", (X * 2).shape)           # (2,3), shape unchanged

s = X.sum(axis=1, keepdims=True)                # (2,1) keeps the collapsed axis
print("keepdims    :", s.shape, "->", s.ravel())
print("row-normalize:\n", (X / s).round(3))     # (2,3)/(2,1) broadcasts per row

print("transpose   :", X.T.shape)               # (3,2)
print("stack        :", np.stack([X, X]).shape) # (2,2,3): a new leading axis
print("concatenate :", np.concatenate([X, X], axis=0).shape)  # (4,3)
print("where>2 :\n", np.where(X > 2, X, 0.0))   # keep big values, else 0
print("clip1..4:\n", np.clip(X, 1, 4))          # bound into [1, 4]
