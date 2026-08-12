# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 39: Appendix E · Python & NumPy primer
# Section: Broadcasting
# Code example 12 of 22 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-python"
# ====================================================================

import numpy as np

X = np.arange(12).reshape(4, 3).astype(float)   # batch: 4 examples, 3 features
b = np.array([100., 200., 300.])                # per-feature bias, shape (3,)
print("X + b (per feature):\n", X + b)          # (4,3) + (3,) aligns on the 3

# a per-example bias of shape (4,) does NOT align with (4,3):
row_bias = np.array([10., 20., 30., 40.])       # shape (4,)
try:
    X + row_bias                                # trailing 3 vs 4 -> mismatch
except ValueError as err:
    print("X + row_bias fails:", err)
    # -> operands could not be broadcast together with shapes (4,3) (4,)

# fix: reshape to a (4,1) column so it broadcasts across the 3 columns
fixed = X + row_bias[:, None]                    # None inserts a length-1 axis
print("fixed with (4,1):\n", fixed)
