# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 39: Appendix E · Python & NumPy primer
# Section: Broadcasting
# Code example 11 of 22 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-python"
# ====================================================================

import numpy as np

X = np.array([[1., 2., 3.],
              [4., 5., 6.]])           # shape (2, 3)
col_mean = X.mean(axis=0)              # shape (3,) -> [2.5 3.5 4.5]
Xc = X - col_mean                     # (2,3) - (3,) broadcasts over rows
print("centered:\n", Xc)
print("column means after:", Xc.mean(axis=0))   # ~ [0 0 0]

# a (2,1) column broadcasts across the 3 columns instead
bias = np.array([[10.], [20.]])       # shape (2, 1)
print("plus per-row bias:\n", X + bias)
