# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 39: Appendix E · Python & NumPy primer
# Section: Reshaping and axis operations
# Code example 9 of 22 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-python"
# ====================================================================

import numpy as np

X = np.arange(6).reshape(2, 3)          # 2 samples, 3 features
print("X:\n", X)
print("sum all     :", X.sum())         # 15
print("sum axis=0  :", X.sum(axis=0))   # down columns -> [3 5 7]
print("sum axis=1  :", X.sum(axis=1))   # across rows  -> [3 12]
print("mean axis=0 :", X.mean(axis=0))  # per-feature mean
print("flatten     :", X.reshape(-1))   # [0 1 2 3 4 5]
