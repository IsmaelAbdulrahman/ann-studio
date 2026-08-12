# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 42: Appendix H · Datasets & further reading
# Section: Exercises
# Code example 7 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-datasets"
# ====================================================================

import numpy as np
series = np.arange(1, 13, dtype=float)
window, horizon = 3, 2

X, y = [], []
for i in range(len(series) - window - horizon + 1):
    X.append(series[i:i + window])
    y.append(series[i + window + horizon - 1])   # 'horizon' steps ahead
X, y = np.array(X), np.array(y)
print("X shape:", X.shape, " y shape:", y.shape)
print("first window:", X[0], "-> target:", y[0])   # [1 2 3] -> 5
