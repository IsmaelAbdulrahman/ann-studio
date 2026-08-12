# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 42: Appendix H · Datasets & further reading
# Section: Time-series datasets
# Code example 4 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-datasets"
# ====================================================================

import numpy as np

series = np.arange(1, 13, dtype=float)      # a toy monthly series 1..12
window = 3

X, y = [], []
for i in range(len(series) - window):
    X.append(series[i:i + window])          # the past 'window' values
    y.append(series[i + window])            # the next value to predict
X, y = np.array(X), np.array(y)

print("X shape:", X.shape, " y shape:", y.shape)
print("first 3 windows:\n", X[:3])
print("their targets  :", y[:3])
