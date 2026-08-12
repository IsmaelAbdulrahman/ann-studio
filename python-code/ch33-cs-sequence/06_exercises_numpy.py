# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 33: Case study: time-series forecasting
# Section: Exercises
# Code example 6 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "cs-sequence"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)
T = 360; t = np.arange(T)
series = np.sin(2*np.pi*t/24) + 0.02*t + 0.30*rng.randn(T)
m = 24                                        # seasonal period
n_tr = 250
idx = np.arange(n_tr, T)                       # test indices (each needs t-1 and t-m)
persist  = series[idx-1]                       # persistence:    next = last value
seasonal = series[idx-m]                       # seasonal-naive: next = value one season ago
rmse = lambda a, f: np.sqrt(np.mean((a-f)**2))
print(f"persistence     RMSE = {rmse(series[idx], persist):.3f}")   # 0.493
print(f"seasonal-naive  RMSE = {rmse(series[idx], seasonal):.3f}")  # 0.654
