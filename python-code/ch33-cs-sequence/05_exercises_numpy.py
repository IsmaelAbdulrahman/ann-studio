# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 33: Case study: time-series forecasting
# Section: Exercises
# Code example 5 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "cs-sequence"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)
T = 360; t = np.arange(T)
series = np.sin(2*np.pi*t/24) + 0.02*t + 0.30*rng.randn(T)
L = 12
X = np.stack([series[i:i+L] for i in range(T-L)]); y = series[L:]
n_tr = 250
yhat = X[:, -1]                              # persistence: next value = last seen value
te = np.mean((yhat[n_tr:] - y[n_tr:])**2)    # on the same chronological test block
print(f"persistence baseline  test MSE = {te:.3f}")
print(f"linear AR (main cell) test MSE = 0.167  ->  AR cuts error by {(1-0.167/te)*100:.0f}%")
