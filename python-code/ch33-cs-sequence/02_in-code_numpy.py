# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 33: Case study: time-series forecasting
# Section: In code
# Code example 2 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "cs-sequence"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)
T = 400
t = np.arange(T)
series = np.sin(2*np.pi*t/24) + 0.02*t + 0.30*rng.randn(T)   # season + trend + noise

L = 12                                        # look-back window
X = np.stack([series[i:i+L] for i in range(T-L)])
y = series[L:]                                # one-step-ahead target
n_tr = 280                                    # chronological split: past -> future
Xtr, ytr = X[:n_tr], y[:n_tr]
Xte, yte = X[n_tr:], y[n_tr:]

A = np.hstack([Xtr, np.ones((len(Xtr), 1))])  # design matrix + a bias column
coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)   # closed-form least-squares AR
w, b = coef[:L], coef[L]

pred  = Xte @ w + b                           # forecast the held-out future
naive = Xte[:, -1]                            # persistence: next = last seen value
rmse  = lambda a, f: np.sqrt(np.mean((a - f)**2))
r_ar, r_naive = rmse(yte, pred), rmse(yte, naive)
print(f"least-squares AR  RMSE = {r_ar:.3f}")      # 0.369
print(f"persistence       RMSE = {r_naive:.3f}")   # 0.452
print(f"skill vs naive    = {(1-r_ar/r_naive)*100:.0f}%   (positive => AR wins)")  # 18%
