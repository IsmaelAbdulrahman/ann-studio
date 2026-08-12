# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 33: Case study: time-series forecasting
# Section: In code
# Code example 3 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "cs-sequence"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)
T = 360; t = np.arange(T)
series = np.sin(2*np.pi*t/24) + 0.02*t + 0.30*rng.randn(T)
L = 12
X = np.stack([series[i:i+L] for i in range(T-L)]); y = series[L:]
n = len(X)

def nn_mse(tr, te):                           # 1-nearest-neighbour forecaster
    d = ((X[te][:, None, :] - X[tr][None, :, :])**2).sum(-1)   # window distances
    j = d.argmin(1)                           # nearest TRAIN window for each test
    return np.mean((y[tr][j] - y[te])**2)     # borrow its next-step value

k = int(0.70*n)
chron = nn_mse(np.arange(k), np.arange(k, n))        # past -> future (honest)
perm  = rng.permutation(n)
shuf  = nn_mse(perm[:k], perm[k:])                   # shuffled first (LEAKS)
print(f"chronological split  test MSE = {chron:.3f}   <- honest")
print(f"shuffled split       test MSE = {shuf:.3f}   <- 'better', but a lie")
print(f"leakage makes the model look {chron/shuf:.1f}x more accurate than it is")
