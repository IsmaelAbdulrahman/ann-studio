# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 12: Regularization & generalization
# Section: Regularization in code
# Code example 1 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "regularization"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)
f = lambda t: np.sin(2.5*t)
xtr = np.sort(rng.uniform(-1, 1, 11))          # 11 unevenly-spaced samples
ytr = f(xtr) + rng.normal(0, 0.15, xtr.shape)  # with observation noise
xte = np.linspace(-1, 1, 200)                  # dense grid of the TRUE curve
V = lambda t: np.vander(t, 11)                 # degree-10 polynomial features
rmse = lambda a, b: float(np.sqrt(np.mean((a-b)**2)))

w_ols = np.linalg.lstsq(V(xtr), ytr, rcond=None)[0]        # no penalty
A = V(xtr); lam = 0.3                                       # L2 / ridge:
w_l2 = np.linalg.solve(A.T@A + lam*np.eye(11), A.T@ytr)    # (AᵀA + λI)w = Aᵀy

for name, w in [("no L2 ", w_ols), ("with L2", w_l2)]:
    print(f"{name}: train RMSE={rmse(V(xtr)@w, ytr):8.3f}   "
          f"test RMSE={rmse(V(xte)@w, f(xte)):8.3f}   max|w|={np.abs(w).max():.1f}")
