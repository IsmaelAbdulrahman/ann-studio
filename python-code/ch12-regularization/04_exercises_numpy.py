# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 12: Regularization & generalization
# Section: Exercises
# Code example 4 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "regularization"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)
f = lambda t: np.sin(2.5*t)
xtr = np.sort(rng.uniform(-1, 1, 11)); ytr = f(xtr) + rng.normal(0, 0.15, 11)
xte = np.linspace(-1, 1, 200)
V = lambda t: np.vander(t, 11)
A = V(xtr); rmse = lambda a, b: float(np.sqrt(np.mean((a-b)**2)))
for lam in [0.0, 0.001, 0.01, 0.1, 0.3, 1.0, 3.0]:
    w = np.linalg.solve(A.T@A + lam*np.eye(11), A.T@ytr)
    print(f"lam={lam:5.3f}  test RMSE={rmse(V(xte)@w, f(xte)):8.3f}")
