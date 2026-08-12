# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 25: Training in practice: the complete recipe
# Section: Exercises
# Code example 8 of 8 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "practice"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)
f = lambda x: 0.5*x**3 - x                          # true (cubic) signal
xtr = np.linspace(-2, 2, 14); ytr = f(xtr) + rng.normal(0, 0.6, 14)
xva = np.linspace(-2, 2, 40); yva = f(xva) + rng.normal(0, 0.6, 40)
for d in [1, 3, 9, 13]:                             # polynomial degree = capacity
    c = np.linalg.lstsq(np.vander(xtr, d+1), ytr, rcond=None)[0]
    tr = np.mean((np.vander(xtr, d+1) @ c - ytr)**2)
    va = np.mean((np.vander(xva, d+1) @ c - yva)**2)
    print(f"degree {d:2d}:  train {tr:6.3f}   val {va:9.3f}")
