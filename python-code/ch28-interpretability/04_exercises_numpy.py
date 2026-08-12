# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 28: Interpretability & explainability
# Section: Exercises
# Code example 4 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "interpretability"
# ====================================================================

import numpy as np
np.random.seed(0)
def f(x): return 2*x[0] + 0*x[1] - x[2] + x[3]**2
x    = np.array([1.0, 5.0, 2.0, 3.0])
mu   = np.array([0.5, 5.0, 1.0, 1.0])     # a "mean" baseline instead of zeros
occ0 = np.array([f(x) - f(np.where(np.arange(4)==i, 0.0,  x)) for i in range(4)])
occm = np.array([f(x) - f(np.where(np.arange(4)==i, mu[i], x)) for i in range(4)])
print("occlusion vs 0    =", np.round(occ0, 4))   # [ 2  0 -2  9 ]
print("occlusion vs mean =", np.round(occm, 4))   # smaller: feature 4 -> 3^2 - 1^2 = 8
