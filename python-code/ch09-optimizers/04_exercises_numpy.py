# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 9: Momentum, RMSProp & Adam
# Section: Exercises
# Code example 4 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "optimizers"
# ====================================================================

import numpy as np
A = np.array([20.0, 1.0]); grad = lambda w: A * w
tol, cap = 1e-3, 5000; w0 = np.array([1.0, 1.0])
w = w0.copy(); s = np.zeros(2)                 # RMSProp on the ravine
for t in range(1, cap + 1):
    g = grad(w); s = 0.9 * s + 0.1 * g * g
    w = w - 0.02 * g / (np.sqrt(s) + 1e-8)
    if np.linalg.norm(w) < tol: break
print("RMSProp :", t, "iters")
    
