# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 9: Momentum, RMSProp & Adam
# Section: Optimizers in code
# Code example 2 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "optimizers"
# ====================================================================

import numpy as np
A = np.array([20.0, 1.0])              # curvatures: steep axis, shallow axis
grad = lambda w: A * w                 # L = 0.5*(20*w0^2 + 1*w1^2), a ravine
tol, cap = 1e-3, 5000
w0 = np.array([1.0, 1.0])

w = w0.copy()                          # --- plain SGD ---
for t in range(1, cap + 1):
    w = w - 0.02 * grad(w)
    if np.linalg.norm(w) < tol: break
print("SGD      :", t, "iters")

w = w0.copy(); v = np.zeros(2)         # --- Momentum ---
for t in range(1, cap + 1):
    v = 0.9 * v + grad(w); w = w - 0.02 * v
    if np.linalg.norm(w) < tol: break
print("Momentum :", t, "iters")

w = w0.copy(); m = np.zeros(2); s = np.zeros(2)   # --- Adam ---
for t in range(1, cap + 1):
    g = grad(w); m = 0.9*m + 0.1*g; s = 0.999*s + 0.001*g*g
    w = w - 0.1 * (m/(1-0.9**t)) / (np.sqrt(s/(1-0.999**t)) + 1e-8)
    if np.linalg.norm(w) < tol: break
print("Adam     :", t, "iters")
