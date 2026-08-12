# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 7: Gradient descent & its variants
# Section: Gradient descent in code
# Code example 2 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "gd"
# ====================================================================

import numpy as np
A = np.array([10.0, 1.0])                 # curvature per axis; condition number = 10
def L(w):  return 0.5*np.sum(A*w**2)
def dL(w): return A*w

w = np.array([1.0, 1.0]); eta = 0.18      # 0.18 < 2/10 = 0.2 stability ceiling
print(f"kappa = {A.max()/A.min():.0f}   ceiling eta < 2/k_max = {2/A.max():.2f}")
for step in range(1, 61):
    w = w - eta*dL(w)
    if step in (1, 2, 3, 5, 10, 30, 60):
        print(f"step {step:2d}: w=[{w[0]:+.4f}, {w[1]:+.4f}]  L={L(w):.6f}")
