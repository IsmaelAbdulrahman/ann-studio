# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 7: Gradient descent & its variants
# Section: Gradient descent in code
# Code example 3 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "gd"
# ====================================================================

import numpy as np
def final_loss(eta, k=4.0, steps=50, t0=1.0):
    t = t0
    for _ in range(steps):
        t = t - eta*(k*t)          # L = 1/2 k t^2, so grad = k t
    return 0.5*k*t*t

print("L = 0.5*4*t^2, start t=1, 50 steps.  Stability ceiling eta = 2/k = 0.5")
print(" eta     final loss")
for eta in [0.05, 0.10, 0.25, 0.45, 0.50, 0.55]:
    print(f"{eta:5.2f}   {final_loss(eta):.3e}")
