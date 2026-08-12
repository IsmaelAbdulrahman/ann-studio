# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 7: Gradient descent & its variants
# Section: Exercises
# Code example 6 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "gd"
# ====================================================================

import numpy as np
def final_loss(eta, k=1.0, steps=200, t0=1.0):
    t = t0
    for _ in range(steps):
        t = t - eta*k*t
    return 0.5*k*t*t
for eta in [1.0, 1.5, 1.9, 1.99, 2.0, 2.01]:
    print(f"eta={eta:4.2f}  final loss={final_loss(eta):.3e}")
print("ceiling = 2/k =", 2.0)
