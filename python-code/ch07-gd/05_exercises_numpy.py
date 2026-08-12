# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 7: Gradient descent & its variants
# Section: Exercises
# Code example 5 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "gd"
# ====================================================================

import numpy as np
def dL(t): return 4*t**3 - 6*t
t, eta = 1.6, 0.1
for step in range(1, 21):
    t = t - eta*dL(t)
    if step in (1, 2, 5, 20):
        print(f"step {step:2d}: theta={t:.4f}")
print("converged near", round(t, 4))
