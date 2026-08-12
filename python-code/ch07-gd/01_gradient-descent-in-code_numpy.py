# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 7: Gradient descent & its variants
# Section: Gradient descent in code
# Code example 1 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "gd"
# ====================================================================

import numpy as np
def L(t):  return t**4 - 3*t**2 + 2      # double-well loss
def dL(t): return 4*t**3 - 6*t           # its gradient

t, eta = 0.5, 0.1
print(f"start: theta={t:.4f}  L={L(t):.5f}")
for step in range(1, 31):
    t = t - eta*dL(t)
    if step in (1, 2, 3, 4, 10, 30):
        print(f"step {step:2d}: theta={t:.4f}  L={L(t):.5f}  grad={dL(t):+.4f}")
print(f"true minimum: theta={np.sqrt(1.5):.4f}  L={L(np.sqrt(1.5)):.4f}")
