# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 10: Initialization & the vanishing gradient
# Section: Exercises
# Code example 6 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "init"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)
n = 512
x  = rng.randn(n)
Wg = rng.randn(n, n) * np.sqrt(1.0 / n)      # Gaussian, variance 1/n
Q, _ = np.linalg.qr(rng.randn(n, n))          # orthogonal: singular values all 1
print("input norm       :", round(float(np.linalg.norm(x)), 4))
print("after Gaussian   :", round(float(np.linalg.norm(Wg @ x)), 4))   # near, but not exact
print("after orthogonal :", round(float(np.linalg.norm(Q @ x)), 4))    # equals input exactly
    
