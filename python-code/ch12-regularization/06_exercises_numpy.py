# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 12: Regularization & generalization
# Section: Exercises
# Code example 6 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "regularization"
# ====================================================================

import numpy as np
w   = np.array([0.60, 0.05, -0.30, 0.01, -0.02])
lam = 0.1
l2 = w / (1 + lam)                                  # ridge: proportional shrink
l1 = np.sign(w) * np.maximum(0.0, np.abs(w) - lam)  # lasso: soft-threshold
print("L2 shrink    =", np.round(l2, 4))
print("L1 threshold =", np.round(l1, 4))
print("exact zeros  -> L2:", int((l2 == 0).sum()), " L1:", int((l1 == 0).sum()))
# expected: L2 zeros 0, L1 zeros 3 (the |w| ≤ 0.1 entries: 0.05, 0.01, -0.02)
