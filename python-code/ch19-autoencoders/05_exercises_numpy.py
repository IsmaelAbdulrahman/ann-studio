# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 19: Autoencoders & representation learning
# Section: Exercises
# Code example 5 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "autoencoders"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)
B = np.linalg.qr(rng.randn(4, 2))[0].T
Z = rng.randn(120, 2); X = Z @ B + 0.05 * rng.randn(120, 4); X -= X.mean(0)
for d in (1, 2):
    We = rng.randn(4, d) * 0.1; Wd = rng.randn(d, 4) * 0.1
    for _ in range(400):
        Zc = X @ We; E = Zc @ Wd - X
        We -= 0.2 * (X.T @ (E @ Wd.T)) * (2 / len(X))
        Wd -= 0.2 * (Zc.T @ E) * (2 / len(X))
    print(f"code size {d}: final error {np.mean(np.sum(E**2, 1)):.4f}")
