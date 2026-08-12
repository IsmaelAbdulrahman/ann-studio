# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 19: Autoencoders & representation learning
# Section: Autoencoders in code
# Code example 2 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "autoencoders"
# ====================================================================

import numpy as np
rng = np.random.RandomState(1)

# 'normal' beats live near a 1-D line through 3-D feature space
line = np.array([[1.0, 0.8, -0.5]])
Xn = rng.randn(300, 1) @ line + 0.05 * rng.randn(300, 3)

We = rng.randn(3, 1) * 0.1               # bottleneck code of size 1
Wd = rng.randn(1, 3) * 0.1
for _ in range(800):
    Z = Xn @ We; Xh = Z @ Wd; E = Xh - Xn
    We -= 0.1 * (Xn.T @ (E @ Wd.T)) * (2 / len(Xn))
    Wd -= 0.1 * (Z.T @ E) * (2 / len(Xn))

def score(x):                            # reconstruction error = anomaly score
    return np.sum((x - (x @ We) @ Wd) ** 2, axis=1)

val = rng.randn(100, 1) @ line + 0.05 * rng.randn(100, 3)   # held-out normal
tau = 5 * score(val).mean()              # threshold = 5x mean normal error
anomaly = np.array([[0.6, 0.5, 0.9]])    # 3rd feature off the line
print(f"mean normal score : {score(val).mean():.4f}")
print(f"threshold  (5x)   : {tau:.4f}")
print(f"anomaly    score  : {score(anomaly)[0]:.4f}")
print("flagged?          :", bool(score(anomaly)[0] > tau))
