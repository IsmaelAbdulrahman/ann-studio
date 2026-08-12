# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 40: Appendix F · PyTorch primer
# Section: Exercises
# Code example 14 of 15 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-pytorch"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)
X = rng.randn(20, 3)
w_true = np.array([2.0, -1.0, 0.5])
y = X @ w_true + 0.1 * rng.randn(20)

w, lr = np.zeros(3), 0.1
for _ in range(300):
    pred = X @ w                              # forward
    grad = (2 / len(y)) * X.T @ (pred - y)    # backward (MSE gradient)
    w = w - lr * grad                         # step
print("learned w:", w.round(3))
print("true    w:", w_true)
