# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 2: The perceptron & linear separability
# Section: Exercises
# Code example 4 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "perceptron"
# ====================================================================

import numpy as np
rng = np.random.RandomState(3)
w_true = np.array([1.0, -2.0])
X = rng.randn(40, 2)
y = (X @ w_true > 0).astype(int)        # separable through the origin
w = np.zeros(2); b = 0.0
for epoch in range(1, 101):
    errs = 0
    for xi, yi in zip(X, y):
        e = yi - (1 if w @ xi + b >= 0 else 0)
        if e:
            w += e * xi; b += e; errs += 1
    if errs == 0:
        print("converged in", epoch, "epochs"); break
acc = np.mean([(1 if w @ xi + b >= 0 else 0) == yi for xi, yi in zip(X, y)])
print("train accuracy:", acc)
