# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 2: The perceptron & linear separability
# Section: Exercises
# Code example 5 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "perceptron"
# ====================================================================

import numpy as np
raw = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = np.array([0, 1, 1, 0])
X = np.column_stack([raw, raw[:, 0] * raw[:, 1]])   # [x1, x2, x1*x2]
w = np.zeros(3); b = 0.0
step = lambda z: 1 if z >= 0 else 0
for epoch in range(1, 51):
    errs = 0
    for xi, yi in zip(X, y):
        e = yi - step(w @ xi + b)
        if e:
            w += e * xi; b += e; errs += 1
    if errs == 0:
        print("separable now -> converged in", epoch, "epochs"); break
print("preds:", [step(w @ xi + b) for xi in X], " target:", y.tolist())
