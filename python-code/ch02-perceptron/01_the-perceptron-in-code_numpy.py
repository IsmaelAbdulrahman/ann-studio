# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 2: The perceptron & linear separability
# Section: The perceptron in code
# Code example 1 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "perceptron"
# ====================================================================

import numpy as np
# Perceptron learning rule on a linearly separable set.
X = np.array([[2., 2.], [1., 3.], [3., 1.],       # class 1
              [-1., -2.], [-2., -1.], [-2., -3.]]) # class 0
y = np.array([1, 1, 1, 0, 0, 0])
w = np.zeros(2); b = 0.0; eta = 1.0
step = lambda z: 1 if z >= 0 else 0
for epoch in range(1, 21):
    errors = 0
    for xi, yi in zip(X, y):
        e = yi - step(np.dot(w, xi) + b)
        if e != 0:
            w += eta * e * xi          # rotate boundary toward the mistake
            b += eta * e
            errors += 1
    if errors == 0:
        break
preds = np.array([step(np.dot(w, xi) + b) for xi in X])
print("converged after", epoch, "epochs")
print("w =", w, " b =", b)
print("accuracy =", (preds == y).mean())
