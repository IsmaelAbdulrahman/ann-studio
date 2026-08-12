# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 2: The perceptron & linear separability
# Section: The perceptron in code
# Code example 2 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "perceptron"
# ====================================================================

import numpy as np
# The same rule on XOR: it can never reach zero mistakes.
X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = np.array([0, 1, 1, 0])                  # XOR
w = np.zeros(2); b = 0.0; eta = 1.0
step = lambda z: 1 if z >= 0 else 0
mistakes = []
for epoch in range(12):
    errs = 0
    for xi, yi in zip(X, y):
        e = yi - step(np.dot(w, xi) + b)
        if e != 0:
            w += eta * e * xi; b += eta * e; errs += 1
    mistakes.append(errs)
print("mistakes per epoch:", mistakes)
print("never 0 -> XOR is not linearly separable; the loop cycles forever")
