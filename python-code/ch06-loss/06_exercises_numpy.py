# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 6: Loss functions & maximum likelihood
# Section: Exercises
# Code example 6 of 8 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "loss"
# ====================================================================

import numpy as np
def softmax_rows(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E/E.sum(axis=1, keepdims=True)
Z = np.array([[1.0, 2.0, 0.5],
              [0.1, 0.2, 3.0]])
y_idx = np.array([1, 2])            # true class per row
P = softmax_rows(Z)
ce = -np.mean(np.log(P[np.arange(len(y_idx)), y_idx]))
print("P =", np.round(P, 3))
print("batch cross-entropy =", round(ce, 4))
