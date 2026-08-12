# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 8: Backpropagation
# Section: Exercises
# Code example 3 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "backprop"
# ====================================================================

import numpy as np
sig = lambda z: 1/(1+np.exp(-z))
X  = np.array([[0.05, 0.10], [0.20, 0.30]])   # two examples, one per row
Y  = np.array([[0.70], [0.10]])
W1 = np.array([[0.15, 0.25], [0.20, 0.30]]); b1 = np.array([0.35, 0.35])
W2 = np.array([[0.40], [0.50]]);             b2 = np.array([0.60])
A1 = sig(X @ W1 + b1)
Yh = A1 @ W2 + b2
D2 = (Yh - Y)                                 # B x 1
gW2 = A1.T @ D2                               # the matmul SUMS over the batch
D1  = (D2 @ W2.T) * A1 * (1 - A1)
gW1 = X.T @ D1
print("gW2 =", gW2.ravel())
print("gb2 =", D2.sum(0), "  gb1 =", D1.sum(0))
print("gW1 =\n", gW1)
    
