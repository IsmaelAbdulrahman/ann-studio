# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 8: Backpropagation
# Section: Exercises
# Code example 4 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "backprop"
# ====================================================================

import numpy as np
sig = lambda z: 1/(1+np.exp(-z))
x = np.array([0.05, 0.10]); y = 0.70
W1 = np.array([[0.15, 0.25], [0.20, 0.30]]); b1 = np.array([0.35, 0.35])
W2 = np.array([[0.40], [0.50]]);             b2 = np.array([0.60])
def loss(W1):
    a1 = sig(x @ W1 + b1); yh = a1 @ W2 + b2
    return 0.5 * (yh - y) ** 2                # array of shape (1,)
e = 1e-5
Wp = W1.copy(); Wp[0, 0] += e
Wm = W1.copy(); Wm[0, 0] -= e
num = (loss(Wp) - loss(Wm)) / (2 * e)
print("numeric dL/dW1[0,0] =", num[0])       # ~0.00210, matches analytic
    
