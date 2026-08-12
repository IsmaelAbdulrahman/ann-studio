# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 3: Multilayer networks & the forward pass
# Section: Exercises
# Code example 7 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "mlp"
# ====================================================================

import numpy as np
X  = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
W1 = np.array([[1., 1.], [1., 1.]]); b1 = np.array([0., -1.])
W2 = np.array([[1.], [-2.]]);        b2 = np.array([0.])
relu = lambda z: np.maximum(0.0, z)
y_relu = (relu(X @ W1 + b1) @ W2 + b2).ravel()
y_lin  = ((X @ W1 + b1) @ W2 + b2).ravel()     # identity hidden = linear net
print("ReLU   hidden ->", np.round(y_relu, 3))  # [0. 1. 1. 0.] = XOR
print("linear hidden ->", np.round(y_lin, 3))   # [2. 1. 1. 0.] != XOR
assert np.allclose(y_relu, [0, 1, 1, 0])
print("XOR solved:", np.allclose(y_relu, [0, 1, 1, 0]))
    
