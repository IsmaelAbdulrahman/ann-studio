# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 4: Activation functions
# Section: Exercises
# Code example 7 of 9 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "activations"
# ====================================================================

import numpy as np
def softplus(z): return np.log1p(np.exp(z))
def sigmoid(z):  return 1/(1+np.exp(-z))
z0 = 0.5; eps = 1e-6
num = (softplus(z0+eps) - softplus(z0))/eps
print("finite-diff softplus'(0.5) =", round(num, 6))
print("sigmoid(0.5)               =", round(sigmoid(z0), 6))
