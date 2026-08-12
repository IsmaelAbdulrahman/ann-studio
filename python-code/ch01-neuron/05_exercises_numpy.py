# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 1: From biological to artificial neurons
# Section: Exercises
# Code example 5 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "neuron"
# ====================================================================

import numpy as np
w = np.array([0.8, 1.1, -3.0]); b = -2.5
x = np.array([4., 2., 1.])
odds = lambda xv: np.exp(w @ xv + b)   # odds = p/(1-p) = e^z
x2 = x.copy(); x2[1] += 1               # one extra link
print("odds ratio for +1 link:", round(odds(x2) / odds(x), 3))
print("e^{w2} =", round(float(np.exp(w[1])), 3))
