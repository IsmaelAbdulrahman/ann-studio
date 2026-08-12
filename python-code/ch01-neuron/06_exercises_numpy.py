# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 1: From biological to artificial neurons
# Section: Exercises
# Code example 6 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "neuron"
# ====================================================================

import numpy as np
sig = lambda z: 1/(1+np.exp(-z))
X = np.array([[0,0],[1,0],[0,1],[1,1]])    # four Boolean inputs (rows)
w = np.array([4.0, 4.0])
for b, name in [(-6.0, "AND"), (-2.0, "OR ")]:
    a = sig(X @ w + b)                      # probability for each input
    print(name, "->", (a > 0.5).astype(int), np.round(a, 3))
# expected:  AND -> [0 0 0 1] ;  OR  -> [0 1 1 1]
