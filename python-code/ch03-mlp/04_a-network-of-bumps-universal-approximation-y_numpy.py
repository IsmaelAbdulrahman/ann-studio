# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 3: Multilayer networks & the forward pass
# Section: A network of bumps: universal approximation you can see
# Code example 4 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "mlp"
# ====================================================================

import numpy as np
# One hidden layer of sigmoids builds any smooth 1-D function as a staircase.
sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))
xs = np.linspace(0, 1, 300)
f  = np.sin(2 * np.pi * xs)             # target function
K, slope = 60, 80.0                     # K hidden units, steep steps
edges   = np.linspace(0, 1, K + 1)
centers = 0.5 * (edges[:-1] + edges[1:])
deltas  = np.diff(np.sin(2 * np.pi * edges))   # height of each step
approx  = f[0] + sum(d * sigmoid(slope * (xs - c))
                     for d, c in zip(deltas, centers))
print("hidden units:", K, " max|approx - f| =",
      round(float(np.max(np.abs(approx - f))), 3))
