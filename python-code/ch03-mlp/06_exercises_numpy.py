# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 3: Multilayer networks & the forward pass
# Section: Exercises
# Code example 6 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "mlp"
# ====================================================================

import numpy as np
W1 = np.array([[1., 2.], [0., 1.]])
W2 = np.array([[1., 0.], [3., 1.]])
x  = np.array([[2., -1.]])
print("stacked  :", (x @ W1) @ W2)
print("collapsed:", x @ (W1 @ W2))
print("W1@W2 =\n", W1 @ W2)
