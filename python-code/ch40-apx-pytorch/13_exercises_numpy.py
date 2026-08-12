# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 40: Appendix F · PyTorch primer
# Section: Exercises
# Code example 13 of 15 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-pytorch"
# ====================================================================

import numpy as np
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
z, h = 0.5, 1e-6
fd = (sigmoid(z + h) - sigmoid(z - h)) / (2 * h)
analytic = sigmoid(z) * (1 - sigmoid(z))
print("finite diff:", round(fd, 6))
print("analytic   :", round(analytic, 6))
