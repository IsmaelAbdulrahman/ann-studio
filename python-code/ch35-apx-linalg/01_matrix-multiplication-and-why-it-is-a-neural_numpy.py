# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 35: Appendix A · Linear algebra refresher
# Section: Matrix multiplication — and why it is a neural layer
# Code example 1 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-linalg"
# ====================================================================

import numpy as np
A = np.array([[1, 2, 3],
              [4, 5, 6]])          # shape (2, 3)
B = np.array([[1, 0],
              [0, 1],
              [1, 1]])             # shape (3, 2)
print("A.shape =", A.shape, "  B.shape =", B.shape)
C = A @ B                          # inner dims 3 and 3 match -> (2, 2)
print("A @ B =\n", C)
print("result shape =", C.shape)
# a single dot product is one neuron's weighted sum
print("dot([1,2,3],[4,5,6]) =", np.dot([1, 2, 3], [4, 5, 6]))
