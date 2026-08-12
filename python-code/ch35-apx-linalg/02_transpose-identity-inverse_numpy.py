# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 35: Appendix A · Linear algebra refresher
# Section: Transpose, identity, inverse
# Code example 2 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-linalg"
# ====================================================================

import numpy as np
A = np.array([[1., 2., 3.],
              [4., 5., 6.]])        # (2, 3)
print("A.T =\n", A.T, "\nshape", A.shape, "->", A.T.shape)
# transpose reverses a product: (A @ B).T == B.T @ A.T
B = np.array([[1., 0.], [0., 1.], [1., 1.]])
print("(A@B).T == B.T@A.T :", np.allclose((A @ B).T, B.T @ A.T))
# inverse undoes a square matrix: M @ inv(M) = I
M = np.array([[1., 2.], [3., 4.]])
print("M @ inv(M) =\n", np.round(M @ np.linalg.inv(M), 6))
