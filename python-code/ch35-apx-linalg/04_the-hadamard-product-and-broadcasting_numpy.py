# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 35: Appendix A · Linear algebra refresher
# Section: The Hadamard product and broadcasting
# Code example 4 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-linalg"
# ====================================================================

import numpy as np
A = np.array([[1., 2.], [3., 4.]])
B = np.array([[10., 20.], [30., 40.]])
print("Hadamard A*B =\n", A * B)      # elementwise, NOT matrix product
print("matmul   A@B =\n", A @ B)      # for contrast
# broadcasting: one bias per feature added to every row
X = np.array([[1., 2., 3.],
              [4., 5., 6.]])          # (2, 3): two samples in rows
b = np.array([10., 20., 30.])         # (3,): stretched across both rows
print("X + b =\n", X + b)
col = np.array([[100.], [200.]])      # (2, 1): stretched across columns
print("X + col =\n", X + col)
