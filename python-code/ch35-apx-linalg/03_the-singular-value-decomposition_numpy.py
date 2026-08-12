# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 35: Appendix A · Linear algebra refresher
# Section: The singular value decomposition
# Code example 3 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-linalg"
# ====================================================================

import numpy as np
np.random.seed(0)

# matrix multiplication is composition; order matters
A = np.array([[1., 2.], [0., 1.]])     # a shear
B = np.array([[0., -1.], [1., 0.]])    # a 90-degree rotation
print("A @ B =\n", (A @ B).astype(int))    # [[2 -1] [1 0]]
print("B @ A =\n", (B @ A).astype(int))    # [[0 -1] [1 2]]  -> A@B != B@A

# vector and matrix norms
v = np.array([2., 2., 1.])
print("L1, L2, Linf =",
      np.linalg.norm(v, 1), np.linalg.norm(v, 2), np.linalg.norm(v, np.inf))  # 5.0 3.0 2.0
M = np.array([[2., 1.], [1., 2.]])
print("Frobenius =", round(float(np.linalg.norm(M, 'fro')), 4))    # sqrt(10) = 3.1623

# eigen-decomposition: symmetric M has real eigenvalues 3 and 1
vals, vecs = np.linalg.eig(M)
print("eigenvalues =", np.round(vals, 4))
print("eigenvectors (columns) =\n", np.round(vecs, 4))

# singular value decomposition  A = U @ diag(S) @ Vt
U, S, Vt = np.linalg.svd(M)
print("singular values =", np.round(S, 4))                  # [3. 1.] == |eigenvalues|
print("reconstructs M:", np.allclose(U @ np.diag(S) @ Vt, M))   # True
