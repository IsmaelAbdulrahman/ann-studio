# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 18: Graph neural networks
# Section: Exercises
# Code example 5 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "gnn"
# ====================================================================

import numpy as np
def gcn(A, X, W):
    At = A + np.eye(len(A)); d = At.sum(1)
    Ahat = np.diag(d ** -0.5) @ At @ np.diag(d ** -0.5)
    return np.maximum(0.0, Ahat @ X @ W)
A = np.array([[0,1,1,0],[1,0,1,0],[1,1,0,1],[0,0,1,0]], float)
X = np.array([[1.,0.],[0.,1.],[1.,-1.],[-1.,0.]])
W = np.array([[1.,1.],[1.,-1.]])
P = np.eye(4)[[3, 1, 2, 0]]                    # permutation: swap nodes 0 and 3
out         = gcn(A, X, W)
out_relabel = gcn(P @ A @ P.T, P @ X, W)
print("max |P*out - out_relabel| =", np.abs(P @ out - out_relabel).max())
