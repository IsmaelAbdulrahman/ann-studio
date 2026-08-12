# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 18: Graph neural networks
# Section: Graph networks in code
# Code example 1 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "gnn"
# ====================================================================

import numpy as np
np.set_printoptions(precision=4, suppress=True)

# 4-node graph: triangle 0-1-2 with a leaf node 3 hanging off node 2
A = np.array([[0, 1, 1, 0],
              [1, 0, 1, 0],
              [1, 1, 0, 1],
              [0, 0, 1, 0]], float)

At   = A + np.eye(4)            # add self-loops:  A~ = A + I
d    = At.sum(1)               # degrees of A~  ->  [3, 3, 4, 2]
Dinv = np.diag(d ** -0.5)      # D~^(-1/2), diagonal
Ahat = Dinv @ At @ Dinv        # symmetric-normalized  A^ = D~^-1/2 A~ D~^-1/2

H = np.array([[ 1.,  0.],      # node features: 4 nodes, 2 features each
              [ 0.,  1.],
              [ 1., -1.],
              [-1.,  0.]])
W = np.array([[ 1.,  1.],      # learnable weights: col 0 = sum, col 1 = difference
              [ 1., -1.]])

Z    = Ahat @ H @ W            # aggregate along graph, then transform
Hnew = np.maximum(0.0, Z)      # ReLU

print("degrees d~       :", d)
print("A^ row 3         :", Ahat[3])          # [0, 0, 0.3536, 0.5]
print("aggregation A^H  :\n", Ahat @ H)        # normalized neighbor sums
print("updated features :\n", Hnew)            # node 3 feature 0 clipped to 0
