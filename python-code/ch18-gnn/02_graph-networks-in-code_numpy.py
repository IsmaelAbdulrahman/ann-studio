# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 18: Graph neural networks
# Section: Graph networks in code
# Code example 2 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "gnn"
# ====================================================================

import numpy as np
np.random.seed(0)
np.set_printoptions(precision=4, suppress=True)

A  = np.array([[0,1,1,0],[1,0,1,0],[1,1,0,1],[0,0,1,0]], float)
At = A + np.eye(4)
d  = At.sum(1)
Ahat = np.diag(d ** -0.5) @ At @ np.diag(d ** -0.5)

H  = np.array([[1.,0.],[0.,1.],[1.,-1.],[-1.,0.]])
W1 = np.random.randn(2, 2)
W2 = np.random.randn(2, 2)
relu = lambda x: np.maximum(0.0, x)

def gcn2(X):                          # two stacked GCN layers
    H1 = relu(Ahat @ X  @ W1)         # after 1 layer: 1-hop information
    H2 = relu(Ahat @ H1 @ W2)         # after 2 layers: 2-hop information
    return H1, H2

H1, H2   = gcn2(H)
Hp       = H.copy(); Hp[0] += [1.0, 1.0]     # perturb ONLY node 0's features
H1p, H2p = gcn2(Hp)

print("A^[3,0]      =", round(Ahat[3, 0], 4), "  (nodes 0 and 3: NOT 1-hop)")
print("(A^@A^)[3,0] =", round((Ahat @ Ahat)[3, 0], 4), "  (they ARE 2-hop)")
print("node 3 change after 1 layer :", np.abs(H1p[3] - H1[3]).sum().round(4))
print("node 3 change after 2 layers:", np.abs(H2p[3] - H2[3]).sum().round(4))
