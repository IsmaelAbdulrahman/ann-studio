# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 18: Graph neural networks
# Section: Exercises
# Code example 4 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "gnn"
# ====================================================================

import numpy as np
np.set_printoptions(precision=4, suppress=True)
A  = np.array([[0,1,1,0],[1,0,1,0],[1,1,0,1],[0,0,1,0]], float)
At = A + np.eye(4); d = At.sum(1)
Asym = np.diag(d ** -0.5) @ At @ np.diag(d ** -0.5)   # symmetric
Arw  = np.diag(1.0 / d) @ At                           # random walk
H = np.array([[1.,0.],[0.,1.],[1.,-1.],[-1.,0.]])
print("symmetric  node0:", (Asym @ H)[0])
print("randomwalk node0:", (Arw  @ H)[0])
print("randomwalk row sums:", Arw.sum(1))             # all exactly 1
