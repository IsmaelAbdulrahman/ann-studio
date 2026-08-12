# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 16: Attention & the transformer
# Section: Attention in code
# Code example 2 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "attention"
# ====================================================================

import numpy as np
np.set_printoptions(precision=4, suppress=True)

def softmax(v):
    v = v - v.max()
    e = np.exp(v)
    return e / e.sum()

# the worked example, in code: one query, three keys, three values
q = np.array([1.0, 0.0])
K = np.array([[1.0, 0.0],
              [0.0, 1.0],
              [1.0, 1.0]])
V = np.array([[1.0, 0.0],
              [0.0, 1.0],
              [1.0, 1.0]])

dk = q.shape[0]                    # key/query dimension = 2
scores = (K @ q) / np.sqrt(dk)     # scaled dot products
alpha  = softmax(scores)           # attention weights, sum to 1
out    = alpha @ V                 # weighted blend of the values

print("scaled scores:", scores)
print("weights      :", alpha, " sum =", alpha.sum())
print("output       :", out)
