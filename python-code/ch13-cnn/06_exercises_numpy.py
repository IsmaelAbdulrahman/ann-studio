# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 13: Convolutional neural networks
# Section: Exercises
# Code example 6 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "cnn"
# ====================================================================

import numpy as np
def conv2d_valid(I, K):
    kh, kw = K.shape
    out = np.zeros((I.shape[0]-kh+1, I.shape[1]-kw+1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(I[i:i+kh, j:j+kw] * K)
    return out

I = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], float)
K = np.array([[1, 0], [0, -1]], float)          # top-left minus bottom-right
print("worked example:\n", conv2d_valid(I, K).astype(int))   # [[-4 -4] [-4 -4]]

rng = np.random.RandomState(0)
X   = rng.randn(8, 8)
Ker = rng.randn(3, 3)
base  = conv2d_valid(X, Ker)                     # conv of original
shift = conv2d_valid(np.roll(X, 1, axis=1), Ker) # conv of input shifted right by 1
# equivariant: shifting the input shifts the output (compare interior columns)
print("equivariant:", np.allclose(shift[:, 1:], base[:, :-1]))   # True
