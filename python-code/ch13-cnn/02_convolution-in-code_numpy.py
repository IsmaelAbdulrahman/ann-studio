# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 13: Convolutional neural networks
# Section: Convolution in code
# Code example 2 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "cnn"
# ====================================================================

import numpy as np
A = np.array([[1, 3, 2, 4],
              [5, 6, 1, 2],
              [7, 2, 9, 1],
              [3, 4, 2, 8]], float)      # a 4x4 feature map

def pool(A, k=2, s=2, mode="max"):
    oh = (A.shape[0]-k)//s + 1
    ow = (A.shape[1]-k)//s + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            win = A[i*s:i*s+k, j*s:j*s+k]
            out[i, j] = win.max() if mode == "max" else win.mean()
    return out

print("max-pool 2x2 /2:\n", pool(A).astype(int))          # -> [[6, 4], [7, 9]]
print("avg-pool 2x2 /2:\n", pool(A, mode="avg"))          # each 2x2 block's mean
