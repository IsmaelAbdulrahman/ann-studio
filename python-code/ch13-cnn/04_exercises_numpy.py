# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 13: Convolutional neural networks
# Section: Exercises
# Code example 4 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
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
img = np.full((5, 5), 10.0); img[3:, :] = 90.0        # top dark, bottom bright
Kh = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], float)
print(conv2d_valid(img, Kh).astype(int))              # -> [[0,0,0],[-320,...],[-320,...]]
