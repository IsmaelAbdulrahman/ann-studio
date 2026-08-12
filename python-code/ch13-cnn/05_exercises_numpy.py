# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 13: Convolutional neural networks
# Section: Exercises
# Code example 5 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
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
rng = np.random.RandomState(0)
img = rng.randint(0, 9, (5, 5)).astype(float)
K = np.ones((3, 3)) / 9.0                       # 3x3 blur kernel
same = conv2d_valid(np.pad(img, 1), K)          # zero-pad 1 -> 'same'
print("input:", img.shape, " same-conv:", same.shape)   # both (5, 5)
