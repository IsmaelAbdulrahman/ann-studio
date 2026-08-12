# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 13: Convolutional neural networks
# Section: Convolution in code
# Code example 1 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "cnn"
# ====================================================================

import numpy as np
img = np.full((5, 5), 10.0)          # a flat dark image …
img[:, 3:] = 90.0                    # … with a vertical edge: left dark, right bright
K = np.array([[1, 0, -1],
              [2, 0, -2],
              [1, 0, -1]], float)    # vertical Sobel kernel

def conv2d_valid(I, K):
    kh, kw = K.shape
    oh, ow = I.shape[0]-kh+1, I.shape[1]-kw+1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i, j] = np.sum(I[i:i+kh, j:j+kw] * K)   # patch ⊙ kernel, summed
    return out

fmap = conv2d_valid(img, K)
print("feature map (valid, 3x3):\n", fmap.astype(int))
print("matches hand calc:", np.allclose(fmap,
      [[0,-320,-320],[0,-320,-320],[0,-320,-320]]))
