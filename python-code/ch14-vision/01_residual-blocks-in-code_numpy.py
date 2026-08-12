# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 14: Modern CNN architectures & computer vision
# Section: Residual blocks in code
# Code example 1 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "vision"
# ====================================================================

import numpy as np
np.random.seed(0)

def conv2d_same(x, W, b):
    # x:(C,H,Wd)  W:(F,C,k,k)  b:(F,)  ->  out:(F,H,Wd) with zero-pad 'same'
    C, H, Wd = x.shape
    F, _, k, _ = W.shape
    p = k // 2
    xp = np.pad(x, ((0, 0), (p, p), (p, p)))
    out = np.zeros((F, H, Wd))
    for f in range(F):
        for i in range(H):
            for j in range(Wd):
                out[f, i, j] = np.sum(xp[:, i:i+k, j:j+k] * W[f]) + b[f]
    return out

relu = lambda t: np.maximum(0.0, t)
C, H, Wd, k = 3, 6, 6, 3
x  = np.random.randn(C, H, Wd)
W1 = 0.10 * np.random.randn(C, C, k, k); b1 = np.zeros(C)
W2 = 0.10 * np.random.randn(C, C, k, k); b2 = np.zeros(C)

Fx = conv2d_same(relu(conv2d_same(x, W1, b1)), W2, b2)   # the residual branch F(x)
y  = Fx + x                                              # y = F(x) + x
print("x shape:", x.shape, " F(x) shape:", Fx.shape, " y shape:", y.shape)
print("shapes match, so x + F(x) is valid:", y.shape == x.shape)

# Zero the LAST conv (BN gamma=0 trick): F(x)=0 exactly -> block is the identity
y0 = conv2d_same(relu(conv2d_same(x, W1, b1)), np.zeros_like(W2), b2) + x
print("F=0 at init  ->  y == x exactly:", np.allclose(y0, x))   # expected: True
