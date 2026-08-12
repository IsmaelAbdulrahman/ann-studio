# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 14: Modern CNN architectures & computer vision
# Section: Residual blocks in code
# Code example 2 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "vision"
# ====================================================================

import numpy as np

def conv1d_same(x, w):                        # 'same' 1-D convolution
    k = len(w); p = k // 2
    xp = np.pad(x, (p, p))
    return np.array([xp[i:i+k] @ w for i in range(len(x))])

n = 41
for L in [1, 2, 3, 5, 7]:
    s = np.zeros(n); s[n // 2] = 1.0          # a single impulse at the centre
    for _ in range(L):
        s = conv1d_same(s, np.ones(3))        # each 3-tap layer widens support by 2
    measured = int(np.count_nonzero(s))       # inputs that influence the centre unit
    print(f"{L} conv layers:  measured RF = {measured:2d}   formula 1+2L = {1 + 2*L}")
