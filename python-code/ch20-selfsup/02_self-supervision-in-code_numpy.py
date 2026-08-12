# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 20: Self-supervised & contrastive learning
# Section: Self-supervision in code
# Code example 2 of 4 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "selfsup"
# ====================================================================

import numpy as np
np.random.seed(1)

def nt_xent(Z, N, tau=0.5):                 # Z is (2N, d); row i pairs with i+N
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    S = (Z @ Z.T) / tau
    np.fill_diagonal(S, -1e9)
    S -= S.max(axis=1, keepdims=True)
    P = np.exp(S) / np.exp(S).sum(axis=1, keepdims=True)
    pos = (np.arange(2*N) + N) % (2*N)
    return -np.log(P[np.arange(2*N), pos]).mean()

N, d = 4, 8
Z = np.random.randn(2*N, d)
before = nt_xent(Z, N)

# gradient-free nudge: pull ONE positive pair (rows 0 and 4) 30% toward their mean
mid = 0.5 * (Z[0] + Z[4])
Z[0] = 0.7 * Z[0] + 0.3 * mid
Z[4] = 0.7 * Z[4] + 0.3 * mid
after = nt_xent(Z, N)

print("loss before nudge:", round(float(before), 4))   # expected: 2.4591
print("loss after  nudge:", round(float(after), 4))    # expected: 2.2902
print("loss dropped?    :", bool(after < before))       # expected: True
