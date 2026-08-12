# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 39: Appendix E · Python & NumPy primer
# Section: Two vectorized building blocks: ReLU and softmax → Worked example — a stable softmax by hand
# Code example 13 of 22 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-python"
# ====================================================================

import numpy as np

def relu(z):
    return np.maximum(0.0, z)              # elementwise, no loop, no mutation

def softmax(z):                            # row-wise and numerically stable
    z = z - z.max(axis=1, keepdims=True)   # shift by row max (safe: shift-invariant)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

rng = np.random.default_rng(0)
X  = rng.standard_normal((4, 3))           # batch of 4 examples, 3 features each
W1 = rng.standard_normal((3, 5)) * 0.5     # layer 1: features -> hidden
b1 = np.zeros(5)                           # bias broadcast across the whole batch
W2 = rng.standard_normal((5, 2)) * 0.5     # layer 2: hidden -> 2 classes
b2 = np.zeros(2)

H = relu(X @ W1 + b1)                       # (4,3)@(3,5) + (5,) -> (4,5)
S = H  @ W2 + b2                            # (4,5)@(5,2) + (2,) -> (4,2) logits
P = softmax(S)                             # (4,2) probabilities

print("shapes  X:", X.shape, " H:", H.shape, " P:", P.shape)
print("logits row 0:", S[0].round(3))      # -> [-0.007 -0.047]
print("probs  row 0:", P[0].round(3))      # -> [0.51  0.49]
print("row sums    :", P.sum(axis=1).round(6))   # every row -> 1.0
