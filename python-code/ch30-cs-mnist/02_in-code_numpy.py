# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 30: Case study: handwritten-digit recognition (MNIST)
# Section: In code
# Code example 2 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "cs-mnist"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)

# three 8x8 templates -> classes 0,1,2  (digits 0, 1, 7)
T = np.array([
 [0,0,1,1,1,1,0,0, 0,1,0,0,0,0,1,0, 0,1,0,0,0,0,1,0, 0,1,0,0,0,0,1,0,
  0,1,0,0,0,0,1,0, 0,1,0,0,0,0,1,0, 0,1,0,0,0,0,1,0, 0,0,1,1,1,1,0,0],  # 0
 [0,0,0,1,1,0,0,0, 0,0,1,1,1,0,0,0, 0,0,0,1,1,0,0,0, 0,0,0,1,1,0,0,0,
  0,0,0,1,1,0,0,0, 0,0,0,1,1,0,0,0, 0,0,0,1,1,0,0,0, 0,0,1,1,1,1,0,0],  # 1
 [1,1,1,1,1,1,0,0, 0,0,0,0,0,1,0,0, 0,0,0,0,1,0,0,0, 0,0,0,1,0,0,0,0,
  0,0,1,0,0,0,0,0, 0,0,1,0,0,0,0,0, 0,0,1,0,0,0,0,0, 0,0,1,0,0,0,0,0]], # 7
 dtype=float)

def make(n_per):                                # noisy samples around each template
    X, y = [], []
    for k in range(3):
        base = np.tile(T[k], (n_per, 1))
        flip = (rng.rand(n_per, 64) < 0.10).astype(float)
        img  = base * (1 - flip) + (1 - base) * flip
        img  = np.clip(img + rng.normal(0, 0.6, (n_per, 64)), 0, 1)
        X.append(img); y.append(np.full(n_per, k))
    X, y = np.vstack(X), np.concatenate(y)
    p = rng.permutation(len(y)); return X[p], y[p]

Xtr, ytr = make(60); Xte, yte = make(30)
Ytr = np.eye(3)[ytr]                            # one-hot targets

# LINEAR softmax classifier: no hidden layer, 64 pixels -> 3 logits
W = rng.randn(64, 3) * 0.1; b = np.zeros(3)
for ep in range(300):
    S = Xtr @ W + b; S -= S.max(1, keepdims=True)
    P = np.exp(S); P /= P.sum(1, keepdims=True)  # softmax
    dS = (P - Ytr) / len(Xtr)                    # cross-entropy gradient  P - y
    W -= 0.5 * (Xtr.T @ dS); b -= 0.5 * dS.sum(0)

acc = lambda X, y: ((X @ W + b).argmax(1) == y).mean()
print(f"linear softmax   train acc {acc(Xtr,ytr):.3f}   test acc {acc(Xte,yte):.3f}")
# expected: linear softmax   train acc 1.000   test acc 0.967
