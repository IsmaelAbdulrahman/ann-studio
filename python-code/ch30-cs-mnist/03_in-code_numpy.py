# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 30: Case study: handwritten-digit recognition (MNIST)
# Section: In code
# Code example 3 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "cs-mnist"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)

# three 8x8 digit templates -> classes 0,1,2  (digits 0, 1, 7)
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
        flip = (rng.rand(n_per, 64) < 0.10).astype(float)   # salt-and-pepper
        img  = base * (1 - flip) + (1 - base) * flip
        img  = np.clip(img + rng.normal(0, 0.6, (n_per, 64)), 0, 1)
        X.append(img); y.append(np.full(n_per, k))
    X, y = np.vstack(X), np.concatenate(y)
    p = rng.permutation(len(y))
    return X[p], y[p]

Xtr, ytr = make(60)
Xte, yte = make(30)
Ytr = np.eye(3)[ytr]                            # one-hot targets

# a 64 -> 16 -> 3 MLP: ReLU hidden, softmax output, trained from scratch
W1 = rng.randn(64, 16) * 0.1; b1 = np.zeros(16)
W2 = rng.randn(16, 3)  * 0.1; b2 = np.zeros(3)
for ep in range(1, 81):
    H = np.maximum(0, Xtr @ W1 + b1)            # forward
    S = H @ W2 + b2; S -= S.max(1, keepdims=True)
    P = np.exp(S); P /= P.sum(1, keepdims=True)  # softmax
    dS = (P - Ytr) / len(Xtr)                    # backprop of cross-entropy
    dW2 = H.T @ dS; db2 = dS.sum(0)
    dH  = (dS @ W2.T) * (H > 0)
    dW1 = Xtr.T @ dH; db1 = dH.sum(0)
    for par, g in [(W1,dW1),(b1,db1),(W2,dW2),(b2,db2)]:
        par -= 0.25 * g                          # gradient step
    if ep in (1, 5, 10, 20, 40, 80):
        print(f"epoch {ep:2d}   train acc {(P.argmax(1)==ytr).mean():.3f}")

pred = (np.maximum(0, Xte @ W1 + b1) @ W2 + b2).argmax(1)
print(f"\ntest accuracy {(pred==yte).mean():.3f}")
C = np.zeros((3,3), int)
for t, p in zip(yte, pred): C[t, p] += 1
print("confusion (rows = true 0/1/7, cols = predicted):")
print(C)
