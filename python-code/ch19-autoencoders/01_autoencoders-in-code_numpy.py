# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 19: Autoencoders & representation learning
# Section: Autoencoders in code
# Code example 1 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "autoencoders"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)

# data that really lives on a 2-D plane inside 4-D space, plus tiny noise
B = np.linalg.qr(rng.randn(4, 2))[0].T   # 2 orthonormal directions
Z = rng.randn(120, 2)
X = Z @ B + 0.05 * rng.randn(120, 4)
X = X - X.mean(axis=0)                    # centre the data

n, d = 4, 2                               # 4 features -> code of size 2
We = rng.randn(n, d) * 0.1               # encoder  x -> z
Wd = rng.randn(d, n) * 0.1               # decoder  z -> x-hat
lr = 0.2
for step in range(401):
    Zc = X @ We                          # codes  (120, 2)
    Xh = Zc @ Wd                         # reconstruction (120, 4)
    E  = Xh - X                          # residual
    loss = np.mean(np.sum(E**2, axis=1))
    We -= lr * (X.T @ (E @ Wd.T)) * (2 / len(X))
    Wd -= lr * (Zc.T @ E) * (2 / len(X))
    if step % 100 == 0:
        print(f"step {step:3d}   reconstruction error {loss:.4f}")
print("floor set by the injected noise is ~0.005")
