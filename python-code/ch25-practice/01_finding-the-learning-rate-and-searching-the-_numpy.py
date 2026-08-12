# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 25: Training in practice: the complete recipe
# Section: Finding the learning rate, and searching the rest
# Code example 1 of 8 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "practice"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)
X = rng.normal(0, 1, 200)                  # one feature
y = 2.0 * X + 0.1 * rng.normal(0, 1, 200)  # target: slope 2, small noise

def sweep(lr, steps=80):                    # GD on MSE from the same start
    w = 0.0
    for _ in range(steps):
        w -= lr * (2/len(y)) * (X * (w*X - y)).sum()
    return ((w*X - y)**2).mean()

curv = (2/len(y)) * (X*X).sum()             # GD is stable only for lr < 2/curv
print(f"stability ceiling 2/curv = {2/curv:.3f}")
for lr in [1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]:
    L = sweep(lr)
    tag = "diverges" if L > 1e3 else ("too slow" if L > 0.05 else "good")
    print(f"lr={lr:<6} final loss={L:12.4f}   {tag}")
