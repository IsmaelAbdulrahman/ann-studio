# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 12: Regularization & generalization
# Section: Exercises
# Code example 5 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "regularization"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)
f = lambda t: np.sin(2.5*t)
xte = np.linspace(-0.9, 0.9, 50)          # evaluate on the interior
def bias_var(deg, ntr=20, reps=400):
    P = []
    for _ in range(reps):
        xtr = np.sort(rng.uniform(-1, 1, ntr))
        ytr = f(xtr) + rng.normal(0, 0.2, ntr)
        w = np.linalg.lstsq(np.vander(xtr, deg+1), ytr, rcond=None)[0]
        P.append(np.vander(xte, deg+1) @ w)
    P = np.array(P)
    return np.mean((P.mean(0)-f(xte))**2), np.mean(P.var(0))
for deg in [1, 3, 7]:
    b, v = bias_var(deg)
    print(f"degree {deg}: bias^2={b:.4f}  variance={v:.4f}")
