# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 11: Normalization: batch, layer & beyond
# Section: Normalization in code
# Code example 1 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "normalization"
# ====================================================================

import numpy as np
np.random.seed(0)

# --- the worked example: ONE feature, a batch of 4 examples ---
x = np.array([[2.], [4.], [6.], [8.]])          # column = one feature over the batch
mu  = x.mean(0)                                  # mu_B  -> [5.]
var = x.var(0)                                   # sigma^2_B (population, /m) -> [5.]
xhat = (x - mu) / np.sqrt(var + 1e-5)
y = 2.0 * xhat + 1.0                             # gamma = 2, beta = 1
print("mu_B =", mu, " var_B =", var)             # [5.] [5.]
print("xhat =", xhat.ravel())                    # [-1.3416 -0.4472 0.4472 1.3416]
print("mean(xhat) =", round(float(xhat.mean()), 6),
      " var(xhat) =", round(float(xhat.var()), 6))   # ~0 and ~1
print("y    =", y.ravel())                       # [-1.6833 0.1056 1.8944 3.6833]

# --- BatchNorm forward on an off-scale (N x D) batch: normalize DOWN each column ---
N, D = 64, 5
X = np.random.randn(N, D) * np.array([3, .5, 10, 2, 7]) + np.array([1, -4, 20, 0, 8])
mu, var = X.mean(0), X.var(0)                    # per-feature stats over the batch
Xhat = (X - mu) / np.sqrt(var + 1e-5)
print("per-feature mean of xhat:", np.round(Xhat.mean(0), 4))   # all ~0
print("per-feature var  of xhat:", np.round(Xhat.var(0), 4))    # all ~1
