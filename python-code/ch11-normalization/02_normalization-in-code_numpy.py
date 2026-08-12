# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 11: Normalization: batch, layer & beyond
# Section: Normalization in code
# Code example 2 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "normalization"
# ====================================================================

import numpy as np
np.random.seed(1)
d = 6
x = np.random.randn(d) * 4 + 10                  # ONE example, 6 features, off-scale
mu  = x.mean()                                   # mean ACROSS the 6 features
var = x.var()                                    # var  ACROSS the 6 features
xhat = (x - mu) / np.sqrt(var + 1e-5)
gamma, beta = np.ones(d), np.zeros(d)
y = gamma * xhat + beta
print("x    =", np.round(x, 3))
print("xhat =", np.round(xhat, 3))
print("mean across features =", round(float(y.mean()), 6))   # ~0
print("var  across features =", round(float(y.var()),  6))   # ~1
