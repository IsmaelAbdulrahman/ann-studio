# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 10: Initialization & the vanishing gradient
# Section: Exercises
# Code example 5 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "init"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)
x = rng.randn(8, 4) * 5 + 3                      # batch 8, 4 features, off-scale
mu = x.mean(0); var = x.var(0)                  # per-feature stats over the batch
xhat = (x - mu) / np.sqrt(var + 1e-8)
gamma, beta = 1.0, 0.0
y = gamma * xhat + beta
print("post-BN mean:", np.round(y.mean(0), 4))  # ~0
print("post-BN std :", np.round(y.std(0), 4))   # ~1
    
