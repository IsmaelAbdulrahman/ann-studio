# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 11: Normalization: batch, layer & beyond
# Section: Exercises
# Code example 4 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "normalization"
# ====================================================================

import numpy as np
np.random.seed(3)
N, D = 5, 8
X = np.random.randn(N, D) * 6 + 20               # 5 examples, 8 features, off-scale
mu  = X.mean(1, keepdims=True)                   # mean ACROSS FEATURES, one per row
var = X.var(1, keepdims=True)                    # var  ACROSS FEATURES, one per row
Xhat = (X - mu) / np.sqrt(var + 1e-5)
print("row means:", np.round(Xhat.mean(1), 4))   # all ~0  (LayerNorm standardizes rows)
print("row vars :", np.round(Xhat.var(1),  4))   # all ~1
print("col means:", np.round(Xhat.mean(0), 3))   # NOT zero: columns are not standardized
    
