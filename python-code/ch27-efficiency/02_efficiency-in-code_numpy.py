# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 27: Efficiency & deployment
# Section: Efficiency in code
# Code example 2 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "efficiency"
# ====================================================================

import numpy as np
np.random.seed(0)

W = np.random.randn(1000)         # a layer's weights (fp32)
x = np.random.randn(1000)         # an input activation vector
p = 30                             # prune the smallest 30% by magnitude

thresh = np.percentile(np.abs(W), p)   # magnitude cutoff at the 30th percentile
mask   = np.abs(W) >= thresh            # keep the large-magnitude weights
Wp     = W * mask                       # smallest 30% set exactly to zero

y_full, y_pruned = W @ x, Wp @ x
print("achieved sparsity  =", round(np.mean(Wp == 0), 3))          # 0.3
print("dense  dot product =", round(float(y_full), 4))            # -30.7258
print("pruned dot product =", round(float(y_pruned), 4))          # -31.4206
print("relative change    =", round(abs(y_pruned - y_full) / abs(y_full), 4))  # 0.0226
