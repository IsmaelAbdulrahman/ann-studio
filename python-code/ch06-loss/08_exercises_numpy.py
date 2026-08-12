# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 6: Loss functions & maximum likelihood
# Section: Exercises
# Code example 8 of 8 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "loss"
# ====================================================================

import numpy as np
def softmax_rows(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E/E.sum(axis=1, keepdims=True)
Z = np.array([[3.0, 0.5, 0.2],       # easy: true class #0 dominates
              [0.2, 0.3, 0.1],       # hard: nearly uniform logits
              [1.0, 2.0, 0.5]])      # true class #1 is not the argmax
y_idx = np.array([0, 0, 1])
P  = softmax_rows(Z)
pt = P[np.arange(3), y_idx]          # true-class probability per row
ce = -np.log(pt)
focal = (1 - pt)**2 * ce             # gamma = 2
print("p_true =", np.round(pt, 3))       # [0.875 0.332 0.629]
print("CE     =", np.round(ce, 3))       # [0.134 1.102 0.464]
print("focal  =", np.round(focal, 3))    # [0.002 0.491 0.064]  easy row crushed
