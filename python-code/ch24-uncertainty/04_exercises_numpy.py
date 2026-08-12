# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 24: Uncertainty, calibration & Bayesian nets
# Section: Exercises
# Code example 4 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "uncertainty"
# ====================================================================

import numpy as np
conf    = np.array([0.6, 0.6, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9])
correct = np.array([1,   0,   1,   1,   0,   1,   1,   1,   1,   0  ], float)

def calib(conf, correct, M):
    idx = np.minimum((conf * M).astype(int), M - 1)
    ece = mce = 0.0
    for b in range(M):
        sel = idx == b
        k = sel.sum()
        if k:
            gap = abs(correct[sel].mean() - conf[sel].mean())
            ece += k / len(conf) * gap
            mce = max(mce, gap)
    return ece, mce

for M in [5, 10, 15, 20]:
    e, m = calib(conf, correct, M)
    print(f"M={M:2d}  ECE={e:.4f}  MCE={m:.4f}")
