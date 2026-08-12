# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 24: Uncertainty, calibration & Bayesian nets
# Section: Uncertainty in code
# Code example 1 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "uncertainty"
# ====================================================================

import numpy as np
np.random.seed(0)

# ten predictions: their confidence (max softmax prob) and whether each was correct
conf    = np.array([0.6, 0.6, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9])
correct = np.array([1,   0,   1,   1,   0,   1,   1,   1,   1,   0  ], float)

def ece(conf, correct, M=10):                 # Expected Calibration Error, M equal bins
    idx = np.minimum((conf * M).astype(int), M - 1)   # bin index of each prediction
    e = 0.0
    for b in range(M):
        sel = idx == b
        k = sel.sum()
        if k:                                 # weighted gap between accuracy and confidence
            e += k / len(conf) * abs(correct[sel].mean() - conf[sel].mean())
    return e

print("ECE =", round(ece(conf, correct), 4))  # expected: 0.11

def softmax(z):
    z = z - z.max()                           # numerical stability (Chapter 6)
    e = np.exp(z)
    return e / e.sum()

z  = np.array([3.0, 1.0, 0.2])                # an over-confident logit vector
p1 = softmax(z)                               # T = 1
p2 = softmax(z / 2.0)                          # temperature scaling, T = 2 (softer)
print("T=1  max prob =", round(p1.max(), 4))  # 0.836  <- over-confident
print("T=2  max prob =", round(p2.max(), 4))  # 0.619  <- calibrated down, class unchanged
