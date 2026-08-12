# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 38: Appendix D · Information theory
# Section: Mutual information: how much X tells you about Y
# Code example 2 of 2 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-infotheory"
# ====================================================================

import numpy as np
np.random.seed(0)

def mutual_information(P):
    P  = np.asarray(P, dtype=float)
    P  = P / P.sum()                          # normalise to a valid joint pmf
    px = P.sum(axis=1, keepdims=True)         # P(X): marginal over Y
    py = P.sum(axis=0, keepdims=True)         # P(Y): marginal over X
    m  = P > 0                                # 0 log 0 := 0, so skip empty cells
    return float(np.sum(P[m] * np.log2(P[m] / (px * py)[m])))

# independent: joint is the outer product of the marginals  ->  I(X;Y) = 0
px, py = np.array([0.6, 0.4]), np.array([0.7, 0.3])
indep  = np.outer(px, py)
print("independent joint =", indep.ravel())
print("I(X;Y) independent =", round(abs(mutual_information(indep)), 6), "bits  (exactly 0)")

# dependent: mass on the diagonal, so knowing X pins down Y  ->  I(X;Y) > 0
dep = np.array([[0.45, 0.05],
                [0.05, 0.45]])
print("I(X;Y) dependent   =", round(mutual_information(dep), 4), "bits  (> 0)")
