# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 29: Ethics, fairness & safety
# Section: Exercises
# Code example 4 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "ethics"
# ====================================================================

import numpy as np
np.random.seed(0)
n = 5000
sA = {1: np.random.randn(n)+1.5, 0: np.random.randn(n)-1.5}   # group A scores
sB = {1: np.random.randn(n)+0.5, 0: np.random.randn(n)-1.5}   # group B positives score lower
def rates(s, t):
    return (s[1] > t).mean(), (s[0] > t).mean(), 0.5*(s[1] > t).mean()+0.5*(s[0] > t).mean()
A = rates(sA, 0.0)
B = rates(sB, 0.0)
print("common t=0.0  A: TPR=%.3f FPR=%.3f sel=%.3f" % A)
print("common t=0.0  B: TPR=%.3f FPR=%.3f sel=%.3f" % B)
print("equal-opportunity gap =", round(abs(A[0]-B[0]), 3))
t = min(np.linspace(-3, 1, 401), key=lambda t: abs(rates(sB, t)[0]-A[0]))  # match A's TPR
Bnew = rates(sB, t)
print("move B threshold to %.2f -> B: TPR=%.3f FPR=%.3f sel=%.3f" % (t, *Bnew))
print("TPR gap now=%.3f  but FPR gap=%.3f" % (abs(A[0]-Bnew[0]), abs(A[1]-Bnew[1])))
    
