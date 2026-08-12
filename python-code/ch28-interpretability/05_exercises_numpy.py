# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 28: Interpretability & explainability
# Section: Exercises
# Code example 5 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "interpretability"
# ====================================================================

import numpy as np
np.random.seed(1)
n = 4
w = np.array([1.5, -2.0, 0.7, 0.0])       # feature 4 (index 3) has zero weight...
A = np.zeros((n, n)); A[0,1] = A[1,0] = 1.0   # ...and appears in no interaction
x = np.array([1.0, 1.0, 1.0, 9.0]); b = np.zeros(n)
def perms(s):
    if len(s) <= 1: return [list(s)]
    return [[s[i]]+p for i in range(len(s)) for p in perms(s[:i]+s[i+1:])]
def f(z): return w @ z + z @ A @ z
phi = np.zeros(n)
for o in perms(list(range(n))):
    z = b.copy(); prev = f(z)
    for i in o:
        z = z.copy(); z[i] = x[i]; cur = f(z); phi[i] += cur - prev; prev = cur
phi /= len(perms(list(range(n))))
print("Shapley values =", np.round(phi, 4))
print("dummy feature 4 value =", round(float(phi[3]), 6))   # exactly 0.0
