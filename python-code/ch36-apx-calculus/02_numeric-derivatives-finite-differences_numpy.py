# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 36: Appendix B · Calculus & the chain rule
# Section: Numeric derivatives: finite differences
# Code example 2 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-calculus"
# ====================================================================

import numpy as np
def f(v):
    x, y = v
    return x**2 + 3*x*y + y**2       # true gradient: (2x + 3y, 3x + 2y)
def num_grad(f, v, h=1e-5):
    v = np.asarray(v, float)
    g = np.zeros_like(v)
    for i in range(len(v)):
        step = np.zeros_like(v); step[i] = h
        g[i] = (f(v + step) - f(v - step)) / (2*h)   # partial i, rest fixed
    return g
p = np.array([1.0, 2.0])
print("numeric  grad:", np.round(num_grad(f, p), 6))
print("analytic grad:", np.array([2*1 + 3*2, 3*1 + 2*2]))   # (8, 7)
