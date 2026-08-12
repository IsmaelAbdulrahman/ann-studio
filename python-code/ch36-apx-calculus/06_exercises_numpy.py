# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 36: Appendix B · Calculus & the chain rule
# Section: Exercises
# Code example 6 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-calculus"
# ====================================================================

import numpy as np
f = lambda v: v[0]**2 * v[1]           # f(x,y) = x^2 y
def num_grad(f, v, h=1e-5):
    v = np.asarray(v, float); g = np.zeros_like(v)
    for i in range(len(v)):
        s = np.zeros_like(v); s[i] = h
        g[i] = (f(v + s) - f(v - s)) / (2*h)
    return g
p = np.array([2.0, 3.0])
print("numeric :", np.round(num_grad(f, p), 6))
print("analytic:", np.array([2*p[0]*p[1], p[0]**2]))   # (12, 4)
