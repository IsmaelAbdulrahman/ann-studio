# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 36: Appendix B · Calculus & the chain rule
# Section: The chain rule is the engine of backprop → Worked example — a two-link chain, forward and backward
# Code example 4 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-calculus"
# ====================================================================

import numpy as np
sig = lambda z: 1/(1 + np.exp(-z))
x, w, b, y = 1.0, 0.5, 0.0, 1.0
def loss(w):
    a = sig(w*x + b)
    return 0.5*(a - y)**2
z = w*x + b; a = sig(z)
analytic = (a - y) * a*(1 - a) * x        # chain rule, three links
numeric = (loss(w + 1e-6) - loss(w - 1e-6)) / (2e-6)
print("a =", round(a, 6), "  loss =", round(loss(w), 6))
print("analytic dL/dw =", round(analytic, 6))
print("numeric  dL/dw =", round(numeric, 6))
