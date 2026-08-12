# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 36: Appendix B · Calculus & the chain rule
# Section: Numeric derivatives: finite differences
# Code example 1 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-calculus"
# ====================================================================

import numpy as np
f = lambda x: x**3 - 2*x            # analytic derivative: 3x^2 - 2
def deriv(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2*h)   # central difference
x0 = 1.5
print("numeric  f'(1.5) =", round(deriv(f, x0), 6))
print("analytic 3x^2-2  =", 3*x0**2 - 2)
# the one-sided forward difference is cruder (error ~ h, not h^2)
fwd = (f(x0 + 1e-5) - f(x0)) / 1e-5
print("forward diff     =", round(fwd, 6))
