# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 36: Appendix B · Calculus & the chain rule
# Section: The chain rule is the engine of backprop → Worked example — a two-link chain, forward and backward
# Code example 3 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-calculus"
# ====================================================================

import numpy as np
g = lambda x: 3*x + 1              # inner function
f = lambda u: u**2                # outer function
comp = lambda x: f(g(x))          # y = (3x + 1)^2
x0 = 2.0
# analytic chain rule: dy/dx = f'(g(x)) * g'(x) = 2*(3x+1) * 3
analytic = 2*(3*x0 + 1) * 3
numeric = (comp(x0 + 1e-6) - comp(x0 - 1e-6)) / (2e-6)
print("analytic chain rule:", analytic)          # 42
print("numeric  derivative:", round(numeric, 4))
