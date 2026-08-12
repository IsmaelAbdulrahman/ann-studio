# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 40: Appendix F · PyTorch primer
# Section: Autograd: gradients for free
# Code example 3 of 15 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-pytorch"
# ====================================================================

import numpy as np

def f(x):
    return x**2 + 3*x            # f'(x) = 2x + 3

x0, h = 2.0, 1e-5
fd = (f(x0 + h) - f(x0 - h)) / (2 * h)   # central difference
print("finite-diff f'(2):", round(fd, 5))
print("exact       f'(2):", 2*x0 + 3)     # 7
