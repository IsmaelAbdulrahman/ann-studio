# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 39: Appendix E · Python & NumPy primer
# Section: NumPy: the array
# Code example 5 of 22 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-python"
# ====================================================================

import numpy as np

a = np.array([1, 2, 3])                 # from a Python list
b = np.zeros((2, 3))                    # a 2x3 block of zeros
d = np.arange(0, 10, 2)                 # start, stop, step -> 0 2 4 6 8
e = np.linspace(0, 1, 5)               # 5 points evenly from 0 to 1

print("a =", a, " shape", a.shape)
print("b shape:", b.shape, " ndim", b.ndim)
print("arange :", d)
print("linspace:", e)
print("b dtype:", b.dtype)              # float64: zeros/ones are float by default
