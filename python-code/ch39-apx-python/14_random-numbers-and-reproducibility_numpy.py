# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 39: Appendix E · Python & NumPy primer
# Section: Random numbers and reproducibility
# Code example 14 of 22 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-python"
# ====================================================================

import numpy as np

rng = np.random.RandomState(0)          # a seeded generator
print("uniform [0,1):", rng.rand(3))
print("standard normal:", rng.randn(3))
print("integers 0..9 :", rng.randint(0, 10, 4))

# the same seed reproduces the identical stream
again = np.random.RandomState(0)
print("reproduced    :", again.rand(3))
