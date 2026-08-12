# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 39: Appendix E · Python & NumPy primer
# Section: Random numbers and reproducibility
# Code example 15 of 22 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-python"
# ====================================================================

import numpy as np

rng = np.random.default_rng(0)          # the modern Generator, seeded
print("uniform      :", rng.random(3).round(3))
print("normal       :", rng.standard_normal(3).round(3))
print("integers 0..9:", rng.integers(0, 10, 4))
print("choice       :", rng.choice([10, 20, 30], size=4))

again = np.random.default_rng(0)        # same seed -> identical stream
print("reproduced   :", again.random(3).round(3))
