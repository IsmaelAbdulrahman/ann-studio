# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 37: Appendix C · Probability & statistics
# Section: Exercises
# Code example 8 of 8 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-probability"
# ====================================================================

import numpy as np
rng = np.random.RandomState(2)
s = rng.normal(5.0, 2.0, size=50000)      # true mean 5, true std 2
print("est mean =", round(s.mean(), 3))   # ~5
print("est std  =", round(s.std(), 3))    # ~2
print("est var  =", round(s.var(), 3))    # ~4
