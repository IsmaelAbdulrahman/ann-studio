# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 37: Appendix C · Probability & statistics
# Section: Independence and the central limit theorem
# Code example 5 of 8 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-probability"
# ====================================================================

import numpy as np
rng = np.random.RandomState(1)
n, trials = 30, 20000
draws = rng.uniform(0, 1, size=(trials, n))   # each row: 30 Uniform(0,1) draws
means = draws.mean(axis=1)                     # sample means -> ~Gaussian by CLT
print("mean of sample-means =", round(float(means.mean()), 4))   # ~0.5
print("std  of sample-means =", round(float(means.std()), 4))    # ~0.053
print("CLT predicts std     =", round(float(np.sqrt((1/12)/n)), 4))
within = np.mean(np.less(np.abs(means - 0.5), means.std()))
print("fraction within 1 sd =", round(float(within), 3))         # ~0.68 (Gaussian)
