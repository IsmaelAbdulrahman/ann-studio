# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 37: Appendix C · Probability & statistics
# Section: Expectation, mean, variance, standard deviation → Worked example — mean and variance of a small sample
# Code example 2 of 8 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-probability"
# ====================================================================

import numpy as np
data = np.array([2., 4., 4., 4., 5., 5., 7., 9.])
mu = data.mean()
var = data.var()                      # population variance (divides by N)
print("mean     =", mu)               # 5.0
print("variance =", var)              # 4.0
print("std      =", np.sqrt(var))     # 2.0
print("E[X^2]-mu^2 =", (data**2).mean() - mu**2)   # same 4.0
# sample estimates converge to the truth as n grows (law of large numbers)
rng = np.random.RandomState(0)
big = rng.normal(10.0, 3.0, size=100000)           # true mean 10, std 3
print("est mean =", round(big.mean(), 3), " est std =", round(big.std(), 3))
