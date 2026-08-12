# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 37: Appendix C · Probability & statistics
# Section: Likelihood: where MSE and cross-entropy come from
# Code example 6 of 8 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-probability"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)
flips = rng.binomial(1, 0.7, size=200)         # true p = 0.7, 200 trials
k, n = int(flips.sum()), len(flips)
print("heads:", k, "of", n, "  sample mean:", round(k/n, 3))
# scan the Bernoulli log-likelihood over candidate p values
grid = np.linspace(0.01, 0.99, 99)
loglik = k*np.log(grid) + (n - k)*np.log(1 - grid)
p_hat = grid[np.argmax(loglik)]
print("MLE p (argmax of log-likelihood):", round(float(p_hat), 3))
print("-> the MLE equals the sample mean")
