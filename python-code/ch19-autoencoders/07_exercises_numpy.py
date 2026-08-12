# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 19: Autoencoders & representation learning
# Section: Exercises
# Code example 7 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "autoencoders"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)
mu    = np.array([0.5, -1.0, 2.0])
sigma = np.array([0.8,  1.0, 0.5])
eps = rng.randn(200000, 3)              # external noise, no parameters
z   = mu + sigma * eps                  # reparameterized samples
print("empirical mean:", z.mean(0).round(3))   # ~ [ 0.5 -1.   2. ]
print("empirical std :", z.std(0).round(3))    # ~ [ 0.8  1.   0.5]
