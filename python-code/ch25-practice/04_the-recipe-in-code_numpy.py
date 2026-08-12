# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 25: Training in practice: the complete recipe
# Section: The recipe in code
# Code example 4 of 8 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "practice"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)

# two features on very different scales
Xtrain = np.column_stack([rng.normal(50, 10, 8), rng.normal(0.5, 0.1, 8)])
Xtest  = np.column_stack([rng.normal(50, 10, 4), rng.normal(0.5, 0.1, 4)])

mu = Xtrain.mean(axis=0)          # statistics from TRAIN ONLY
sd = Xtrain.std(axis=0)
Xtr = (Xtrain - mu) / sd
Xte = (Xtest  - mu) / sd          # reuse train mu, sd -- no peeking at test

print("train mean after scaling:", Xtr.mean(0).round(3))
print("train std  after scaling:", Xtr.std(0).round(3))
print("test  mean after scaling:", Xte.mean(0).round(3))
print("test  std  after scaling:", Xte.std(0).round(3))
