# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 12: Regularization & generalization
# Section: Regularization in code
# Code example 2 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "regularization"
# ====================================================================

import numpy as np
a    = np.array([2.0, -1.0, 4.0, 0.5])   # a hidden layer's activations
p    = 0.5; keep = 1.0 - p                # drop rate p -> keep probability (1-p)
mask = np.array([1., 0., 1., 0.])         # one sampled keep/drop mask

print("scale 1/(1-p) =", 1/keep)
print("kept & scaled =", a * mask / keep)          # survivors amplified -> [4, 0, 8, 0]

# averaged over many random masks the expected activation is unchanged:
rng = np.random.RandomState(0)
M = rng.binomial(1, keep, size=(200000, a.size)) / keep    # random inverted masks
print("E[dropped]    ~", np.round((M * a).mean(0), 3), " vs a =", a)
