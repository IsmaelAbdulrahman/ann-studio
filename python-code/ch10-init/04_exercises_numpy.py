# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 10: Initialization & the vanishing gradient
# Section: Exercises
# Code example 4 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "init"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0); n = 256
relu = lambda z: np.maximum(0.0, z)
x = rng.randn(512, n)
for l in range(6):
    W = rng.randn(n, n) * np.sqrt(2.0 / n)      # He initialization
    x = relu(x @ W)
    print(f"layer {l+1}: std = {x.std():.3f}")
    
