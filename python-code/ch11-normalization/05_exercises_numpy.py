# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 11: Normalization: batch, layer & beyond
# Section: Exercises
# Code example 5 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "normalization"
# ====================================================================

import numpy as np
np.random.seed(7)
D, true_sd = 4, 2.0
for m in [2, 8, 32, 256]:
    B = np.random.randn(2000, m, D) * true_sd + 3.0   # 2000 batches of size m
    est = B[:, :, 0].mean(1)                           # each batch's estimate of feature-0 mean
    print(f"batch m={m:4d}: std of estimated mean = {est.std():.3f}"
          f"   (1/sqrt(m) law: {true_sd/np.sqrt(m):.3f})")
    
