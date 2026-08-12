# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 26: Data-centric deep learning
# Section: Exercises
# Code example 4 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "data"
# ====================================================================

import numpy as np
counts = np.array([900, 100])
N, K = counts.sum(), len(counts)
w = N / (K * counts)                 # w_c = N / (K * n_c)
mass = counts * w                    # total weight each class puts on the loss
print("weights =", w.round(4))       # [0.5556 5.    ]
print("mass    =", mass.round(1))    # [500. 500.] -> equal -> balanced
    
