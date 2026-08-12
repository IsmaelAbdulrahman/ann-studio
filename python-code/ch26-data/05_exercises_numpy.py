# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 26: Data-centric deep learning
# Section: Exercises
# Code example 5 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "data"
# ====================================================================

import numpy as np
pt = np.array([0.9, 0.5, 0.1])
for g in [0, 1, 2, 5]:
    print(f"gamma={g}: (1-pt)^g at pt=0.9,0.5,0.1 =", ((1 - pt) ** g).round(5))
# gamma=0 -> all 1 (plain cross-entropy); gamma=5 -> easy example ~1e-5, essentially ignored
    
