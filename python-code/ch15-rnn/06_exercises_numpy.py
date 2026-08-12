# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 15: Recurrent networks & sequences
# Section: Exercises
# Code example 6 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "rnn"
# ====================================================================

import numpy as np
def clip(g, theta):
    n = np.linalg.norm(g)
    return g * (theta / n) if n > theta else g   # rescale only when too big

theta = 1.0
for g in [np.array([3.0, 4.0]), np.array([0.3, 0.4])]:
    c = clip(g, theta)
    print(f"||g||={np.linalg.norm(g):.1f}  ->  clipped ||g||={np.linalg.norm(c):.3f}")
