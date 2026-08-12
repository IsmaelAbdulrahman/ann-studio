# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 25: Training in practice: the complete recipe
# Section: Exercises
# Code example 7 of 8 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "practice"
# ====================================================================

import numpy as np
Xtr = np.array([[2., 100.], [4., 200.], [6., 300.], [8., 400.]])
mu = Xtr.mean(0)
sd = Xtr.std(0)
Xte = np.array([[10., 500.]])            # a new test point
print("train mu:", mu, " train sd:", sd.round(3))
print("scaled test point:", ((Xte - mu) / sd).round(3))
