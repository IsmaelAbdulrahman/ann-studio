# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 6: Loss functions & maximum likelihood
# Section: Exercises
# Code example 7 of 8 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "loss"
# ====================================================================

import numpy as np
def huber(r, d=1.0):
    a = np.abs(r)
    return np.where(a <= d, 0.5*r**2, d*(a - 0.5*d))
r = np.array([0.5, 1.0, 2.0])
print("Huber(r) =", huber(r))
print("at r=1: quad 0.5*1^2 =", 0.5*1**2, " linear 1*(1-0.5) =", 1*(1-0.5))
