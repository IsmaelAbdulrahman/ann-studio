# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 19: Autoencoders & representation learning
# Section: Exercises
# Code example 6 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "autoencoders"
# ====================================================================

import numpy as np
# reconstruction errors from an autoencoder on six new points
err = np.array([0.02, 0.03, 0.015, 0.50, 0.025, 0.90])
tau = 3 * np.median(err)              # threshold at 3x the median normal error
print("threshold     :", round(tau, 4))
print("flagged points:", np.where(err > tau)[0].tolist())
