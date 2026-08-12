# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 4: Activation functions
# Section: Exercises
# Code example 9 of 9 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "activations"
# ====================================================================

import numpy as np
from math import erf
erf_v = np.vectorize(erf)
z = np.linspace(-4, 4, 801)
exact  = z*0.5*(1 + erf_v(z/np.sqrt(2)))                       # z * Phi(z)
approx = 0.5*z*(1 + np.tanh(np.sqrt(2/np.pi)*(z + 0.044715*z**3)))
print("max |exact - tanh-approx| =", round(float(np.max(np.abs(exact-approx))), 6))
