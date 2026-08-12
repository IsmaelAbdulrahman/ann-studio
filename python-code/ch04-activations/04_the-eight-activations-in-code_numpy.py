# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 4: Activation functions
# Section: The eight activations in code
# Code example 4 of 9 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "activations"
# ====================================================================

import numpy as np
from math import erf
erf_v = np.vectorize(erf)

def sigmoid(z): return 1/(1+np.exp(-z))
z = np.array([-2.0, 0.0, 2.0])
s = sigmoid(z)
phi = np.exp(-z**2/2)/np.sqrt(2*np.pi)         # standard-normal pdf
Phi = 0.5*(1+erf_v(z/np.sqrt(2)))              # standard-normal cdf

d = [("sigmoid'", s*(1-s)),          ("tanh'", 1-np.tanh(z)**2),
     ("relu'",    (z>0).astype(float)), ("leaky'", np.where(z>0, 1.0, 0.01)),
     ("elu'",     np.where(z>0, 1.0, np.exp(z))), ("softplus'", s),
     ("gelu'",    Phi + z*phi),       ("swish'", s + z*s*(1-s))]
print("             z = -2      0     +2")
for name, g in d:
    g = np.round(g, 3)
    print(f"{name:>10}: {g[0]:7.3f} {g[1]:6.3f} {g[2]:6.3f}")
