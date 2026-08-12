# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 4: Activation functions
# Section: In this app's engine → Worked example — four activations and their slopes at z = 1
# Code example 6 of 9 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "activations"
# ====================================================================

import numpy as np
from math import erf
z = 1.0
sig = 1/(1+np.exp(-z))
Phi = 0.5*(1+erf(z/np.sqrt(2)))                 # standard-normal CDF  Phi(1)=0.8413
phi = np.exp(-z**2/2)/np.sqrt(2*np.pi)          # standard-normal pdf  phi(1)=0.2420
rows = [("sigmoid", sig,         sig*(1-sig)),
        ("tanh",    np.tanh(z),  1-np.tanh(z)**2),
        ("relu",    max(0.0, z), 1.0),
        ("gelu",    z*Phi,       Phi + z*phi)]
print("activation   g(1)     g'(1)")
for name, g, dg in rows:
    print(f"{name:>9}  {g:6.4f}  {dg:6.4f}")
# expected: sigmoid 0.7311/0.1966  tanh 0.7616/0.4200  relu 1.0000/1.0000  gelu 0.8413/1.0833
