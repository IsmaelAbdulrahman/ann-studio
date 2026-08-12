# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 8: Backpropagation
# Section: Exercises
# Code example 5 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "backprop"
# ====================================================================

import numpy as np
z    = np.array([-1.0, 2.0, 0.5, -3.0])
abar = np.array([0.4, -0.2, 0.7, 0.1])      # adjoint arriving from above
relu = lambda z: np.maximum(0.0, z)
zbar = abar * (z > 0)                        # VJP: gate the adjoint
print("zbar =", zbar)                        # [ 0.  -0.2  0.7  0. ]
# check coordinate 2 against a central difference of L = abar . relu(z)
e = 1e-6; i = 2
zp = z.copy(); zp[i] += e; zm = z.copy(); zm[i] -= e
num = (abar @ relu(zp) - abar @ relu(zm)) / (2 * e)
print("coord 2: analytic", zbar[i], " numeric", round(float(num), 6))  # 0.7  0.7
    
