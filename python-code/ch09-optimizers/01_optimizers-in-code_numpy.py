# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 9: Momentum, RMSProp & Adam
# Section: Optimizers in code
# Code example 1 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "optimizers"
# ====================================================================

import numpy as np
g, eta = 0.4, 0.1                     # constant gradient, learning rate
v = s = m = vv = 0.0                   # every optimizer's state starts at 0
b1, b2, rho, eps = 0.9, 0.999, 0.9, 1e-8
print(" t |   SGD   Moment  RMSProp   Adam")
for t in range(1, 6):
    sgd = eta * g
    v   = 0.9 * v + g;                 mom = eta * v
    s   = rho * s + (1 - rho) * g * g; rms = eta * g / (np.sqrt(s) + eps)
    m   = b1 * m + (1 - b1) * g
    vv  = b2 * vv + (1 - b2) * g * g
    mh  = m / (1 - b1 ** t);  vh = vv / (1 - b2 ** t)
    adam = eta * mh / (np.sqrt(vh) + eps)
    print(f"{t:2d} | {sgd:.4f}  {mom:.4f}  {rms:.4f}  {adam:.4f}")
