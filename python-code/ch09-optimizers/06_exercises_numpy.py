# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 9: Momentum, RMSProp & Adam
# Section: Exercises
# Code example 6 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "optimizers"
# ====================================================================

import numpy as np
g, eta, eps = 0.4, 0.1, 1e-8
r = 0.0; s = 0.0; rho = 0.9
for t in range(1, 8):
    r += g * g                               # AdaGrad: running SUM (never forgets)
    ada = eta * g / (np.sqrt(r) + eps)
    s = rho * s + (1 - rho) * g * g          # RMSProp: running AVERAGE (forgets)
    rms = eta * g / (np.sqrt(s) + eps)
    print(f"t={t}: AdaGrad {ada:.4f}   RMSProp {rms:.4f}")
    
