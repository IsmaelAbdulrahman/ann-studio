# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 9: Momentum, RMSProp & Adam
# Section: Exercises
# Code example 5 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "optimizers"
# ====================================================================

import numpy as np
g, eta, lam = 0.4, 0.1, 0.01                    # gradient, lr, weight decay
theta = 0.5; m = v = 0.0
for t in range(1, 4):
    m = 0.9 * m + 0.1 * g
    v = 0.999 * v + 0.001 * g * g
    step  = eta * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    decay = eta * lam * theta                   # decoupled: uses current theta
    theta = theta - step - decay                # AdamW update
    print(f"t={t}: data step={step:.4f}, decay={decay:.5f}, theta={theta:.4f}")
    
