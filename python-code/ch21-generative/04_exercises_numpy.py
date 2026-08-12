# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 21: Generative models: GANs to diffusion
# Section: Exercises
# Code example 4 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "generative"
# ====================================================================

import numpy as np
betas = np.linspace(0.01, 0.2, 10)   # a 10-step increasing noise schedule
abar = np.cumprod(1 - betas)
for t in range(10):
    print(f"t={t+1:2d}  abar={abar[t]:.3f}  kept sqrt(abar)={np.sqrt(abar[t]):.3f}")
