# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 21: Generative models: GANs to diffusion
# Section: Exercises
# Code example 6 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "generative"
# ====================================================================

import numpy as np
eps_uncond = np.array([0.20, -0.50, 0.10])   # noise predicted with no prompt
eps_cond   = np.array([0.35, -0.20, 0.40])   # noise predicted given the prompt c
for w in (0.0, 1.0, 3.0):
    eps_tilde = eps_uncond + w * (eps_cond - eps_uncond)
    print(f"w={w:.0f}:  {eps_tilde.round(3)}")
# w=0 -> unconditional; w=1 -> conditional; w=3 -> extrapolated beyond the prompt
