# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 19: Autoencoders & representation learning
# Section: In this app's engine → Worked example — the KL cost of a small latent
# Code example 4 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "autoencoders"
# ====================================================================

import numpy as np
mu    = np.array([0.5, 0.0, -1.0])     # per-unit posterior means
sigma = np.array([0.8, 1.0,  0.5])     # per-unit posterior std devs
kl_per_unit = 0.5 * (mu**2 + sigma**2 - 1.0 - np.log(sigma**2))
print("per-unit KL :", kl_per_unit.round(4))          # [0.1681 0.     0.8181]
print("total KL    :", round(float(kl_per_unit.sum()), 4), "nats")   # 0.9863
