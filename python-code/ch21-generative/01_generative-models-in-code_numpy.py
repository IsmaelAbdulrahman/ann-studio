# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 21: Generative models: GANs to diffusion
# Section: Generative models in code
# Code example 1 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "generative"
# ====================================================================

import numpy as np

x0 = 2.0
betas = [0.2, 0.5]
eps   = [0.3, -1.1]

x = x0
for step, (b, e) in enumerate(zip(betas, eps), start=1):
    x = np.sqrt(1 - b) * x + np.sqrt(b) * e     # one forward noising step
    print(f"step {step}: x_{step} = {x:.4f}")

abar = np.prod([1 - b for b in betas])          # cumulative kept signal
print(f"abar_2                = {abar:.4f}")
print(f"kept  sqrt(abar)      = {np.sqrt(abar):.4f}")
print(f"noise sqrt(1 - abar)  = {np.sqrt(1 - abar):.4f}")
eps_eq = (x - np.sqrt(abar) * x0) / np.sqrt(1 - abar)
print(f"equivalent single eps = {eps_eq:.4f}")
