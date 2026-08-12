# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 21: Generative models: GANs to diffusion
# Section: Exercises
# Code example 5 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "generative"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)
def gauss(x, mu, s=1.0):
    return np.exp(-(x - mu) ** 2 / (2 * s * s)) / np.sqrt(2 * np.pi * s * s)
def V(gmu, n=100000):
    r = rng.normal(3, 1, n); f = rng.normal(gmu, 1, n)
    Dr = gauss(r, 3) / (gauss(r, 3) + gauss(r, gmu))
    Df = gauss(f, 3) / (gauss(f, 3) + gauss(f, gmu))
    return np.log(Dr).mean() + np.log(1 - Df).mean()
print("mismatch gmu=0:", round(V(0.0), 3))
print("match    gmu=3:", round(V(3.0), 3), " ~ 2 ln 0.5 =", round(2 * np.log(0.5), 3))
