# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 1: From biological to artificial neurons
# Section: Exercises
# Code example 4 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "neuron"
# ====================================================================

import numpy as np
z = np.linspace(-6, 6, 13)
s = 1.0 / (1.0 + np.exp(-z))
ds = s * (1.0 - s)                 # sigmoid derivative
for zi, di in zip(z, ds):
    print(f"z={zi:+.1f}  s'={di:.3f}")
print("max slope", round(ds.max(), 3), "at z =", z[ds.argmax()])
