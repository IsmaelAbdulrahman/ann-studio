# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 4: Activation functions
# Section: Temperature — sharpening and softening softmax
# Code example 2 of 9 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "activations"
# ====================================================================

import numpy as np
def softmax(z, T=1.0):
    z = z/T
    z = z - z.max()                 # subtract max for numerical stability
    e = np.exp(z)
    return e/e.sum()

logits = np.array([2.0, 1.0, 0.1])  # same scores, different temperatures
print("temperature   softmax probabilities        peak")
for T in [0.5, 1.0, 2.0, 5.0]:
    p = softmax(logits, T)
    print(f"   T = {T:<4}  {np.round(p,3)}   {p.max():.3f}")
