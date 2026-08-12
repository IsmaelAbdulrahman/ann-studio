# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 4: Activation functions
# Section: Exercises
# Code example 8 of 9 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "activations"
# ====================================================================

import numpy as np
def softmax(z, T):
    z = z/T; z = z - z.max(); e = np.exp(z); return e/e.sum()
logits = np.array([2.0, 1.0, 0.1])
print("T=0.01 (cold) ->", np.round(softmax(logits, 0.01), 3))   # approaches argmax
print("T=100  (hot)  ->", np.round(softmax(logits, 100), 3))    # approaches uniform
