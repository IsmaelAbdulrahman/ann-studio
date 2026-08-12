# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 1: From biological to artificial neurons
# Section: The neuron in code
# Code example 2 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "neuron"
# ====================================================================

import numpy as np

def neuron(x, w, b, activation=lambda z: 1/(1+np.exp(-z))):
    z = np.dot(w, x) + b         # weighted sum + bias
    return z, activation(z)       # pre-activation, activation

# the spam detector from the worked example
x = np.array([4, 2, 1])           # CAPS words, links, known-sender?
w = np.array([0.8, 1.1, -3.0])    # learned weights
b = -2.5                          # learned bias

z, a = neuron(x, w, b)
print(f"z = {z:.3f}   (evidence)")
print(f"a = {a:.3f}   -> {a*100:.1f}% spam")

# counterfactual: an unknown sender flips the verdict
_, a2 = neuron(np.array([4, 2, 0]), w, b)
print(f"unknown sender -> {a2*100:.1f}% spam")
