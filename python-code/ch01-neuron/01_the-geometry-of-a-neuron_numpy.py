# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 1: From biological to artificial neurons
# Section: The geometry of a neuron
# Code example 1 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "neuron"
# ====================================================================

import numpy as np
# One neuron's boundary is the hyperplane  w.x + b = 0.
# The signed distance of a point to it is (w.x + b)/||w||.
w = np.array([0.8, 1.1, -3.0]); b = -2.5
pts = np.array([[4, 2, 1],      # borderline spam
                [4, 2, 0],      # unknown sender
                [0, 0, 1]])     # clean note from a contact
z = pts @ w + b                 # pre-activations (signed evidence)
dist = z / np.linalg.norm(w)    # signed distance to the boundary
for p, zi, di in zip(pts, z, dist):
    side = "spam" if zi >= 0 else "ham "
    print(f"x={p}  z={zi:+.2f}  dist={di:+.2f}  -> {side}")
