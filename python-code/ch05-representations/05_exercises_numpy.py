# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 5: Features, embeddings & what a network learns
# Section: Exercises
# Code example 5 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "representations"
# ====================================================================

import numpy as np
np.random.seed(0)
vocab = ["king", "queen", "man", "woman", "prince", "apple"]
V = np.array([[0.9, 0.9, 1.0, 0.0],   # king
              [0.9,-0.9, 1.0, 0.0],   # queen
              [0.1, 0.8, 1.0, 0.0],   # man
              [0.1,-0.8, 1.0, 0.0],   # woman
              [0.7, 0.7, 1.0, 0.0],   # prince
              [0.0, 0.0, 0.0, 1.0]])  # apple
def vec(w): return V[vocab.index(w)]
q   = vec("king") - vec("man") + vec("woman")           # analogy vector
cos = (V @ q) / (np.linalg.norm(V, axis=1) * np.linalg.norm(q))
for w in ["king", "man", "woman"]:                      # exclude the query words
    cos[vocab.index(w)] = -np.inf
best = int(np.argmax(cos))
print("king - man + woman  ->", vocab[best], " cos =", round(float(cos[best]), 3))
print("runner-up cos       =", round(float(np.sort(cos)[-2]), 3))
