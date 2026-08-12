# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 3: Multilayer networks & the forward pass
# Section: The forward pass in code
# Code example 1 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "mlp"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)
# a 3 -> 4 -> 2 MLP applied to a batch of 5 examples
X  = rng.randn(5, 3)
W1 = rng.randn(3, 4) * 0.5; b1 = np.zeros(4)
W2 = rng.randn(4, 2) * 0.5; b2 = np.zeros(2)
relu = lambda z: np.maximum(0.0, z)
Z1 = X @ W1 + b1;  A1 = relu(Z1)       # hidden layer
Z2 = A1 @ W2 + b2                       # output logits
print("shapes:  X", X.shape, "-> A1", A1.shape, "-> out", Z2.shape)
print("output row 0:", np.round(Z2[0], 3))
