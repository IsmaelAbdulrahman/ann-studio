# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 3: Multilayer networks & the forward pass
# Section: Exercises
# Code example 5 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "mlp"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)
dims = [3, 5, 4, 2]                        # in -> hidden -> hidden -> out
Ws = [rng.randn(dims[i], dims[i+1]) * 0.3 for i in range(len(dims) - 1)]
relu = lambda z: np.maximum(0.0, z)
A = rng.randn(8, 3)                        # batch of 8 examples
for i, W in enumerate(Ws):
    A = A @ W                              # biases taken as zero here
    if i < len(Ws) - 1:                    # ReLU on hidden layers only
        A = relu(A)
params = sum(dims[i]*dims[i+1] + dims[i+1] for i in range(len(dims) - 1))
print("output shape:", A.shape, " total parameters:", params)
