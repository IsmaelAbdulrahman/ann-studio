# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 3: Multilayer networks & the forward pass
# Section: The forward pass in code
# Code example 2 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "mlp"
# ====================================================================

import numpy as np
rng = np.random.RandomState(1)
x  = rng.randn(1, 3)
W1 = rng.randn(3, 4); W2 = rng.randn(4, 2)
two_linear = (x @ W1) @ W2                 # two stacked linear layers
one_linear = x @ (W1 @ W2)                 # collapsed to one matrix W1@W2
print("two linear layers vs one matrix, max|diff| = {:.1e}".format(
      float(np.max(np.abs(two_linear - one_linear)))))
nonlin = np.tanh(x @ W1) @ W2              # a nonlinearity between them
print("with tanh inserted, output moves by {:.3f}".format(
      float(np.max(np.abs(nonlin - two_linear)))))
