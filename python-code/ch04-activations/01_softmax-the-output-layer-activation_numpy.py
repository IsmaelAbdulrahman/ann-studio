# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 4: Activation functions
# Section: Softmax: the output-layer activation
# Code example 1 of 9 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "activations"
# ====================================================================

import numpy as np
def softmax(z):
    z = z - z.max(); e = np.exp(z); return e/e.sum()
z = np.array([2.0, 1.0, 0.1])
p = softmax(z)
J = np.diag(p) - np.outer(p, p)      # Jacobian dp/dz = diag(p) - p p^T
y = np.array([0.0, 1.0, 0.0])        # one-hot target: true class = index 1
dL_dp = -y/p                         # gradient of cross-entropy wrt probabilities
dL_dz = J.T @ dL_dp                  # chain rule through the softmax
print("p          =", np.round(p, 4))
print("row sums(J) =", np.round(J.sum(1), 6))    # each 0: probabilities sum to 1
print("dL/dz via J =", np.round(dL_dz, 4))
print("p - y       =", np.round(p - y, 4))       # the famous fused result matches
