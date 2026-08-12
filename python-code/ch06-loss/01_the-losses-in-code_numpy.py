# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 6: Loss functions & maximum likelihood
# Section: The losses in code
# Code example 1 of 8 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "loss"
# ====================================================================

import numpy as np
y    = np.array([3.0, -1.0, 2.0, 0.5])     # targets
yhat = np.array([2.5, -0.2, 2.0, 1.5])     # predictions
m = len(y)

def mse(p, t): return 0.5*np.mean((p-t)**2)
grad = (yhat - y)/m                          # analytic dL/dyhat

eps = 1e-6                                    # finite-difference check on yhat[0]
yh2 = yhat.copy(); yh2[0] += eps
num0 = (mse(yh2, y) - mse(yhat, y))/eps

print("MSE            =", round(mse(yhat, y), 4))
print("analytic grad  =", np.round(grad, 4))
print("numeric grad[0]=", round(num0, 4), " (matches", round(grad[0], 4), ")")
