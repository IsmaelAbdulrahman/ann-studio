# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 39: Appendix E · Python & NumPy primer
# Section: Exercises
# Code example 21 of 22 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-python"
# ====================================================================

import numpy as np
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
Z = np.array([[1., 2., 3.], [1., 1., 1.]])
P = softmax(Z)
print("rows sum :", P.sum(axis=1))                     # [1. 1.]
print("shift ok :", np.allclose(softmax(Z), softmax(Z + 100)))   # True
