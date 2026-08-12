# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 6: Loss functions & maximum likelihood
# Section: The losses in code
# Code example 2 of 8 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "loss"
# ====================================================================

import numpy as np
def softmax(z):
    z = z - z.max()                 # numerical stability (subtract max logit)
    e = np.exp(z)
    return e/e.sum()

z = np.array([1.0, 2.0, 0.5])       # logits for 3 classes
y = np.array([0.0, 1.0, 0.0])       # one-hot: true class is #1
p = softmax(z)
ce = -np.sum(y*np.log(p))
print("p            =", np.round(p, 4))
print("cross-entropy=", round(ce, 4))
print("fused p - y  =", np.round(p - y, 4))

eps = 1e-6; g = np.zeros(3)          # numeric gradient of CE wrt each logit
for i in range(3):
    z2 = z.copy(); z2[i] += eps
    g[i] = (-np.sum(y*np.log(softmax(z2))) - ce)/eps
print("numeric grad =", np.round(g, 4))
