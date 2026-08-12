# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 6: Loss functions & maximum likelihood
# Section: In this app's engine → Worked example — categorical cross-entropy and its gradient (3 classes)
# Code example 5 of 8 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "loss"
# ====================================================================

import numpy as np
def softmax(z):
    z = z - z.max()                     # numerical stability (subtract max logit)
    return np.exp(z)/np.exp(z).sum()
z = np.array([1.0, 2.0, 0.5])           # logits for 3 classes
y = np.array([1.0, 0.0, 0.0])           # one-hot: true class is #0
p = softmax(z)
print("p        =", np.round(p, 4))     # [0.2312 0.6285 0.1402]
print("CE       =", round(float(-np.sum(y*np.log(p))), 4))   # 1.4644
print("grad p-y =", np.round(p - y, 4)) # [-0.7688  0.6285  0.1402], sums to 0
