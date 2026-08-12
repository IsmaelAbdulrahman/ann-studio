# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 29: Ethics, fairness & safety
# Section: Fairness and attacks in code
# Code example 2 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "ethics"
# ====================================================================

import numpy as np
np.random.seed(0)
sig = lambda z: 1/(1+np.exp(-z))

# a toy logistic-regression classifier   p = sigmoid(w . x + b)   on d features
d = 100
w = np.random.randn(d)                      # weights
b = 0.0
x = np.random.randn(d)
x = 2.20 * x / (w @ x)                       # scale x so the clean logit is +2.20
y = 1.0                                      # true label (matches the clean prediction)

p0 = sig(w @ x + b)
print("clean : logit = %+.3f   p(class 1) = %.4f  -> class %d" % (w@x+b, p0, p0 > 0.5))

# FGSM.  For binary cross-entropy dL/dx = (p - y)*w, so step x to RAISE the loss:
eps  = 0.06                                  # tiny per-feature budget (features ~0.43)
grad = (p0 - y) * w                          # gradient of the loss wrt the input
xadv = x + eps * np.sign(grad)               # x' = x + eps * sign(grad_x L)
p1   = sig(w @ xadv + b)
print("adv   : logit = %+.3f   p(class 1) = %.4f  -> class %d" % (w@xadv+b, p1, p1 > 0.5))
print("logit shift = %+.3f  ( = -eps*||w||_1 = %+.3f )" %
      (w@xadv - w@x, -eps*np.abs(w).sum()))
# clean class 1 @ 90.0%  ->  adversarial class 0 @ 93.5%   from an invisible eps=0.06
