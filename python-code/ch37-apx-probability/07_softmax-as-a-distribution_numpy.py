# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 37: Appendix C · Probability & statistics
# Section: Softmax as a distribution
# Code example 7 of 8 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-probability"
# ====================================================================

import numpy as np
def softmax(z):
    z = z - z.max()                 # shift-invariant: subtract max for safety
    e = np.exp(z)
    return e / e.sum()
z = np.array([2.0, 1.0, 0.1])
p = softmax(z)
print("logits :", z)
print("softmax:", np.round(p, 4))       # positive, sums to 1, ordered like z
print("sums to:", round(float(p.sum()), 6))
# huge logits would overflow exp() without the max-subtraction; here they don't
print("big logits ->", np.round(softmax(np.array([1000., 1001., 1002.])), 4))
