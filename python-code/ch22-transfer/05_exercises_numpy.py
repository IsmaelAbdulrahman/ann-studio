# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 22: Transfer learning & fine-tuning
# Section: Exercises
# Code example 5 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "transfer"
# ====================================================================

import numpy as np
def softmax_T(z, T):
    z = np.asarray(z) / T
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

z = np.array([4.0, 2.0, 1.0, 0.0])
for T in [1, 2, 5, 100]:
    p = softmax_T(z, T)
    print("T =", T, " max prob =", round(float(p.max()), 4), " probs =", np.round(p, 3))
print("uniform would be", round(1 / len(z), 4))   # 0.25
# T=1 max 0.831 ; T=2 0.579 ; T=5 0.375 ; T=100 0.256  -> approaching 0.25
