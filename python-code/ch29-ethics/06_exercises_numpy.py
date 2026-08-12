# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 29: Ethics, fairness & safety
# Section: Exercises
# Code example 6 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "ethics"
# ====================================================================

import numpy as np
np.random.seed(0)
for d in [10, 100, 1000]:
    w = np.random.randn(d)
    x = np.random.randn(d); x = 2.2 * x / (w @ x)   # fix the clean logit at +2.2
    l1 = np.abs(w).sum()
    print("d=%4d  ||w||_1=%8.2f  eps to flip = %.4f" % (d, l1, 2.2/l1))
# eps to flip: ~0.22 (d=10) -> ~0.025 (d=100) -> ~0.003 (d=1000)
    
