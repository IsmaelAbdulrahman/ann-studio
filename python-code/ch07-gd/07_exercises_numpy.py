# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 7: Gradient descent & its variants
# Section: Exercises
# Code example 7 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "gd"
# ====================================================================

import numpy as np
eta0, warm, total = 0.1, 10, 60
def sched(t):
    if t < warm:
        return eta0 * (t + 1) / warm             # linear warmup 0 -> eta0
    p = (t - warm) / (total - warm)              # 0..1 after warmup
    return 0.5 * eta0 * (1 + np.cos(np.pi * p))  # cosine down to 0
for t in [0, 4, 9, 10, 20, 35, 59]:
    print(f"step {t:2d}: eta = {sched(t):.4f}")
# eta climbs 0.01 -> 0.10 by step 9, then cosine-decays toward 0
