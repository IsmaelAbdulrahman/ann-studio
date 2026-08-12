# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 10: Initialization & the vanishing gradient
# Section: Initialization in code
# Code example 1 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "init"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)
n, depth = 256, 6
x0 = rng.randn(512, n)                 # a batch of 512 inputs, unit variance
for label, gain in [("too big   Var(w)=4/n  ", np.sqrt(4.0 / n)),
                    ("too small Var(w)=1/4n  ", np.sqrt(0.25 / n)),
                    ("Xavier    Var(w)=1/n   ", np.sqrt(1.0 / n))]:
    x = x0.copy(); stds = []
    for l in range(depth):
        W = rng.randn(n, n) * gain
        x = x @ W                       # LINEAR relay: pure variance flow
        stds.append(round(float(x.std()), 3))
    print(label, "std/layer:", stds)
