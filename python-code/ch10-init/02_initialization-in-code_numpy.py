# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 10: Initialization & the vanishing gradient
# Section: Initialization in code
# Code example 2 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "init"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)
n = 256
def relay(gain, act):                  # 6-layer relay -> per-layer std, final sat%
    x = rng.randn(512, n); stds = []
    for l in range(6):
        x = act(x @ (rng.randn(n, n) * gain))
        stds.append(round(float(x.std()), 2))
    sat = 100 * np.mean(np.abs(x) > 0.99)
    return stds, sat
relu = lambda z: np.maximum(0.0, z)
s, sat = relay(1.0, np.tanh)
print("tanh naive std=1 :", s, f" saturated={sat:.0f}%")   # frozen at the rails
s, sat = relay(np.sqrt(1.0 / n), np.tanh)
print("tanh Xavier 1/n  :", s, f" saturated={sat:.0f}%")   # healthy mid-range
print("ReLU naive std=1 :", relay(1.0, relu)[0])            # explodes
print("ReLU He   2/n    :", relay(np.sqrt(2.0 / n), relu)[0])
