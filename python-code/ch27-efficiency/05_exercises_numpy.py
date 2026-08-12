# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 27: Efficiency & deployment
# Section: Exercises
# Code example 5 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "efficiency"
# ====================================================================

import numpy as np
def quantize(w, nbits):
    qmin, qmax = -(2**(nbits-1)), 2**(nbits-1) - 1
    s = (w.max() - w.min()) / (qmax - qmin)
    z = int(np.clip(np.round(qmin - w.min()/s), qmin, qmax))
    q = np.clip(np.round(w/s) + z, qmin, qmax)
    return np.max(np.abs(w - s*(q - z))), s

w = np.array([-1.0, -0.4, 0.0, 0.3, 0.7, 1.5])
for b in (8, 4):
    err, s = quantize(w, b)
    print(f"int{b}: levels={2**b:3d}  scale={s:.5f}  max_abs_err={err:.5f}")
# int8: max_abs_err ~ 0.00392   int4: max_abs_err ~ 0.06667
