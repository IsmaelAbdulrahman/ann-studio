# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 27: Efficiency & deployment
# Section: Efficiency in code
# Code example 1 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "efficiency"
# ====================================================================

import numpy as np
np.set_printoptions(suppress=True)

w = np.array([-1.0, -0.4, 0.0, 0.3, 0.7, 1.5])   # a small fp32 weight vector
qmin, qmax = -128, 127                             # signed int8 grid (256 codes)

s = (w.max() - w.min()) / (qmax - qmin)            # scale = range / 255
z = int(np.clip(np.round(qmin - w.min()/s), qmin, qmax))   # integer zero-point

q   = np.clip(np.round(w/s) + z, qmin, qmax).astype(np.int8)   # quantize
deq = s * (q.astype(np.float64) - z)                          # dequantize

print("scale       =", round(s, 6))          # 0.009804
print("zero_point  =", z)                     # -26
print("int8 codes  =", q)                     # [-128 -67 -26  5  45 127]
print("dequantized =", deq.round(4))          # close to w, snapped to the grid
print("max abs err =", round(np.max(np.abs(w - deq)), 6))   # 0.003922  (<= s/2)
print("compression =", 32 // 8, "x  (fp32 -> int8)")        # 4 x
