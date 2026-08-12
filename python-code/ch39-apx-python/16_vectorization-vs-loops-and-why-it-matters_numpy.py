# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 39: Appendix E · Python & NumPy primer
# Section: Vectorization vs loops (and why it matters)
# Code example 16 of 22 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-python"
# ====================================================================

import numpy as np, time

n = 1000000
xs = np.random.RandomState(0).rand(n)

t0 = time.perf_counter()
total = 0.0
for v in xs:                    # pure-Python loop
    total += v * v
t1 = time.perf_counter()

t2 = time.perf_counter()
total_vec = np.sum(xs * xs)     # vectorized: one compiled call
t3 = time.perf_counter()

print("sum of squares :", round(float(total_vec), 1))
print("results match  :", bool(np.isclose(total, total_vec)))
print("loop time  (s) :", round(t1 - t0, 4))
print("vector time(s) :", round(t3 - t2, 4))
print("speedup    (x) :", round((t1 - t0) / (t3 - t2), 1))
