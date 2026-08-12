# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 37: Appendix C · Probability & statistics
# Section: Sampling and the law of large numbers
# Code example 4 of 8 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-probability"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)
x = rng.normal(0.0, 1.0, size=10000)       # 10k standard-normal samples
edges = np.arange(-4, 5)                     # integer bin edges -4..4
counts, _ = np.histogram(x, bins=edges)
for lo, hi, c in zip(edges[:-1], edges[1:], counts):
    print(f"[{int(lo):+d}, {int(hi):+d})  {int(c):5d}  " + "#" * (int(c) // 100))
print("total counted:", int(counts.sum()))
