# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 35: Appendix A · Linear algebra refresher
# Section: Norms: measuring the size of a vector
# Code example 6 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-linalg"
# ====================================================================

import numpy as np
v = np.array([3., -4.])
print("L1 =", np.sum(np.abs(v)))          # 3 + 4 = 7
print("L2 =", np.sqrt(np.sum(v**2)))      # sqrt(9 + 16) = 5
print("np.linalg.norm(v, 1) =", np.linalg.norm(v, 1))
print("np.linalg.norm(v, 2) =", np.linalg.norm(v, 2))
# L2 length of each ROW of a matrix (per-sample sizes)
M = np.array([[3., 4.], [5., 12.]])
print("row L2 norms:", np.linalg.norm(M, axis=1))   # [5, 13]
