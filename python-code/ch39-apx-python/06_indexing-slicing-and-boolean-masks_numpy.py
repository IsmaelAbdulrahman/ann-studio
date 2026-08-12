# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 39: Appendix E · Python & NumPy primer
# Section: Indexing, slicing and boolean masks
# Code example 6 of 22 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-python"
# ====================================================================

import numpy as np

M = np.arange(12).reshape(3, 4)         # 3 rows, 4 columns: 0..11
print(M)
print("element [1,2]:", M[1, 2])        # row 1, col 2 -> 6
print("row 0       :", M[0])            # [0 1 2 3]
print("column 3    :", M[:, 3])         # [3 7 11]
print("top-left 2x2:\n", M[:2, :2])     # rows 0-1, cols 0-1
print("last row    :", M[-1])           # [8 9 10 11]
