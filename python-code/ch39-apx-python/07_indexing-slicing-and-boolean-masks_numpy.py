# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 39: Appendix E · Python & NumPy primer
# Section: Indexing, slicing and boolean masks
# Code example 7 of 22 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-python"
# ====================================================================

import numpy as np

x = np.array([-2.0, 0.5, -1.0, 3.0, -0.2, 4.0])
mask = x > 0                            # a boolean array
print("mask     :", mask)
print("positives:", x[mask])           # keep where True
print("count > 0:", mask.sum())        # True counts as 1

x[x < 0] = 0.0                         # ReLU: clamp negatives to zero
print("relu(x)  :", x)
