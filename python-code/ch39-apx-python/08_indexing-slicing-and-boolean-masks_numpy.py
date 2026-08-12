# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 39: Appendix E · Python & NumPy primer
# Section: Indexing, slicing and boolean masks
# Code example 8 of 22 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-python"
# ====================================================================

import numpy as np

v = np.array([10, 20, 30, 40])
idx = np.array([3, 0, 0, 2])            # reorder and repeat freely
print("reorder :", v[idx])             # [40 10 10 30]

scores = np.array([[2.0, 1.0, 0.1],
                   [0.5, 2.5, 0.3],
                   [1.2, 0.7, 3.1]])
rows   = np.arange(3)                   # [0 1 2]
labels = np.array([0, 1, 2])           # the "correct" column of each row
print("gather  :", scores[rows, labels])   # one per row -> [2.0 2.5 3.1]
