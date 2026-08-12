# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 5: Features, embeddings & what a network learns
# Section: Exercises
# Code example 4 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "representations"
# ====================================================================

import numpy as np
np.random.seed(0)
E = np.array([[3., 0., 1.], [3., 0., 2.], [0., 3., 2.], [0., 3., 3.]])
idx = np.array([2, 0, 3, 0, 1])            # five tokens (a "sentence")
gather = E[idx]                            # direct row gather, shape (5, 3)
OH     = np.eye(4)[idx]                    # one-hot rows, shape (5, 4)
matmul = OH @ E                            # one-hot . E
print("gather shape:", gather.shape, " matmul shape:", matmul.shape)
print("identical:", np.allclose(gather, matmul))     # expected: True
