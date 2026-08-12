# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 39: Appendix E · Python & NumPy primer
# Section: Exercises
# Code example 19 of 22 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-python"
# ====================================================================

import numpy as np
X = np.array([[1., 2.], [3., 6.], [5., 10.]])
Xs = (X - X.mean(axis=0)) / X.std(axis=0)
print(Xs)
print("means:", Xs.mean(axis=0).round(6), " stds:", Xs.std(axis=0).round(6))
