# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 39: Appendix E · Python & NumPy primer
# Section: Exercises
# Code example 20 of 22 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-python"
# ====================================================================

import numpy as np
xs = np.array([3, 1, 4, 1, 5, 9])
acc, out = 0, []
for v in xs:
    acc += v
    out.append(acc)
print("loop  :", np.array(out))
print("cumsum:", np.cumsum(xs))
print("equal :", np.array_equal(np.array(out), np.cumsum(xs)))
