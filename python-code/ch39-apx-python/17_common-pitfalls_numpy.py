# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 39: Appendix E · Python & NumPy primer
# Section: Common pitfalls
# Code example 17 of 22 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-python"
# ====================================================================

import numpy as np

# 1) integer arrays truncate on assignment (no automatic upcast in place)
a = np.array([1, 2, 3])                 # dtype int64
a[0] = 3.9                             # 3.9 is TRUNCATED to 3, not rounded
print("int assign  :", a)              # [3 2 3]

# 2) float equality is unreliable; compare with a tolerance
print("== on floats:", 0.1 + 0.2 == 0.3)            # False!
print("np.isclose  :", np.isclose(0.1 + 0.2, 0.3))  # True

# 3) a 1-D vector is neither row nor column: .T is a no-op on it
u = np.array([1., 2., 3.])
print("u.T no-op   :", np.array_equal(u + u, u + u.T))    # True
print("real outer  :", (u[:, None] + u[None, :]).shape)   # (3, 3), as intended
