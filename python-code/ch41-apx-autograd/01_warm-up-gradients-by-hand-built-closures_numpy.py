# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 41: Appendix G · Autograd from scratch
# Section: Warm-up: gradients by hand-built closures
# Code example 1 of 4 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-autograd"
# ====================================================================

import numpy as np

# f = a*b + a, differentiated by hand-built "backward" steps.
a, b = 3.0, 4.0
p = a * b                 # p = a*b
f = p + a                 # f = p + a   (a fans out: used in p AND here)

# adjoints: seed df/df = 1, then push gradient back through each node
f_bar = 1.0
p_bar = f_bar * 1.0                 # node f = p + a:  df/dp = 1
a_bar = f_bar * 1.0                 # node f = p + a:  df/da = 1  (first path)
a_bar += p_bar * b                  # node p = a*b:    a FANS OUT, so accumulate (+=)
b_bar  = p_bar * a                  # node p = a*b:    dp/db = a

print("df/da =", a_bar, " (expect b+1 =", b + 1, ")")   # 5.0
print("df/db =", b_bar, " (expect a   =", a, ")")       # 3.0

# finite-difference sanity check on df/da (central difference)
h = 1e-6
fd = ((a + h) * b + (a + h) - ((a - h) * b + (a - h))) / (2 * h)
print("finite-diff df/da =", round(fd, 5))              # 5.0
