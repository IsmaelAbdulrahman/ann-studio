# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 22: Transfer learning & fine-tuning
# Section: Exercises
# Code example 4 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "transfer"
# ====================================================================

import numpy as np
np.random.seed(0)
d, r, alpha = 512, 4, 8
A = 0.01 * np.random.randn(r, d)          # r x d
B = 0.01 * np.random.randn(d, r)          # d x r
dW = (alpha / r) * (B @ A)                # d x d effective update
print("LoRA params 2*d*r =", 2 * d * r)                       # 4096
print("full params d*d   =", d * d)                            # 262144
print("percent           =", round(100 * 2 * d * r / (d * d), 3), "%")   # 1.562 %
print("rank(B@A)         =", np.linalg.matrix_rank(dW))        # 4  (= r)
print("||dW||_F          =", round(float(np.linalg.norm(dW)), 4))   # small: 0.1981
